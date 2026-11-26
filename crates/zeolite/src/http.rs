//! HTTP gateway: deployment, replicated execution, webhooks and micro-VM boots.

// Handlers return early with a ready-made `Response`. Boxing it would add an allocation to
// every error path in exchange for nothing, since these functions are never hot.
#![allow(clippy::result_large_err)]

use crate::node::{Node, ProposeFailure, TransferFailure};
use crate::state::{Command, Output, VmImage, Workload, valid_name};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State as AxumState};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use hmac::{Hmac, KeyInit, Mac};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Semaphore;
use zeolite_faas::{Function, Limits as FaasLimits, Runtime};
use zeolite_vmm::{Payload, VmConfig, VmError};

/// Largest accepted request body, which bounds a deployed module or a webhook payload.
pub const MAX_BODY_BYTES: usize = 32 << 20;
const MAX_CONCURRENT_PER_TENANT: usize = 32;
const MAX_VM_MEMORY_BYTES: usize = 512 << 20;

pub struct Gateway {
    pub node: Arc<Node>,
    pub runtime: Arc<Runtime>,
    /// SHA-256 of the admin token. When set, mutating endpoints require it.
    pub admin_token_hash: Option<[u8; 32]>,
    /// HTTP address of each peer, for telling clients where the leader is.
    pub peer_http: BTreeMap<u64, String>,
    pub limits: FaasLimits,
    compiled: Mutex<HashMap<[u8; 32], Arc<Function>>>,
    tenant_slots: Mutex<HashMap<String, Arc<Semaphore>>>,
    wasm_invocations: AtomicU64,
    wasm_fuel_used: AtomicU64,
    vm_boots: AtomicU64,
    zel_queries: AtomicU64,
}

impl Gateway {
    pub fn new(
        node: Arc<Node>,
        runtime: Arc<Runtime>,
        admin_token: Option<&[u8]>,
        peer_http: BTreeMap<u64, String>,
    ) -> Arc<Gateway> {
        Arc::new(Gateway {
            node,
            runtime,
            admin_token_hash: admin_token.map(|t| Sha256::digest(t).into()),
            peer_http,
            limits: FaasLimits::default(),
            compiled: Mutex::new(HashMap::new()),
            tenant_slots: Mutex::new(HashMap::new()),
            wasm_invocations: AtomicU64::new(0),
            wasm_fuel_used: AtomicU64::new(0),
            vm_boots: AtomicU64::new(0),
            zel_queries: AtomicU64::new(0),
        })
    }

    fn slots(&self, tenant: &str) -> Arc<Semaphore> {
        let mut slots = self.tenant_slots.lock().unwrap_or_else(|e| e.into_inner());
        slots
            .entry(tenant.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(MAX_CONCURRENT_PER_TENANT)))
            .clone()
    }

    /// Compiles on first use. The cache key is the content hash, so a redeployed function
    /// can never run stale code.
    fn compile(&self, wasm: &[u8]) -> Result<Arc<Function>, String> {
        let key: [u8; 32] = Sha256::digest(wasm).into();
        if let Some(function) = self
            .compiled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Ok(function.clone());
        }
        let function = Arc::new(self.runtime.deploy(wasm).map_err(|e| e.to_string())?);
        let mut cache = self.compiled.lock().unwrap_or_else(|e| e.into_inner());
        // Bounded by replacement, so that a stream of deployments cannot grow it forever.
        if cache.len() >= 1024 {
            cache.clear();
        }
        cache.insert(key, function.clone());
        Ok(function)
    }
}

pub fn router(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/metrics", get(metrics))
        .route("/v1/status", get(status))
        .route("/v1/digest", get(digest))
        .route("/v1/leadership/transfer", post(transfer_leadership))
        .route("/v1/functions", get(list_workloads))
        .route(
            "/v1/functions/{tenant}/{name}",
            get(get_workload).put(deploy).delete(remove),
        )
        .route("/v1/functions/{tenant}/{name}/invoke", post(invoke))
        .route(
            "/v1/functions/{tenant}/{name}/execute/{function}",
            post(execute),
        )
        .route(
            "/v1/functions/{tenant}/{name}/query/{function}",
            post(query_zel),
        )
        .route("/v1/functions/{tenant}/{name}/boot", post(boot))
        .route("/hooks/{tenant}/{name}", post(webhook))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(gateway)
}

fn error(code: StatusCode, message: impl std::fmt::Display) -> Response {
    (code, axum::Json(json!({ "error": message.to_string() }))).into_response()
}

fn require_admin(gateway: &Gateway, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = &gateway.admin_token_hash else {
        return Ok(());
    };
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let hash: [u8; 32] = Sha256::digest(presented.unwrap_or("").as_bytes()).into();
    // Compares fixed-size digests without an early exit, so timing reveals nothing.
    let difference = hash
        .iter()
        .zip(expected.iter())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b));
    if presented.is_none() || difference != 0 {
        return Err(error(StatusCode::UNAUTHORIZED, "missing or invalid token"));
    }
    Ok(())
}

fn check_names(tenant: &str, name: &str) -> Result<(), Response> {
    if valid_name(tenant) && valid_name(name) {
        Ok(())
    } else {
        Err(error(
            StatusCode::BAD_REQUEST,
            "tenant and name must be 1-64 characters of [A-Za-z0-9_-]",
        ))
    }
}

fn not_leader(gateway: &Gateway, leader: Option<u64>) -> Response {
    let leader_http = leader.and_then(|id| gateway.peer_http.get(&id).cloned());
    (
        StatusCode::MISDIRECTED_REQUEST,
        axum::Json(json!({
            "error": "this node is not the leader",
            "leader": leader,
            "leader_http": leader_http,
        })),
    )
        .into_response()
}

fn propose_failure(gateway: &Gateway, failure: ProposeFailure) -> Response {
    match failure {
        ProposeFailure::NotLeader { leader } => not_leader(gateway, leader),
        ProposeFailure::Unknown => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "leadership changed; the outcome is unknown, and a retry may apply twice",
        ),
        ProposeFailure::Encoding(e) => error(StatusCode::BAD_REQUEST, e),
        ProposeFailure::Stopped => error(StatusCode::SERVICE_UNAVAILABLE, "the node is stopping"),
    }
}

fn output_response(output: Output) -> Response {
    match output {
        Output::Deployed => axum::Json(json!({ "status": "deployed" })).into_response(),
        Output::Removed => axum::Json(json!({ "status": "removed" })).into_response(),
        Output::Executed { result, gas_used } => match result {
            Ok(value) => {
                axum::Json(json!({ "result": value, "gas_used": gas_used })).into_response()
            }
            Err(trap) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                axum::Json(json!({ "trap": trap, "gas_used": gas_used })),
            )
                .into_response(),
        },
        Output::Rejected(reason) => error(StatusCode::BAD_REQUEST, reason),
    }
}

async fn status(AxumState(gateway): AxumState<Arc<Gateway>>) -> Response {
    match gateway.node.status().await {
        Some(s) => axum::Json(json!({
            "id": s.id,
            "role": format!("{:?}", s.role),
            "term": s.term,
            "leader": s.leader,
            "commit": s.commit,
            "applied": s.applied,
            "last_index": s.last_index,
            "snapshot_index": s.snapshot_index,
        }))
        .into_response(),
        None => error(StatusCode::SERVICE_UNAVAILABLE, "the node is stopping"),
    }
}

async fn digest(AxumState(gateway): AxumState<Arc<Gateway>>) -> Response {
    let state = gateway.node.state();
    let applied = gateway.node.status().await.map(|s| s.applied);
    let (digest, workloads) = {
        let state = state.read().unwrap_or_else(|e| e.into_inner());
        (state.digest(), state.len())
    };
    axum::Json(json!({ "digest": digest, "applied": applied, "workloads": workloads }))
        .into_response()
}

/// Prometheus text exposition metrics for Raft state, workload inventory, and execution counters.
async fn metrics(AxumState(gateway): AxumState<Arc<Gateway>>) -> Response {
    use std::fmt::Write as _;
    let Some(s) = gateway.node.status().await else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "the node is stopping");
    };
    let role_code = match s.role {
        zeolite_raft::Role::Follower => 0,
        zeolite_raft::Role::PreCandidate => 1,
        zeolite_raft::Role::Candidate => 2,
        zeolite_raft::Role::Leader => 3,
    };
    let (zel_count, wasm_count, vm_count) = {
        let state = gateway.node.state();
        let state = state.read().unwrap_or_else(|e| e.into_inner());
        state.counts()
    };
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "# HELP zeolite_raft_term Current Raft term.");
    let _ = writeln!(out, "# TYPE zeolite_raft_term gauge");
    let _ = writeln!(out, "zeolite_raft_term {}", s.term);
    let _ = writeln!(
        out,
        "# HELP zeolite_raft_role Current Raft role (0=follower, 1=pre_candidate, 2=candidate, 3=leader)."
    );
    let _ = writeln!(out, "# TYPE zeolite_raft_role gauge");
    let _ = writeln!(out, "zeolite_raft_role {role_code}");
    let _ = writeln!(
        out,
        "# HELP zeolite_raft_commit_index Committed Raft log index."
    );
    let _ = writeln!(out, "# TYPE zeolite_raft_commit_index gauge");
    let _ = writeln!(out, "zeolite_raft_commit_index {}", s.commit);
    let _ = writeln!(
        out,
        "# HELP zeolite_raft_applied_index Applied Raft log index."
    );
    let _ = writeln!(out, "# TYPE zeolite_raft_applied_index gauge");
    let _ = writeln!(out, "zeolite_raft_applied_index {}", s.applied);
    let _ = writeln!(out, "# HELP zeolite_raft_last_index Last Raft log index.");
    let _ = writeln!(out, "# TYPE zeolite_raft_last_index gauge");
    let _ = writeln!(out, "zeolite_raft_last_index {}", s.last_index);
    let _ = writeln!(
        out,
        "# HELP zeolite_raft_snapshot_index Last compacted snapshot index."
    );
    let _ = writeln!(out, "# TYPE zeolite_raft_snapshot_index gauge");
    let _ = writeln!(out, "zeolite_raft_snapshot_index {}", s.snapshot_index);
    let _ = writeln!(
        out,
        "# HELP zeolite_workloads_total Deployed workloads by tier."
    );
    let _ = writeln!(out, "# TYPE zeolite_workloads_total gauge");
    let _ = writeln!(out, "zeolite_workloads_total{{kind=\"zel\"}} {zel_count}");
    let _ = writeln!(out, "zeolite_workloads_total{{kind=\"wasm\"}} {wasm_count}");
    let _ = writeln!(out, "zeolite_workloads_total{{kind=\"vm\"}} {vm_count}");
    let _ = writeln!(
        out,
        "# HELP zeolite_wasm_invocations_total Total Wasm invocations served by this node."
    );
    let _ = writeln!(out, "# TYPE zeolite_wasm_invocations_total counter");
    let _ = writeln!(
        out,
        "zeolite_wasm_invocations_total {}",
        gateway.wasm_invocations.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        out,
        "# HELP zeolite_wasm_fuel_used_total Cumulative Wasm fuel consumed on this node."
    );
    let _ = writeln!(out, "# TYPE zeolite_wasm_fuel_used_total counter");
    let _ = writeln!(
        out,
        "zeolite_wasm_fuel_used_total {}",
        gateway.wasm_fuel_used.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        out,
        "# HELP zeolite_vm_boots_total Total micro-VM boot requests served by this node."
    );
    let _ = writeln!(out, "# TYPE zeolite_vm_boots_total counter");
    let _ = writeln!(
        out,
        "zeolite_vm_boots_total {}",
        gateway.vm_boots.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        out,
        "# HELP zeolite_zel_queries_total Total read-only Zel queries served by this node."
    );
    let _ = writeln!(out, "# TYPE zeolite_zel_queries_total counter");
    let _ = writeln!(
        out,
        "zeolite_zel_queries_total {}",
        gateway.zel_queries.load(Ordering::Relaxed)
    );
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    (StatusCode::OK, headers, out).into_response()
}

#[derive(Deserialize)]
struct ListQuery {
    tenant: Option<String>,
}

async fn list_workloads(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Query(query): Query<ListQuery>,
    headers: HeaderMap,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    if let Some(ref tenant) = query.tenant
        && !valid_name(tenant)
    {
        return error(
            StatusCode::BAD_REQUEST,
            "tenant must be 1-64 characters of [A-Za-z0-9_-]",
        );
    }
    let workloads = {
        let state = gateway.node.state();
        let state = state.read().unwrap_or_else(|e| e.into_inner());
        state.list(query.tenant.as_deref())
    };
    axum::Json(json!({ "count": workloads.len(), "workloads": workloads })).into_response()
}

async fn get_workload(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    if let Err(r) = check_names(&tenant, &name) {
        return r;
    }
    let summary = {
        let state = gateway.node.state();
        let state = state.read().unwrap_or_else(|e| e.into_inner());
        state.summary(&tenant, &name)
    };
    match summary {
        Some(summary) => axum::Json(summary).into_response(),
        None => error(StatusCode::NOT_FOUND, "no such workload"),
    }
}

#[derive(Deserialize)]
struct TransferBody {
    to: u64,
    #[serde(default = "default_transfer_timeout")]
    timeout_ms: u64,
}

fn default_transfer_timeout() -> u64 {
    5000
}

const MAX_TRANSFER_TIMEOUT_MS: u64 = 60_000;

/// Hands leadership to another node, for example before taking this one down for
/// maintenance, and waits until this node sees the target lead.
async fn transfer_leadership(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    let request: TransferBody = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let to = request.to;
    match gateway.node.transfer_leadership(to).await {
        Ok(()) => {}
        Err(TransferFailure::NotLeader { leader }) => return not_leader(&gateway, leader),
        Err(e @ TransferFailure::UnknownPeer(_)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e @ TransferFailure::Stopped) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    }
    let timeout = Duration::from_millis(request.timeout_ms.min(MAX_TRANSFER_TIMEOUT_MS));
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let Some(status) = gateway.node.status().await else {
            return error(StatusCode::SERVICE_UNAVAILABLE, "the node is stopping");
        };
        let done = if to == status.id {
            status.role == zeolite_raft::Role::Leader
        } else {
            status.leader == Some(to) && status.role != zeolite_raft::Role::Leader
        };
        if done {
            return axum::Json(json!({
                "status": "transferred",
                "leader": to,
                "leader_http": gateway.peer_http.get(&to),
                "term": status.term,
            }))
            .into_response();
        }
        if tokio::time::Instant::now() >= deadline {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(json!({
                    "error": "leadership did not move to the target in time",
                    "leader": status.leader,
                    "term": status.term,
                })),
            )
                .into_response();
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[derive(Deserialize)]
struct DeployQuery {
    kind: String,
    #[serde(default = "default_memory")]
    memory_bytes: u64,
    webhook_secret: Option<String>,
    arch: Option<String>,
}

fn default_memory() -> u64 {
    1 << 16
}

async fn deploy(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name)): Path<(String, String)>,
    Query(query): Query<DeployQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    if let Err(r) = check_names(&tenant, &name) {
        return r;
    }
    let command = match query.kind.as_str() {
        "zel" => {
            let Ok(source) = String::from_utf8(body.to_vec()) else {
                return error(StatusCode::BAD_REQUEST, "Zel source must be UTF-8");
            };
            Command::DeployZel {
                tenant,
                name,
                source,
                memory_bytes: query.memory_bytes,
            }
        }
        "wasm" => {
            // Validating here keeps broken modules out of the replicated log entirely.
            if let Err(e) = gateway.runtime.deploy(&body) {
                return error(StatusCode::BAD_REQUEST, e);
            }
            Command::DeployWasm {
                tenant,
                name,
                wasm: body.to_vec(),
                webhook_secret: query.webhook_secret.map(String::into_bytes),
            }
        }
        "vm" => {
            let image = match query.arch.as_deref() {
                Some("x86-long") => VmImage::RawX86LongMode(body.to_vec()),
                Some("x86-real") => VmImage::RawX86RealMode(body.to_vec()),
                Some("aarch64") => VmImage::RawAarch64(body.to_vec()),
                _ => {
                    return error(
                        StatusCode::BAD_REQUEST,
                        "arch must be one of x86-long, x86-real or aarch64",
                    );
                }
            };
            Command::DeployVm {
                tenant,
                name,
                image,
            }
        }
        other => return error(StatusCode::BAD_REQUEST, format!("unknown kind `{other}`")),
    };
    match gateway.node.propose(&command).await {
        Ok(output) => output_response(output),
        Err(failure) => propose_failure(&gateway, failure),
    }
}

async fn remove(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    match gateway
        .node
        .propose(&Command::Remove { tenant, name })
        .await
    {
        Ok(output) => output_response(output),
        Err(failure) => propose_failure(&gateway, failure),
    }
}

#[derive(Deserialize)]
struct ExecuteBody {
    #[serde(default)]
    args: Vec<i64>,
    #[serde(default = "default_gas")]
    gas: u64,
}

fn default_gas() -> u64 {
    10_000_000
}

async fn execute(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name, function)): Path<(String, String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    let parsed: ExecuteBody = if body.is_empty() {
        ExecuteBody {
            args: Vec::new(),
            gas: default_gas(),
        }
    } else {
        match serde_json::from_slice(&body) {
            Ok(parsed) => parsed,
            Err(e) => return error(StatusCode::BAD_REQUEST, e),
        }
    };
    let command = Command::Execute {
        tenant,
        name,
        function,
        args: parsed.args,
        gas: parsed.gas,
    };
    match gateway.node.propose(&command).await {
        Ok(output) => output_response(output),
        Err(failure) => propose_failure(&gateway, failure),
    }
}

/// Evaluates a Zel function locally against a snapshot of linear memory without mutating state.
async fn query_zel(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name, function)): Path<(String, String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    if let Err(r) = check_names(&tenant, &name) {
        return r;
    }
    let parsed: ExecuteBody = if body.is_empty() {
        ExecuteBody {
            args: Vec::new(),
            gas: default_gas(),
        }
    } else {
        match serde_json::from_slice(&body) {
            Ok(parsed) => parsed,
            Err(e) => return error(StatusCode::BAD_REQUEST, e),
        }
    };
    let slots = gateway.slots(&tenant);
    let Ok(_permit) = slots.try_acquire_owned() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "tenant concurrency limit reached",
        );
    };
    gateway.zel_queries.fetch_add(1, Ordering::Relaxed);
    let state = gateway.node.state();
    let result = tokio::task::spawn_blocking(move || {
        let guard = state.read().unwrap_or_else(|e| e.into_inner());
        guard.query(&tenant, &name, &function, &parsed.args, parsed.gas)
    })
    .await;
    match result {
        Ok(Some(Output::Executed { result, gas_used })) => match result {
            Ok(value) => axum::Json(json!({
                "result": value,
                "gas_used": gas_used,
                "read_only": true,
            }))
            .into_response(),
            Err(trap) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                axum::Json(json!({
                    "trap": trap,
                    "gas_used": gas_used,
                    "read_only": true,
                })),
            )
                .into_response(),
        },
        Ok(Some(Output::Rejected(reason))) => error(StatusCode::BAD_REQUEST, reason),
        Ok(Some(other)) => output_response(other),
        Ok(None) => error(StatusCode::NOT_FOUND, "no such Zel workload"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// A function's module bytes and its webhook secret.
type WasmEntry = (Arc<Vec<u8>>, Option<Vec<u8>>);

/// Fetches a Wasm function from the local replica, along with its webhook secret.
fn wasm_of(gateway: &Gateway, tenant: &str, name: &str) -> Option<WasmEntry> {
    let state = gateway.node.state();
    let state = state.read().unwrap_or_else(|e| e.into_inner());
    match state.get(tenant, name)? {
        Workload::Wasm {
            wasm,
            webhook_secret,
        } => Some((wasm.clone(), webhook_secret.clone())),
        _ => None,
    }
}

async fn run_wasm(
    gateway: Arc<Gateway>,
    tenant: String,
    wasm: Arc<Vec<u8>>,
    input: Bytes,
) -> Response {
    let slots = gateway.slots(&tenant);
    // One noisy tenant runs out of its own slots rather than starving everyone else.
    let Ok(_permit) = slots.try_acquire_owned() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "tenant concurrency limit reached",
        );
    };
    let task_gateway = gateway.clone();
    let result = tokio::task::spawn_blocking(move || {
        let function = task_gateway.compile(&wasm)?;
        Ok::<_, String>(
            task_gateway
                .runtime
                .invoke(&function, &input, &task_gateway.limits),
        )
    })
    .await;
    let outcome = match result {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    gateway.wasm_invocations.fetch_add(1, Ordering::Relaxed);
    gateway
        .wasm_fuel_used
        .fetch_add(outcome.fuel_used, Ordering::Relaxed);

    let mut headers = HeaderMap::new();
    let mut put_header = |name: &'static str, value: String| {
        if let Ok(value) = HeaderValue::from_str(&value) {
            headers.insert(name, value);
        }
    };
    put_header("x-zeolite-fuel-used", outcome.fuel_used.to_string());
    put_header("x-zeolite-micros", outcome.elapsed.as_micros().to_string());
    put_header(
        "x-zeolite-cold-start-micros",
        outcome.instantiate.as_micros().to_string(),
    );
    match outcome.result {
        Ok(output) => (StatusCode::OK, headers, output).into_response(),
        Err(e) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            headers,
            axum::Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn invoke(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name)): Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    let Some((wasm, _)) = wasm_of(&gateway, &tenant, &name) else {
        return error(StatusCode::NOT_FOUND, "no such Wasm function");
    };
    run_wasm(gateway, tenant, wasm, body).await
}

/// Public trigger. If the function has a secret, the caller must prove knowledge of it by
/// signing the body: `x-zeolite-signature: sha256=<hex hmac>`.
async fn webhook(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name)): Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some((wasm, secret)) = wasm_of(&gateway, &tenant, &name) else {
        return error(StatusCode::NOT_FOUND, "no such Wasm function");
    };
    if let Some(secret) = secret {
        let signature = headers
            .get("x-zeolite-signature")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("sha256="))
            .and_then(decode_hex);
        let Some(signature) = signature else {
            return error(StatusCode::UNAUTHORIZED, "missing or malformed signature");
        };
        let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(&secret) else {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "invalid webhook secret");
        };
        mac.update(&body);
        if mac.verify_slice(&signature).is_err() {
            return error(StatusCode::UNAUTHORIZED, "signature does not match");
        }
    }
    run_wasm(gateway, tenant, wasm, body).await
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

#[derive(Deserialize)]
struct BootQuery {
    #[serde(default = "default_vm_memory")]
    memory_bytes: usize,
    #[serde(default = "default_vm_timeout")]
    timeout_ms: u64,
}

fn default_vm_memory() -> usize {
    16 << 20
}

fn default_vm_timeout() -> u64 {
    5000
}

async fn boot(
    AxumState(gateway): AxumState<Arc<Gateway>>,
    Path((tenant, name)): Path<(String, String)>,
    Query(query): Query<BootQuery>,
    headers: HeaderMap,
) -> Response {
    if let Err(r) = require_admin(&gateway, &headers) {
        return r;
    }
    if query.memory_bytes > MAX_VM_MEMORY_BYTES {
        return error(StatusCode::BAD_REQUEST, "memory_bytes is too large");
    }
    let image = {
        let state = gateway.node.state();
        let state = state.read().unwrap_or_else(|e| e.into_inner());
        match state.get(&tenant, &name) {
            Some(Workload::Vm { image }) => image.clone(),
            _ => return error(StatusCode::NOT_FOUND, "no such micro-VM workload"),
        }
    };
    let slots = gateway.slots(&tenant);
    let Ok(_permit) = slots.try_acquire_owned() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "tenant concurrency limit reached",
        );
    };
    let payload = match image {
        VmImage::RawX86LongMode(b) => Payload::RawX86LongMode(b),
        VmImage::RawX86RealMode(b) => Payload::RawX86RealMode(b),
        VmImage::RawAarch64(b) => Payload::RawAarch64(b),
    };
    let mut cfg = VmConfig::new(payload);
    cfg.memory_bytes = query.memory_bytes;
    cfg.timeout = Duration::from_millis(query.timeout_ms.min(60_000));
    gateway.vm_boots.fetch_add(1, Ordering::Relaxed);
    let result = tokio::task::spawn_blocking(move || zeolite_vmm::run(cfg)).await;
    match result {
        Ok(Ok(outcome)) => axum::Json(json!({
            "exit": format!("{:?}", outcome.exit),
            "serial": String::from_utf8_lossy(&outcome.serial),
            "elapsed_micros": outcome.elapsed.as_micros() as u64,
            "exits": outcome.exits,
        }))
        .into_response(),
        Ok(Err(VmError::Unavailable(why))) => error(StatusCode::NOT_IMPLEMENTED, why),
        Ok(Err(e)) => error(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}
