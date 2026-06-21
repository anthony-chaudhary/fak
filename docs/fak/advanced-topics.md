# Advanced topics: performance, scaling, multi-region, and HA

This guide covers running `fak serve` beyond a single-process dogfood: tuning it for
throughput, spreading it across replicas, deploying it in more than one region, and
keeping it available through restarts and failures.

Every flag, env var, route, and metric named below is verified against the source in
this repository. Where `fak`'s design draws a hard boundary — most importantly, that its
two pieces of in-process state (the vDSO cache and the per-trace IFC ledger) are
**process-local and not shared across replicas** — this guide says so plainly rather
than implying a cluster feature that does not exist. Building a production topology on a
wrong mental model of what is shared is the one mistake that actually bites.

The companion references:

- [server-config.md](server-config.md) — every `fak serve` flag and env var, in full.
- [observability.md](observability.md) — the `/metrics`, `/debug/vars`, and access-log
  surfaces this guide tells you to alert on.

---

## The one architectural fact everything else follows from

`fak serve` is a **security gate**, not a stateful application server. A single tool-call
adjudication (`/v1/fak/adjudicate`, `/v1/fak/syscall`, `/v1/fak/admit`) is decided
against the capability-floor manifest loaded at boot, and the manifest is the same on
every replica that loads the same `--policy` file. That makes the *decision* path
effectively stateless and trivially replicable.

Two things are **not** stateless, and both live entirely inside one process:

| In-process state | What it is | Lifetime / scope | If a replica loses it |
|---|---|---|---|
| **vDSO tier-2 cache** | cross-agent read-dedup cache, keyed `tool:argHash:epoch` (agent-blind — carries no trace id) | process-global (`vdso.Default`), in-memory | a cold cache — pure performance, never a correctness change |
| **Per-trace IFC ledger** | the taint marks a trace accumulates as it reads untrusted data, used to gate later egress | process-local, keyed by `trace_id` | the trace's accumulated taint is gone — a later egress call on that trace is judged with no memory of what it read |

Everything in the sections below is a consequence of this table. The cache being
process-local is why horizontal scaling *dilutes* the cross-agent hit rate. The IFC
ledger being process-local-per-trace is why a multi-call IFC flow needs **sticky routing
by `trace_id`** to stay correct across replicas.

---

## 1. Performance optimization

### 1.1 Timeout tuning for different model backends

`fak serve` runs three independent HTTP timeouts plus a separate upstream-call timeout.
The defaults are conservative for a *network-exposed proxy*; a *slow local model* needs
the write timeout raised because **`WriteTimeout` bounds the whole handler, and a live
model round-trip rides inside it** — a multi-thousand-token CPU prefill can take minutes
and will otherwise be cut off mid-stream.

| Setting | Default | What it bounds | Tune when |
|---|---|---|---|
| `ReadHeaderTimeout` | `10s` (fixed) | time to receive request headers (slow-loris guard) | not tunable; not normally a concern |
| `FAK_HTTP_READ_TIMEOUT_S` | `30` | time to receive the whole request body | very large request bodies |
| `FAK_HTTP_WRITE_TIMEOUT_S` | `90` | **the entire handler**, including the upstream/in-kernel model turn | **slow local backend** — raise generously, or `0` to disable |
| `FAK_HTTP_IDLE_TIMEOUT_S` | `120` | keep-alive idle between requests | high-RTT clients reusing connections |
| `FAK_PLANNER_TIMEOUT_S` | `60` | one upstream model HTTP call (proxy mode) | slow or far upstream provider |

Backend-specific starting points:

```sh
# Hosted API upstream (fast, predictable) — defaults are fine, maybe trim the planner timeout.
fak serve --addr 0.0.0.0:8080 --provider openai \
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model gpt-4o-mini
#   FAK_PLANNER_TIMEOUT_S=30 is reasonable here.

# Local in-kernel GGUF on CPU (slow prefill) — give the handler room so a long turn isn't truncated.
FAK_HTTP_WRITE_TIMEOUT_S=600 \
  fak serve --addr 0.0.0.0:8080 --gguf ./model.gguf --require-key-env FAK_TOKEN
```

Setting any of the `FAK_HTTP_*_TIMEOUT_S` knobs to `0` disables that timeout entirely.
Do this only for local dogfood serving — on a network-reachable gateway an unbounded
read or idle timeout is a slow-loris / idle-keepalive resource-exhaustion vector, which
is exactly why the defaults are bounded.

### 1.2 Connection handling (Nagle / TCP_NODELAY)

The gateway disables Nagle's algorithm (`TCP_NODELAY`) on every accepted TCP connection.
Without it, the kernel coalesces small writes and adds **40–200 ms** of buffering to
streamed chat-completion deltas and the small fak-native verdict replies. This is
automatic and requires no configuration — but it's worth knowing when you measure tail
latency: that source of jitter is already removed at the gateway, so any remaining
small-write latency lives in your load balancer or upstream, not here.

There is no upstream HTTP connection *pool* to size — proxy-mode calls go through the
standard Go HTTP client. The lever that matters for upstream behavior is
`FAK_PLANNER_TIMEOUT_S` (above) and keeping the gateway network-close to the model
([§3.2](#32-latency-optimization)).

### 1.3 vDSO cache tuning

The vDSO fast path deduplicates **reads across agents** sharing one gateway. Its tier-2
key is `tool:argHash:epoch` and deliberately carries no trace id, so a read warmed by
agent A is served to agents B and C for free — no second engine call. The tradeoff is
write invalidation, and the `--invalidation` granularity is the dial:

| `--invalidation` (or `FAK_VDSO_GRANULARITY`) | A write invalidates… | Cross-agent hit rate under writes |
|---|---|---|
| `global` (default) | the whole tier-2 cache | lowest — one write strands every peer's cached read |
| `namespace` | only entries in the written namespace | middle ground |
| `resource` | only the written entity's own epoch | highest — a peer's read of a *different* entity stays warm |

**Recommendation:** in a read-heavy fleet with occasional writes, `resource` granularity
is what turns cross-agent sharing from a net loss under writes into a net gain — a write
to one entity bumps only that entity's epoch, leaving every peer's unrelated cached read
hot. Use `global` only when you cannot reason about write blast radius and want the
safest (most aggressive) invalidation.

```sh
fak serve --addr 0.0.0.0:8080 --gguf ./model.gguf \
  --vdso --invalidation resource --require-key-env FAK_TOKEN
```

Watch the effect on `/metrics`:

```promql
# cross-agent dedup effectiveness — higher is better
fak_gateway_vdso_hit_ratio
rate(fak_vdso_hits_total[5m])
rate(fak_vdso_invalidations_total[5m])   # write-driven strandings; spikes here erode the hit ratio
```

`--vdso=false` disables the fast path entirely (every read hits the engine). Only do this
to isolate a correctness question — the cache is a performance feature with no bearing on
a verdict.

> **Scaling caveat (carried forward to [§2.3](#23-cross-agent-kv-and-read-sharing)):** this
> cache is process-global. The cross-agent uplift is real *within one gateway process*.
> Spreading the same agents across N replicas splits their reads across N independent
> caches and reduces the hit rate accordingly.

### 1.4 Model selection strategies

`fak serve` has three serving modes, selected by which model flags you pass:

| Mode | How to select | Use it for |
|---|---|---|
| **Proxy** | `--base-url` + `--provider` + `--api-key-env` | front a hosted model (OpenAI/Anthropic/Gemini/xAI wire) with adjudication |
| **In-kernel** | `--gguf` (no `--base-url`) | self-host the model fused into the gate; `/v1/chat/completions` and `/v1/messages` serve it directly using the GGUF's embedded tokenizer |
| **Offline mock** | neither `--base-url` nor `--gguf` | tests and policy dry-runs — the mock planner needs no network and no weights |

In-kernel load and compute tuning (all from [server-config.md](server-config.md), verified
against the serve path):

- **`FAK_Q4K=1`** selects the direct-resident-Q4_K load path for Qwen3.6-27B Q4_K_M
  weights — it holds eligible matmul tensors raw and engages the int8-SDOT decode GEMV,
  for **~10× faster load** than the default lean-Q8 round-trip. The default path stays
  byte-identical when the env is unset.
- **`FAK_BACKEND`** picks the compute backend (`cuda`, `metal`, `vulkan`, `cpu`); it is
  auto-detected if unset.
- **`FAK_WORKERS`** caps matmul parallelism (defaults to `GOMAXPROCS`) — pin it to leave
  cores for other tenants on a shared box.
- **`FAK_INKERNEL_MAX_TOKENS`** (default `256`), **`FAK_INKERNEL_TEMP`** (default `0`),
  **`FAK_INKERNEL_SEED`** bound and shape in-kernel generation. A lower max-tokens cap
  directly bounds worst-case handler time and pairs with the write-timeout tuning above.

The eager GGUF load happens **before the listener binds**, so its cost is measured as
part of `fak_gateway_time_to_ready_seconds` and broken out per phase on `/metrics` rather
than paid lazily on the first request. This is what makes `/healthz` usable as a
readiness gate — see [§4.1](#41-health-check-patterns).

---

## 2. Horizontal scaling

### 2.1 What replicates cleanly, and what doesn't

Run N identical `fak serve` processes behind a load balancer, each with the **same
`--policy` manifest** and the same flags. The adjudication decision for any single call
is identical on every replica (same floor → same verdict), so the verdict path scales
horizontally with no coordination.

The two process-local states from the [opening table](#the-one-architectural-fact-everything-else-follows-from)
are what shape your routing:

- **vDSO cache** — losing or splitting it costs hit rate, never correctness.
- **Per-trace IFC ledger** — splitting a *single trace's* calls across replicas is a
  **correctness** problem, because the replica handling a later egress call won't have
  the taint the trace accumulated on an earlier replica.

### 2.2 Load-balancer configuration and sticky sessions

| Traffic shape | Routing | Why |
|---|---|---|
| Independent single-call syscalls (each call self-contained) | round-robin / least-conn | stateless; any replica gives the same verdict |
| A multi-call IFC flow on one `trace_id` (read untrusted → … → egress) | **sticky by `trace_id`** | the IFC ledger that gates the egress is process-local to wherever the trace's earlier reads landed |
| Read-heavy fleet wanting max dedup | sticky by a stable agent/tenant key | keeps an agent's reads landing on one warm cache |

`fak serve` honors an inbound **`X-Trace-Id`** header and echoes it on the response
(it mints one when absent — see [observability.md §1](observability.md)). That header is
the natural sticky-session key: configure your LB to hash on `X-Trace-Id`.

```nginx
# nginx: hash on the caller's trace id so every call of one trace lands on one replica,
# keeping that trace's IFC ledger and warm vDSO entries co-located.
upstream fak_gateways {
    hash $http_x_trace_id consistent;
    server gw1.internal:8080;
    server gw2.internal:8080;
    server gw3.internal:8080;
}
server {
    location /v1/ {
        proxy_pass http://fak_gateways;
        proxy_set_header X-Trace-Id $http_x_trace_id;
    }
    location /healthz {            # health-check the pool members directly, not hashed
        proxy_pass http://fak_gateways;
    }
}
```

If your callers don't yet set `X-Trace-Id`, have the LB or your client assign a stable id
per logical agent session before this works as a stickiness key.

### 2.3 Cross-agent KV and read sharing

The vDSO tier-2 cache *is* the cross-agent sharing mechanism, and it is **per-process**.
The implication for scaling is counter-intuitive and worth stating directly:

- **One gateway, many agents** → maximum sharing. Agent A's read warms the cache for B
  and C; the served hit-rate is the cross-agent uplift.
- **Many gateways, agents spread by round-robin** → the same set of agents now reads
  across N independent caches, so each cache sees ~1/N of the warming traffic and the
  hit rate drops.

**Recommendation:** don't over-shard. Scale up (a bigger gateway fronting more agents)
before you scale out, so the cross-agent cache stays dense. When you do scale out, route
agents that share a working set (same tenant, same repo, same resource namespace) to the
same replica via the sticky key above, so their reads keep landing on one warm cache
instead of being scattered. There is no cross-replica cache-coherence bus — the
`/v1/fak/changes` feed and `/v1/fak/revoke` operate **within a single process**, not
across the fleet.

### 2.4 Statelessness considerations

To keep replicas as close to stateless as the design allows:

- **Ship the policy as an immutable artifact.** The manifest file is the only
  configuration that determines verdicts; bake it into the image or mount it read-only so
  every replica is provably identical. Validate it offline first with
  `fak serve --policy floor.json --policy-check` (binds no listener; exits non-zero on a
  bad manifest).
- **Treat the vDSO cache as ephemeral.** Never persist or try to replicate it — a cold
  start is just a cold cache.
- **Bound the per-process ledgers.** The IFC taint ledger and the rate-limit counters are
  process-lifetime, bounded structures. They reset on restart, which is fine; just don't
  assume a trace's taint or a key's quota survives a replica replacement (it doesn't — see
  [§5.3](#53-rate-limiting-strategies)).
- **Reset a trace deliberately when you reuse its id.** `POST /v1/fak/trace/reset` clears
  one trace's process-local taint mark, so a recycled `trace_id` doesn't inherit stale
  taint.

---

## 3. Multi-region deployment

Multi-region is "independent single-region stacks, each complete, with shared
*configuration* but not shared *runtime state*." Because the only shared input that
matters is the policy manifest, multi-region is mostly a config-distribution and
routing problem.

### 3.1 Cross-region model routing

Co-locate each gateway with the model it fronts; route callers to the nearest region.

- **In-kernel (`--gguf`):** the model is fused into the gate, so the model is wherever the
  gateway is — deploy the same image per region and you're done.
- **Proxy (`--base-url`):** point each region's `--base-url` at that region's model
  endpoint so adjudication and inference stay in-region. Avoid a gateway in region A
  proxying to a model in region B — you pay the cross-region RTT on every turn, inside the
  write-timeout budget.

```sh
# us-east replica
fak serve --addr 0.0.0.0:8080 --provider openai \
  --base-url https://us-east.models.internal/v1 --api-key-env MODEL_KEY \
  --policy /etc/fak/floor.json --require-key-env FAK_TOKEN

# eu-west replica — same policy artifact, region-local model URL
fak serve --addr 0.0.0.0:8080 --provider openai \
  --base-url https://eu-west.models.internal/v1 --api-key-env MODEL_KEY \
  --policy /etc/fak/floor.json --require-key-env FAK_TOKEN
```

### 3.2 Latency optimization

- **Keep the model in-region** (above) — the dominant latency term is the model turn, and
  it rides inside `WriteTimeout`. A cross-region model hop is the easiest way to blow the
  latency budget.
- **`TCP_NODELAY` is already on** ([§1.2](#12-connection-handling-nagle--tcp_nodelay)), so
  streamed deltas aren't Nagle-buffered at the gateway.
- **Let each region keep its own warm vDSO cache.** Cross-region cache sharing does not
  exist by design; that's the right behavior — a remote cache lookup would cost more RTT
  than the engine call it saves.
- **Tune timeouts per region** if a far upstream needs more headroom: raise
  `FAK_PLANNER_TIMEOUT_S` and `FAK_HTTP_WRITE_TIMEOUT_S` only in the region that needs it.

### 3.3 Policy synchronization

The capability floor is the one thing every region must agree on. The gateway gives you a
clean two-step rollout that needs no restart:

1. **Validate** the new manifest before it touches any region:
   `fak serve --policy new-floor.json --policy-check` (prints the floor it admits and the
   confirmation that every deny cites a closed-vocabulary reason; exits non-zero if not).
2. **Reload in place** on each running replica: `POST /v1/fak/policy/reload` re-reads the
   `--policy` file from disk and swaps both the adjudicator floor and the IFC
   configuration atomically — no dropped connections, no restart.

```sh
# roll a validated policy to a region, replica by replica
fak serve --policy /etc/fak/new-floor.json --policy-check || exit 1   # gate first
# distribute the file to each replica's --policy path, then:
for gw in gw1 gw2 gw3; do
  curl -fsS -X POST -H "Authorization: Bearer $FAK_TOKEN" \
    "http://$gw.eu-west.internal:8080/v1/fak/policy/reload"
done
```

The reload route is only mounted when the gateway was started with `--policy` (a
gateway on the built-in default floor returns an error to a reload call — it has no
manifest path to re-read). Distribute the *file* through your normal config pipeline
(GitOps, config map, signed artifact); `fak` reloads from whatever is on disk at the
`--policy` path.

### 3.4 Observability across regions

Every surface in [observability.md](observability.md) is per-process, so multi-region
observability is "scrape every replica, label by region, aggregate centrally."

- **Pin the deployed build per region** with the `fak_gateway_build_info{version,engine,
  model,vdso}` gauge — add a `region` label at scrape time and you have a single panel
  showing exactly what's running where, which is how you catch a region that didn't get
  the rollout.
- **Propagate `trace_id` across hops.** The gateway honors an inbound `X-Trace-Id` and
  threads it into the verdict log and the response header, so a request that crosses a
  region boundary keeps one id end-to-end — your cross-region traces stitch together
  without the gate ever logging a request body, tool argument, or result content.
- **Alert per region, roll up globally:**

```promql
# per-region error rate (add region via relabeling on the scrape job)
sum by (region, route) (rate(fak_gateway_http_requests_total{status=~"5.."}[5m]))
  / sum by (region, route) (rate(fak_gateway_http_requests_total[5m]))

# any region down
min by (region) (fak_gateway_up) == 0
```

---

## 4. High availability

### 4.1 Health-check patterns

`/healthz` is an **unauthenticated** liveness endpoint (it's the one route exempt from
`--require-key-env`), returning `{"ok":true,"engine":...,"model":...}` with `200`.

Crucially, the eager GGUF load completes **before the listener binds**
([§1.4](#14-model-selection-strategies)), so a successful `/healthz` already implies the
weights are resident and the gateway is ready to serve — **`/healthz` doubles as a
readiness gate**, no separate endpoint needed.

| Probe | Use | Source signal |
|---|---|---|
| **Liveness / readiness** | LB pool membership, orchestrator probe | `GET /healthz` → `200` (answers only after bind, which is after weight load) |
| **Scrape-level liveness** | alerting | `fak_gateway_up == 0` or scrape failure |
| **Cold-start budget** | deploy gating, regression alerts | `fak_gateway_time_to_ready_seconds` (0 until ready) and the per-phase `fak_gateway_startup_phase_duration_seconds` |
| **Saturation / stuck requests** | autoscaling, incident triage | `fak_gateway_inflight_requests` (+ its max-age) |

```sh
# Kubernetes-style probe
livenessProbe:  { httpGet: { path: /healthz, port: 8080 }, periodSeconds: 10 }
readinessProbe: { httpGet: { path: /healthz, port: 8080 }, periodSeconds: 5 }
```

### 4.2 Graceful shutdown

On **`os.Interrupt` (SIGINT / Ctrl-C)** the gateway stops accepting new connections and
drains in-flight requests within a **bounded 5-second window** before exiting (it calls
`http.Server.Shutdown` with a 5s deadline). In-flight adjudications get up to 5s to
finish; anything still running at the deadline is cut.

Operational notes, stated precisely so you don't build on a wrong assumption:

- The handler installed is for **`os.Interrupt`** — on Unix that is **SIGINT**. The
  process does not install a separate SIGTERM handler. Orchestrators that send `SIGTERM`
  on pod termination will hit the runtime's default signal behavior, not this 5s drain,
  unless you arrange for `SIGINT` to be delivered (e.g. a wrapper/`STOPSIGNAL SIGINT`, or
  send `SIGINT` from your stop hook).
- Set your orchestrator's **termination grace period to at least the 5s drain window**
  (a little more for headroom) so the drain can complete before a `SIGKILL`.
- Keep individual turns bounded (`FAK_INKERNEL_MAX_TOKENS`, the write timeout) so a
  request in flight at shutdown can actually finish inside 5s.

```dockerfile
# Make container stop deliver the signal the gateway drains on.
STOPSIGNAL SIGINT
```

### 4.3 Zero-downtime deployments

The stateless verdict path makes rolling deploys straightforward; the two state caveats
set the rules:

1. **Roll replicas one at a time** behind the LB, gating each new replica into the pool on
   `GET /healthz` `200` (which, per [§4.1](#41-health-check-patterns), already means
   weights-resident).
2. **Drain before replace.** Take a replica out of the LB pool, let in-flight traces
   finish, then signal shutdown — so no trace loses its IFC ledger mid-flow. Sticky
   routing ([§2.2](#22-load-balancer-configuration-and-sticky-sessions)) plus connection
   draining keeps a multi-call IFC flow intact across the rollout.
3. **Prefer a policy reload to a restart** for floor changes:
   `POST /v1/fak/policy/reload` swaps the floor with zero dropped connections
   ([§3.3](#33-policy-synchronization)) — no rollout needed at all.
4. **Pre-warm matters for cold-start.** With `--gguf`, `time_to_ready` includes the weight
   load; size your readiness timeout above the observed
   `fak_gateway_startup_phase_duration_seconds{phase="model-load"}` so a replica isn't
   pulled for being slow to boot.

### 4.4 Failover strategies

- **Replica failure:** the LB health check drops a dead replica; survivors serve every
  call identically (same policy floor). The failed replica's vDSO cache is lost — a cold
  cache on its replacement, nothing more.
- **In-flight traces on a failed replica:** their process-local IFC ledger dies with the
  process. A retried call on that trace lands on a fresh replica with no accumulated
  taint. For IFC-sensitive flows, retry the *whole* trace from a known-clean point rather
  than resuming mid-flow, and treat `POST /v1/fak/trace/reset` as the explicit
  "start this trace's taint accounting over" control.
- **Region failure:** route callers to another region's stack (each is complete and
  in-region per [§3](#3-multi-region-deployment)). There is no shared runtime state to
  reconcile — only the policy artifact, which every region already has.
- **Upstream model failure (proxy mode):** bounded by `FAK_PLANNER_TIMEOUT_S`; a timeout
  surfaces as an error response and shows up on `fak_gateway_http_requests_total{status=~"5.."}`.
  Put retry/failover to a backup model endpoint at the layer in front of the gateway
  (see circuit breakers, [§5.4](#54-circuit-breakers)).

---

## 5. Production patterns

### 5.1 Blue-green deployments

Run two complete pools (blue and green), each a set of `fak serve` replicas, and cut the
LB from one to the other. The `fak_gateway_build_info{version}` gauge is your proof of
which pool is live — pin it in a deploy panel and you can confirm the cutover at a glance.
Validate the green pool's policy artifact with `--policy-check` before it takes traffic;
keep blue warm until green's error rate and latency
([observability.md](observability.md) PromQL) match.

### 5.2 Canary testing

Weight a small slice of traffic to a canary replica running the new build or new policy:

- **New policy floor, same binary:** the cleanest canary — start one replica with the new
  `--policy` file (or `POST /v1/fak/policy/reload` it on one replica), send it a traffic
  slice, and compare its verdict mix. Watch `fak_verdict_total` by kind
  (`ALLOW`/`DENY`/`TRANSFORM`/`QUARANTINE`/`WITNESS`): a canary that suddenly denies far
  more (or far less) than the baseline is a misconfigured floor caught before full
  rollout.
- **New binary:** distinguish canary from baseline by `fak_gateway_build_info{version}`
  and compare error rate, p99 latency, and the verdict mix side by side.

```promql
# canary vs baseline deny-rate, split by build version label
sum by (version) (rate(fak_verdict_total{kind="DENY"}[5m]))
  / sum by (version) (rate(fak_verdict_total[5m]))
```

### 5.3 Rate-limiting strategies

`fak serve` has a built-in throttle that runs as an **early, cheap load-shed** (a
rank-8 adjudicator — it sheds an over-cap call before the expensive trust checks run) and
denies with the closed-vocabulary reason `RATE_LIMITED` and a `WAIT` disposition (retry
after a wait). It is **off unless you set the env vars**:

| Env var | Effect |
|---|---|
| `FAK_RATELIMIT_MAX_CALLS` | per-key admitted-call quota |
| `FAK_RATELIMIT_MAX_COST` | per-key cumulative cost budget (≈ argument bytes ≈ tokens) |
| `FAK_RATELIMIT_KEY` | bucket dimension: `trace` (default), `tool`, or `global` |

```sh
# 1000 admitted calls per trace, bucketed per-trace
FAK_RATELIMIT_MAX_CALLS=1000 FAK_RATELIMIT_KEY=trace \
  fak serve --addr 0.0.0.0:8080 --gguf ./model.gguf --require-key-env FAK_TOKEN
```

**The counters are per-process.** This interacts with horizontal scaling and demands a
deliberate choice:

- The effective fleet-wide limit is `per-replica quota × number of replicas` *only if*
  the keyspace is spread evenly across replicas — which round-robin does not guarantee.
- For a *true* per-trace or per-tool cap, route that key to a single replica (sticky by
  `trace_id`, [§2.2](#22-load-balancer-configuration-and-sticky-sessions)) so one counter
  sees all of that key's calls. Otherwise a per-trace cap of N becomes "up to N on each
  replica the trace happens to touch."
- For a coarse, fleet-aggregate throttle that doesn't need to be exact, enforce the real
  ceiling at the LB / API-gateway layer in front of `fak` and use `FAK_RATELIMIT_*` as a
  per-replica backstop.

### 5.4 Circuit breakers

`fak serve` does not ship a configurable upstream circuit breaker — and shouldn't fake
one. What it *does* give you to build on:

- **Fail-closed by default.** With `posture: fail_closed` (the default), anything not
  explicitly allowed is `DEFAULT_DENY` — the gate fails *safe*, not open, when in doubt.
  A misconfiguration or an unknown tool is refused, not waved through.
- **A bounded upstream call.** `FAK_PLANNER_TIMEOUT_S` caps every upstream model call, so
  a hung provider can't pin a handler indefinitely — failures surface promptly as 5xx on
  `fak_gateway_http_requests_total`.
- **Quarantine-driven cache reset.** In proxy mode, a quarantined tool result can trigger
  a remote serving-engine K/V reset (`--engine-cache-engine sglang|vllm` with
  `--engine-cache-base-url` / `--engine-cache-admin-key-env`) so poisoned context doesn't
  persist in the upstream's cache after the gate walls it off.

Put the *breaker* itself — open-on-error-threshold, half-open probing, failover to a
backup model — at the proxy or service-mesh layer in front of the gateway, and drive it
off the gateway's own error signal:

```promql
# feed this to your mesh/proxy breaker: per-route 5xx rate over the last minute
sum by (route) (rate(fak_gateway_http_requests_total{status=~"5.."}[1m]))
```

That keeps `fak` doing the one job it's authoritative for — adjudicating each call
against the floor — while the breaker logic lives where retries and failover belong.

---

## See also

- [server-config.md](server-config.md) — the full flag/env/policy-manifest reference
  behind every knob used above.
- [observability.md](observability.md) — the `/metrics`, `/debug/vars`, and access-log
  surfaces, with the metric families and PromQL these patterns alert on.
- [security.md](security.md) — auth, network exposure, and the threat model for a
  network-reachable gateway.
- [`fak/GETTING-STARTED.md`](../../GETTING-STARTED.md) — the route table and a guided
  first session.
