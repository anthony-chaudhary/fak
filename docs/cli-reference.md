---
title: "fak CLI Reference — Verbs of the Agent Kernel"
description: "The fak CLI reference: every verb (serve, run, preflight, bench, turntax, agent, recall, dream, debug, policy, hook) for the in-process agent kernel."
---

# fak — the agent kernel

`fak` is the runnable implementation of the Fused Agent Kernel: one Go binary
that puts a policy and quarantine boundary between an AI agent and its tools — with
an external dependency set of two `golang.org/x` modules, so the gateway, the
capability gate, the quarantine, the
audit log, and the metrics all live in that single process (no Python/CUDA stack, no
sidecars). The practical use is gateway-first: keep your existing model or agent host,
front it with `fak serve`, and make tool calls cross a reviewable capability floor
before anything dangerous executes.

Under the hood, every tool call becomes a syscall-like request: adjudicated
in-process, served from a local **tool vDSO** when possible, screened by a
**pre-flight + grammar ladder** before it fires, and admitted through a
**context-MMU** before tool results enter model context.

## `runtime-capabilities`: inspect the deployable runtime before payload load

```text
fak runtime-capabilities [--receipt-schema fak-runtime-capabilities/1|fak-execution-mode-receipt/1] [--backend NAME] [--prefer-backend NAME] [--fallback-policy pin_or_refuse|local_cpu_degraded] [--cpu-envelope ID] [--placement local_only|prefer_local|remote_allowed] [remote gate flags]
```

The default remains `fak-runtime-capabilities/1`. Opt into `--receipt-schema fak-execution-mode-receipt/1` for the uniform execution record. Its closed modes are `local_accelerator`, `local_cpu_degraded`, `remote_backed`, `offline_control_mock`, `offline_model_backed`, `control_only`, and `refused`; status and audit views carry explicit evidence states; independently observed views must agree or the receipt refuses, while the pre-payload projection marks both views `unwitnessed`. Model-backed records require exact engine, model, and local backend or remote provider evidence. A native/performance record is invalid unless `engine` is exactly `fak-native`, so engine substitution cannot masquerade as native execution. Unknown and unavailable facts are emitted as `unknown` or `unwitnessed`, not inferred.

`--execution-mode-fixture MODE` is available only with that opt-in schema. It emits deterministic schema coverage for one of the seven modes and stamps `witness.status: "fixture"` plus `witness.certification: "unwitnessed"`; it is not clean-host, device, provider, or model-execution evidence. Receipt collection is a bounded projection of the already-built capabilities report and performs no payload load or outbound probe.

Emits stable, machine-readable `fak-runtime-capabilities/1` JSON. The command performs no model-weight load and keeps three states separate: the executable runs, the governed control plane runs, and a fak-native model backend can execute for the requested policy. `--backend` remains an exact lookup: unknown, uncompiled, unavailable, or platform-unsupported backends return structured reasons and remediation without falling back to `cpu-ref` or another engine.

`--prefer-backend` is the explicit degraded-mode seam. When the preferred backend is unavailable, `--fallback-policy local_cpu_degraded` may select `cpu-ref` only if an exact `--cpu-envelope` row from `supported_cpu_envelopes` matches the host and the pre-load host-RAM budget clears. Unsupported or over-budget CPU fallback requests refuse before payload load, and the report emits a `local_cpu_degraded` receipt naming the requested backend, the selected `cpu-ref` path, the supported envelope, and the accelerator refusal reason. The optional `--goos`, `--goarch`, `--host-total-ram-bytes`, and `--host-free-ram-bytes` flags are diagnostic overrides for witness capture.

`--placement remote_allowed` adds an explicit remote admission path after an unavailable `--prefer-backend`; `local_only`, `prefer_local`, exact `--backend`, and pinned-native requests never use it. The caller must exactly match `--remote-target` with `--authorize-remote-target` and declare provider, engine, model, endpoint class/region when applicable, boundary state classes, egress, credential name/presence, TLS/proxy, reachability, timeout, retry ceiling, and budget. The machine-readable `remote_placement` receipt names local `fak` control-plane ownership and the remote execution trigger and gates. This command performs no outbound traffic and accepts no credential secret. It never silently chooses llama.cpp, Ollama, a provider, or a future fak cloud target.

The report includes `goos`, `goarch`, build tags, host memory, registered backend names/classes/tiers, portable `cpu-ref` status, supported CPU envelopes, `engine: "fak-native"` when execution is available, and `payload_compatibility: "supported" | "refused" | "not_checked"` for explicit CPU-envelope admission. This is distinct from `fak preflight`, which adjudicates a tool call against policy and grammar.
## `native-performance`: query the raw-model hill climb

`fak native-performance` renders the committed Qwen3.8-27B optimization graph as a
human checklist/table. It preserves the original Metal P32/T64 aggregate rungs and
adds independently attributable levers in separate pinned Metal and CUDA/A100
envelopes.

```bash
fak native-performance
fak native-performance --json
fak native-performance --baseline metal.command-buffer-amortization > baseline.json
fak native-performance --attach-receipt RECEIPT --system-baseline ATTESTATION --out NEXT
fak native-performance --compare baseline.json --candidate candidate.json
fak native-performance --profile profile.json
fak native-performance --profile-next profile.json
fak native-performance --gate gate-request.json
fak native-performance --next
fak native-performance --dot
```

`--json` emits the validated typed graph. `--next` deterministically prints the first
dependency-ready unwitnessed lever, its owning issue, and exact receipt requirement.
`--dot` emits deterministic Graphviz DOT with dependency/conflict edges and separate
envelope clusters; fak does not invoke Graphviz.

`--baseline LEVER` emits the versioned pre-change receipt template for that lever and
pinned envelope. `--compare BASELINE --candidate CANDIDATE` validates both receipts and
emits a deterministic JSON comparison. It refuses envelope/control drift, absent
repetitions or native execution identity, nonzero fallback, private path/host details,
and undeclared multi-axis changes. Templates contain `FILL_*` placeholders and are not
benchmark evidence until capture fills every required field and validation succeeds.

`--profile FILE` strictly decodes `fak-native-performance-profile/v1`, validates it, and
emits a deterministic bottleneck classification as JSON. `--profile-next FILE` also
returns the recommended lever and preserves any accepted `selection_override` in the
JSON result. These modes already select JSON output, require a nonempty path, and cannot
be combined with `--json` or another output mode. A recommendation that contradicts the
selected envelope's dependency-ready graph-order lever is refused unless
`selection_override` carries a positive issue number and a nonempty, scrubbed
issue-backed reason. A recommendation whose own dependencies are not enabled is never
returned as “next,” even with an override.

Profile execution must name `fak-native`, use zero fallback, and match the selected
envelope's `forward_path` exactly. The six phase starts are finite and non-negative,
durations are finite and positive, and phases may not overlap. Every backend counter is
required: count and capacity fields that form the capture are positive; non-negative
elapsed or achieved values may be zero only where zero is semantically meaningful; CUDA
occupancy stays in `[0,100]`, peaks are positive, and achieved values cannot exceed their
matching peaks.

Metal and CUDA counter blocks remain separate and are never normalized into a synthetic
cross-backend score. Manifest-declared `counter_comparisons` are unsupported in v1 and
fail closed. Dispatch attribution must either contain nonempty, single-lever records for
the selected envelope or use `attribution_unavailable` with one of the closed typed
reasons and a scrubbed detail; omission is not treated as unavailability.

Profile v1 requires the complete backend counter block and never substitutes a missing
counter with numeric zero. A capture tool that cannot expose a required counter may still
retain its scrubbed artifact, but it remains explicitly unclassified until a compatible
capture or a future typed counter-unavailability schema exists.

Each lever reports a stable ID, platform/envelope applicability, independent enabled
state and present/partial/absent status, dependency and conflict IDs, a
provenance-labelled expected planning effect, a separately receipt-backed witnessed
effect or `null`, owning issue, and next witness. Validation fails closed on cycles,
conflicts, cross-envelope edges, invalid applicability, duplicate IDs, and
expected/witnessed evidence conflation.

The command performs no model execution. Every execution envelope remains fak-native;
llama.cpp appears only as an explicitly selected parity/reference comparison. The
current 3.3 tok/s Metal fak-native witness and 6.966061 tok/s llama.cpp comparison
remain separately classified and only approximately comparable until #8697 captures
a joint matched receipt. CUDA #8635 targets are hypotheses, not measurements, and are
never combined with the Metal throughput curve. Graph semantics, the update checklist,
strict synthetic fixture status, and source provenance are in
[`NATIVE-PERFORMANCE-HILLCLIMB.md`](benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md).

## `bench system-baseline`: qualify one benchmark repetition

Wrap exactly one repetition, including its quiet pre-command baseline, and retain the
versioned attestation:

```bash
fak bench system-baseline --baseline-duration 2s --interval 250ms --max-sampler-duty-percent 10 --out system-rep-1.json -- <one benchmark repetition>
fak bench system-baseline --verify system-rep-1.json
fak native-performance --attach-receipt RECEIPT --system-baseline ATTESTATION --out NEXT
```

`--verify` strictly decodes and validates an existing attestation without running a
command. A `clean` verdict means the required CPU attribution is available and the
configured contamination ceiling was met; `investigate` means contamination was observed
or required attribution is unknown; `invalid` means the window or command cannot be used.
The attestation reports sampler wall duty separately for the baseline and command windows.
Duty above `--max-sampler-duty-percent` produces `investigate`; the wrapper never corrects
CPU or throughput for sampler cost and never silently drops a sample.

Run the attachment command once per repetition, passing each `NEXT` receipt as the next
call's `RECEIPT`; attestations append in repetition order. The first attachment upgrades a
legacy `fak-native-performance-receipt/v1` to v2, and attachment refuses to overfill the
1:1 `system_baselines`/`repetitions` sequence. Each repetition stores the exact attached
attestation digest; digest mismatch or reuse is rejected.

A strict `fak-native-performance-gate-policy/v2` must set both
`require_system_baseline: true` and `allow_sampled_system_baseline: true` to accept the v1
sampler's `sampled_pid_ppid_tree` coverage. Without that explicit sampled-coverage opt-in,
even a valid `clean` attestation produces an `investigate` gate result.

By default the artifact contains aggregate data only. `--top-consumers` opts into as many
as five scrubbed executable-image and PID records; this is local, high-cardinality evidence
and should be omitted from public artifacts unless needed. V1 samples host totals and
PID/PPID-tree CPU and RSS on Linux and Windows. It does not yet capture PSI, cgroup or Job
Object accounting, GPU activity, or complete short-lived-descendant attribution.

## `dup cache-maintain`: bound the shared token-window cache

`fak dup cache-maintain` runs the same nonblocking retention seam used after a
cache-backed duplication index build. Human output and `--json` receipts include exact
before/after bytes and entry counts, removed entries, stale atomic-write temps, skipped
locked files, configured limits, and a typed verdict.

```bash
fak dup cache-maintain --repo .
fak dup cache-maintain --repo . --json
fak dup cache-maintain --repo . --max-bytes 268435456 --max-entries 10000 --temp-grace 24h --json
```

The default cache is under Git's common directory, bounded to 256 MiB and 10,000 JSON
entries; `.entry-*.tmp` files receive a 24-hour active-write grace. See
[`docs/token-cache.md`](token-cache.md) for environment overrides, concurrency behavior,
disable semantics, and rollback.

## `goal`: canonical intent across harnesses

`fak goal` keeps durable user intent separate from an execution root or session. Bind only stable identities explicitly supplied by a harness; never derive identity from a title, prompt, task, issue text, or registration ID.

```bash
# Create one canonical goal, then bind harness-native identities to it.
fak goal create --title "Improve fleet observability" --actor operator --authority user
fak goal bind --id goal_<hex> --namespace claude:goal --external-id <claude-goal-id> --actor claude --authority harness
fak goal bind --id goal_<hex> --namespace codex:goal  --external-id <codex-goal-id>  --actor codex  --authority harness

# Resolve at a launch boundary. Read env.FAK_GOAL_ID from the versioned JSON and
# export it for the child; an unknown or ambiguous binding exits nonzero.
fak goal resolve --namespace claude:goal --external-id <claude-goal-id>
```

Canonical lifecycle is separate from execution completion. New goals default to `independent_witness_required`: harness, agent, and operator assertions may be recorded elsewhere but cannot terminalize canonical truth. Supply a durable independent witness, and reopen explicitly before replacing an incompatible terminal outcome.

```bash
fak goal transition --id goal_<hex> --lifecycle achieved --evidence-class independent_witness --evidence-author <judge> --evidence-ref <artifact>
fak goal reopen --id goal_<hex> --evidence-author operator --evidence-ref <decision-record>
fak goal show --id goal_<hex> # includes append-only outcome_evidence
```

Inspect the exact execution topology behind a canonical goal without scanning the journal. The versioned JSON groups registrations by `root_registration_id` and preserves each runtime, session/thread, attempt, state, and witness field.

```bash
fak goal topology --id goal_<hex>
```

Historical roots stay unbound unless an operator supplies an independent witness. Preview the exact root records first, then apply; the command also records a typed `fak:session-root` binding in the canonical registry.

```bash
fak goal backfill-root --id goal_<hex> --root-registration-id <root-id> --witness <artifact-ref> --actor operator --authority operator-declared
fak goal backfill-root --id goal_<hex> --root-registration-id <root-id> --witness <artifact-ref> --actor operator --authority operator-declared --apply
```

The same lookup contract supports explicitly witnessed `fak:trajctl`, `github:issue`, and `dos:unit` bindings. An optional `--revision` selects one external revision; omitting it fails closed if several revisions match. Fak does not introspect Claude or Codex state automatically. If a harness supplies no stable identity, launch without `FAK_GOAL_ID` and the execution root remains explicitly unbound.

## `microcontextdemo`: bounded logical-context fabric

`microcontextdemo` is the research/demo CLI for one immutable agent base serving bounded physical model slots across 100, 1,000, and 10,000 logical contexts. It is a separate demo binary, so invoke it with `go run ./cmd/microcontextdemo` rather than as a `fak` subcommand.

```bash
# Offline LCD floor: deterministic scheduler/shared-base semantics, not model tokens/s.
go run ./cmd/microcontextdemo -selfcheck -contexts 10000 -workers 64

# Verify captured controlled witnesses.
go run ./cmd/microcontextdemo -verify-kernel-prefix-ab experiments/microcontext/s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json
go run ./cmd/microcontextdemo -verify-quality experiments/microcontext/s5-gcp-1000-cuda-outcomes-2026-08-07.json
go run ./cmd/microcontextdemo -verify-health-scorecard experiments/microcontext/s5-gcp-1000-cuda-health-scorecard-2026-08-07.json
```

Core live flags match `microcontextdemo -h`: `-endpoint`, `-model`, `-provider`, and `-hardware` declare endpoint provenance; `-contexts` and `-workers` separate logical orchestration from bounded physical execution; `-request-timeout` and `-run-timeout` bound calls and runs. The API-only admission envelope uses `-api-concurrency`, `-api-rpm`, `-api-tpm`, and `-api-spend-micros`. Real in-kernel compatibility batches use `-gguf`, `-compat-batch-hardware`, `-compat-batch-size`, and `-compat-batch-execution`.

All artifact modes have explicit `-verify-*` counterparts in `-h`. The research contract and claim boundaries are in [`docs/research/micro-context-fabrics.md`](research/micro-context-fabrics.md); captured artifacts are indexed in [`docs/research/README.md`](research/README.md).

## Use fak with your coding agent (Claude Code, Cursor, …)

If you drive a coding agent, fak fits in two ways:

- **In front of the model** — point your agent's `ANTHROPIC_BASE_URL` (or OpenAI
  base URL) at `fak serve`, and every tool call the model proposes is
  denied / repaired / quarantined by the kernel before your agent runs it. No
  agent-side changes; the Anthropic `/v1/messages` and OpenAI
  `/v1/chat/completions` wires are both adjudicated, and a dropped or repaired
  call comes back with an in-band `[fak]` note so the agent adapts instead of
  looping. Witnessed live on macOS + Windows with the real Claude Code CLI:
  [`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md) (one command — `scripts/dogfood-claude.sh`,
  or `scripts/dogfood-claude.ps1` on Windows).
- **As an MCP server** — `fak serve --stdio` exposes the kernel's verbs
  (`fak_adjudicate`, `fak_syscall`, `fak_admit`, …) as MCP tools, so your agent
  can ask the kernel for a verdict before it runs a tool, or screen a result it
  executed itself through the exfil floor. Copy-paste config + the tool catalog:
  [`examples/mcp/`](https://github.com/anthony-chaudhary/fak/tree/main/examples/mcp) (drop [`examples/mcp/.mcp.json`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/.mcp.json)
  in your project root for Claude Code).

Both put the **same reviewable capability floor** (`--policy floor.json`) on every
tool call. New here? The fastest "feel it" is the no-credential boundary below;
the fastest "use it for real" is the dogfood path above.

## Console settings

`fak console settings` is the dedicated view for persisted console preferences. It
lists every safe registry-backed setting even before `~/.fak/console.json` exists,
including its effective value, source, allowed options, and copyable set/reset
commands. Use this five-stage workflow:

```bash
# 1. Find the canonical settings pane in the console command map.
fak console help

# 2. Inspect effective values and whether each is built-in or saved.
fak console settings

# 3. Save one preference.
fak console settings --set-default issues.top=40

# 4. Reset that preference to its built-in value.
fak console settings --unset-default issues.top

# 5. Verify the effective value and source after either change.
fak console settings
```

The default file is `~/.fak/console.json`; `FAK_CONSOLE_FILE` changes it, and
`--path FILE` selects one file for a settings command. Mutations validate the full
configuration before atomically replacing that file with user-only permissions.
`fak console config` remains a compatibility alias, but help and pane discovery
advertise only `settings`. The [captured 72×24 render](./_witnesses/tui-settings/settings-render.txt)
shows a loaded file, built-in values, allowed options, copyable actions, and one
saved value without a key, model, network call, or GPU.

On the in-front-of-the-model path, an upstream rate limit **never sleeps past your
client**. fak absorbs short waits in-handler — the transient backoff schedule caps
at 30s, and a provider-named wait that fits under the ceiling is simply slept, which
is the right UX for a throttle. But when the honored wait is longer than a wrapped
client can hold a request open, sleeping is structurally uncompletable: your agent's
own request timeout (~300s for Claude Code) fires first, so the wait burns futile
client-side retries and then dies with an opaque timeout that names none of the
rate-limit truth fak already holds. Past the ceiling fak therefore stops retrying and
hands the truthful response downstream instead — the real upstream status (429, or
529 for an overload), the provider's own `Retry-After` header verbatim, and the
distinct `upstream_retry_ceiling` error code, so a Retry-After-honoring harness backs
off correctly and a supervisor can park the turn across the reset rather than
hammering it. `FAK_INHANDLER_WAIT_CEILING` sets the boundary (any Go duration,
default `90s` — safely under that ~300s client timeout — clamped to `[0, 1h]`; `0`
disables the ceiling and restores absorb-everything). The total retry budget
(`FAK_PLANNER_RETRY_BUDGET`, default 4h) still bounds the loop overall, but on this
proxy path it is deliberately **not** reachable in-handler: riding out a longer
window is a supervisor's job (`fak manage`), not a sleeping request handler's.

## Try the boundary first

Run the no-credential proof before reading the architecture:

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
go run ./cmd/fak agent --offline
```

This shows the adoption posture in one screen: a dangerous support action is
denied, a benign search is allowed, and the offline injection A/B blocks the
poisoned instruction while the task still completes.

## Try the live gateway

The boundary above is offline. To put the gate **in front of a real model** over
OpenAI-compatible HTTP, run `fak serve`: it fronts any OpenAI-compatible upstream
(Ollama / vLLM / llama.cpp / a cloud provider), and on every `/v1/chat/completions`
it denies / repairs / quarantines the tool calls the model proposes before returning
the survivors.

```bash
ollama serve &                         # any OpenAI-compatible server on :11434
ollama pull qwen2.5:1.5b
go run ./cmd/fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
# from a second terminal:
curl -s http://127.0.0.1:8080/healthz                       # liveness
curl -s http://127.0.0.1:8080/metrics | head                # Prometheus scrape
```

Point any OpenAI client at `http://127.0.0.1:8080/v1`. `fak serve` also speaks the
Anthropic `/v1/messages` wire, exposes `/metrics` + `/debug/vars`, and takes
`--policy floor.json` + `--require-key-env VAR` to harden it. One caveat worth
knowing up front: `stream:true` SSE is **synthesized from the finished,
already-adjudicated turn** — well-formed, but not true token-by-token streaming. (The
client's `max_tokens`/`temperature`/`top_p`/`stop` are now forwarded per request, so
long completions are no longer truncated.) Full walkthrough (Tiers 0–2):
[`GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md).

> **Status: shipped & benchmarked** (release version comes from the single source of truth at
> the root [`VERSION`](https://github.com/anthony-chaudhary/fak/blob/main/VERSION) file). `go build`/`go vet`/`go test ./...` green across the
> internal packages, CI green, and the A/B benchmark gate passes. Confirmed not by self-report
> but by the DOS truth syscall (`dos_verify`): the shipped line runs from **`v0.1.0`** (the
> syscall skeleton, sha `c72ddf1`) through **`v0.2.0`** (model fusion + security substrate +
> gateway) and **`v0.2.1`** (gateway hardening + the turn-tax bench). Note: `dos verify` on
> the v0.2.0/v0.2.1 release-bump tags returns `shipped:false` because those commits touch only
> `VERSION` + release notes; the actual code ships across the commits each tag caps (see
> `../STATUS.md` §1 for the caveat). The truth syscall confirms the *code* ships, not the
> version bump — see `../docs/releases/`.
>
> **What v0.2 added on top of the v0.1 syscall skeleton** (each its own witnessed lane):
> a **real model fused into the kernel** (a pure-Go SmolLM2-135M forward pass, every rung
> proven bit-for-bit vs HuggingFace, then made parity-fast — `MODEL-BASELINE-RESULTS.md`);
> a **kernel-authored security substrate** (information-flow control, a trust/provenance
> classifier, plan-CFI, an effect-verifying witness gate, and a normalize-and-rescan
> admission driver — the kernel stops believing the model's self-reports); a **gateway**
> (`fak serve`) fronting the syscall boundary over OpenAI-compatible HTTP + MCP; a durable
> **session core-dump + context debugger + dream cleanup** (`recall` / `fak debug` /
> `fak dream` — a quarantine that survives the process boundary, plus an offline
> re-screen/prune pass, `RECALL-RESULTS.md` / `CDB-RESULTS.md` /
> `MEMORY-DREAM-CLEANUP-RESULTS.md`); and a dedicated
> **turn-tax benchmark** (`fak turntax`) pricing the extra error-code model turn the
> 1-shot kernel deletes. The `fak agent` verb still drives the **live** turn-count A/B
> (real Gemini + local Qwen2.5, transcript-hashed) — see `LIVE-RESULTS.md`.

### Live dashboards

The default listener serves a lightweight live dashboard at [http://127.0.0.1:8080/](http://127.0.0.1:8080/). It polls `/healthz` and `/metrics` every five seconds and keeps its last good values through a failed refresh.

**Rich dashboards are lazy.** Selecting one shows startup progress, reuses a healthy Grafana at `http://localhost:3000`, or asks Docker Compose to start the bundled `tools/grafana/docker-compose.yml` stack on demand. FAK does **not** start Docker Desktop or another Docker daemon. Once Grafana is ready, the same route redirects to the selected dashboard. No Docker or Grafana probe runs before the first click.

For an owned bundled start, FAK derives the Prometheus target from the gateway's actual bound listener. A gateway on `127.0.0.1:61666`, for example, is scraped through Docker's host bridge on port `61666`; FAK does not widen the gateway bind to `0.0.0.0` or otherwise expose it off-host. Startup writes a temporary Prometheus config and passes it to Compose through `FAK_PROMETHEUS_CONFIG`; the tracked `tools/grafana/prometheus.yml` remains unchanged.

Environment controls:

- `FAK_GRAFANA_URL=https://grafana.example` reuses an existing HTTP(S) Grafana endpoint instead of Docker.
- `FAK_GRAFANA_COMPOSE=/path/to/docker-compose.yml` selects a Compose bundle for on-demand startup. The bundle must retain the `${FAK_PROMETHEUS_CONFIG:-./prometheus.yml}` mount used by the shipped Compose file.
- `FAK_DASHBOARDS=off` disables only Rich dashboards; the lightweight live dashboard stays available.

Lifecycle and ownership:

- If Grafana is already healthy at `http://localhost:3000`, FAK adopts it without running Compose and leaves it running when the gateway exits.
- If the click starts the bundled Compose stack, that gateway owns it, stops it when the gateway exits, and removes its temporary Prometheus config after Compose is down.
- FAK never starts Docker Desktop. If Docker is stopped, start Docker and click again (or set `FAK_GRAFANA_URL`).
- After a Docker or host restart, restart Docker and the gateway, then click a rich dashboard again so the on-demand check/start path runs against the live gateway address.

Missing Docker, missing bundle files, an unavailable gateway listener address, startup errors, readiness timeout, and unsafe URLs render an actionable failure page instead of silently changing behavior.
## Syscall subsystem check — useful, not the headline KPI (call-mix-independent)

`fak bench --suite tau2-smoke` replays a frozen tool-call trace through the one
binary and times the **in-process tool-call adjudication boundary** against a
**spawned-hook baseline** measured on the same machine (the same `Fold` decide, two
transports — apples-to-apples):

```
in-process adjudication p50 : 2,427 ns
spawned-hook        p50     : 6.913 ms  (process-per-decide, this machine, n=100)
SUBSYSTEM CHECK (gate_primary): pass   (~2,849x fusion speedup)
```

This is a **subsystem regression sentinel**, deliberately **not** the headline
product KPI: it is independent of the tool mix — it times the adjudication fold,
not the workload, so a low vDSO hit-rate can never turn it red — but it does not
prove production readiness, model quality, serving throughput, or the 45x fleet
claim. (For speedup figures: 45× = Phase-0 batched-decode gate currently failing at
40.98×; ~60× = headline session wall-time vs naive stateless; ~1.5–4× = realistic
gain vs tuned warm-cache stack.) The vDSO hit-rate and token savings are reported
as **soft UPSIDE secondaries**, never the gate. See `STATUS.md` §2 and `CLAIMS.md`
(unit 82).

## `fak harness model-set`

`fak harness model-set` is the generated-harness model dependency lifecycle. It composes the shipped strict intent, normalized inventory, deterministic resolver, canonical lock, and startup-receipt packages; it does not download artifacts, contact providers, or construct model servers.

```bash
# Resolve the generated harness's role requirements into its build/init handoff.
fak harness model-set resolve \
  --intent harness.model-set.json \
  --inventory model-inventory.json \
  --out harness.model-set.lock.json

# Read and verify the lock without changing it.
fak harness model-set inspect --lock harness.model-set.lock.json --json

# Re-evaluate current witnessed facts at startup/CI and persist the decision.
fak harness model-set selfcheck \
  --lock harness.model-set.lock.json \
  --inventory model-inventory.json \
  --receipt harness.model-set.receipt.json \
  --as-of 2030-01-02T12:00:00Z
```

`resolve` defaults evaluation to the inventory's authored `as_of`, so identical inputs rerun byte-for-byte. It writes `harness.model-set.expectation.json` beside the conventional lock and publishes the canonical lock last as the commit point; an incompatible or malformed resolution never replaces the prior lock. Target defaults are the current Go OS/architecture with `accelerator=none` and `runtime=mixed-runtime`; generated cross-target builds should set `--os`, `--arch`, `--accelerator`, and `--runtime` explicitly. `--expectation` overrides the expectation-sidecar path.

`selfcheck` defaults `--intent` to `harness.model-set.json` and the expectation to `harness.model-set.expectation.json` beside the lock. It reads only local files, independently re-runs compatibility at `--as-of` (current UTC when omitted), writes a canonical receipt for both compatible and incompatible decisions, and never probes the network. Exit `0` means compatible, `3` means a typed incompatibility was captured in the receipt, `2` is usage, and `1` is invalid input or I/O. Unknown, stale, malformed, mismatched, or unresolved required facts fail closed. The receipt is the startup/CI gate; the lock remains unchanged.

`inspect` verifies the lock digest and optional receipt, then renders target, resolver, role selections, immutable artifact identities, rejection counts, and receipt status. It is read-only. `--json` schemas are `fak.harness-model-set-resolve/1`, `fak.harness-model-set-inspect/1`, and `fak.harness-model-set-selfcheck/1`.

## `fak ultracode`

`fak ultracode` is the first-class front door for a bounded concurrent coding-agent fleet. It routes through the canonical orchestration runtime with the `ultracode` profile fixed: workers require lane leases, checkable effects require independent readback, and a lead reconciles the results. Planning does not launch a model or spend tokens.

```bash
fak ultracode --task-text "implement two disjoint checks and reconcile them"
fak ultracode --task task.json --launch --json
fak ultracode status --json
fak ultracode --selfcheck
```

The offline selfcheck proves plan shape and safety invariants, not a speedup or intelligence gain. A real value claim needs paired single-agent and fleet runs over the same accepted-outcome workload, reporting wall time, billed/input/output/cache tokens, spend, and witness acceptance. Use `fak orchestration plan --profile off|auto|ultracode` when comparing profiles explicitly.

### `fak ultracode bench`

`fak ultracode bench` folds an identical single-agent/fleet pair into one replayable `fak-ultracode-paired/1` report:

```bash
fak ultracode bench --selfcheck
fak ultracode bench --pair paired-run.json --json
```

The report keeps concurrency, caching/token, and quality evidence separate, then couples them as accepted effects per wall second and per billed token. It emits `GAIN` only when both witnessed runs pass the same acceptance workload without retries, the fleet preserves quality, and both coupled efficiency axes improve. Unequal identities, outcomes, missing witnesses, or retries return an error or `ABSTAIN`; a faster but lower-quality or token-worse fleet returns `NO_GAIN`. The selfcheck is an offline fixture and is explicitly not live model-performance evidence.

## `fak orchestration status`

`fak orchestration status` joins the newest local orchestration launch receipt, worker process liveness, and JSONL turn progress into one read-only fleet view. Select an older launch with `--session ID`; use `--home DIR` when runtime artifacts live outside the current directory.

```bash
fak orchestration status
fak orchestration status --session 01a015da-c527-7ed0-b8e8-3e0d9b7da5ed
fak orchestration status --json
```

Human output leads with the run verdict and running/completed/exited counts. `--json` emits `fak.codex_orchestration_status.v1` with the same per-worker PID, process, log freshness, last-event, and turn-count evidence. The command observes workers; it does not stop, replace, or mutate them.

## `fak coordinate`: governed agentic compute

`fak coordinate` turns one harness task into a content-free whole-path plan and receipt. It joins harness-neutral task intent with managed context/cache actions, **fak-native** compute placement, serve admission/backpressure, required effects, and accepted outcomes. This is governed agentic compute: unlike a raw model-serving receipt, it proves the task-level controls and outcome evidence needed for acceptance. Raw-model-only input is therefore reported as insufficient and cannot yield an accepted receipt.

```bash
fak coordinate --demo        # deterministic built-in two-worker receipt
fak coordinate --json < in.json
```

Equivalent Claude, Codex, and fak-native inputs normalize to the same plan identity. The JSON schema is `fak.coordinate/1`; input and output carry identifiers and control metadata, not prompts, responses, or other task content. Unknown or malformed fields fail closed. Setting `coordination` to `false` delegates explicitly to existing harness behavior instead of claiming a coordinated result.
## `fak agent` response profiles

`fak agent --output-style <selection>` opts the owned agent loop into a named response shape. The default is `full` (no shaping). Run `fak agent profiles` or `fak agent profiles --json` to list shipped, reserved, and not-yet selections. The recommended concise compatibility setting is `caveman:medium`; it canonicalizes to the fak-authored `caveman:native:medium`. See [Response profiles](response-profiles.md) for intensity guidance, native-versus-original provenance, preservation rules, and the current harness boundary.

```bash
fak agent profiles
fak agent --output-style caveman:medium --task "Explain this failure"
fak agent --output-style full
```

Unknown or not-yet selections fail before the run. Response profiles do not change work policy or tool authorization.

`fak agent --work-profile standard|ponytail:{low|medium|high}` independently selects implementation-policy pressure. `standard` is the off/default state; `ponytail:*` expands to the fak-native canonical form. Mix it with `--output-style` without either axis implying the other. See [Work profiles](work-profiles.md).

## Verbs

The command surface has two executable boundaries. The shipped **`fak` runtime** is
the product — what an adopter or operator touches (`manage`, `serve`, `agent`,
`run`, `preflight`, `policy`, `attest`, `audit`, `egress`, `info`, `session`,
`ps`/`top`, `signal`, `resume`, `doctor`, `recover`, `model`, `codex`,
`self-update`, `version`, `help`). `fak help` and `fak help --all` describe that
runtime. Repository workflows, scorecards, benches, and issue/docs tooling live
in the separate **`fak-dev`** executable; `fak-dev help` lists them. The legacy
`fak dev <verb>` spelling is a compatibility handoff to a sibling or
`PATH`-installed `fak-dev`, not a dev namespace linked into the runtime.

The deep verb documentation and the kernel map moved to linked sub-pages so
this reference stays the front door; every section is preserved verbatim:

- the per-verb deep catalog (idempotency, architecture, and the rest of the shipped runtime): [verbs.md](cli/verbs.md)
- the appended verb contracts (launch, workpattern, stale-work, skill compile, the study-* verbs, server, new-model): [verbs-extended.md](cli/verbs-extended.md)
- the kernel map: file table, one-breath contract, witness-closed shipped table, honest limits, and wave-0 build history: [kernel-map.md](cli/kernel-map.md)

## `fak study-inventory`

The `fak study-inventory` contract moved to
[docs/cli/verbs-extended.md](cli/verbs-extended.md#fak-study-inventory); its
report does not pretend to contain symbols or forge history.
