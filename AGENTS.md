# AGENTS.md: orientation for coding agents

> Automated-contributor instructions for the shared checkout. Humans: read [`README.md`](README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md); curated doc map: [`llms.txt`](llms.txt).

## First-class multi-platform invariant (any OS, any GPU)

Keep `fak` co-equal across **macOS, Linux, and Windows** and architected for first-class native inference across **heterogeneous GPUs** behind `internal/compute.Backend` (current top-3 focus: **Apple Metal, NVIDIA CUDA, and AMD ROCm/Vulkan**, plus the `cpu-ref` oracle), rather than hyper-fitting the engine to a single OS or vendor. Vendor-specific kernels, layouts, and fast paths stay behind the HAL and dispatch on `Caps`; never leak single-vendor assumptions into the model loop, scheduler, KV/context MMU, policy floor, or user contract, and never extrapolate a single-box witness across platforms. Canonical doctrine and deterministic docs guard: [`docs/multi-platform-doctrine.md`](docs/multi-platform-doctrine.md).

## What this project is

**fak** is an *agent kernel*: one Go binary that sits between an AI agent and its tools, handling every call before it runs to reuse shared setup, route calls, serve repeats locally, and shed old turns while provider caches survive. It is first a **performance gate** and, on the same checkpoint, a **security gate** using `default_open` (permissive default bounded by a structural safety floor; `fail_closed` is an explicit operator lever; see [`POLICY.md`](POLICY.md)).

### Default scope and priorities

Default broad or underspecified requests in this checkout to **FAK work in `fak`**; touch sibling repositories only when named or required by a `pkg/*` or companion contract. Under the **Companion-Aware Dev Harness** ([`docs/architecture/companion-dev-harness.md`](docs/architecture/companion-dev-harness.md)), `fak` pre-commit hooks run 5-gate boundary checks (`cmd/fak-boundary` in companion `fak-private`) and leak audits (`tools/scrub_public_copy.py`), while `fak-dev gate|provenance|audit-leak|ticket` bridge to companion tooling.

All new core engine work (inference runtime in `cmd/fak/serve.go` and `internal/engine`, context MMU in `internal/ctxmmu`, compute HALs in `internal/compute/`, weight loaders in `internal/model/`, tool capability floors, `pkg/abi`, `pkg/fakclient`, `pkg/scorecard`) belongs in public `fak` by default; commercial policy and autonomous factory tooling stay in `fak-private`.

Prioritize work in four tiers:
1. **fak all in one** (`fak up`): turnkey single-binary serving, harness, and durable memory.
2. **fak serving only** (`fak serve`): inference runtime, disaggregated gateway, KV/MMU acceleration, batching.
3. **fak harness only** (`fak guard`): standalone governance substrate, `default_open` structural floor, policy adjudication.
4. **other**: peripheral utilities and off-spine tooling ([routing policy](docs/notes/AGENT-DELEGATION-AND-ROUTING-2026-11-11.md)).

## Build, test, and first proof

Module root requires Go 1.26+ (`GOTOOLCHAIN=auto`; install: `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`). Repo layout: `cmd/fak/` (CLI), `internal/` (`adjudicator`, `policy`, `vdso`, `engine`, `gateway`, `ctxmmu`, `model`), `examples/`, `docs/`, [`docs/private-comms-channel.md`](docs/private-comms-channel.md).

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
```

Full walkthrough: [`docs/repro-packet.md`](docs/repro-packet.md). Build profiles: [`docs/dev-tooling.md`](docs/dev-tooling.md#build-profiles). Common gates: `make build`, `make smoke`, `make test-fast`, `make mac-perf`, `make test-race`, `make test`, `make ci`. On this Windows host, run `./test.ps1` under WSL and follow [`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md).

Use isolated verbs on the permanently peer-dirty checkout (compile checks never write an in-tree binary):
- `fak-dev ci-preflight`: verify committed trunk builds and is gofmt-clean.
- `fak validate --mine <path>... [--smoke]`: build, vet, and test trunk plus your paths.
- `fak-dev buildcheck [--vet] [--mine <file>]`: compile-check while hiding peer WIP (`--isolate=false` for literal tree).
- `fak hooks pre-push`: verify push graph safety.

The workstation is the control point, not the compute boundary: dispatch genuine GPU, accelerator, or heavy-CPU work to a sanctioned fleet node in [`docs/fleet-compute-nodes.md`](docs/fleet-compute-nodes.md), or hand the operator the ready-to-run command sequence. `fak hwgate-lint` and `fak guard --hardware-gate enforce` enforce this redirect.

## Delegate real work; keep the coordinator context clean

Use guarded headless agents or an equivalent isolated worker for every substantive unit of work. The depth-0 coordinator decomposes requests into pairwise tree-disjoint lanes (researcher/explore/scout -> worker/deep-reason -> cross-validator/tester/issue-auditor), arbitrates leases (`dos arbitrate`), preserves only decisions and compact evidence, and independently witness worker results before landing or reporting. Depth-1 leaf workers stay within 1–3 files and package-scoped tests without nested `task` calls. Reserve `gpt-6-astra` for pre-specified formal-mathematical leaves ([routing policy](docs/notes/AGENT-DELEGATION-AND-ROUTING-2026-11-11.md)).

Scope smaller or resource-constrained models to atomic S0/S1 leaves (1–3 files, one witness command). On high-difficulty subsystems (concurrency lock invariants, `internal/abi`, low-level SIMD/CUDA kernels, security gates), emit a structured `ABSTAIN` with the exact boundary while landing independent safe sub-components. Keep tests atomic and declare cost-shaped tests with `// fak-test:runtime <class> est=<duration> [lane=<lane>]`.

## Native inference and local-agent milestone

**Useful local agents, accelerated automatically.** Read [`docs/local-agent-milestone.md`](docs/local-agent-milestone.md) before planning native performance, caching, or local UX work; name workflow, bottleneck, seam, device, expected default, witness, and missing evidence in worker packets.

Keep native-inference and performance work **fak-native all the way**: fak owns kernels, memory, scheduling, cache, adaptation, and ops. `llama.cpp` is permitted only when explicitly selected for benchmarks, parity diagnosis, migration/interop, or borrowing; omit silent `auto` or recovery fallbacks to `llama.cpp`. Canonical rules and docs guard: [`docs/native-inference-goal.md`](docs/native-inference-goal.md).

New native-performance work prefers Qwen3.8.
Qwen3.6 is allowed only when the task states an explicit task-specific exception, such as regression, compatibility, historical comparison, or a hardware/artifact constraint.
Preserve historical Qwen3.6 artifacts; do not rename or rewrite them as Qwen3.8 evidence.

For tuning, identify candidates by source, config, and workload; re-measure references on objective/load changes; prefer one-variable comparisons; obtain independent validity and budget review before expensive runs.

## Track work in GitHub before implementation

Use a GitHub issue as the durable tracker for every substantive unit of work whenever
reasonably possible. Before editing code or docs, changing configuration, or launching an
implementation worker, search open and closed issues for duplicates, then claim the matching
issue or create one stating the problem, intended outcome, and witness. Normalize task dependencies with `Start blocked by: #N`, `Coordinates with: #N`, or `Promotion requires: #N`.

Scoping, reproduction, and read-only triage may precede the issue when needed to write an
honest ticket. Skip advance ticketing only when it would be unreasonable (no repo change, trivial edit, urgent safety/outage, or GitHub unavailable); create or reconcile the issue as soon as the constraint clears. Do not use plans, TODOs, commit messages, or chat as substitutes when GitHub tracking is available.

Qualify cross-repo issue references: bare `#<num>` means public `fak`; cite private issues as `fak-private#<num>` (`anthony-chaudhary/fak-private#<num>`).

### Proof by default and spine-first workflow

Match the proof artifact to the failure class before claiming a fix:
- **TUI / visual**: captured render-witness test (e.g. `cmd/fak/watchdog_autoheal_test.go` `TestWatchdogAutohealKeepsAgentPaneClean`) plus before/after screenshot for live UIs.
- **Logic / behavior**: repro test that fails before the fix and passes after.
- **Runtime / CLI / integration**: real path execution via `fak validate --mine <paths> --smoke`, `make smoke`, `make dogfood-test`, or `make test-integration`.
- **Hardware / performance**: physical silicon witness (`fak hil`, `make mac-perf`, on-device GPU receipt); simulations are early indicators only.
- **Shipped claims**: witnessed commit (`dos verify` + `(fak <leaf>)` trailer); keep proof artifacts under 5 MiB (`TestRootModuleZipBudget`).

For new work, check P1-P4 in [`docs/problems-we-solve.md`](docs/problems-we-solve.md), state **For / Problem / Today / Better because / Witness**, ship the smallest working end-to-end spine first (`gen/now` issue if it cannot ship), and fan out follow-ons with `fak-dev issue fanout --title T --leaf L --spine <sha|cmd|doc> --json` ([`docs/spine-first-defaults.md`](docs/spine-first-defaults.md)).

Promote reusable scratch via **scratchpad -> committed Go tool -> `fak` verb -> captured knowledge**; allocate and reap scratch with `fak tree-doctor` ([`docs/generated-output-defaults.md`](docs/generated-output-defaults.md)). Frame diagnostics around forward momentum and the next checkable lever.

There is no blocked state: name the next action (spawn / queue / route around / wait / hand off), never a terminal block.
Checked by `fak headless-lint --next-action`.
At run end, dedupe and file every real leftover as an open issue; otherwise say nothing remains.
Close operator-facing turns with verdict-first bullets, one claim and inline evidence per line;
make the final line the next checkable step.

Cite evidence as `module@rev` (`fak version modules`; [`docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md`](docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md)).

## Hard rules and kernel recovery

1. **Trunk and worktrees**: Stay on `main`; off-trunk commits fail `OFF_TRUNK`. Use only detached worker worktrees via `fak worktree worker prepare|land|reap|list` ([`docs/managed-worker-worktrees.md`](docs/managed-worker-worktrees.md)); reap stale scratch worktrees with `tools/worktree_doctor.py`.
2. **Sync, commit, and push unprompted**: Pre-flight sync with `fak sync check` / `fak sync apply` (`--ff-only`) / `fak sync reconcile` (check `.git/MERGE_HEAD` first; with `fak-private`, sync remotes, `go work sync`, and `fak-dev audit-leak --staged`). Commit one issue per commit by explicit path using `fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply` (automatic DCO sign-off via `-s`, Conventional-Commits subject, `(fak <leaf>)` trailer). Raw `git commit -s -m ... -- <paths>` is emergency-only when `fak` is unbuilt; avoid `git add -A` and force-pushes. Push green work via `fak sync push` or `fak commit --push`.
3. **Shared-tree buildability and CI contracts**: Fence incomplete cross-file `.go` WIP with `//go:build wip_<feature>`; update CI/schema consumers atomically and verify with `fak-dev ci-preflight` ([`docs/ci/ci-spec-change-migration.md`](docs/ci/ci-spec-change-migration.md)).
4. **Native Go tooling only**: Extend via `fak new-leaf` and native Go leaves/modules; adding `.ps1`, `.sh`, `.bat`, or `.cmd` scripts is refused by `FILE_ADMISSION` (except bare-metal bootstrap).
5. **Claims, ABI, and boundaries**: Tag every `CLAIMS.md` row (`[SHIPPED]`, etc.), keep `internal/abi` frozen unless explicitly scoped, respect [`docs/repo-guard.md`](docs/repo-guard.md) (`OUT_OF_TREE_WRITE` external targets), keep private host/credential details in `fak-private`, avoid broad `find` on `/`, `~`, `/mnt`, `/proc`, and verify no dependent sessions are active before restarting an inference server.

### Kernel refusal recovery

A guard refusal names a token from [`dos.toml`](dos.toml); treat it as per-call feedback. Run `dos man wedge <TOKEN> --explain` instead of carrying the full recovery cookbook in every agent context:

```bash
dos man wedge <TOKEN> --explain  # summary, category, fix, and references
fak recover <TOKEN>              # dry-run recovery plan (--execute to apply; --list for all)
```

For transient locks (`MERGE_IN_PROGRESS`, `COLLISION_RISK`, `LOCK_BUSY`), run `git restore --staged`, inspect `dos top`, checkpoint with `fak wip checkpoint`, or pivot to a disjoint subtask. For `FILE_ADMISSION` or `POLICY_BLOCK`/`SELF_MODIFY`, use sanctioned Go tools or escalate the bounded path. Verify setup with `python tools/extend_preflight.py` and [`CONTRIBUTING.md`](CONTRIBUTING.md). File genuine wrong refusals or workflow catch-22s with `fak complain` ([`docs/notes/CONCEPT-AGENT-FRICTION-COMPLAINT-CHANNELS-2026-06-29.md`](docs/notes/CONCEPT-AGENT-FRICTION-COMPLAINT-CHANNELS-2026-06-29.md)).

## Releasing, planning, and doc map

Use `/release` for full releases ([`docs/releases-channel.md`](docs/releases-channel.md), [`docs/on-demand-releases.md`](docs/on-demand-releases.md)); rollback runbook: [`docs/ROLLBACK.md`](docs/ROLLBACK.md). Keep phased deliverables accurate across product, evidence, and queue axes ([`docs/progress-state-defaults.md`](docs/progress-state-defaults.md), `fak plan-audit`), and update plans in the same commit as the work.

| Need | Read |
|---|---|
| CLI reference & install tiers | [`docs/cli-reference.md`](docs/cli-reference.md) · [`GETTING-STARTED.md`](GETTING-STARTED.md) · [`LEARNING-PATH.md`](LEARNING-PATH.md) |
| Mac local models & integrations | [`docs/fak/mac-local-models.md`](docs/fak/mac-local-models.md) · [`docs/integrations/`](docs/integrations/) · [`examples/mcp/`](examples/mcp/) |
| Policy floor & extending kernel | [`POLICY.md`](POLICY.md) · [`examples/README.md`](examples/README.md) · [`EXTENDING.md`](EXTENDING.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| SOTA prior art & prose scorer | [`docs/sota/README.md`](docs/sota/README.md) · `fak score negframe --suggest` |
| Feature status, claims, benchmarks | [`docs/supported/features.md`](docs/supported/features.md) · [`CLAIMS.md`](CLAIMS.md) · [`STATUS.md`](STATUS.md) · [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) |
| Full doc map & contributor contract | [`llms.txt`](llms.txt) · [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`docs/ROLLBACK.md`](docs/ROLLBACK.md) |

License: [Apache-2.0](LICENSE).
