---
title: "fak learning path — You've finished the path"
description: "A completion appendix that inventories the shipped fak surface and gives final verification commands after the staged learning path."
---

# You've finished the path — the shipped-surface appendix

**Wrap-up** · prev: [L600 — Mastery](mastery.md) · back to the [overview and L100–L200](../../LEARNING-PATH.md)

loading a payload; `fak up --help` prints the shared serve contract without starting a
listener; and the scorecard plus focused package tests complete locally. Put every
`fak study search` flag before the query because the query is positional.

**Checkpoint:** Given a new native benchmark finding, state where you would (a) retain
its external provenance, (b) decide whether it transfers to another envelope, (c) prove
the dashboard does not turn missing evidence into zero, and (d) measure whether the
performance-learning loop improved. Then explain why neither `fak agentic` nor
`fak runtime-capabilities` launches the work it describes.

---

## You've finished the path

If you can pass the checkpoints through **FAK 619**, you can: stand up and harden the
gateway in front of any OpenAI- or Anthropic-compatible model; author and review a
capability floor; explain the write-time quarantine and the IFC taint lattice; read the
in-kernel model's forward pass and its oracle-parity ledger; tell an honest benchmark
from a strawman; and land a new optimization into the kernel through the three-gate leaf
pattern (**FAK 615**) — prove it correct, prove it faster, earn the keep-bit.

Where to go from there:

- **Contribute.** Pick up the leaf pattern (**FAK 615**) and the witness-gated dispatch
  loop (**FAK 616**); the contract is in [`EXTENDING.md`](../../EXTENDING.md) and
  [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
- **Audit the honesty.** Re-run the repro packet (**FAK 603**,
  [`docs/repro-packet.md`](../repro-packet.md)) and check every number against
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) and the claims ledger
  [`CLAIMS.md`](../../CLAIMS.md).
- **Go deep on the math.** The per-module correctness proofs are the graduate seminar:
  [`docs/proofs/README.md`](../proofs/README.md).

Found a course whose reading no longer matches what the code does? That is a doc bug —
please [open an issue](https://github.com/anthony-chaudhary/fak/issues).

## Appendix — the full shipped surface (operator, contributor, and package map)

**Who this is for.** The courses above teach the load-bearing ideas end to end, but the
binary ships far more than 100 courses can each drill into: a fleet-operations toolbox, a
set of self-measuring scorecards, model/hardware benchmarks, and a stack of internal
leaves you will meet the moment you open the tree. This appendix is the orientation index
for that surface — one honest line per verb, standalone binary, and internal package,
each anchored to the code that implements it. It is a reference, not a lesson: skim it to
know a tool *exists*, then reach for `--help` or the cited source when you need it.

**Try any verb.** Every verb is reachable through the single binary; append `-h`/`--help`
or the JSON flag most carry:

```bash
go run ./cmd/fak <verb> --help      # each verb prints its own usage/subcommands
go run ./cmd/fak demo               # if you want a zero-flag proof first, start here
```

Deploying onto rented GPU hardware rather than fronting a hosted API? The operator
quickstart is [`docs/fak/neo-cloud-deploy.md`](../fak/neo-cloud-deploy.md).

### A. Running the gateway and the shared-trunk fleet

| Verb | What it does | Try |
| --- | --- | --- |
| `fak service` | Install/run fak as a system service (systemd on Linux, launchd on macOS) via `internal/systemservice`. | `fak service -h` |
| `fak watchdog` | Operator surface over the default OS-scheduled watchdog monitors (resume, supervisor, dos-dispatch, stale-work garden). | `fak watchdog status` |
| `fak schedscan` | Observability window onto the fleet's Windows Scheduled Tasks (FleetResumeWatchdog / FakFleetJanitor) beyond "is it Ready?". | `fak schedscan` |
| `fak host-crash` | Samples host crash / soft-fault signals over an interval and writes them for the crash-profile ledger. | `fak host-crash -h` |
| `fak host-relaunch-broker` | Drains the machine-control-plane→desktop relaunch request spool and launches Windows Terminal; `--validate` prints the argv without launching. | `fak host-relaunch-broker --validate` |
| `fak conpty` | Searchable status surface for the pwsh `0xE9` ConPTY FailFast crash class: resolves the ConPTY pair on PATH and compares its FileVersion to a known-good floor. | `fak conpty --json` |
| `fak stallscan` | Reads the churn signals (soft-fault storm, scheduler/syscall thrash, spawn bursts) that reveal a low-usage machine stall. | `fak stallscan` |
| `fak growthgate` | The standing-footprint twin of `stallscan`: classifies where the disk/IO went and emits a census + verdict. | `fak growthgate --json` |
| `fak git-maint` | Guarded, lock-aware "consolidate-never-prune" object-DB maintenance (multi-pack-index / commit-graph) the destructive-git guard otherwise only defers. | `fak git-maint -h` |
| `fak git-daily` | The scheduled daily tick that makes `git-maint` actually fold: reaps the orphaned locks the maintenance tiers correctly defer on, THEN consolidates; once-a-day deduped, so a coarse OS trigger is safe. | `fak git-daily --dry-run` |
| `fak clean-bins` | Safe, idempotent, witnessed prune of the stray `go build` binaries dropped at the module root. | `fak clean-bins` |
| `fak buildcheck` | Concurrency-safe compile check for a fleet editing one shared trunk; never drops a binary in the tree. | `fak buildcheck` |
| `fak ci-preflight` | Answers "is the committed trunk tip CI-buildable and gofmt-clean, and if not which files" without trusting the working tree. | `fak ci-preflight -h` |
| `fak go` | A poison-free `go` passthrough for the permanently peer-dirty shared trunk (builds in an isolated temp copy). | `fak go build ./...` |
| `fak validate` | Answers whether the committed tree builds, a distinct question from a live working-tree build. | `fak validate -h` |
| `fak trunk-build-probe` | Read-only diagnosis of whether the release gate's red trunk is a forgotten `git add`. | `fak trunk-build-probe -h` |
| `fak trunk-red` | Folds the pre-existing trunk-red witness ledger into the distinct shared breaks the build gate admitted over (a peer's red, not yours), worst-first. | `fak trunk-red -h` |
| `fak worktree` | `fak worktree worker prepare\|land\|reap` — the detached per-worker build-isolation worktree that lands its diff back on `main` under a lane lease. | `fak worktree worker -h` |

### B. Session, budget, and spend accounting

| Verb | What it does | Try |
| --- | --- | --- |
| `fak budget` | Inline per-task budget readout ("spent N, budget M, here's where it went") from real gateway usage records. | `fak budget --json` |
| `fak spend` | Cross-account spend rollup; gate-fails on unlabeled spend figures. | `fak spend -h` |
| `fak savings` | The Track-2 observed-$ cache-savings audit trio over the local savings ledger. | `fak savings audit` |
| `fak balance` | Night-balance readout: resume recovery-vs-stranding and gardening-vs-throughput side by side; exits non-zero on an underwater recovery budget. | `fak balance -h` |
| `fak footprint` | Always-sent MCP tool-schema floor scorecard: prices fak's registered `tools/list` floor offline. | `fak footprint -h` |
| `fak assume` | Thin shell over the `internal/assumecheck` kernel: gathers evidence for a launch-seat assumption, prints the verdict, maps it to an exit code. | `fak assume -h` |
| `fak sessionjournal` | Crash-survivable session-registration journal (open/beat/close JSONL) and its monitoring/resume sidecar. | `fak sessionjournal -h` |
| `fak wip` | Working-tree checkpoint/restore spine: a gc-safe snapshot of uncommitted tracked changes under `refs/fak/wip/<session>`. | `fak wip -h` |
| `fak idempotency` | Retry-safe executor for non-idempotent tool ops (create issue, push, ledger append) keyed by op + token. | `fak idempotency -h` |
| `fak goal-park` | Status/claim a durable goal under a supervisor lease (`internal/goalpark`). | `fak goal-park status -h` |
| `fak tasks` | Thin shell over `internal/taskgraph`: the shared task list as a typed table with lease-gated claims. | `fak tasks -h` |
| `fak multisubmit` | Multi-submission planner: once an issue is resolved once, plans cheap differentiated bonus takes on the same issue. | `fak multisubmit -h` |
| `fak cachesweep` | Sweeps the cache-savings knee: finds the threshold as a fraction of the infinite-cache ceiling and emits the sweep result. | `fak cachesweep --json` |

### C. Guard, audit, and safety surfaces

| Verb | What it does | Try |
| --- | --- | --- |
| `fak guard-audit` | Prunes repo-local guard audit journals only after a durable logvault mirror proves their bytes (`internal/guardaudit`). | `fak guard-audit prune -h` |
| `fak guard-commit-gate` | Claude Code PreToolUse boundary that binds a git-commit's stamp/paths (hook actuator). | `fak guard-commit-gate -h` |
| `fak guard-sessionstart` | Claude Code SessionStart hook actuator installed by `fak guard`. | `fak guard-sessionstart -h` |
| `fak guard-stops` | Folds the typed Stop-hook decision ledger into a tally (clean stops, bounded stand-downs, fail-open stops). | `fak guard-stops -h` |
| `fak guard-stops-slack` | Durable update-in-place Slack scoreboard feeder for the Stop decision ledger. | `fak guard-stops-slack -h` |
| `fak chatops` | The inbound READ-ONLY control door: drives fak from one Slack channel through a closed verb grammar and a fail-closed admin allowlist. It answers the read verbs (`help`, `ping`, `status`, `fleet`) inline and DECLINES the act verbs (`dispatch`, `resume`, `bench`, `halt`) — the mutating path detaches through guarded dispatch, so the door itself can never start or stop work. Channel text is only ever parsed as a closed verb, never executed. | `fak chatops --dry-run` |
| `fak knownbad` | Impure shell over `internal/knownbad`: record/match/claim/resolve blast-radius "known-bad tree" containment entries. | `fak knownbad report` |
| `fak blast` | `fak blast estimate` — blast-radius estimate for a change (the containment epic's planner). | `fak blast estimate -h` |
| `fak headless-lint` | Scans an agent's final-output text for operator-directed "pesky notes" ("do you want me to push?") — the sensor-side dual of choice-triage. | `fak headless-lint -h` |
| `fak conformance` | Standalone safety-conformance suite: re-adjudicates the shipped dogfood verdict matrix against the compiled kernel — a CI-gateable, third-party-runnable attestation. | `fak conformance` |
| `fak egresslist` | Maintenance surface for the bundled egress filter lists the adjudicator's egress rung compiles. | `fak egresslist refresh --dry-run` |
| `fak eve` | Eve integration bridge: mechanical security preflight over Eve's MCP/OpenAPI connections. | `fak eve -h` |

### D. Self-measuring scorecards, RSI loops, and contributor tooling

| Verb | What it does | Try |
| --- | --- | --- |
| `fak score` | Parent verb grouping the meta-scorecards / RSI loops; `fak score <name>` routes to each legacy `*-scorecard`/`*-score` handler. | `fak score -h` |
| `fak quality` | The missing-middle quality ladder: run one case through a reference path and an engine path, compare, score against a rubric, emit a replayable failure bundle. | `fak quality -h` |
| `fak mlp-score` | Witnessed grade of the "first lovable cut" for the all-in-one agent-runtime epic. | `fak mlp-score -h` |
| `fak antipattern-scorecard` | The unifying work-loss card: folds REDUNDANT_REWORK + UNWIRED_PKG + ORPHAN_FUNC into one `antipattern_debt`. | `fak antipattern-scorecard` |
| `fak checkpoint-scorecard` | Which long-running subsystems persist durable resumable WIP and expose a witnessed status surface. | `fak checkpoint-scorecard` |
| `fak checkpoint-debt-dispatch` | Files one deduped issue per un-checkpointed subsystem, capped. | `fak checkpoint-debt-dispatch -h` |
| `fak unwired-scorecard` | Which code-complete internal packages are wired into a default path vs orphaned (imported by no `.go`). | `fak unwired-scorecard` |
| `fak unwired-debt-dispatch` | Files one deduped issue per unwired package, capped. | `fak unwired-debt-dispatch -h` |
| `fak qa-process-debt-dispatch` | The E-testing-quality "is our test process honest?" fan-out (regression_catch). | `fak qa-process-debt-dispatch -h` |
| `fak mode-debt-dispatch` | Permission-regime fan-out: one deduped issue per un-lifted permission dial. | `fak mode-debt-dispatch -h` |
| `fak harness-debt-dispatch` | Routes the model-strength classifier's already-produced verdict into the backlog: keeps the REDUNDANT/HOBBLING harness scaffolds and plans at most `--cap` deletion issues. Dry-run mutates nothing; `--live` is required to file. | `fak harness-debt-dispatch -h` |
| `fak concept` | The conceptbench disambiguation surface: `fak concept position\|classify\|validate\|freshness\|admission` over the concept catalog. | `fak concept -h` |
| `fak negate` | The negation operator: `fak negate detect\|resolve\|reframe` over `internal/negframe`. | `fak negate detect -h` |
| `fak signals` | Plain-English behavioral signals (NL prompt + verdict schema + sample rate) judged over an agent's turns. | `fak signals -h` |
| `fak question-ledger` | Deterministic labeling authority for `docs/questions/asked.jsonl` that `/question-loop` defers to. | `fak question-ledger -h` |
| `fak refactor-verify` | Read-only proof that a god-split / code-motion refactor dropped no top-level declaration. | `fak refactor-verify -h` |
| `fak godsplit-plan` | Read-only, doc-comment-aware boundary + hazard planner for a behavior-preserving Go split (consumed by `/modularize`). | `fak godsplit-plan --json <file>` |
| `fak hwgate-lint` | Lints that no local-hardware blocker is declared as terminal, keeping the fleet hardware-portable. | `fak hwgate-lint` |
| `fak tier-calibrate` | Offline calibration fold over recorded tier decisions and witnessed outcomes; proposes threshold moves, mutates nothing live. | `fak tier-calibrate -h` |
| `fak dispatch-aging` | Read-only anti-starvation diagnostic: which ready issues are starving and in what pick order. | `fak dispatch-aging` |
| `fak dispatchlat` | Read-only dispatch-latency diagnostic. | `fak dispatchlat --json` |
| `fak execution-route` | Composes harness/model/session routing into one inspectable execution decision (`internal/executionroute`). | `fak execution-route -h` |
| `fak steer` | `fak steer prs` — folds the pending dev→release delta into operator-legible PR-sized units, worst-attention-first. | `fak steer prs -h` |
| `fak project` | The ProjectsV2 board control-pane fold: makes the board an operator-visible report/verdict/Slack-ready dimension. | `fak project -h` |
| `fak trajctl` | Declare/scope your own trajectory-corpus views (scope confinement for `trajquery`). | `fak trajctl declare -h` |
| `fak trajquery` | Query your own trajectory corpus with a small scoped SQL `SELECT`; the validator refuses any scope escape. | `fak trajquery -h` |
| `fak shadowgit` | Non-invasive per-step write ledger over a worktree via a separate git dir; attributes each step's changed files. | `fak shadowgit -h` |
| `fak wiki` | The fak-native, witness-verified repo-wiki core (structure + content). | `fak wiki -h` |
| `fak skill` | Queried skill loader: `fak skill query\|residency\|swap` over `.claude/skills` (+ the MCP resolver). | `fak skill query -h` |
| `fak demo` | The zero-flag 60-second offline proof: fak's scenario through the real kernel, one live verdict per call class. | `fak demo` |
| `fak init` | Emit a minimal, valid `fak.toml` deployment manifest. | `fak init` |
| `fak micro` | The native in-process Go microagent runtime front door. | `fak micro -h` |
| `fak doomloop` | Two-axis doom-loop guard: classifies live workers on effort vs verified progress and wires the reversible-first correction. | `fak doomloop -h` |
| `fak llms-full` | Generate the full `llms-full.txt` corpus digest from the git tree. | `fak llms-full -h` |

### E. Model and hardware benchmarks

| Verb | What it does | Try |
| --- | --- | --- |
| `fak macbench` | Apple-silicon decode-longgen + prefill-sweep benchmark driver. | `fak macbench -h` |
| `fak macfit` | Models Apple unified-memory capacity for many concurrent agents (`internal/macfit`). | `fak macfit -h` |
| `fak mac` | Crisp handle for the mac path (long form `fak claude-mac-fak`); both spellings route to one handler. | `fak mac -h` |
| `fak deepseekbench` | DeepSeek V4 Pro/Flash TTFT/TPOT/context-scaling scorecard (thin wire over `internal/deepseekbench`). | `fak deepseekbench -h` |
| `fak glm52-prefill-sweep` | GLM-5.2 pure-fak prefill-latency sweep driver; `--dry-run` prints the plan, `--endpoint` runs it live. | `fak glm52-prefill-sweep --dry-run` |
| `fak kvbm` | Replays a KV-block-manager trace and proves the #2666 validation shape; exit 1 unless proven. | `fak kvbm replay` |
| `fak bench-ingest` | Folds checked-in benchmark snapshot fixtures (Terminal-Bench, SWE-bench, FrontierSWE) into a provenanced `modelscore` registry, refusing any unprovenanced row. | `fak bench-ingest -h` |
| `fak microbench` | Turns "ultra-light memory/CPU, thousands of agents in one process" into a measured number with zero provider spend: boots the real `internal/microagent` host at N agents, reports RSS/agent against a guarded-CLI process-pair baseline, and appends the row as JSONL. | `fak microbench -h` |

### F. Standalone binaries under `cmd/`

Not everything is a `fak` sub-verb; a handful of engines and fixtures ship as their own
binary so they compose without importing the whole kernel.

| Binary | What it does |
| --- | --- |
| `cmd/agentdojoprobe` | Scores attacker-PROPOSED prompt injections through the real red-team stack: attack specs in on stdin, per-attack outcomes plus detection-only vs full-stack (IFC) ASR out. Exit 0 only if the full stack held on every in-scope attack. |
| `cmd/auditreceipt` | Verifies a model-route audit-receipt ledger (`fak-auditreceipt-verification/v1`) against its HEAD-bound cursor. |
| `cmd/codesearch` | Standalone front door to the code-intelligence engine: trigram regex/literal search, AST shape queries, call-graph traversal, and RRF-fused feature retrieval. |
| `cmd/crossauditcalibrate` | Calibrates a cross-audit run manifest against per-arm observation files. |
| `cmd/crossauditdogfood` | Folds cross-audit receipt envelopes (author/auditor/verdict) for dogfooding the cross-model audit. |
| `cmd/crossauditfixture` | Test fixture for the cross-audit ABI: PASS/FAIL exact contract comparison of a candidate against a base64 contract. |
| `cmd/customlintfixture` | Hostile-behavior test fixture for the custom-linter ABI. |
| `cmd/devfresh` | Scans docs for unpointed version claims (`docfreshrsi.ScanVersionClaims`); `--selfcheck` runs the built-in witness. |
| `cmd/ggufmeta` | Prints a GGUF checkpoint's metadata header — the CLI front door to `internal/ggufload`'s full-metadata export. `-json` emits the lossless, byte-identical snapshot for diffing two checkpoints; the default is sorted `key=value` lines, narrowable with `-grep`. |
| `cmd/negframescan` | Throwaway harness (not shipped) that exercises `internal/negframe` against real repo prose when `cmd/fak` is wedged. |
| `cmd/wdcheck` | Throwaway verifier for the `info_watchdog.go` pure mappers. |
| `cmd/wgscan` | Scans a tree via `internal/windowgate` for un-suppressed window-popping exec calls. |
| `cmd/zaitask` | Bounded non-streaming Z.AI task runner CLI (emits content + a usage receipt). |

### G. Internal package map — the leaves you will meet in the tree

Each leaf is a tiered, single-purpose fold (see **FAK 209** for the tier DAG). The map is
grouped by the job the leaf does, because that is how you will look one up — you will
arrive knowing "something supervises the host" or "something prices the cache", not the
package name. One honest line each.

#### G.1 — Fleet, host, and service supervision

| Package | Purpose |
| --- | --- |
| `internal/fleetspine` | Networking-aware self-discovery for the fleet-control pane: UDP-multicast heartbeats let machines find each other live on the LAN instead of only through the git-mediated machine-dir snapshot. |
| `internal/fleetbottleneck` | Ranks which class is the fleet's binding constraint right now — seats, throttle, resume backlog, host load, or auth — from one fleet snapshot. |
| `internal/fleetreap` | Bounded retention and footprint measurement for per-session fleet artifacts; never follows directories or removes a file outside the caller's explicit set. |
| `internal/fleetmemory` | The cross-agent lessons ledger and its write-time duplicate guard: a workaround one agent learns is published once with a trigger context and injected to any peer whose session matches it. |
| `internal/fleetverify` | A throwaway compile-verification of the `operator.go` fleet helpers, isolated from the churning `cmd/fak` so the loopfleet/loopmgr API usage type-checks with no duplicate-symbol risk. |
| `internal/ghexec` | Builds deadlined `gh` invocations: every construction carries a context deadline, disables interactive prompting and the update notifier, and suppresses the Windows console window — so a wedged `gh` cannot wedge its caller. |
| `internal/hostfault` | Names the HOST-level failure classes that destabilize the fleet without being a child tool process crashing: a failed Windows Update install, an orchestrator fault, a GPU live-kernel/TDR watchdog event, an app-termination dump. |
| `internal/hostplacement` | Deterministic multi-host worker placement: a per-host headroom heartbeat registry plus a pure function that picks the least-saturated, non-stale host to spill a dispatch worker onto, or stays local when none is eligible. |
| `internal/hostresurrect` | Turns a host-fault crash class plus the guard-session index into a bounded, rate-limited relaunch request, so a crashed box brings its sessions back without a relaunch storm. |
| `internal/linkstate` | The general "what state is my channel with a peer in right now?" record — a three-phase CLEAR / WORKING / WAITING model replacing a five-state vocabulary that left agents unsure which not-ready they were in. |
| `internal/loaddebounce` | Publishes a per-worker load signal only when it CHANGES, and coalesces a burst behind a short reset-on-every-change window so only the latest value survives to be emitted. |
| `internal/promalert` | Parses an Alertmanager webhook payload and renders it into compact Slack text — the inbound half of the Prometheus-alerts-to-Slack wiring that `fak slack alert` enqueues into the durable outbox. |
| `internal/scmbridge` | One desired-state / read-back contract shared by the Windows SCM control plane and the Scheduled-Task recovery bridge, so machine services, interactive agents, boot recovery, and crash recovery cannot diverge or double-launch. |
| `internal/servicespec` | The portable `fak.service.v1` desired-state and restart-semantics contract every service surface renders from: identity, command, cwd, environment references, readiness, restart/backoff, checkpoint/resume, dependencies, intentional stop. |
| `internal/servicelease` | The lease / generation / incarnation fencing layer over that contract: the durable facts that let a control plane decide whether a disconnected remote node still owns a leased workload, so a partition retry cannot create a second owner. |
| `internal/serviceledger` | The portable append-only observed-state event ledger for services: Windows crash rows, SCM state changes, systemd journal entries, launchd exits, and fak's own supervisor receipts folded into one correlated schema. |
| `internal/stallpage` | Turns stallscan's reboot high-water decision into a durable, deduped operator page. A reboot drops every live session, so the choice explicitly names operator approval; the publisher never kills a process or reboots the host. |
| `internal/terminalrisk` | Reports whether this box carries the Windows Terminal Direct2D render-crash risk (an AMD adapter plus a prior render crash) and can write the settings key that avoids it. |

#### G.2 — Loops, sessions, and worker isolation

| Package | Purpose |
| --- | --- |
| `internal/chatopsdetach` | Pure detached-execution decision kernel for chatops ACT verbs: ack-now / witnessed-completion / stall-escalation routing. |
| `internal/doomloop` | Two-axis doom-loop classifier: an effort-vs-verified-progress window → a closed verdict + reversible-first correction. |
| `internal/dormancysim` | The deterministic time-travel harness for the dormancy organs: one injectable clock threaded through the pure measure and rehydrate leaves, so a test fast-forwards an agent's dormancy from hours to months with zero real waits. |
| `internal/executionroute` | Composes harness, model, and session routing into one inspectable execution decision. |
| `internal/guardrotate` | The pure decision core for `fak guard`'s cooldown-aware seat selection: without it a bare `fak guard` launches against an account the launcher just watched bounce off its own cap. |
| `internal/guardsessions` | The local, queryable index of `fak guard` sessions — one appended row per launch (handle, trace id, wrapped agent, pid, cwd, journal path, start time), and prefix resolution to the one session it names. |
| `internal/lookahead` | The witness-gated Lesson core: a fork-rollout's outcome distilled into a lesson whose assertive authority is bounded by the witness rung its evidence earned — a self-report rollout may assert nothing at all. |
| `internal/looporphan` | The pure duplicate-loop-supervisor reaper: folds a process census of loop and drainer supervisors into a closed keep/reap plan that never strands live work. |
| `internal/loopunblock` | The generic head-of-line unblocker for any worklist-draining loop — the always-on rung that keeps a loop which has already selected its next move from stalling on a move it cannot make. |
| `internal/microagent` | Hosts many agent loops in one process: a worker pool sharing one in-process kernel gateway. |
| `internal/resumebackoff` | The pure resume-storm containment fold. |
| `internal/resumemetrics` | The process-global expvar surface for the resume/heal watchdog, so a renamed or rotated status ledger can no longer read as "zero activity" and be mistaken for a healthy-but-idle watchdog. |
| `internal/seatpark` | A bounded park-and-retry fold for the no-seat transient: consecutive parks plus a clock become SEAT_READY or SEAT_PARKED, instead of bursting preflight probes at a wall only a peer finishing can move. |
| `internal/sessionctl` | The control-op vocabulary spine of the out-of-band operator control plane: every shipped write op (pause, resume, cancel, throttle, budget, pace, priority, steer, redirect) named once with what "applied" means and how it refuses. |
| `internal/sessionread` | The read/query/observe-op vocabulary spine — the outbound twin of `sessionctl`: what each shipped read DISCLOSES, under whose right, on what evidence, and how it refuses when the read is illegal. |
| `internal/sessionledger` | The durable per-trace hash chain the gateway and the RSI loop append to at every turn boundary, as append-only JSONL — one bounded line per append instead of re-marshalling whole-state JSON on every write. |
| `internal/sessionreplay` | Freezes one turn's regime-conditioned harness decision into a checked-in, deterministically replayable regression fixture, so a mode/regime bug is reproducible with its ambient state pinned, not just its transcript. |
| `internal/sessionsearch` | Witnessed cross-session recall over the guard session journal: full-text recall whose relevance is MEASURED, so an index that quietly starts returning irrelevant hits is caught instead of trusted. |
| `internal/sessionsteer` | The steering and admission half of the zero-knob automatic-context doctrine: a content-free snapshot of a long session's context-value advice becomes one typed directive the guard hooks turn into a start-time rule and a stop-time persist decision. |
| `internal/supervisoragent` | The closed, payload-free INPUT CONTRACT a supervisor agent consumes. The meta-loop actor may CONSUME a witnessed signal and pick a move; it may never MANUFACTURE a health signal by reading transcripts. |
| `internal/workerworktree` | The per-worker git worktree isolation primitive the live dispatch spawn wires in, so N concurrent workers stop sharing one working tree, one index, and one build cache on the trunk. |
| `internal/worktreewitness` | Runs a command inside a transient detached worktree pinned at `origin/main`, so the verdict reflects the trunk tip and not a peer's dirty working tree. |

#### G.3 — Trunk hygiene, work-in-progress, and shared plumbing

| Package | Purpose |
| --- | --- |
| `internal/buildwitness` | A structural CI guard that fails when `cmd/fak` does not compile with the DEFAULT build tags — the recurring break where an uncommitted or tagless sibling file makes the package green only on its author's disk. |
| `internal/godfileceiling` | Ratcheting god-file LOC ceiling gate — the merge-time "no" that stops god-files from growing. |
| `internal/godsplitplan` | The doc-comment-aware boundary and hazard planner for a behavior-preserving Go split; the error-prone part is cutting at the right line, because a declaration's doc comment sits above its keyword. |
| `internal/refactorverify` | Proves a god-split or code-motion refactor dropped NO top-level definition — the question `go build` cannot answer, since it only catches a REFERENCED symbol that went missing. |
| `internal/jsonlledger` | The shared JSONL-ledger row helpers the report packages each used to copy-paste: `Parse` scans a ledger into typed rows, `LatestBefore` finds the newest prior row. |
| `internal/patchcommit` | Commits one explicitly supplied unified patch through a temporary git index — the hunk-scoped sibling of safecommit's path commit. |
| `internal/privatepath` | Resolves private operator artifacts outside the public checkout. |
| `internal/refutil` | Foundation-level helpers for materializing ABI refs. |
| `internal/tokencache` | The persisted, content-addressed backing store for clonescan's per-file tokenization, so an unchanged file is a file read instead of a re-lex — which is what lets the push rung and the CI duplicate job scale. |
| `internal/trunkbuildprobe` | Diagnoses *why* the release gate's ci-fast subset is red. The commonest cause is a coherence break: a commit lands a caller whose definition lives only in an uncommitted sibling file, so the whole tree builds on the author's disk. |
| `internal/wipattr` | The pure attribution core: for every dirty working-tree hunk it decides whether a session OWNS it (that session's checkpoint records the identical edit) or it is an ORPHAN a peer's sweeping stage could silently destroy. |
| `internal/wipfence` | Applies and removes the shared-trunk WIP build fence — a `//go:build wip_<slug>` first line that keeps not-yet-compiling work on the trunk without reddening the default build. Pure text: no git, no filesystem. |
| `internal/wiprecon` | The pure reconciliation core for a crashed session's orphaned checkpoint: DISCARD_WITNESSED (the delta already landed in HEAD), RECLAIM (unlanded but applies cleanly), or QUARANTINE (unlanded and conflicting). |
| `internal/wipref` | The pure core of `fak wip`: the ref-name grammar, the stamp encode/decode, and the status fold for the checkpoint ledger under `refs/fak/wip/<session>`, with zero git I/O of its own. |
| `internal/tuiplugin` | In-process extension seam for fak console panes (Register-from-init). |
| `internal/versionskew` | Turns "I can't tell which fak is running" into a structured, refusable version-skew verdict. |
| `internal/wiki` | The fak-native, witness-verified repo-wiki core (structure L1 + content). |
| `internal/xprobe` | A throwaway `Ping()` symbol used as the end-to-end fallback probe for buildcheck. |

#### G.4 — Guard, policy, and safety

| Package | Purpose |
| --- | --- |
| `internal/blastradius` | The pure join blast-radius containment is built on: given a broken package, which live leases and queued issues intersect its transitive dependency radius — so a hold stops being all-or-nothing guesswork. |
| `internal/egresslist` | Adblock-style site allow/block layer above the hardwired cloud-metadata egress floor and below the WebFetch research allowlist. |
| `internal/egressrefresh` | Re-fetches the bundled egress filter lists from their provenance URLs and rewrites the checked-in artifact + pinned checksum. |
| `internal/ghspam` | Reusable GitHub-comment abuse match families — the release-archive download lure and the fake patch/fix lure. Each needs two independent signals, so a genuine outsider bug report that merely says "fix" does not match. |
| `internal/guardaccuracy` | Folds a labeled command corpus through the real guard reversibility classifier and scores the boundary itself: the false-positive rate (benign calls escalated) and the false-negative rate (dangerous calls let through). |
| `internal/guardaudit` | Bounds repo-local guard audit journals only after a verified logvault mirror proves durability. |
| `internal/guardcompile` | Compiles one authoring-time model extraction into a review-only policy patch (data, not runtime enforcement). |
| `internal/guardcorpus` | Folds a guarded session's decision journal into the durable, policy-attributable guard-session dataset: one record per session plus replayable, redacted example rows that survive the journal reaper. |
| `internal/guardvars` | The canonical wire shapes shared by the `/debug/vars` producer (`internal/gateway`) and the `fak info` consumer, defined once so a new producer field can no longer drop silently on the floor. |
| `internal/knownenv` | The fleet-wide known-ENVIRONMENT-failure registry: an error-text needle and/or exit code mapped to a not-your-fault verdict. It matches by tool OUTPUT where `knownbad` matches by TREE. |
| `internal/market` | Validates discoverable extension descriptors without executing extension code. |
| `internal/planresolve` | Oracle-driven plan-content adjudicator. |
| `internal/reachdelta` | Categorizes what a policy change actually made REACHABLE: a newly permitted tool or tool prefix, a newly permitted egress host, a widened default posture, a removed explicit deny, a lost self-modify protection. |
| `internal/rsl` | Git Reference State Log: a forge-independent, append-only, hash-chained record of trunk ref transitions — the offline no-force-push proof. |
| `internal/verifierexposure` | Ranks the gameability of fak's own verification gates. |

#### G.5 — Scorecards, debt dispatchers, and RSI folds

| Package | Purpose |
| --- | --- |
| `internal/agentreadinessscore` | Grades the one thing the sibling cards do not: can an autonomous coding agent DISCOVER fak, WANT to adopt it, and do so easily — scored over the git-tracked tree on twenty-three mechanical KPIs. |
| `internal/antipattern` | The unifying registry for the agentic-dev anti-patterns whose common shape is "work that did not convert into progress": work REDONE that was already done, and work LANDED but connected to nothing. |
| `internal/brittleness` | The detector-and-capture for seams that "got lucky": process, commit, and test outcomes that worked only by timing, chance, or a symptom patch that did not hold — and the regressions those seams throw. |
| `internal/catchupscore` | The catch-up lens: not how much happened over a window, but how far BEHIND the dev system is right now at each level — intake, measurement, and the rest — as a number a human glances at and a gate ratchets on. |
| `internal/checkpointscore` | The deterministic checkpoint-readiness card: does each long-running process persist durable resumable state AND expose a witnessed status surface a peer can read without tailing its logs? |
| `internal/closureaudit` | The pure grader half of the issue-closure audit: binds commits to issue numbers from commit text, then grades each issue into exactly one witness bucket from the per-SHA verdicts the caller supplies. |
| `internal/conceptcatalog` | The typed catalog behind `fak concept`: the concept families with their roots and exclusions, plus the authoring, freshness, and separation folds over the disambiguation data file. |
| `internal/ctxknobs` | The manual-overlay COUNTER for the zero-knob automatic-context doctrine: it enumerates every surviving knob whose only purpose is context management and ratchets the count, so a new one cannot land silently. |
| `internal/ctxplans` | The context-plan-required lint: every context-touching verb or skill declares, as a structured directive co-located with the code, what enters the window, what pages out, and what warms. |
| `internal/envconfiglint` | The config-not-env ratchet: behavioral configuration must stay out of the environment (which is for secrets), enforced as a machine-checked gate on environment READS rather than by a human reading the diff. |
| `internal/findingsink` | A general sink seam for scorecard findings: a producer folds its debt into neutral Findings without knowing whether the sink is a dry-run terminal, a durable local ledger, or GitHub issues. |
| `internal/focusscore` | Grades whether the fleet as a WHOLE is converging on its live goal or fanning out — many objectives active at once, detours run past budget while their parents sit paused. The aggregate the per-objective fold cannot see. |
| `internal/headlesslint` | The sensor-side dual of `choicetriage`: scans a worker's final output for operator-directed notes ("do you want me to push?") that page a person who, in a headless run, is not there to answer. |
| `internal/hwgatelint` | The sensor for the "local machine is the compute boundary" regression. A laptop with no GPU is the CONTROL point, not the boundary; the work dispatches to the machine that can witness it. |
| `internal/issuehygiene` | The pure KPI core behind `fak score issue-hygiene`: how well the default GitHub issue surface is created and tagged, folded into one issue-hygiene-debt integer. |
| `internal/knobcensus` | The knob census over the whole user-facing surface: each knob classified as intent the system cannot infer versus a dial that should be automatic, so the context ratchet and the control ratchet read one inventory. |
| `internal/mcpfootprint` | Prices the always-sent MCP tool-schema floor — the fixed per-turn token tax every registered tool adds to every call, whether or not it is ever selected. Deterministic and offline, so it can be ratcheted down. |
| `internal/modedebt` | The consumer half of the mode-debt pair: reads the permission-dial scorecard and maps every HARD un-lifted dial onto the existing backlog bridge, adding no new issue-filing code of its own. |
| `internal/mutationefficacy` | A bounded, SOFT mutation-testing probe: it applies standard operator mutants and counts SURVIVORS — the one question coverage cannot ask, would the suite actually FAIL if the code were wrong? |
| `internal/orphanscan` | A syntactic detector for the "built but never wired up" smell: an unexported top-level function defined but referenced nowhere in its own package — the shape work takes when it was authored and then dropped. |
| `internal/promptlint` | The durable freshness monitor for the dispatch worker-issue prompts, whose executable claims (which verbs to run, which refusal tokens gate the commit) silently rot when a verb is renamed or a token retired. |
| `internal/qaprocessscore` | The QA-process card's KPI folds — the "is our test process honest?" signals, each a pure function over git and history facts, so the fold is deterministic and fixture-testable with no toolchain in the loop. |
| `internal/scdiff` | The shared diff-scoping seam for shift-left scorecards: given the ref the caller based its work on, exactly which paths changed — so a card can skip whole corpora instead of rescanning the tree after every edit. |
| `internal/sensecheck` | The "does this actually make sense?" side-car — a common-sense smell battery over claims the hard rungs already passed: a cache hit rate above 100%, a "fixed" commit over a visible non-zero exit, a test asserting a tautology. |
| `internal/seoaeoscore` | The discoverability measuring stick: will a reader — or an answer engine — find fak at all, and cite it correctly? Folded into an seo-debt integer, orthogonal to whether the docs are correct once found. |
| `internal/skillvalue` | The per-skill outcome-VALUE ledger: attributes the witnessed outcome delta of sessions that LOADED a skill against matched sessions of the same task class that did not. Staleness is not value. |
| `internal/mlpscore` | Grades the "first lovable cut" contract from committed, machine-checkable witness manifests. |
| `internal/unwiredscore` | The unwired-code card for "code complete but not wired into the default path": a package that compiles, carries its own tests, reads clean — and that nothing ever imports. |

#### G.6 — Issues, projects, and operator reporting

| Package | Purpose |
| --- | --- |
| `internal/choicetriage` | Decenters the human from a surfaced "choice" — most surfaced choices are fake and resolvable without a person. |
| `internal/dispatchaging` | The deterministic anti-starvation term the dispatch order was missing: among READY work, which unit a worker picks FIRST when raw priority alone would let a low-priority unit wait forever. |
| `internal/dispatchcache` | A content-addressed, in-memory TTL cache with an injectable clock for the routed-backlog inputs that successive dispatch ticks share. |
| `internal/issuededup` | The write-time near-duplicate gate shared by every issue producer: simhash embed plus top-K over the title and title+body axes, catching the paraphrase a producer's own seen-cache cannot. Advisory, never blocking. |
| `internal/issuestriage` | Folds one surfaced issue action — close a dormant question, mark an issue stale, review an under-labeled issue — into its decenter-the-human disposition, in one tested place instead of inline at every pane. |
| `internal/milestoneburndown` | The milestone SCHEDULE dimension: due dates, open/closed counts, and trailing closure velocity folded into ON_TRACK / AT_RISK / OVERDUE / NO_DUE_DATE / DONE, with a projected drain date compared against the due date. |
| `internal/operatorquestion` | Harness-agnostic operator-question normalization. |
| `internal/operatorresolve` | Evidence-first operator-question resolver. |
| `internal/projectcompletion` | Folds the board's project-work readouts into completion buckets and points per standard, so "how much of this project is actually done" becomes a derived number instead of a status opinion. |
| `internal/projectreport` | Folds a GitHub ProjectsV2 board into the same control-pane envelope the milestone report uses, so the board becomes an operator-visible dimension instead of a write-only sync target. |
| `internal/questionledger` | The deterministic labeling authority for the question-loop ledger: unambiguous ids, a closed category vocabulary, a closed status lifecycle, and no leaked host, path, or email — rules prose in a skill file cannot enforce. |
| `internal/spendrollup` | Builds the cross-account `fak spend` rollup and gate-fails any figure that forgets its valuation basis or its WITNESSED/OBSERVED provenance label — the conflation card's discipline applied to money. |
| `internal/steerpr` | Folds continuous-merge trunk commits into operator-legible PR-sized units banded by where operator attention is owed. |
| `internal/worklog` | The unified agent-work change feed: the commit, the diff-witnessed verdict, the lease epoch, and a later verdict flip folded into ONE append-only, cursor-drained feed a consumer tails by offset. |

#### G.7 — Context, cache, and token economics

| Package | Purpose |
| --- | --- |
| `internal/cacheprice` | The one source of truth for the provider prompt-cache price multipliers — the cost of a cached-prefix read or write relative to a base input token — so no layer re-declares the literals and drifts. |
| `internal/cachevalue` | Folds the persisted cache-savings ledger into per-session cache-efficiency metrics (hit rate, write amplification) and flags the churny sessions whose prefix keeps mutating. |
| `internal/cvregress` | Per-session cache-efficiency (hit% + write-amp) regression flagging over the cache-savings ledger. |
| `internal/computeadmit` | The one shared admission kernel over the compute partitioners — the compute-plane twin of the lane and region admitters, so a compute-region overflow refuses in the same closed vocabulary as a lane collision. |
| `internal/deadlineadmit` | A pure admission policy that orders pending work earliest-deadline-first and sheds degradation-eligible items it predicts will miss, so the survivors keep their SLO instead of everything missing together. |
| `internal/flowcredit` | The receiver-granted credit ledger for cross-node KV block transfer backpressure: a sender must reserve credit before transmitting, so a fast prefill node cannot overrun a slow decode node's KV-ingest buffers. |
| `internal/guideddecode` | A sound, byte-level compiler that constrains a model's decode to a valid tool-call envelope. This slice constrains the tool-NAME enum and the fixed skeleton around it; full argument-schema enforcement is a later slice. |
| `internal/kvbudget` | A pure, GPU-free calculator for the KV-cache VRAM budget of concurrent decode streams: the closed-form KV-bytes-per-token sizing math, with the hardware A/B left owed by a GPU node. |
| `internal/l3kv` | Durable L3 KV residency backend: `StageSpan`/`RestoreSpan` persist a demoted span by digest behind a durable manifest. |
| `internal/stepbaton` | The pre-resume step-advice stamp: the durable, cross-restart carrier of the managed-context decision captured while the trace is still live, so a resuming successor can read what the window pressure WAS. |
| `internal/stepbatoncapture` | The live-side producer of that stamp: it reads the gateway's managed-context report while the trace is alive and projects it into a durable stamp — kept out of `stepbaton` so that core never imports the gateway. |
| `internal/stripeload` | Fans a single logical read across N byte-identical mirrors of the same file, sized by relative bandwidth. It is a `ReaderAt`, so the model loader needs no changes to stripe across several NVMes or sources. |
| `internal/toon` | A general JSON/TOON codec whose correctness spine is a lossless, type-preserving round-trip. A uniform array of flat objects collapses to one header plus one line per row, so field names are not repeated per row. |
| `internal/turnkind` | Classifies the latest user turn from message STRUCTURE alone — which content-block types it carries, never their content — so a routine tool-result continuation is told apart from a fresh instruction. |

#### G.8 — Model, kernel, and hardware benchmarks

| Package | Purpose |
| --- | --- |
| `internal/benchauthority` | The typed, in-binary source of truth for the primary benchmark NUMBERS fak claims — the number, its baseline, its provenance, its retraction history — replacing a hand-typed table of run-on cells. |
| `internal/benchckpt` | The shared per-cell write-ahead checkpoint the compute-bench executors write through, so a crash at cell N does not discard the cells already measured. |
| `internal/deepseekv4kv` | A pure, weight-free block-accounting fixture for the DeepSeek V4 heterogeneous KV plane and its prefix-reuse policies. It reasons in normalized units, so unpublished head dimensions are never fabricated. |
| `internal/deepseekv4moe` | A pure, weight-free synthetic model of V4's all-MoE dispatch that compares naive per-expert scheduling against grouped/fused scheduling in work-units, never in fabricated milliseconds. |
| `internal/deploymanifest` | Defines the unified `fak.toml` all-in-one deployment manifest and its fail-closed loader — one reviewable declarative artifact in place of flag-soup, a policy file, environment variables, and service registration. |
| `internal/dsparity` | The pure, offline parity-harness SPECIFICATION for future DeepSeek-V4 native kernels: what a batch-invariant, bit-reproducible decode would have to prove before any such kernel is trusted. |
| `internal/glm52prefillsweep` | The GLM-5.2 pure-fak prefill-latency sweep driver: a prefill-dominant request at each prompt length, landing time-to-first-token and prefill throughput per length as discoverable benchmark-ledger artifacts. |
| `internal/macfit` | Models Apple unified-memory capacity for many concurrent agents. |
| `internal/modelaccept` | Evaluates versioned exact-model capability corpora without letting missing evidence or averages authorize a tier. |
| `internal/modelops` | Folds exact-model canary observations into capability-safe promotion / rollback / hold decisions. |
| `internal/roofline` | Folds a lab drive's run artifacts into one current-vs-target-vs-ceiling dashboard, joining the ceiling note that carries the targets with the run artifacts that carry the measurements. |
| `internal/zaitask` | Bounded non-streaming Z.AI task runner. |

#### G.9 — Code intelligence and big-doc paging

| Package | Purpose |
| --- | --- |
| `internal/agentsindex` | A view over `AGENTS.md` that parses it into a deterministic section model, so a worker holds a compact resident table of contents and faults in only the section its task needs. It changes no content — only the load path. |
| `internal/astquery` | Structural (AST-shape) search over Go source with metavariables: a pattern is Go with `$NAME` holes. A repeated hole must bind consistently, so `$X == $X` matches `a == a` but not `a == b` — which no regex can express. |
| `internal/atif` | Projects the redacted trajectory corpus onto ATIF, the Agent Trajectory Interchange Format, so a fak session round-trips into a portable artifact a standard eval or fine-tuning pipeline can consume. |
| `internal/codegraph` | A directed code knowledge-graph with breadth-first traversal: forward reachability ("what does this end up calling") and reverse reachability ("what breaks if I change this"), each as a shortest deterministic path. |
| `internal/codexlifecycle` | Folds a native Codex rollout transcript into an exactly-once task lifecycle keyed by turn id, and decodes the structured tool envelope into a closed outcome vocabulary so expected negatives are not counted as failures. |

#### G.10 — Trajectory, tool, and partner-runtime analytics

| Package | Purpose |
| --- | --- |
| `internal/evebridge` | The connection-security preflight for the fak/Eve bridge: a mechanical fold over Eve's connection manifest (auth posture, tool filters, approval policies, scopes) into typed pass/fail diagnostics, so a scoping mistake fails closed. |
| `internal/eveimport` | The read-only importer that folds saved Eve observability artifacts into fak's session-ledger row shape, consuming only framework-owned workflow tags — never free-form assistant text. |
| `internal/eveparity` | Runs a fixture Eve-shaped eval suite raw and fak-routed and proves the two arms agree — in particular that fak never silently downgrades a hard Eve gate FAILURE into a soft observation. |
| `internal/toolrollup` | Folds a corpus of individual tool-call records into a per-tool-TYPE rollup: call count, token totals and means, wall duration, error count and rate. |
| `internal/toolseq` | Turns ordered per-session tool-call sequences into a tool-transition graph and its most common contiguous variants. Transitions never span a session boundary, so unrelated sessions cannot manufacture an adjacency. |
| `internal/toolshape` | Fingerprints the SHAPE of one tool call's input and output — the redaction-safe structural record the analytics chain consumes, where the trajectory turn itself carries only opaque content digests. |
| `internal/tooltrend` | Folds a sequence of per-session tool-call buckets into a tool-mix and output-shape TREND: which tools and which response shapes are rising or falling across N sessions. |
| `internal/trajctlhook` | The impure call-site assembly that binds the pure trajectory-control turn-boundary fold to a running session's host evidence — whether a claimed commit SHA still resolves, the analyzed audit rows, the wall-clock stamp. |

### H. FAK 618 field map — choosing the right shipped entry point

These are the surfaces added after the last learning snapshot. Read each table as a route,
not a vocabulary list: begin with the operator question, use the public verb when one
exists, drop to a standalone command for its bounded artifact workflow, and edit the
internal leaf only when the reusable contract itself must change.

#### H.1 — Agent, fleet, evidence, and model verbs

| Verb | Use it when you need to… |
| --- | --- |
| `fak agent-queue` | Reconcile an agent-pool snapshot into explicit start or hold actions. |
| `fak agents` | Query live and historical sessions with constrained SQL, grouping, counts, or JSON. |
| `fak architecture` | Inspect dependency tiers, violations, fan-out, depth, and blast radius before changing a leaf. |
| `fak armbench` | Run provenance-locked paired benchmark arms from one immutable manifest. |
| `fak borrow-provenance` | Pin an external source and later verify its bytes against the recorded digest. |
| `fak breath` | Check the counted, mechanically enforceable half of the one-breath documentation contract. |
| `fak capabilities` | Find token, turn, cache, routing, session-control, and supporting-floor outcomes by intent. |
| `fak codex-resume` | Resume Codex sessions under bounded deadlines and report their rollout outcomes. |
| `fak component` | Check component contracts and workload coverage from a declared root. |
| `fak compute-trace` | Turn compute events into a bounded trace for placement and performance diagnosis. |
| `fak config` | Locate configuration guidance and audit a deployed posture against it. |
| `fak disambiguation` | Query, audit, or regenerate the canonical concept-separation index. |
| `fak dormancy` | Classify dormant loop work from the loop ledger instead of guessing from age alone. |
| `fak enroll` | Pin, inspect, or revoke a host's opt-in organization trust anchor. |
| `fak fanout` | Fold nightrun fan-out reuse receipts into a trend report. |
| `fak gitd` | Serve provenance-bearing, content-addressed git reads through the resident repo broker. |
| `fak goal` | Manage canonical goal state, bindings, evidence, lifecycle transitions, and execution topology. |
| `fak guard-goal-question` | Enforce the active-goal boundary before a guarded agent asks an operator question. |
| `fak harness` | Compose, inspect, resolve, and verify reusable agent-harness stacks. |
| `fak hostdiag` | Correlate privacy-safe host resource symptoms with fak work before prescribing recovery. |
| `fak launch` | Reversibly install, enable, inspect, or remove provider routing through fak. |
| `fak learning-observation` | Trace an observation through candidate, witness, and verdict rather than calling it learned early. |
| `fak lifecycle` | Inspect or control phase-aware capability lifecycle state. |
| `fak m` | Use the short alias for `fak manage` when wrapping a harness interactively. |
| `fak manage` | Put a harness behind the managed-agent door so tool calls can be denied, repaired, or quarantined. |
| `fak model-default` | Read the default model identity together with its dated evidence fold. |
| `fak model-observe` | Proxy, summarize, or verify model-performance and cache-transition observations. |
| `fak native-benchmarks` | List required fak-native witnesses and fail when an obligation is still missing. |
| `fak native-first-lint` | Catch prose that treats missing local hardware as a terminal native-inference blocker. |
| `fak native-performance` | Query, compare, or choose the next profile in the committed native optimization graph. |
| `fak org` | See organization-policy posture and which control channel owns each capability. |
| `fak progress` | Detect fleet stalls by comparing a recent window with its declared baseline. |
| `fak provider-cost` | Import, report, and reconcile provider-cost ledgers against registered sessions. |
| `fak quantbench` | Run the quantization benchmark contract or emit its self-test matrix. |
| `fak quantwatch` | Collect bounded arXiv and GitHub quantization updates, offline or live. |
| `fak schedule-held` | Evaluate held hardware-job schedules and measure admission-policy overhead. |
| `fak scratch-janitor` | Preview or remove abandoned session scratch while respecting age and resume guards. |
| `fak search` | Search the tracked repository corpus with bounded results and machine-readable output. |
| `fak shellprov` | Capture shell-command provenance so later evidence names what actually ran. |
| `fak speed-ab` | Replay a captured speed A/B manifest and emit its benchmark witness. |
| `fak stale-work` | Discover and rank stale repository work against open issues within a bounded scan. |
| `fak temp-artifacts` | Preview or quarantine aged temporary artifacts, then recheck before permanent reap. |
| `fak terminal-relief` | Measure terminal-host pressure and relaunch only restorable dashboards when armed. |
| `fak test-quality` | Score defects in test source against a shrinks-only baseline and emit repair candidates. |
| `fak token-profile` | Price forecast or observed token classes into dollars and scheduler-weight units. |
| `fak tool-width` | Fold tool-width observations and ratchet the rate of batched turns. |
| `fak trajectory` | Audit Claude or Codex logs into scrubbed JSONL and Markdown summaries. |
| `fak turnavoid` | Replay whole-model-turn avoidance traces with strict input and net-true attribution. |
| `fak turntax` | Measure the extra recovery turns a baseline loop fires against fak's one-shot path. |
| `fak value-chain` | Hold a value-chain manifest against observed artifacts and their evidence. |
| `fak watchdog-audit-run` | Run the watchdog's own liveness/productivity audit rather than trusting scheduled-task status. |
| `fak work-delivery` | Track, transition, and diagnose a work unit across declared delivery stages. |
| `fak workpattern` | Mine recurring work shapes from source or recorded trajectories. |
| `fak worktype` | Attribute session token spend and witnessed outcomes to classified work types. |

Start read-only: `fak architecture`, `fak agents`, `fak capabilities`, and the report
subcommands expose state; `fak enroll`, `fak launch`, and reap modes cross a mutation
boundary and should follow their preview/status path first.

#### H.2 — Bounded standalone commands

| Command | Artifact or boundary it owns |
| --- | --- |
| `cmd/amoprofpub` | Converts an AMOProf directory or archive into Confluence storage XHTML plus an attachment manifest. |
| `cmd/caveman-pairwise-judge` | Judges paired caveman benchmark outputs against an immutable source manifest and versioned receipt protocol. |
| `cmd/fak-dev` | Hosts contributor-only, shared-trunk-safe diagnostics and project-maintenance helpers outside the product CLI. |
| `cmd/fak-dos` | Applies journaled DOS decision changes through the writable host adapter; inspect/list before add or remove. |
| `cmd/fak-project-assets` | Syncs generated project assets and checks parity with their canonical sources. |
| `cmd/framevisibility` | Probes whether harness frames remain observable across the adapter boundary. |
| `cmd/kvdepth` | Measures reusable prefix depth per observation instead of collapsing reuse into one hit-rate scalar. |
| `cmd/managedinventory` | Generates or checks the managed-agent portability inventory without reading live credentials. |
| `cmd/modelperfobs` | Captures OpenAI-compatible timings, renders reports, and verifies cache-state benchmark receipts. |
| `cmd/nvidia-nemo-api-demo-audit` | Audits the NVIDIA NeMo API demo inputs and outputs into a reproducible, scrubbed receipt. |
| `cmd/pagescheck` | Checks documentation source, freshness, discoverability, and built publication artifacts. |
| `cmd/portability-adapter-selfcheck` | Exercises adapter registration or prints a skeleton without launching a real workload. |
| `cmd/portability-lab` | Runs the clean-room portability acceptance lab and writes its authoritative JSON report. |
| `cmd/portabilitycontract` | Explains, identities, or validates a packaged portability contract and its round trip. |
| `cmd/qwen38campaign` | Runs the frozen Qwen3.8 soak or oracle evidence campaign from declared config and corpus. |
| `cmd/streamcapture` | Records a real provider stream, scrubs it, and verifies the fixture before hermetic replay. |
| `cmd/supportwitness` | Applies a support witness to a capability graph and writes the promoted graph separately. |
| `cmd/testenv` | Runs a test command after removing credential-shaped environment variables. |
| `cmd/uxjourneyproxy` | Scores a deterministic user-journey corpus for proxy-path cognitive load. |
| `cmd/verbsdoc` | Renders the source-derived verb and refusal reference so documentation follows the registry. |

These binaries are narrow by design. If the same job has a `fak` verb, teach and automate
the verb; invoke `go run ./cmd/<name>` when reproducing the specific artifact workflow.

#### H.3 — Internal leaves behind the new surface

| Package | Contract to understand before editing it |
| --- | --- |
| `internal/archreport` | Derives a queryable architecture report from the tier/source registry. |
| `internal/cloudroute` | Detects request-signed cloud model routes without confusing provider routing with model identity. |
| `internal/codexresume` | Drives bounded Codex headless resumes to rollout-witnessed terminal outcomes. |
| `internal/corelockgate` | Owns the hard-self core-lock question so one boundary decides whether it may be opened. |
| `internal/customizationindex` | Indexes supported customization seams so operators can discover the owning guide and contract. |
| `internal/depthadmit` | Folds witnessed plan depth and admits continuation only while declared closure can still be reached. |
| `internal/dispatchdoa` | Detects workers that died before completing useful dispatch work. |
| `internal/docrender` | Renders repository Markdown into print-ready HTML and related publication artifacts. |
| `internal/fleetbus` | Carries typed fleet-control messages rather than treating terminal text as a control protocol. |
| `internal/fp4runtime` | Negotiates versioned FP4 and microscaling artifacts at the runtime boundary. |
| `internal/generationctl` | Coordinates live generation epochs, steering directives, and their acknowledgements. |
| `internal/gitbroker` | Centralizes guarded git execution and the resident content-addressed read broker. |
| `internal/gitdaily` | Runs the deduped daily lock-reap and safe object-database consolidation fold. |
| `internal/harnessmodelset` | Declares strict, role-indexed model requirements for a harness. |
| `internal/harnessmodelsetconformance` | Captures the end-to-end witness that a resolved role/model set actually conforms. |
| `internal/harnessserver` | Binds a harness to an externally owned, already-ready inference server. |
| `internal/hostdiag` | Correlates Windows resource warnings with privacy-safe fak workload facts. |
| `internal/httptrust` | Resolves one declared corporate CA bundle source for outbound HTTP clients. |
| `internal/humanctl` | Indexes operator-requested outcomes and the control surfaces that can satisfy them. |
| `internal/kvquantmeta` | Defines provider-neutral KV-cache quantization descriptors and compatibility rules. |
| `internal/leasequeue` | Queues region-admission waiters with explicit ownership instead of busy retry. |
| `internal/lightgapport` | Audits claimed portability swap points against committed CI witnesses. |
| `internal/managedocs` | Ratchets the canonical managed-agent documentation against its shipped surfaces. |
| `internal/mixedprecision` | Defines deterministic layerwise mixed-precision policy and accounting contracts. |
| `internal/modelinventory` | Normalizes model artifacts and runtime observations into one inventory. |
| `internal/modelperfobs` | Measures OpenAI-compatible requests and folds cache-state observations into reports. |
| `internal/modelsetlock` | Persists canonical model-set selections with stable identity and locking. |
| `internal/modelsetreceipt` | Independently attests what harness model set was resolved and launched. |
| `internal/modelsetresolve` | Deterministically binds harness roles to compatible model candidates. |
| `internal/ociartifact` | Implements the activation-neutral OCI 1.1 collection profile. |
| `internal/portabilityswitch` | Coordinates context changes with lifecycle state during a portability switch. |
| `internal/projectassets` | Resolves canonical project assets and proves generated copies remain in parity. |
| `internal/quantdetect` | Performs bounded, weight-free detection of quantization formats. |
| `internal/quantpolicy` | Evaluates explicit constraints over a proposed quantization choice. |
| `internal/quantprov` | Carries neutral provenance for quantized artifacts and their transformations. |
| `internal/scratchmark` | Detects source files whose leading contract marks them as disposable scratch. |
| `internal/serveradapter` | Renders configuration for supported external inference servers and probes readiness. |
| `internal/serverlifecycle` | Owns one local server instance from configuration through readiness and teardown. |
| `internal/serverproduct` | Defines the secret-free boundary between a server product description and its adapter. |
| `internal/sessionintent` | Represents provider-neutral session-level operator intent. |
| `internal/skilleffectiveness` | Measures whether loading a skill improved witnessed outcomes for matched work. |
| `internal/skillfootprint` | Prices the resident skill-description context floor. |
| `internal/streamrules` | Matches incremental provider output against regex rules without waiting for stream completion. |
| `internal/studydrift` | Detects when a pinned external study no longer matches the source revision it analyzed. |
| `internal/studymonitor` | Folds study provenance and drift checks into an operator-readable monitoring verdict. |
| `internal/taskvc` | Binds enabled fleet Scheduled Tasks to versioned installers or scrubbed captures. |
| `internal/tempartifact` | Inventories and conservatively reaps direct fak temporary artifacts. |
| `internal/tokenprofile` | Classifies forecast and observed tokens by economic and scheduling duty. |
| `internal/toolcallcontrol` | Applies deterministic pre-execution checks to proposed tool calls. |
| `internal/toolcatalog` | Separates executable tool registration from the catalog metadata agents discover. |
| `internal/ultracodebench` | Evaluates paired single-agent and fleet coding runs with accepted-effect accounting. |
| `internal/ultracodenegcontrol` | Evaluates predeclared negative controls so fleet gains cannot be self-attributed. |
| `internal/valuechain` | Verifies that promised value-chain stages resolve to observed, evidence-bearing artifacts. |

The recurring pattern is shell → pure fold → witness: for example,
`fak model-observe` or `cmd/modelperfobs` gathers facts, `internal/modelperfobs` folds
them, and the receipt makes the result replayable. Keep those responsibilities separate
when adding a new surface.

**Checkpoint:** Name the verb you would reach for to (a) prove the committed trunk builds
without trusting your working tree, (b) see which ready issues are starving, and (c) get
an offline per-task spend readout mid-run. (Answers: `fak validate` / `fak ci-preflight`,
`fak dispatch-aging`, `fak budget`.)

## Current shipped-surface map (August 2026)

Use this compact map after the core course when a current issue or receipt names a newer leaf. Each entry names the implementation surface and its operator-facing purpose; follow the linked package or command help before changing behavior.

- Desktop trust helpers: `cmd/localappcert` provisions the local app certificate contract, while `cmd/localapphelper` hosts the narrowly scoped desktop helper.
- Coordination and recovery: run `fak coordinate --help` for the coordinator surface and `fak watchdog-audit-health --help` for the watchdog health audit.
