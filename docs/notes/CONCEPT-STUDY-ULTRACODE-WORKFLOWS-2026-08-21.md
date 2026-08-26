# Ultracode workflow and orchestration borrow study — 2026-08-21

**Issue:** [#8484](https://github.com/anthony-chaudhary/fak/issues/8484)  
**Observed:** 2026-08-21 20:12 PDT  
**Machine companion:** [`CONCEPT-STUDY-ULTRACODE-WORKFLOWS-2026-08-21.json`](CONCEPT-STUDY-ULTRACODE-WORKFLOWS-2026-08-21.json)  
**Deterministic witness:** `internal/ultracodeborrow/check_test.go`

## Verdict

Fak already has the stronger orchestration spine: a typed plan, bounded concurrency,
leases, task progress/cancellation, an effect-successor receipt, independent-witness
requirements, a generic witness-keyed resume journal, and a paired value schema. The
current Ultracode frontier is **PARTIAL**, not absent: its latest live status showed three
exited workers, zero completed turns, and no observed effect readback, independent
witness, or reconciliation. A launch is therefore not a ship.

The public Claude Code examples make coordinator-authored parallel workflows easy to
express, LangGraph supplies the best inspected selective-checkpoint/resume implementation,
and stable A2A supplies the best inspected cross-vendor task/message/artifact envelope.
None supplies fak's lease plus independent effect-witness contract. Borrow their narrow
mechanisms and keep their product boundaries explicit.

The dedupe left two implementation gaps. Effective per-child activation is already owned
by [#8488](https://github.com/anthony-chaudhary/fak/issues/8488). The only ownerless gap,
joining an Ultracode launch graph to fak's shipped workflow-resume journal, is now the
contract-ready [#8490](https://github.com/anthony-chaudhary/fak/issues/8490). No other
unfiled implementation remainder leaves this study.

## Value frame and problem checklist

**For:** operators who need a multi-agent run to produce an accepted repository effect,
not merely several child processes. **Problem:** the field offers useful workflow pieces,
but their words—task, artifact, checkpoint, reviewer, resume—do not carry the same proof
or ownership semantics. **Today:** an operator can over-read a public example or a process
exit as proof of orchestration. **Better because:** every borrowed mechanism is pinned to
source, mapped to fak's current seam and owner, and bound to a falsifiable benchmark.
**Witness:** this note's companion fails deterministically if provenance, license,
mechanism, status, posture, owner, benchmark denominator, falsifier, or privacy boundary
is missing.

Problem centrality is **Enabling**. P1 managed context advances through typed role inputs
and durable artifacts rather than transcript replay. P2 net-true efficiency is admitted
only when accepted work improves per wall-time/token/cache denominator. P3 adaptation is
bounded by graph epoch, leases, capabilities, budgets, and witness keys. P4 integrated
operations joins activation, task state, effects, reconciliation, recovery, usage, and
quality instead of reporting process activity as outcome.

## Evidence and license ledger

All revisions are immutable pins. Observation time is distinct from the source's commit,
release, or installation event. `INSPIRE-ONLY` means concepts and citations only; no
external code was copied.

| Source | Revision and source event | Public license / provenance boundary | Use |
|---|---|---|---|
| `anthropics/claude-code` | [`16440d0f6ee8c47f34169687044b89eafa8b0f8d`](https://github.com/anthropics/claude-code/tree/16440d0f6ee8c47f34169687044b89eafa8b0f8d), commit 2026-08-21 19:54:17Z; release `v2.1.239` published 2026-08-21 19:54:23Z | Root [`LICENSE.md`](https://github.com/anthropics/claude-code/blob/16440d0f6ee8c47f34169687044b89eafa8b0f8d/LICENSE.md#L1) reserves rights and points to commercial terms. Public examples/changelog are evidence; runtime source is not public. | `INSPIRE-ONLY` |
| `langchain-ai/langgraph` | [`f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f`](https://github.com/langchain-ai/langgraph/tree/f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f), commit 2026-08-20 07:02:48Z; release `1.2.11` published 2026-08-11 14:00:50Z | [MIT](https://github.com/langchain-ai/langgraph/blob/f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f/LICENSE); no root NOTICE or submodules observed. Implementation and tests inspected. | `INSPIRE-ONLY`; preserve notice if code is ever adapted |
| `a2aproject/A2A` | current main [`16ba52690519bf55b9388e34d4db356efa88aa51`](https://github.com/a2aproject/A2A/tree/16ba52690519bf55b9388e34d4db356efa88aa51), commit 2026-08-18 17:09:55Z; stable [`v1.0.1`](https://github.com/a2aproject/A2A/releases/tag/v1.0.1) at `3303592588e388e62e0f69f701af531d2f4e3991`, published 2026-05-28 11:34:36Z | [Apache-2.0](https://github.com/a2aproject/A2A/blob/16ba52690519bf55b9388e34d4db356efa88aa51/LICENSE#L2-L122); no root NOTICE or submodules observed. Canonical protobuf/spec plus official SDK/sample behavior inspected. | `INSPIRE-ONLY`; standard transport is not the local scheduler |
| Installed Claude Code | version `2.1.237`; installation event 2026-08-20 20:28:50Z; observed 2026-08-21 | Proprietary black box. Only `--help`, version, and a redacted structural aggregate were used. No decompilation, extraction, or redistribution. | `DO-NOT-USE` as implementation material |
| Current fak | [`cb72859bce041963185d8cc83703c933509f49cc`](https://github.com/anthony-chaudhary/fak/tree/cb72859bce041963185d8cc83703c933509f49cc), commit 2026-08-21 17:10:12Z | Repository Apache-2.0; committed trunk plus read-only selfchecks. Dirty peer paths excluded. | Authored baseline |

The installed CLI help structurally exposed agent configuration, continuation/resume,
session identity, streaming/JSON/schema output, budget, permission mode, and debug
controls. That is black-box surface evidence, not evidence of how the proprietary runtime
implements them.

One local trajectory was reduced before publication to a structure digest and counts:
`sha256:b618b92d3ca15b53372ba28f9787dcd4fedb7fb0495eb7aa94ae2cb16a28a2a4`,
517 records (`assistant` 125, `user` 123, `attachment` 199, `queue-operation` 8,
`atis-latch` 31, `last-prompt` 31) and three `Agent` tool events. Raw paths, prompts,
message bodies, accounts, hosts, process/session identifiers, and credentials are absent.

## What the sources actually optimize

### Claude Code public examples

The feature workflow uses parallel explorers, architects, implementers, and reviewers,
then returns curated findings to a coordinator
([source](https://github.com/anthropics/claude-code/blob/16440d0f6ee8c47f34169687044b89eafa8b0f8d/plugins/feature-dev/commands/feature-dev.md#L20-L123)).
The review workflow fans out redundant reviewers and per-finding validators
([source](https://github.com/anthropics/claude-code/blob/16440d0f6ee8c47f34169687044b89eafa8b0f8d/plugins/code-review/commands/code-review.md#L30-L77)).
Advanced command guidance illustrates checkpoints, recovery state, and locks
([source](https://github.com/anthropics/claude-code/blob/16440d0f6ee8c47f34169687044b89eafa8b0f8d/plugins/plugin-dev/skills/command-development/references/advanced-workflows.md#L279-L635)).

The worldview is declarative, customizable, coordinator-centric workflow composition,
with human gates and reviewer redundancy. The public repository does not establish
runtime lease, durable effect-receipt, or independent completion-witness semantics.
Changelog entries are release claims, not public implementation evidence.

### LangGraph

LangGraph prepares and deterministically orders tasks and writes
([source](https://github.com/langchain-ai/langgraph/blob/f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f/libs/langgraph/langgraph/pregel/_algo.py#L392-L624)),
supports `Send`/`Command` fan-out and checkpoint-linked child state
([source](https://github.com/langchain-ai/langgraph/blob/f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f/libs/langgraph/langgraph/types.py#L637-L824)),
bounds concurrent execution and cancellation
([source](https://github.com/langchain-ai/langgraph/blob/f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f/libs/langgraph/langgraph/pregel/_executor.py#L122-L217)),
and has selective interrupt/resume tests
([source](https://github.com/langchain-ai/langgraph/blob/f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f/libs/langgraph/langgraph/types.py#L851-L974)).

Its Pregel-style barrier and checkpoint worldview favors deterministic state-machine
recovery. Pending writes provide replay/idempotency material, but remain runtime-authored;
the inspected core has no exclusive worker/effect lease or independent effect witness.
Selective recovery is the useful borrow; checkpoint existence is not a completion receipt.

### A2A stable protocol

A2A standardizes task/context identity and submitted/working/completed/failed/canceled/
input-required/rejected/auth-required state
([schema](https://github.com/a2aproject/A2A/blob/16ba52690519bf55b9388e34d4db356efa88aa51/specification/a2a.proto#L18-L218)),
typed messages, artifacts, and status/artifact events
([schema](https://github.com/a2aproject/A2A/blob/16ba52690519bf55b9388e34d4db356efa88aa51/specification/a2a.proto#L221-L322)),
and current-snapshot-before-subscription recovery
([spec](https://github.com/a2aproject/A2A/blob/16ba52690519bf55b9388e34d4db356efa88aa51/docs/specification.md#L182-L311)).
Messages are not a reliable critical-data channel after reconnect, so durable outputs
belong in artifacts; push delivery can duplicate and consumers must be idempotent
([spec](https://github.com/a2aproject/A2A/blob/16ba52690519bf55b9388e34d4db356efa88aa51/docs/specification.md#L679-L885)).

A2A's worldview is a thin, compatible, cross-vendor envelope around opaque agents. Stable
A2A is not a planner, dependency scheduler, ownership/lease system, durable cursor log,
effect-receipt system, or independent-witness protocol. `contextId` and referenced task
IDs provide related-work traceability, not an enforced parent/child DAG. Its best-effort
cancel does not reverse effects. Generation/OCC, coherent history, trace conventions, and
cascade cancellation remain WATCH proposals, not stable shipped behavior.

## Twelve-axis borrow matrix

`PRESENT` means the pinned fak baseline has the mechanism. `PARTIAL` means a typed seam
exists but its live integration/readback/value witness is incomplete. Posture describes
where the completed mechanism belongs in fak, not what an external project defaults.

| Axis | Field mechanism and limit | Current fak evidence | Status | Posture | Owner / decision |
|---|---|---|---|---|---|
| Activation | Claude exposes agent/settings controls; public source does not prove effective child activation. | `fak ultracode --selfcheck` passes and the plan resolves; live per-child acknowledgement is absent. | PARTIAL | DEFAULT | [#8488](https://github.com/anthony-chaudhary/fak/issues/8488): requested/resolved/injected/observable receipt; abstain on unknown. |
| Plan/task graph | Claude prompt workflows and LangGraph typed task preparation. | `internal/orchestration` carries roles, edges, budgets, lease, witness, reconcile, and interaction policy. | PRESENT | DEFAULT | [#5964](https://github.com/anthony-chaudhary/fak/issues/5964): retain one harness-neutral typed graph. |
| Child context | Claude curates child findings; LangGraph checkpoints child lineage. | Role inputs are typed; per-node admitted/returned context is not exposed end to end. | PARTIAL | DEFAULT | [#8334](https://github.com/anthony-chaudhary/fak/issues/8334): role-minimal context and node readback. |
| Parallelism | Claude fans out roles; LangGraph bounds executor concurrency. | Max workers/capabilities are typed; latest live run exited without a verified completed turn. | PARTIAL | DEFAULT | [#5970](https://github.com/anthony-chaudhary/fak/issues/5970): count accepted work, not spawned processes. |
| Messaging | Claude release claims, LangGraph `Send`/`Command`, A2A messages/artifacts. | `internal/a2achan` and interaction policy exist; current adapter delivery/readback is unwitnessed. | PARTIAL | DEFAULT | #5970: typed progress messages, durable critical artifacts. |
| Ownership/leases | LangGraph checkpoint identity and A2A authorization are not leases. | Plan lease policy plus snapshot-epoch/lease-bound effect successor checks. | PARTIAL | DEFAULT | #8334: expose the already-stronger per-node lease/readback seam. |
| Progress | A2A lifecycle/events and LangGraph streams. | `taskmgr` has snapshots/progress; current Ultracode status is process-oriented. | PARTIAL | DEFAULT | #8334: task/node state distinct from liveness. |
| Cancellation | LangGraph cooperative cancellation; A2A best-effort task cancel plus reread. | `taskmgr` cancellation exists; adapter orphan cleanup and terminal readback remain open. | PARTIAL | DEFAULT | [#5973](https://github.com/anthony-chaudhary/fak/issues/5973): never describe cancel as effect compensation. |
| Effect receipts | LangGraph pending writes and A2A artifacts are self/runtime-authored. | Effect successor binds snapshot epoch, lease identity, and readback contract. | PARTIAL | DEFAULT | #8334: artifacts/checkpoints are evidence inputs, not proof alone. |
| Independent witness | Claude reviewer redundancy finds issues; it is not deterministic completion proof. | Task witness records and plan requirements exist; latest live run observed none. | PARTIAL | DEFAULT | #5964: independent readback remains the completion gate. |
| Recovery/resume | LangGraph selective resume; A2A snapshot-before-resubscribe; Claude recovery recipes. | Generic witness-keyed journal/resume shipped in [#2444](https://github.com/anthony-chaudhary/fak/issues/2444); Ultracode launch is not joined. | PARTIAL | DEFAULT | [#8490](https://github.com/anthony-chaudhary/fak/issues/8490): persist graph receipts, re-lease incomplete nodes. |
| Usage/quality observability | Claude exposes usage; A2A offers IDs/events; neither proves accepted repository value. | `ultracodebench` has paired `GAIN/NO_GAIN/ABSTAIN`; selfcheck is offline and live outcome unverified. | PARTIAL | DEFAULT | [#8168](https://github.com/anthony-chaudhary/fak/issues/8168), [#8337](https://github.com/anthony-chaudhary/fak/issues/8337), [#8404](https://github.com/anthony-chaudhary/fak/issues/8404): live accepted-outcome, role frontier, trajectory attribution. |

Two discovery misses matter operationally. `fak capabilities "agent message task artifact"`
returned no direct card even though `internal/a2achan` exists. The capability search for
witnessed workflow resume returned general session cards, while `fak-dev feature query`
found the shipped workflow leaf. These are lexical discovery limits, not absent mechanisms.

## Dedupe ledger

| Issue | Existing contract; why this study does not duplicate it |
|---|---|
| [#5964](https://github.com/anthony-chaudhary/fak/issues/5964) | Parent harness-neutral plan/execution/witness/reconcile envelope. |
| [#5970](https://github.com/anthony-chaudhary/fak/issues/5970) | Codex parallelism, messaging, cancel, and cleanup adapter. |
| [#5973](https://github.com/anthony-chaudhary/fak/issues/5973) | Claude capability negotiation, cancel, and cleanup adapter. |
| [#8156](https://github.com/anthony-chaudhary/fak/issues/8156) | First-class Ultracode CLI plan/selfcheck; shipped commit exists although issue is open. |
| [#8168](https://github.com/anthony-chaudhary/fak/issues/8168) | Live paired speed/token/cache/quality proof; offline fixture is not closure. |
| [#8334](https://github.com/anthony-chaudhary/fak/issues/8334) | Per-node child context, lease, effect, witness, and progress status. |
| [#8337](https://github.com/anthony-chaudhary/fak/issues/8337) | Read-only scout versus effectful worker value frontier. |
| [#8404](https://github.com/anthony-chaudhary/fak/issues/8404) | Queryable cross-harness trajectory efficiency audit. |
| [#2444](https://github.com/anthony-chaudhary/fak/issues/2444) | Closed: generic witness-keyed journal/resume. #8490 integrates it; it does not replace it. |
| [#8488](https://github.com/anthony-chaudhary/fak/issues/8488) | Effective per-child activation receipt and causal benchmark join; found before duplicate filing. |

## Source-to-benchmark contract

Every axis has one treatment/control row in the companion. The shared rules are:

- **Workload:** treatment and control use the same model/provider, repository start
  revision, task, tools, budget, and acceptance contract unless an axis explicitly varies
  one of them.
- **Quality oracle:** an independently accepted repository/effect result; activation
  unknown, missing readback, or unequal outcome forces `ABSTAIN`.
- **Speed denominator:** accepted outcomes, graph obligations, packets, effects, or
  transitions per end-to-end wall-second—never spawned workers per second.
- **Token denominator:** the same accepted unit per billed or admitted input/output/control
  tokens, with the exact unit declared per row.
- **Cache denominator:** accepted reused/cache-read tokens divided by provider-eligible or
  otherwise explicitly eligible repeated input tokens; never a bare hit rate.
- **Falsifier:** the row states the observation that defeats the borrow, including
  collision, duplicate effect, false done, missing activation, graph-epoch crossing,
  worse accepted latency/token economy, or quality regression.

The sharpest experiments are: typed graph versus an unstructured coordinator; bounded
disjoint parallelism versus sequential execution; minimal child context versus transcript
copy; typed message plus artifact versus transcript polling; exact leases versus concurrent
unowned effects; witness-required completion versus self-report; selective witnessed
resume versus restart-from-zero; and a live acknowledged Ultracode treatment versus a
matched non-Ultracode control. The full per-axis treatment, control, workload, oracle,
denominators, and falsification conditions are machine-readable in the companion.

## Borrow portfolio

**DEFAULT:** typed plan/task identity; bounded child context; capability-gated concurrency;
typed progress plus durable critical artifacts; exact ownership/effect leases; terminal
cancel readback; effect-successor receipt; independent completion witness; snapshot before
stream/recovery; witness-keyed selective resume; and joined activation/outcome/usage/cache/
quality evidence.

**WATCH:** A2A v1.1 generation/OCC and coherent timeline proposals, standardized trace
context, and cascade cancellation. They address real gaps but were not stable released
semantics at the pin.

**EXCLUDE:** treating Claude prompt examples as a runtime protocol; treating LangGraph
pending writes as an independent effect receipt; treating A2A context/task references as
an enforced dependency DAG; treating messages, artifacts, process exits, or AgentCard
signatures as execution proof; and treating best-effort cancellation as compensation.

The field-borrow decision is therefore additive and narrow. Fak should keep its current
strong proof/lease model, make the partial integrations observable, connect recovery, and
measure the accepted-outcome frontier before expanding the architecture.

## Completeness and refresh boundary

Checked source classes: pinned public source and tests; licenses, NOTICE/submodule
boundaries, releases, changelogs/history, public issue/PR/proposal surfaces; installed CLI
help and redacted structural behavior; current fak code, capability/feature queries,
selfchecks, live status, issues, and shipped generic resume.

Unavailable: proprietary Claude runtime source, private provider telemetry, and nonpublic
design work. These absences bound all negative claims. Other A2A language SDKs were not
exhaustively read because the canonical schema/spec plus official Python SDK exercised the
ownership/cancel/reconnect seams used here. Unreleased A2A proposals are WATCH only.

Refresh this note when any pinned repository revision materially changes the cited seams,
when A2A publishes a stable generation/replay release, when the installed Claude Code
version changes, or when #8488/#8490/#8334/#8168 land. The deterministic companion checker
guards shape and privacy; it does not convert stale evidence into current evidence.

### LangGraph exhaustive inventory refresh (2026-08-25)

The LangGraph denominator is now explicit and reproducible at the original pin
`f09cfe8ffc1eeffd68f4b628ed69c30f7cad229f` (2026-08-20T07:02:48Z). The generated
map at [`docs/research/inventory/langchain-ai-langgraph.json`](../research/inventory/langchain-ai-langgraph.json)
walks all 672 regular files (13,719,867 bytes; 318 runtime, 237 test, and 33 docs
files) across five top-level subsystems, skipping only `.git`. Architecture and
roadmap were not separate tree classes at this revision, so those claims are
bounded to README/docs, runtime, tests, commit history, releases, and public
tracker evidence rather than inferred from absent documents.

The non-tree audit exhaustively paged the public GitHub surfaces observed on
2026-08-25: 1,534 issues (480 open), 5,642 pull requests (242 open), zero
Discussions, 560 releases, 731 tags, and eight branches. The pinned clone contains
7,045 commits through the checked revision and confirms the repository-level MIT
license. These live counts establish the audited denominator; source conclusions
remain tied to the checked revision rather than silently advancing to the later
upstream head.

Candidate adjudication did not change the portfolio:

- **Already owned:** deterministic Pregel task preparation/write ordering maps to
  fak's typed DAG, exact leases, and effect-successor receipts.
- **Already shipped:** checkpoint pending writes inspired selective recovery, but
  fak keeps independent effect proof; witnessed graph resume shipped in #8490.
- **Still tracked:** checkpoint lineage strengthens the case for per-node context,
  lease, effect, and witness visibility already owned by #8334.
- **No new issue:** #8168 already supplies the paired value witness, and no
  source-validated candidate survived dedupe as an unowned borrow.

Fak self-queries for deterministic writes/checkpoint lineage, durable resume, and
graph orchestration returned native carry-forward, live-control, context-reuse,
attribution, fleet-monitoring, and capability-floor surfaces. The refreshed
candidate matrix therefore keeps LangGraph as implementation evidence for
selective recovery, not as a graph runtime to port and not as evidence that a
persisted pending write independently proves an external effect.
