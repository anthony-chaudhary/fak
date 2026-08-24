# External Repository Study Report: `Mchicao/endless`

## Verdict

**No evidence-backed borrow recommendation can be made from this halted investigation.** The initial source-read command was aborted before returning content, and the stop instruction prohibited further tool calls. Consequently, no files from `endless`, its history, or the requested canonical FAK skills were successfully inspected.

The only established repository facts are those supplied directly in the task:

- Repository: `https://github.com/Mchicao/endless`
- Pinned revision: `bbf449ca624aca90d8cda0bc21100f2a73a56bf4`
- Local source location: `C:\work\fak\_scratch\study-endless\source\endless-main`
- Declared license: MIT
- Comparison checkout: `C:\Users\USER\AppData\Local\Fleet\worker-worktrees\fak-worker-wt-study-repo-16e99b3a19ca`

**No repository changes were made and no issues were filed.**

---

## 1. Evidence boundary

### Successfully available

The session included the detached FAK checkout’s `AGENTS.md` as supplied context. It establishes these relevant FAK architectural and operating facts:

| FAK surface | Evidence available from supplied `AGENTS.md` |
|---|---|
| Kernel role | FAK is an agent kernel placed between an agent and its tools. |
| Core execution seam | Every tool call is intercepted before execution for adjudication, routing, reuse, context management, and security enforcement. |
| Primary module layout | `cmd/fak/`, `internal/adjudicator/`, `internal/policy/`, `internal/vdso/`, `internal/engine/`, `internal/gateway/`, `internal/ctxmmu/`, and `internal/model/`. |
| Extension policy | Capabilities should be introduced as leaves and durable Go tooling, generally through `fak new-leaf`, rather than by directly expanding core registries. |
| Proof standard | Behavioral changes require a failing-before/passing-after witness; completion claims require independent artifact or commit evidence. |
| External-study doctrine | Proposed work must identify mechanism, measurable axis, next-best alternative, witness, centrality, and P1–P4 status. |
| Performance invariant | Native inference must remain FAK-native except for explicitly selected reference, parity, migration, or interoperability work. |
| Source integrity | Claims and benchmark gains must be independently witnessed and operating-envelope constrained. |

### Not successfully gathered

The attempted command to read the following files returned no content because it was aborted:

- `C:\work\fak\AGENTS.md`
- `C:\work\fak\.claude\skills\study-repo\SKILL.md`
- `C:\work\fak\.claude\skills\field-borrow\SKILL.md`

No successful inspection was made of:

- `endless` source files
- `endless` tests
- `endless` documentation
- Build or dependency manifests
- CI configuration
- Security configuration
- Git history
- Commit metadata
- FAK source line numbers
- FAK issue or documentation indexes
- Existing GitHub issues
- FAK query/index command output

Any detailed claim about those surfaces would therefore be fabricated.

---

## 2. Architecture and execution flow

### `endless`

**Status: not established.**

There is no admitted evidence identifying:

- Entry points
- Runtime language
- Package or module graph
- Control loop
- State model
- Persistence layer
- Tool-execution model
- Agent lifecycle
- Retry or termination mechanism
- Concurrency model
- Configuration loading
- Error propagation
- Observability
- Test architecture

The repository name alone is not evidence that the project implements an endless agent loop, autonomous continuation, workflow runner, or any other specific mechanism.

### FAK comparison baseline

From the supplied FAK orientation, the high-level FAK flow is:

1. An agent proposes a tool call.
2. The FAK kernel intercepts the call.
3. Policy and adjudication determine whether the operation may proceed.
4. Shared setup, routing, model selection, or repeat-serving mechanisms may process it.
5. The underlying engine executes allowed work.
6. Result-side context and trust handling determine what may be admitted back to the agent.
7. Durable receipts, journals, tests, or registry evidence support later verification.

Relevant declared FAK seams include:

- CLI and process entry points: `cmd/fak/`
- Admission and decisions: `internal/adjudicator/`
- Capability policy: `internal/policy/`
- Locally served/reused operations: `internal/vdso/`
- Model/tool execution: `internal/engine/`
- Provider or request boundary: `internal/gateway/`
- Context admission and shedding: `internal/ctxmmu/`
- Model abstraction: `internal/model/`

This baseline cannot yet be compared against `endless` because its execution flow was not observed.

---

## 3. File-by-file evidence

### `endless`

No file contents were successfully read. The honest evidence table is therefore empty:

| Source path | Observed construct | Behavioral implication | Candidate borrow |
|---|---|---|---|
| None | No source evidence admitted | No conclusion possible | None |

### FAK

The supplied orientation names architectural paths but does not provide stable line numbers. It would be misleading to invent `file:line` locations.

| FAK path | Established role | Exact line |
|---|---|---|
| `cmd/fak/` | CLI verbs and executable entry surface | Not established |
| `internal/adjudicator/` | Tool-call adjudication | Not established |
| `internal/policy/` | Policy enforcement | Not established |
| `internal/vdso/` | Repeat/local-serving mechanisms | Not established |
| `internal/engine/` | Execution engines | Not established |
| `internal/gateway/` | Gateway and provider boundary | Not established |
| `internal/ctxmmu/` | Managed context and result admission | Not established |
| `internal/model/` | Model abstraction | Not established |
| `CLAIMS.md` | Tagged claim inventory | Not established |
| `BENCHMARK-AUTHORITY.md` | Benchmark source of truth | Not established |
| `docs/native-inference-goal.md` | Native-inference invariant | Not established |
| `docs/spine-first-defaults.md` | Spine-first and follow-on doctrine | Not established |

---

## 4. Dependency assessment

### `endless`

**Unassessed.**

No manifest or lockfile was inspected. Therefore the following are unknown:

- Direct dependencies
- Transitive dependencies
- Pinned versus floating versions
- Runtime requirements
- Native libraries
- Network clients
- Database drivers
- Shell-command usage
- Optional integrations
- Dependency freshness
- Known vulnerability exposure
- Vendored code
- Generated code
- Reproducibility of builds

No dependency may be characterized as safe, risky, heavy, portable, or reusable without evidence.

### FAK compatibility implications

FAK’s supplied orientation states that the root Go module has zero external dependencies and no `go.sum`. Any direct port from another language or dependency-heavy runtime would therefore need to justify:

1. Why a standard-library Go implementation is insufficient.
2. Whether introducing a dependency weakens FAK’s deployability or trust boundary.
3. Whether the mechanism belongs in the kernel, a leaf, an external adapter, or documentation.
4. Whether the dependency’s license and transitive tree are compatible.
5. Whether the mechanism can preserve FAK-native execution and result-side admission.

Because `endless` dependencies were not observed, no compatibility judgment is available.

---

## 5. License assessment

### Established fact

The task identifies `endless` as **MIT-licensed** at the pinned revision.

### What MIT would ordinarily permit

Subject to confirmation from the actual pinned `LICENSE` file, MIT generally permits:

- Use
- Modification
- Distribution
- Sublicensing
- Commercial use

It ordinarily requires preservation of the copyright and permission notice in substantial copied portions.

### Unresolved license questions

The actual repository was not inspected, so these remain open:

- Whether the pinned commit contains the expected MIT text
- Copyright holder identity and years
- Whether individual files carry different headers
- Whether dependencies use compatible licenses
- Whether assets, fixtures, generated material, or copied snippets have separate terms
- Whether documentation or examples contain third-party material
- Whether git history reveals code imported under another license

### Classification impact

Even if the MIT declaration is correct:

- **Direct port** would require preserving applicable notices and tracing copied portions.
- **Adaptation** may still require attribution when substantial expression is retained.
- **Inspiration** based only on an abstract technique is preferable when the mechanism is simple enough to reimplement independently.
- License compatibility alone does not establish architectural suitability or security.

---

## 6. Security assessment

### `endless`

**Unassessed.** No security-relevant code or configuration was inspected.

The following threat surfaces must not be presumed absent:

- Arbitrary shell execution
- Path traversal
- Unsafe workspace mutation
- Environment-variable or credential leakage
- Prompt or tool-result injection
- Network exfiltration
- Unbounded retries or runaway autonomous loops
- Untrusted configuration parsing
- Insecure temporary files
- Command-string construction
- Symlink traversal
- Missing capability checks
- Unsafe subprocess inheritance
- Log or transcript secret exposure
- Race conditions
- Denial of service through uncontrolled concurrency
- Inadequate stop conditions
- Unsigned or unverified updates

### FAK acceptance bar

Any imported mechanism would need to preserve FAK’s declared properties:

1. Every external action remains behind adjudication.
2. A worker’s self-report is never treated as completion evidence.
3. Context returned by tools remains subject to result-side trust handling.
4. Autonomous continuation has a typed and witnessed stopping condition.
5. File-tree ownership and lane leases remain enforceable.
6. Native-performance work never silently falls back to another engine.
7. Security-sensitive operations retain explicit capability and policy checks.
8. Logs and receipts do not expose credentials or private infrastructure.

No security-positive claim about `endless` can presently clear this bar.

---

## 7. Candidate-borrow inventory

### Evidence-backed candidates

**None.**

A candidate borrow requires at least:

- A concrete source construct
- Its source path
- Its execution mechanism
- Its measurable axis
- A FAK seam
- A duplicate search
- A falsifiable witness

No source construct was observed.

### Classification table

| Candidate | Mechanism | Classification | Decision |
|---|---|---|---|
| None established | No observed mechanism | Exclusion pending evidence | Do not port |
| Hypothetical continuation loop | Not observed | Exclusion | Repository name is insufficient evidence |
| Hypothetical retry policy | Not observed | Exclusion | No source or tests |
| Hypothetical workflow orchestration | Not observed | Exclusion | No architecture evidence |
| Hypothetical context compaction | Not observed | Exclusion | Could duplicate `internal/ctxmmu/`; no comparison available |
| Hypothetical tool-call guard | Not observed | Exclusion | Could duplicate adjudicator/policy; no source evidence |
| Hypothetical agent supervision | Not observed | Exclusion | Could duplicate guard/watchdog/DOS surfaces |
| Hypothetical terminal UI | Not observed | Exclusion | No render surface or witness |
| Hypothetical persistence format | Not observed | Exclusion | No schema or migration evidence |
| Hypothetical provider adapter | Not observed | Exclusion | No compatibility or trust-boundary evidence |

---

## 8. Mechanism, ablation, axis, and worldview

No `endless` mechanism was observed, so a genuine mechanism analysis is unavailable. The required analytical frame for any later candidate would be:

### Mechanism

Identify the smallest causal construct, not the project-level feature name. Examples of valid granularity would be:

- A lease-renewal transition
- A retry-budget calculation
- A persisted continuation cursor
- A termination predicate
- A bounded transcript renderer
- A deduplication key
- A supervisor heartbeat
- A structured failure envelope

### Ablation

For each candidate, remove only that mechanism and test whether the claimed benefit disappears. A valid ablation must distinguish the mechanism from:

- More retries
- A larger token budget
- Added concurrency
- Better prompts
- Different models
- Hidden provider caching
- Changed task selection
- Lower safety standards

### Axis

The candidate needs one primary measurable axis, such as:

- Verified tasks completed per billed token
- Time to independently witnessed completion
- False-done refusal rate
- Recovery success after worker death
- Duplicate tool-call avoidance
- Context retained per admitted byte
- Operator interventions per completed issue
- Unsafe operation catch rate
- Lease collision rate
- Provider-cache survival across compaction

### Worldview

The candidate must express a distinct operating belief rather than merely add machinery. Relevant worldview questions include:

- Is continuation earned from evidence or asserted by the worker?
- Does a loop optimize activity or verified progress?
- Is state reconstructed from artifacts or trusted from narration?
- Is retrying the default, or does each retry require new evidence?
- Are autonomous workers isolated by capability and file tree?
- Is context treated as trusted memory or untrusted input?
- Is completion a semantic claim or a witnessed state transition?

No answer can be attributed to `endless` without inspecting it.

---

## 9. Direct port, adaptation, inspiration, and exclusion

| Disposition | Current result | Reason |
|---|---|---|
| **Direct port** | No candidates | No code, tests, dependencies, or notice text inspected |
| **Adaptation** | No candidates | No mechanism identified |
| **Inspiration** | No candidates | Even abstract techniques were not established |
| **Exclusion** | Entire uninspected surface | Fail closed rather than infer from repository name or task framing |

This is an evidence exclusion, not a judgment that the repository lacks useful ideas.

---

## 10. Exact FAK seams

No exact `file:line` seam can be responsibly supplied. Only provisional path-level destinations are available from the supplied orientation:

| If an observed candidate concerned… | Provisional FAK seam | Current disposition |
|---|---|---|
| Tool admission or denial | `internal/adjudicator/`, `internal/policy/` | No candidate |
| Repeat suppression or local serving | `internal/vdso/` | No candidate |
| Provider/model execution | `internal/engine/`, `internal/gateway/`, `internal/model/` | No candidate |
| Context retention or result admission | `internal/ctxmmu/` | No candidate |
| CLI and operator workflow | `cmd/fak/` | No candidate |
| Agent supervision or autonomous continuation | Relevant `cmd/fak/` guard/watchdog/agent verbs, exact files unknown | No candidate |
| Claims or benchmark evidence | `CLAIMS.md`, `BENCHMARK-AUTHORITY.md` | No candidate |
| Native inference | `docs/native-inference-goal.md` and corresponding engine leaves | No candidate |
| External-study documentation | Research/concept documentation cluster, exact index unknown | No candidate |

These are search starting points, not implementation recommendations.

---

## 11. Self-query evidence

No FAK query or index command completed.

Accordingly, there is no evidence from:

- `fak index`
- `fak_index_docs`
- `fak_index_claims`
- `fak_index_leaves`
- `fak_index_verbs`
- `fak_feature_query`
- `fak_capabilities`
- `fak version modules`
- Repository grep
- Claims indexes
- Plan indexes
- Issue views
- Git history

No assertion of novelty, absence, duplication, or prior implementation is supported.

---

## 12. Duplicate issue and documentation searches

**Not performed.**

The stop instruction arrived before any issue or documentation query could be run. Therefore:

- No GitHub issue was identified as a duplicate.
- No closed issue was identified as prior art.
- No FAK plan or program was identified as overlapping.
- No concept note was identified as covering the same mechanism.
- No `CLAIMS.md` row was matched.
- No CLI verb was confirmed to provide an equivalent feature.
- No ticket recommendation should be filed from this report.

---

## 13. Bounded-superset portfolio

A bounded-superset portfolio should include every independently valuable technique found in the source, while deduplicating techniques that share the same mechanism and measurement axis.

Because zero source techniques were established, the current portfolio is intentionally empty:

```text
Portfolio size: 0
Direct ports:   0
Adaptations:    0
Inspirations:   0
Exclusions:     all uninspected source constructs
```

This empty set is the only portfolio consistent with the available evidence.

---

## 14. Ticket recommendations

### Evidence-backed tickets

**None.**

Creating or recommending implementation tickets without identifying the underlying source mechanism would violate the requested proof standard.

### Gated study ticket template

The only defensible future unit would be a renewed read-only study, not a feature implementation:

#### Study the pinned `Mchicao/endless` revision

- **For:** FAK maintainers evaluating external mechanisms for verified autonomous execution.
- **Problem:** The pinned repository has not yet been examined, so its architecture, mechanisms, risks, and overlap with FAK are unknown.
- **Today:** Only the repository identity, revision, local path, and declared MIT license are known.
- **Better because:** A source- and history-backed study would permit small causal techniques to be accepted or rejected independently rather than borrowing a project-level design by reputation.
- **Witness:** A report containing source `file:line`, FAK `file:line`, dependency and license inventories, test-backed execution flow, duplicate-query output, and one ablation per retained candidate.
- **Centrality:** Stewardship until a Core or Enabling mechanism is actually established.
- **P1 — Real user/problem:** Provisionally passes only as research hygiene; no product problem has yet been connected to a source mechanism.
- **P2 — Next-best alternative:** Continue using FAK’s existing guard, watchdog, DOS, context-MMU, and workflow surfaces.
- **P3 — Working spine:** Not applicable until a concrete technique is found; the smallest spine is the read-only evidence map.
- **P4 — Witnessability:** Passes if every conclusion is bound to pinned source lines, duplicate searches, and falsifiable measurements.

Per the user’s instruction, this is a recommendation in the report only; no issue was filed.

---

## 15. Explicit exclusions

The following are explicitly excluded from recommendation:

1. **Any inference from the name “endless.”**  
   A repository name does not establish an autonomous loop, retry system, or continuation protocol.

2. **Any claimed architecture for `endless`.**  
   No entry point, package graph, runtime, or execution trace was read.

3. **Any direct code port.**  
   No code or copyright notice was inspected.

4. **Any dependency adoption.**  
   No manifest or lockfile was inspected.

5. **Any security-positive claim.**  
   No subprocess, filesystem, network, credential, or capability handling was examined.

6. **Any performance claim.**  
   No benchmark, workload, baseline, accounting model, or operating envelope was examined.

7. **Any novelty claim.**  
   FAK’s code, docs, claims, issues, and history were not queried for duplicates.

8. **Any exact FAK implementation seam.**  
   The necessary source line inspection did not occur.

9. **Any issue filing.**  
   The user expressly prohibited it, and the evidence would not support one.

10. **Any repository mutation.**  
    The requested work was read-only; no changes were made.

---

## Final assessment

- **Result:** Investigation halted before source evidence was admitted.
- **Borrow decision:** No candidate accepted.
- **Risk decision:** Fail closed on all uninspected mechanisms.
- **Repository state:** Unchanged.
- **Issues:** None filed.
- **Evidence strength:** The repository identity, pin, local location, and declared MIT license come from the task; FAK’s path-level architecture comes from the supplied orientation; all requested source-level comparisons remain unverified.