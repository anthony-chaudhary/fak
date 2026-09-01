---
title: "TensorBuild study: evidence identity and artifact liveness"
description: "A hash-pinned local study of how TensorBuild represents build identity, evidence strength, artifact reachability, and outcome economics."
---

# TensorBuild deep study: evidence identity, artifact liveness, and outcome economics

**Observed:** 2026-08-15
**Source:** local snapshot at `C:\work\tb\tensor-build-main`
**Snapshot identity:** SHA-256 `bf4dd9267f31dea48b925602e3d1326f65ca3a1e02d3062afecf414af1614288` over 6,358 sorted path/content digests (218,328,563 bytes)
**Module declaration:** owner-qualified private path, redacted from the public copy
**Revision caveat:** this copy is an uncommitted snapshot nested inside an unrelated enclosing `C:\work` Git checkout. Its source repository is unavailable through the current GitHub credentials, so no trustworthy source commit, issue, PR, release, or blame history could be recovered. File hashes and the whole-snapshot digest above are the reproducibility pin.
**License disposition:** no root `LICENSE`, `COPYING`, or `NOTICE` was present and the declared GitHub repository was unavailable. This pass borrows concepts only; it copies no TensorBuild implementation or prose.

## Verdict first

TensorBuild's most transferable lesson is not a TensorRT optimization. It is that **an engineering result becomes reusable only after identity, evidence strength, liveness, and operational cost are all explicit**. The repository repeatedly turns those concerns into typed objects and fail-closed checks:

1. engine identity includes model, builder, hardware, toolchain, flags, and content digests;
2. result documents preserve evidence tier and distinguish unknown from negative;
3. artifact indexes expose reachability and supersession instead of treating file presence as currency;
4. the same one-binary surface serves humans and agents with exit-code verdicts and JSON;
5. audits account for token spend by work class rather than celebrating aggregate activity.

fak already has strong pieces of this philosophy: content-addressed cache keys, effect receipts, claims tags, named documentation reachability, work-type classification, and per-session usage. The underlearned opportunity is **joining those pieces**. Three gaps survived current-tree and issue deduplication and are filed as #6874, #6875, and #6876.

## What was actually studied

This was a second, broader pass over the local snapshot rather than another README skim.

| Surface | Evidence inspected | What it establishes |
|---|---|---|
| Front doors | `README.md`, `AGENTS.md`, `llms.txt`, `docs/INDEX.md`, `CHANGELOG.md` | Product shape, human/agent symmetry, documentation routing, and stated status as of 2026-07-31. |
| Architecture | `internal/enginekey`, `internal/results`, `internal/conn`, `internal/hull`, `internal/adjudicate`, `internal/servegate` | Typed identity, result/evidence contracts, operator control plane, and decision-boundary machinery. |
| Designs | `docs/design/DESIGN-repo-index-graph.md`, `DESIGN-agentic-ground-truth-engine.md`, `DESIGN-gt-agent-tools.md`, `DESIGN-conn-control-interface.md` | Intended graph, ground-truth, tool, and control-plane semantics. |
| Audits and retrospectives | `docs/audits/AUDIT-index-reachability.md`, `AUDIT-agent-token-spend-by-work-class.md`, `AUDIT-build-where-run-where.md`, `AUDIT-repo-weight-and-payload-migration.md`, `RETRO-build-proveout.md` | Where the project found its own abstractions insufficient and how it measured the gaps. |
| Research | `docs/research/INSIGHT-tensorrt-decoupling.md`, `INSIGHT-two-levels-of-quality-loss.md`, `REPORT-llm-first-data-engine.md` | Separation of build/run concerns, quality decomposition, and agent-assisted data operations. |
| Confusion control | `DISAMBIGUATION.md` and `docs/disambiguation/` | A durable numbered registry for terms that repeatedly cause wrong decisions. |
| Proof density | 2,337 Go files, including 1,389 package tests across 32 `internal/` leaves | The contracts have adversarial shape, omission, provenance, and fail-closed tests. |
| Results corpus | `results/`, including phase ledgers, manifests, hashes, logs, images, and negative outcomes | Captured artifacts are part of the product, not disposable run residue. |

The enclosing Git history contains only two unrelated commits and Git-note commits; it is not evidence about this snapshot's development. The unavailable remote also prevents a defensible open/closed issue or PR survey. Those are explicit evidence gaps, not silently replaced with inference.

## Exhaustive inventory refresh (2026-08-25)

[`inventory/local-tensor-build.json`](../research/inventory/local-tensor-build.json) is the pinned machine denominator generated with `fak study-inventory` at `snapshot-sha256:bf4dd9267f31dea48b925602e3d1326f65ca3a1e02d3062afecf414af1614288`. The raw snapshot still contains **6,358 regular files / 218,328,563 bytes**. The inventory indexes **6,318 files / 217,599,107 bytes / 1,482,659 text lines** after deliberately skipping 40 generated files (729,456 bytes) under `internal/build` and `internal/conn/web/dist`; every other regular file is grouped into 18 top-level subsystem rows. This reconciles, rather than changes, the original snapshot pin.

The non-tree audit is explicit in the map's `non_tree_study` block:

- **History, changelog, and releases:** `CHANGELOG.md` is indexed. The snapshot is uncommitted inside an unrelated enclosing checkout, so source Git history, release history, and blame cannot be attributed to TensorBuild.
- **Issues, PRs, and discussions:** templates exist, but no trustworthy forge identity is recoverable with the available credentials. Open/closed issues, pull requests, discussions, and releases therefore remain unavailable, not silently treated as empty.
- **Roadmap and unfinished-work markers:** no dedicated roadmap or unfinished-work-marker path exists. Prose future-work mentions were sampled, but are not promoted to a canonical roadmap.
- **License and provenance:** module manifests are present, but no repository `LICENSE`, `COPYING`, or `NOTICE` file exists. They establish module/dependency provenance, not permission to copy TensorBuild source; this study remains **concepts only**.
- **FAK self-query and dedupe:** three replayable `fak capabilities` queries and the source/issue checks behind the candidate matrix are recorded in the map. The refresh found no novel unowned borrow. #6874 is now closed; #6875 and #6876 remain the open owners.
- **Completeness critic:** the map records every skipped tree, the raw-versus-indexed denominator, all unavailable non-tree surfaces, the candidate dispositions, and issue tracking. The largest residual uncertainty is provenance: without source history or a license, code transfer remains out of scope.

The refresh also corrects the original table's issue-number association: artifact reachability maps to **#6876**, work-class/token-cost attribution maps to **#6874**, and the negative experiment ledger maps to **#6875**.

## Iterative experimentation update (2026-08-27)

This update asks a narrower question prompted by #6875: how can performance work stay cheap and
creative while it is searching, yet remain rigorous at the point where fak promotes a result or
closes a claim? It extends this study instead of creating a second note because the TensorBuild
candidate already owns scoped negative and inconclusive experiment memory.

### Dated primary-source facts and inference

| Kind | Observation | Transfer to fak |
|---|---|---|
| **SOURCE FACT — NIST** | The NIST/SEMATECH Engineering Statistics Handbook recommends a sequence of small experiments rather than one large experiment. Its first stage screens many possible factors; later focused experiments follow the direction found and characterize the optimum. The accompanying DOE checklist says to preserve raw data and record what happened. Observed 2026-08-27; pages last modified 2026-04-22. Sources: [iterative nature](https://www.itl.nist.gov/div898/handbook/pri/section2/pri223.htm) and [DOE steps](https://www.itl.nist.gov/div898/handbook/pri/section1/pri13.htm). | **INFERENCE:** exploration and claim closure need different evidence rungs. A cheap screen may choose the next experiment; it must not by itself close a performance claim. |
| **SOURCE FACT — Google Benchmark** | At pinned revision [`6408acf50aa7a157a8bc561c367baef0721ccd38`](https://github.com/google/benchmark/blob/6408acf50aa7a157a8bc561c367baef0721ccd38/docs/user_guide.md), committed 2026-05-27, `--benchmark_dry_run` forces one iteration and one repetition to check that a benchmark runs. Full repetitions report summary statistics, while individual repetitions are reported by default and can remain in file output even when only aggregates are displayed. Observed 2026-08-27. | **INFERENCE:** a screening receipt should preserve what ran, its exact identity, and its outcome without pretending that one cheap run is stable performance evidence. Repetition, quality, noise, ambient-state, and matched-envelope gates remain promotion requirements. |

**License disposition: INSPIRE-ONLY.** Google Benchmark is Apache-2.0 at the pinned revision, and
NIST describes employee-authored works as generally outside US copyright protection. This update
copies neither implementation nor expressive prose; it independently specifies only the staged
mechanism.

### Current-fak witness: PARTIAL overall

- **PRESENT — strict promotion:** `internal/nativeperf/receipt.go` binds the exact artifact,
  machine, revision, controls, repetitions, quality, memory, execution identity, commands, and
  profiler artifacts. `internal/nativeperf/gate.go` admits only a comparable candidate under the
  declared repetition, noise, quality, ambient-evidence, and throughput policy. This remains the
  claim-closing boundary; `docs/native-inference-goal.md` likewise permits unmatched results for
  diagnosis but not for closing a native-performance claim.
- **PARTIAL — preserved exploration:** `internal/benchscore` already distinguishes accepted,
  negative, and exploratory rows, while `internal/nativeperf/aggregate.go` preserves all
  repetitions plus exclusion reasons. Those outcomes were not previously queryable by hypothesis
  and exact revision/environment/artifact identity.
- **ABSENT before #6875 — a reusable screening ledger:** `fak study search` returned no durable
  record for `staged iterative performance experimentation`, `negative inconclusive experiment
  ledger`, or `NIST design of experiments Google Benchmark repetitions`; `fak capabilities`
  returned no on-point card. The older `fak experiments` registry could find model/backend
  collisions but carried no hypothesis, verdict, evidence class, or supersession.

The #6875 spine implements the missing screening rung without weakening promotion. Every
`fak-experiment-receipt/1` row is `evidence_class=screening` and lookup reports
`claim_eligible=false`. `won`, `lost`, `inconclusive`, and `invalid` stay distinct; a loss applies
only to an exact hypothesis + revision + environment + environment digest + artifact digest.
Identity mismatch is not a loss, and `next_action` turns a negative or inconclusive screen into a
focused follow-up instead of a stop. Superseding receipts preserve the history as assumptions or
environments change.

The operator surface in the #6875 implementation is:

```text
fak experiments record --file <receipt.json> [--ledger <path>] [--json]
fak experiments lookup --file <identity.json> [--ledger <path>] [--json]
```

The resulting boundary is **screen → focus → strict promotion**: record cheap scoped evidence;
use exact lookup and `next_action` to choose the next run; then use the existing matched native
performance receipt and `fak native-performance --gate gate-request.json` before making a promoted
performance claim. Screening stays permissive enough to try new ideas, while the promotion seam
keeps the full rigor in the one place where it changes what fak asserts or ships by default.

Refresh this comparison when either NIST page's `Last-Modified` value changes, Google Benchmark
changes its dry-run or repetition semantics, #6875 changes its receipt contract, or the native
performance gate changes its claim-admission inputs. Expected rediscovery terms are `staged
iterative performance experimentation`, `screening confirmatory promotion`, `benchmark dry run
repetitions`, `negative inconclusive experiment ledger`, and `NIST DOE Google Benchmark`.

## Architecture map

### 1. Identity before optimization

`internal/enginekey/enginekey.go` models canonical identity over model, target class, backend, precision, shapes, workspace, builder flags, toolchain, and silicon. Companion files (`contentid.go`, `provenance.go`, `pinidentity.go`, `storepath.go`) separate raw content identity, provenance, and storage location. Tests cover injectivity, omitted source state, contradictory parameters, missing raw digests, aliases, and store-path precision.

The spirit is stronger than “hash the model”: **a cache or benchmark key must include every axis capable of changing the artifact or verdict, and absence must remain distinguishable from a measured value**.

### 2. Evidence is typed, not merely attached

`internal/results/schema.go`, `tiering.go`, `tiermint.go`, and `scoredoc.go` encode result shape, evidence tiers, provenance, operating points, geometry, preprocessing, costs, and absence coverage. Tests exercise missing schema, unfired guards, untracked membership, symlinked subtrees, provenance, and unknown operating points.

The durable lesson is to keep four states separate: witnessed positive, witnessed negative, inconclusive/invalid, and no evidence. Collapsing the last three creates false confidence and causes dead experiments to be rerun.

### 3. Presence is weaker than reachability

`docs/design/DESIGN-repo-index-graph.md` defines a typed graph whose nodes include docs, results, plans, issues, and artifacts. `docs/audits/AUDIT-index-reachability.md` checks whether those objects are reachable from declared roots. The useful abstraction is not “generate an index”; it is **name the root set and resolver, preserve the denominator, and expose orphaned or superseded evidence**.

fak implemented the documentation portion in closed #5937. TensorBuild shows the next layer: claim/issue/run roots should point to current proof artifacts, not merely to documents that mention one another.

### 4. One control plane, two consumers

The README treats a human terminal user and an agent as two consumers of one `tb` surface. Exit status is the verdict, `--json` is the machine form, and `AGENTS.md`/`llms.txt` route agents into the same verbs. `internal/conn` extends this into an operator control plane with authority, eval firewall, answer packs, events, and source-aware help.

fak has absorbed much of this pattern. The remaining lesson is not to add a parallel agent API: preserve one semantic operation and adapt rendering/transport around it.

### 5. Cost is meaningful only when joined to work and outcome

`docs/audits/AUDIT-agent-token-spend-by-work-class.md` asks which work classes consumed tokens, how much attribution coverage exists, and what artifacts resulted. This avoids optimizing the largest bucket without knowing its yield and extrapolating from the classified subset without its denominator.

fak has both halves but not the join: `internal/worktype` emits deterministic classifications, while closed #3329 added per-session token/cost accounting.

### 6. Confusions deserve stable IDs

`DISAMBIGUATION.md` is a front-door registry of numbered ambiguity entries linked to focused pages. Recurring conflations become searchable objects rather than repeated prose corrections. fak has individual disambiguation notes and scoring skills, but no comparable global registry.

This remains a watch candidate rather than a new issue: a registry-only change would be documentation architecture before a witnessed retrieval failure.

## Candidate matrix and current-fak witnesses

| Candidate mechanism | Source fact | Current fak evidence | Verdict | Disposition |
|---|---|---|---|---|
| Claim-to-proof artifact reachability | Typed nodes/edges and reachability from named roots. | Closed #5937 measures docs, but searches for `artifact reachability`, `artifact graph`, and `orphaned docs` found no proof-liveness equivalent. | **PARTIAL** | Filed #6876; open at refresh. |
| Work-class × token-cost × outcome join | Agent spend is audited by work class with coverage and delivered artifacts. | `internal/worktype` classifies work; closed #3329 reports cost; no joined issue was found. | **SHIPPED** | Filed #6874; closed at refresh. |
| Scoped negative experiment ledger | Result schema and engine identity preserve evidence class and exact environment. | Individual negatives exist (for example closed #5852), but no structured failed-experiment lookup was found. | **PARTIAL** | Filed #6875; open at refresh. |
| Complete cache/build identity | Artifact keys cover model, flags, toolchain, and silicon. | fak has content-addressed keys, build-profile SSOT, runtime receipts, model fingerprints, and effect safety. | **PRESENT** | Retain as audit heuristic. |
| Evidence-tiered verified claims | Evidence grades refuse omitted/unknown states. | Open #3949, #4084, and #4085 cover tiers, basis quorum, and reproducibility. | **ALREADY FILED** | Deduped. |
| Documentation reachability denominator | Named roots/resolvers report reachable/total. | Closed #5937 shipped named `R-LINK`/`R-MENTION` semantics and denominators. | **PRESENT** | #6874 starts above docs. |
| Human/agent one-binary parity | One verb surface, exit-code verdicts, and JSON. | Already a core fak convention. | **PRESENT** | No issue. |
| Build-where/run-where decoupling | Build hosts and execution hosts are separate and receipted. | fak hardware gates, fleet nodes, build profiles, and remote witnesses encode this. | **PRESENT** | No issue. |
| Global disambiguation registry | Recurring confusions receive stable IDs and pages. | fak has notes, `disambiguate-section`, `disambiguation-score`, and labels, but no registry. | **PARTIAL, weak witness** | Watch. |
| LLM-first ground-truth/data engine | Deterministic tools govern data/evaluation boundaries. | fak adjudication, effect receipts, tool routing, corpus grading, and managed-context work cover the transferable seam. | **PRESENT / DOMAIN-SPECIFIC** | No issue. |
| Repository payload separation | Heavy payload migration uses hashes/manifests. | fak has generated-output defaults, scratch allocation, artifact paths, and ledgers. | **PRESENT** | No issue. |

## Filed borrows

### #6874 — attribute session token cost to work class and witnessed outcome

Smallest spine: join existing session usage to `fak-worktype/1` and a minimal explicit outcome vocabulary, preserving classified-session and classified-token denominators. It observes before it optimizes.

### #6875 — queryable scoped ledger for negative and inconclusive results

Smallest spine: append-only typed receipts keyed by hypothesis and relevant environment/revision/artifact identity, with `won`, `lost`, `inconclusive`, and `invalid` kept distinct. One existing real negative result must be migrated as the live witness.

### #6876 — typed artifact reachability from active claims to current proof

Smallest spine: a checked manifest and `fak evidence reachability --json` that rejects duplicate/dangling IDs and reports active, superseded, and orphaned proof artifacts under named roots. This extends rather than duplicates #5937.

All three issues are labeled, milestoned to **Generation G1 - Next Gen**, and contain Core-through-line, Gold-plating-boundary, P1-P4, value frame, current-fak witness, and done conditions.

## Best-default frontier and bounded supersets

| Axis | Best default for fak | Bounded superset worth preserving |
|---|---|---|
| Artifact navigation | Generated indexes and named doc reachability. | Typed proof graph only for artifacts backing claims/issues/runs; do not graph every file. |
| Cost reporting | Per-session usage. | Join to deterministic worktype/outcome when trace coverage exists; expose unknown coverage. |
| Experiment memory | Dated benchmark reports. | Structured negative receipts for exact lookup; prose remains explanation, not query index. |
| Identity | Subsystem-local cache/model/runtime fingerprints. | Share identity components only where cross-subsystem mismatch is witnessed. |
| Disambiguation | Focused notes and scoring skills. | Global registry only after measured retrieval or repeated-conflation evidence. |

## What not to copy

- **The repository's breadth.** Thirty-two large internal packages reflect TensorBuild's domain, not a target package count.
- **A graph before roots exist.** Typed edges help only when active claims/issues/runs define liveness.
- **Source prose or code.** Licensing is unresolved; this study carries concepts and independently specified effects only.
- **Domain-specific vision semantics.** Hull geometry, SAM/COCO label generation, detector operating points, and tactics are peripheral unless a fak seam independently needs them.
- **Aggregate activity as success.** The token-spend audit matters precisely because volume is not outcome.

## Problem and value check

- **Centrality:** Enabling. The study improves proof reuse and operating economics, not the kernel hot path itself.
- **P1 managed context:** 6,358 files are compressed into source-anchored mechanisms and three dispatchable gaps.
- **P2 net-true efficiency:** no gain is claimed. #6875 requires denominators and a baseline before intervention.
- **P3 bounded adaptation:** proposed classifiers/lookups preserve unknown and inconclusive states and explicit scope.
- **P4 integrated operations:** borrows join existing claims, worktype, dispatch, benchmark, and receipt seams.

**For** fak maintainers and coding agents; **Problem:** the local TensorBuild snapshot contained more operational knowledge than prior references captured; **Today:** only one inherited fleet-wave rationale named it; **Better because:** mechanisms, witnesses, dedupes, and three spines are durable and dispatchable; **Witness:** this pinned study, current-tree/issue queries, and #6874-#6876.

## Reproduction

The snapshot pin was computed by sorting every relative path, hashing each file, then hashing `path + NUL + file_digest + newline` for all 6,358 files. Current-fak dedupe used `git grep` plus `gh issue list --state all` queries for artifact reachability, worktype/work-class cost, experiment receipts, and failed-experiment ledgers.

## Limits and next check

This study cannot make source-history, upstream-roadmap, issue, PR, release, or license claims because the snapshot has no trustworthy repository metadata and the declared remote was unavailable on 2026-08-15. If a canonical TensorBuild checkout or remote becomes available, refresh this note against its exact revision and inspect history around `internal/enginekey`, `internal/results`, the reachability design/audit pair, and the token-spend audit before treating this as a complete upstream study.
