---
title: "Coding-workload vocabulary (cited proposal v0.1.0)"
description: "A cited, four-axis vocabulary separating agent coding-workload patterns from subpatterns, mechanisms, and antipatterns, with rejected conflations and a deterministic machine-readable companion."
---

# Coding-workload vocabulary — cited proposal v0.1.0

**Status: proposal (research maturity `hypothesis`).** This is a defensible vocabulary
proposal grounded in the cited corpus as of **2026-08-10**. It is **not** a claim of
field-wide consensus, and no term here is a maintained fak contract. Issue
[#6209](https://github.com/anthony-chaudhary/fak/issues/6209), parent
[#6208](https://github.com/anthony-chaudhary/fak/issues/6208); generation stream
`gen/next`.

**Machine-readable companion:** [`coding-workload-vocabulary.json`](coding-workload-vocabulary.json)
(`fak.workload-vocabulary.v1`). Every identifier below appears there verbatim.

**Next action:** before reusing a name from this page, check its provenance label and its
inclusion/exclusion criteria; if your case fails the exclusion test, it is a different
element, not a variant of this one.

## What this page is for

The naming risk this addresses is stated in #6209: inventing terms from local intuition
either relabels an established concept or fuses distinctions that the literature already
keeps apart. So every element carries one of three provenance labels:

| Provenance | Meaning |
|---|---|
| `borrowed` | The name and the concept both come from a cited source; we adopt them unchanged. |
| `adapted` | The concept is cited; the name, the boundary, or the scope is ours. |
| `new-synthesis` | No cited source names this element; it is proposed here and the citation supports only its ingredients. |

### Citation verification method and its limit

Each source's title, authors, venue, year, and identifier were checked on **2026-08-10**
against indexed publisher/preprint metadata and abstract text. Direct page fetch was
unavailable in this environment (the `WebFetch` capability is refused at the capability
floor), so **entries are verified at metadata-and-abstract granularity, not full text**.
Claims attributed to a source below are restricted to what its abstract or publicly
indexed summary states. Where an antipattern maps to a source, the source names the
*nearest documented class*, not necessarily the same term — that mapping is our reading
and is labelled accordingly.

## The four axes

The central proposal is that these are **four independent axes**, not four names for the
same thing. An instance of work has a value on each. Collapsing any two is the source of
the conflations rejected below.

| Axis | ID | The question it answers | Example values |
|---|---|---|---|
| Workload shape | `ax.workload-shape` | What is the job, and what artifact counts as done? | the eleven `wp.*` patterns |
| Orchestration topology | `ax.orchestration-topology` | How is the work arranged across steps, agents, and turns? | single interactive loop; fixed staged pipeline; fan-out/fan-in; tree search; human-in-the-loop |
| Verification strategy | `ax.verification-strategy` | What evidence closes the loop? | executable oracle; derived/pinned oracle; relational (metamorphic) oracle; held-out oracle; independent judge; citation resolution; none |
| Failure mode | `ax.failure-mode` | How does it degrade into a plausible wrong answer? | the nine `ap.*` antipatterns |

Two sources make the axis split load-bearing rather than decorative. Agentless solves the
same issue-resolution shape as an agent loop using a fixed three-phase pipeline of
localization, repair, and patch validation — the shape is constant while the topology
changes ([s.agentless]). And MAST's fourteen failure modes are organised into system
design, inter-agent misalignment, and task verification — failure classes that recur
across workloads rather than belonging to any one of them ([s.mast]).

## Patterns — the workload-shape axis

Eleven candidate patterns. A pattern names a **goal and its done-artifact**, nothing else.

| ID | Name | Definition | Aliases | Include when | Exclude when | Provenance | Sources |
|---|---|---|---|---|---|---|---|
| `wp.issue-to-patch` | Issue-to-Patch Repair | Turn a reported defect into a change that alters exactly the reported behavior. | issue resolution; bug fix; test-suite-based program repair | A defect claim exists and the done-artifact is a diff | No prior defect claim; or the check itself is the deliverable | borrowed | [s.swe-bench], [s.apr-bibliography], [s.autocoderover] |
| `wp.spec-to-feature` | Specification-to-Feature Construction | Add an externally visible capability described in prose while preserving existing behavior. | feature implementation; NL-to-code | The described capability does not exist yet | The capability exists but misbehaves (→ `wp.issue-to-patch`) | adapted | [s.swe-bench], [s.agentless], [s.grounded-copilot] |
| `wp.behavior-preserving-restructure` | Behavior-Preserving Restructuring | Change internal structure while asserting that observable behavior is unchanged. | refactoring; cleanup; modularization | The success oracle is "nothing observable changed" | Any observable output is intended to change | borrowed | [s.swe-at-google-lsc], [s.parallel-change] |
| `wp.mechanical-sweep` | Mechanical Fleet-Wide Sweep | Apply one verified rewrite rule across many sites that cannot land as a single atomic change. | large-scale change (LSC); codemod; migration sweep | The rule is decidable per site and site count dominates the cost | One site, or per-site human judgment is required | borrowed | [s.swe-at-google-lsc] |
| `wp.interface-migration` | Interface Migration Under Live Consumers | Change an interface incompatibly while its consumers keep working throughout. | expand-and-contract; parallel change; strangler migration | Consumers exist and cannot be updated atomically | No consumer outside the change's own footprint | borrowed | [s.parallel-change], [s.strangler-fig] |
| `wp.env-remediation` | Environment and Dependency Remediation | Restore a runnable build/test environment when the defect is in the environment, not the program. | build fix; dependency bump; "make it run" | The failure reproduces with no program logic being wrong | Program logic is wrong under one toolchain (→ `wp.issue-to-patch`) | new-synthesis | [s.swe-bench-verified], [s.swe-bench] |
| `wp.oracle-construction` | Oracle Construction | Produce the check itself: tests, properties, metamorphic relations, or fixtures. | test writing; coverage raise; property authoring | The delivered artifact *is* the check | A failing test written as a step toward a fix (→ `sp.reproduce-first`) | adapted | [s.oracle-problem], [s.metamorphic-survey], [s.codet] |
| `wp.comprehension-report` | Comprehension and Explanation Report | Answer a question about a system; the deliverable is a cited explanation, not a diff. | code archaeology; investigation; "how does X work" | No source mutation is required for success | A diff is required for success | new-synthesis | [s.swe-agent], [s.oracle-problem] |
| `wp.triage-and-route` | Triage and Routing | Decide what happens to an item — route, assign, dedupe, defer — without changing the program. | bug triage; assignment; prioritization | The deliverable is a decision about an item | The deliverable is the fix itself | borrowed | [s.who-should-fix] |
| `wp.change-adjudication` | Change Adjudication | Judge a proposed change and its evidence, returning a verdict and findings. | code review; patch-correctness assessment | Input is a candidate change plus its evidence | The same actor also authored the change | adapted | [s.apr-overfitting], [s.mast] |
| `wp.perf-under-invariant` | Performance Optimization Under Invariant | Move a measured quantity while a behavior pin holds. | optimization; tuning; performance-regression hunt | Success requires a measurement against a named baseline | Accepted on a plausibility argument with no measurement | adapted | [s.git-bisect], [s.oracle-problem] |

`wp.perf-under-invariant` is grounded rather than invented: the `git bisect` manual states
the search is not limited to bugs and explicitly gives "the commit that caused a
benchmark's performance to improve" as a target property ([s.git-bisect]) — a measured
quantity is treated as a first-class searchable property.

## Subpatterns — reusable moves

Twenty-two subpatterns. A subpattern is a **move**: it recurs inside more than one
pattern, and naming it as a workload is the error rejected in `cf.subpattern-as-pattern`.

| ID | Name | Definition | Aliases | Include when | Exclude when | Provenance | Sources |
|---|---|---|---|---|---|---|---|
| `sp.reproduce-first` | Reproduce First | Build a deterministic check that fails now and would pass on success, before editing. | failing-test-first; repro harness | A behavioral claim can be executed | The claim is about structure or prose, not behavior | borrowed | [s.swe-bench], [s.apr-bibliography] |
| `sp.input-minimization` | Input Minimization | Shrink a failing input while the failure is preserved, down to a 1-minimal remainder. | delta debugging; test-case reduction; shrinking | The failure is input-triggered and re-runnable | The failure depends on unreproducible external state | borrowed | [s.delta-debugging] |
| `sp.history-bisection` | History Bisection | Binary-search revisions for the first commit at which a monotone property flips. | git bisect; regression hunt | History is navigable and the property is testable per revision | The property is not decidable at an arbitrary revision | borrowed | [s.git-bisect] |
| `sp.spectrum-localization` | Spectrum Localization | Rank suspect code by coverage difference between passing and failing runs. | spectrum-based fault localization; suspiciousness ranking | A suite with both passing and failing runs exists | No executable suite exists | borrowed | [s.fault-localization-survey], [s.autocoderover] |
| `sp.hierarchical-localization` | Hierarchical Localization | Narrow file → class/function → edit site in fixed stages, with no interactive loop. | staged localization; localization funnel | The repo is large and each stage is cheap | Localization needs feedback from execution | borrowed | [s.agentless] |
| `sp.structured-code-search` | Structured Code Search | Retrieve context through program structure (AST/symbols) rather than text similarity. | AST search; symbol retrieval | A parseable program representation is available | The artifact is prose or opaque data | borrowed | [s.autocoderover] |
| `sp.graft-search` | Graft Search | Look for existing in-repo code that already implements the change before writing new code. | plastic surgery; donor search; reuse-first | The change plausibly resembles code already present | The change is genuinely novel to the repository | borrowed | [s.plastic-surgery] |
| `sp.expand-migrate-contract` | Expand, Migrate, Contract | Add the new interface, move consumers, then delete the old one — never broken in between. | parallel change; expand-and-contract | Consumers cannot be moved atomically | A single atomic change is already safe | borrowed | [s.parallel-change] |
| `sp.facade-and-drain` | Facade and Drain | Put a facade in front of the old path, redirect call sites incrementally, then remove the old path. | strangler fig; gradual legacy replacement | The old component must keep serving during replacement | The old component can be switched off at once | borrowed | [s.strangler-fig] |
| `sp.rule-then-sweep` | Rule Then Sweep | Validate the rewrite rule on a small sample, then apply and verify shard by shard. | codemod-then-verify; sharded LSC | The rule is mechanical and the sites are many | Sites need individual judgment | borrowed | [s.swe-at-google-lsc] |
| `sp.behavior-pinning` | Behavior Pinning | Capture current observable behavior as a check before changing anything. | characterization test; golden master; approval test | Current behavior is trusted and re-observable | Current behavior *is* the defect | adapted | [s.oracle-problem] |
| `sp.metamorphic-relation` | Metamorphic Relation | Assert a relation between outputs of related inputs when no expected output is known. | metamorphic testing; relational oracle | Expected output is unavailable but a relation is known | A concrete expected output is available | borrowed | [s.metamorphic-survey], [s.oracle-problem] |
| `sp.generate-and-filter` | Generate and Filter | Sample many candidates and keep those surviving an executable filter or cross-agreement. | sample-and-rank; dual execution agreement | Candidates are cheap and a filter is executable | Each candidate is expensive or unverifiable | borrowed | [s.codet] |
| `sp.execute-observe-repair` | Execute, Observe, Repair | Run, read the actual observation, patch, repeat under an explicit step budget. | agent loop; reason-act loop; self-debug | Execution feedback is available and cheap | Feedback is absent, delayed, or systematically misleading | borrowed | [s.react], [s.swe-agent] |
| `sp.verbal-retrospective` | Verbal Retrospective | Carry a written failure summary into the next attempt instead of re-sampling blind. | reflection memory; verbal reinforcement | Attempts are repeatable and failures are describable | The failure signal cannot be attributed to a step | borrowed | [s.reflexion] |
| `sp.independent-adjudication` | Independent Adjudication | A verifier that did not author the change judges it, with author-side rationale withheld. | blind review; judge panel | The change's correctness is contestable | The available check is a total decidable oracle | adapted | [s.mast], [s.apr-overfitting] |
| `sp.held-out-oracle` | Held-Out Oracle | Reserve a check that the producing step never saw and never optimized against. | held-out tests; unseen regression suite | A check can be withheld without blocking production | Every available check must guide production | adapted | [s.apr-overfitting], [s.swe-bench-verified] |
| `sp.interface-shaping` | Interface Shaping | Change the tools and affordances the agent acts through, not the prompt or the model. | agent-computer interface design; harness design | Failures trace to action/observation format | Failures trace to missing knowledge | borrowed | [s.swe-agent] |
| `sp.blast-radius-scoping` | Blast-Radius Scoping | Declare the write footprint before editing so concurrent work provably cannot collide. | write lease; lane scoping | More than one worker can write the tree | A single serialized writer | new-synthesis | [s.mast], [s.swe-at-google-lsc] |
| `sp.escalation-checkpoint` | Escalation Checkpoint | Hand control to a human at a named decision boundary, not at a step-count limit. | mixed-initiative handoff; human-in-the-loop gate | The decision's cost is asymmetric and its outcome uncertain | The decision is cheap and reversible | borrowed | [s.mixed-initiative] |
| `sp.citation-binding` | Citation Binding | Bind every assertion to a resolvable pointer that is checked apart from the assertion. | provenance binding; `file:line` citation | The deliverable is prose or a report | The deliverable is executable and self-checking | new-synthesis | [s.oracle-problem], [s.mast] |
| `sp.budgeted-exploration` | Budgeted Exploration | Spend a bounded budget in explicit explore mode before committing to one option. | explore-then-accelerate; option sampling | The approach is genuinely unknown | The next step is already known (accelerate mode) | borrowed | [s.grounded-copilot] |

`sp.budgeted-exploration` borrows a measured distinction rather than a slogan: Barke et al.
report that programmer interaction with a code-generating model is **bimodal** —
*acceleration*, where the programmer knows what to do next, and *exploration*, where they
do not and use the model to survey options, with exploration being slower, more deliberate,
and requiring more validation ([s.grounded-copilot]).

## Mechanisms

A mechanism is a **substrate capability**, not a move and not a goal. It is what a
subpattern needs in order to run at all. Listing mechanisms separately is what stops
"we have a sandbox" from being mistaken for "we do reproduce-first".

| ID | Mechanism | Enables |
|---|---|---|
| `mx.execution-sandbox` | Run untrusted code and observe its effects | `sp.reproduce-first`, `sp.execute-observe-repair`, `sp.generate-and-filter` |
| `mx.test-runner` | Select and run checks, reporting per-check status | `sp.reproduce-first`, `sp.behavior-pinning`, `sp.held-out-oracle` |
| `mx.coverage-spectra` | Per-test coverage or execution traces | `sp.spectrum-localization` |
| `mx.vcs-history` | Addressable revisions with checkout and diff | `sp.history-bisection`, `sp.rule-then-sweep` |
| `mx.program-index` | AST/symbol index over the repository | `sp.structured-code-search`, `sp.hierarchical-localization`, `sp.graft-search` |
| `mx.patch-applier` | Apply and revert a diff safely | `sp.rule-then-sweep`, `sp.execute-observe-repair` |
| `mx.write-lease` | Exclusive claim over a declared file region | `sp.blast-radius-scoping` |
| `mx.step-budget` | Bounded, observable step and token accounting | `sp.execute-observe-repair`, `sp.budgeted-exploration` |
| `mx.judge-channel` | A second model or human that never sees author-side rationale | `sp.independent-adjudication`, `sp.escalation-checkpoint` |
| `mx.reference-resolver` | Resolve a citation to its target and compare | `sp.citation-binding` |

## Antipatterns — the failure-mode axis

Nine antipatterns. Each is stated with a **detector**, because a failure mode you cannot
detect is a mood, not an axis value. Source column names the nearest documented class.

| ID | Name | Definition | Aliases | Detector | Provenance | Sources |
|---|---|---|---|---|---|---|
| `ap.oracle-overfit` | Oracle Overfit | The change satisfies the checks it was optimized against and breaks untested but intended behavior. | patch overfitting; test overfitting | A held-out oracle disagrees with the optimized oracle | borrowed | [s.apr-overfitting] |
| `ap.oracle-laundering` | Oracle Laundering | The gate is weakened, narrowed, or deleted so that it passes. | test gaming; check tampering | The diff touches the check and the check's strictness decreased | adapted | [s.apr-overfitting], [s.mast] |
| `ap.silent-scope-narrowing` | Silent Scope Narrowing | Delivered work covers less than was asked and the report does not say so. | quiet descoping | A requested acceptance item has neither an artifact nor an explicit deferral | adapted | [s.mast] |
| `ap.unverified-completion` | Unverified Completion | "Done" is declared with no verification step executed. | claim without witness | No gate invocation appears in the trace | borrowed | [s.mast] |
| `ap.premature-termination` | Premature Termination | The run ends before producing the information needed to close the task. | early stop | Handoff conditions are unmet at exit | borrowed | [s.mast] |
| `ap.no-progress-oscillation` | No-Progress Oscillation | Equivalent actions repeat without changing observable state. | step repetition; thrash loop | Repeated action/observation pairs above a threshold | borrowed | [s.mast] |
| `ap.confabulated-reference` | Confabulated Reference | A cited file, symbol, or source does not exist, or does not support the claim made of it. | hallucinated citation | The reference resolver fails, or the resolved target contradicts the claim | adapted | [s.mast], [s.oracle-problem] |
| `ap.benchmark-validity-drift` | Benchmark Validity Drift | The measuring instrument, not the system, explains the score. | benchmark artifact; unsolvable items | A human re-screen of the items changes the ranking | adapted | [s.swe-bench-verified] |
| `ap.atomicity-overreach` | Atomicity Overreach | A sweep is attempted as one atomic change at a scale where it cannot land. | mega-commit | The change cannot pass presubmit as a single unit | borrowed | [s.swe-at-google-lsc] |

MAST supplies the empirical spine here: 1,600+ annotated traces across seven multi-agent
frameworks, 14 failure modes in three categories, with inter-annotator agreement reported
at kappa = 0.88 ([s.mast]). `ap.unverified-completion`, `ap.premature-termination`, and
`ap.no-progress-oscillation` are its "No or Incomplete Verification", "Premature
Termination", and "Step Repetition" modes under our names.

## Rejected conflations

Six conflations that are tempting, common, and wrong. Each gets a counterexample that a
reader can check.

### `cf.shape-is-topology` — "multi-agent is a workload pattern"

**Counterexample.** `wp.mechanical-sweep` runs as a single scripted pass and as a fan-out of
per-shard workers. The goal, the done-artifact, the oracle, and the dominant failure mode
(`ap.atomicity-overreach`) are identical in both. Conversely "fan-out" says nothing about
what done means, so it cannot be a workload. Agentless makes the same point from the other
side: it solves the issue-resolution shape with a fixed localize/repair/validate pipeline
rather than an agent loop ([s.agentless]).

### `cf.shape-is-verification` — "if it has no test, it is not real engineering work"

**Counterexample.** `wp.comprehension-report` has no executable oracle by construction; its
verification strategy is citation resolution. Meanwhile a fully test-green
`wp.issue-to-patch` result can still be wrong: Smith et al. evaluated repair tools on bugs
that each had a human-written patch and found that patches which pass the available tests
can overfit them and break untested desired functionality ([s.apr-overfitting]). Test-green
is neither necessary nor sufficient, so it cannot define the shape.

### `cf.subpattern-as-pattern` — "bisect is a workload"

**Counterexample.** `sp.history-bisection` appears inside `wp.issue-to-patch` (find the
breaking commit), inside `wp.perf-under-invariant` (find the regressing commit — the
`git bisect` manual gives benchmark movement as a target property), and inside
`wp.env-remediation` (find the breaking dependency bump) ([s.git-bisect]). A move reusable
across shapes cannot itself be the shape.

### `cf.workflow-patterns-transfer` — "the Workflow Patterns catalogue already names these"

**Counterexample.** Van der Aalst et al.'s workflow patterns are control-flow constructs
that capture workflow functionality for comparing workflow management systems — they
describe how tokens route through a process model ([s.workflow-patterns]). "Parallel Split"
describes a mechanical code sweep and a document-translation pipeline identically, because
it is silent about what the work is and what makes it correct. Borrowing that catalogue for
`ax.workload-shape` imports a topology name where a goal name is required. It is the right
vocabulary for `ax.orchestration-topology` and the wrong one for `ax.workload-shape`.

### `cf.repair-is-restructure` — "refactor to fix the bug"

**Counterexample.** `wp.behavior-preserving-restructure` asserts that observable behavior is
unchanged — its oracle is that the pin still holds. `wp.issue-to-patch` asserts that exactly
one observable behavior changed. A change that does both has no single passing oracle: the
pin fails at the fixed site, and the repair witness cannot distinguish "fixed" from "moved".
They must be sequenced, not merged. The `expand / migrate / contract` discipline exists
precisely because the phases are kept separate and the code is never broken between them
([s.parallel-change]).

### `cf.environment-is-logic` — "a red build is a bug in the program"

**Counterexample.** OpenAI's SWE-bench Verified screening found development environments
that were difficult to set up reliably, causing unit tests to fail regardless of the
solution, alongside tests too specific to the original patch and under-specified issue
descriptions ([s.swe-bench-verified]). Those are `wp.env-remediation` and
`ap.benchmark-validity-drift`, not defects in the program under test. Routing them to
`wp.issue-to-patch` produces edits to correct code.

## Evidence table

| Source | What it establishes (per its abstract/summary) | What it grounds here |
|---|---|---|
| [s.workflow-patterns] | A catalogue of workflow control-flow patterns used to compare workflow management systems. | `ax.orchestration-topology`; `cf.workflow-patterns-transfer` |
| [s.delta-debugging] | Delta Debugging minimizes a failing test case to a 1-minimal failure-inducing input and isolates the passing/failing difference. | `sp.input-minimization` |
| [s.oracle-problem] | Frames the test-oracle problem; surveys specified, derived, implicit, and human oracles, including metamorphic testing. | `ax.verification-strategy`; `sp.behavior-pinning`; `wp.oracle-construction` |
| [s.metamorphic-survey] | Correctness can be judged by transforming an input and observing how the output changes, when a concrete expected output is unavailable. | `sp.metamorphic-relation` |
| [s.fault-localization-survey] | Surveys fault-localization techniques that guide developers to fault locations with minimal human intervention. | `sp.spectrum-localization` |
| [s.apr-bibliography] | Structures automatic repair into behavioral and state repair, and catalogues bug oracles and repair operators. | `wp.issue-to-patch`; `sp.reproduce-first` |
| [s.apr-overfitting] | Patches that pass the available tests can overfit them and break untested but desired functionality. | `ap.oracle-overfit`; `sp.held-out-oracle`; `cf.shape-is-verification` |
| [s.plastic-surgery] | Commits are graftable from code already present in the project to a degree largely independent of commit size and type. | `sp.graft-search` |
| [s.who-should-fix] | Bug reports must be triaged and assigned; a learned model reaches 57%/64% precision on Eclipse/Firefox. | `wp.triage-and-route` |
| [s.swe-bench] | 2,294 real GitHub issue/PR tasks; resolution frequently requires coordinated edits across functions, classes, and files. | `wp.issue-to-patch`; `wp.spec-to-feature`; `sp.reproduce-first` |
| [s.swe-bench-verified] | 500 human-screened samples; original items suffered under-specified issues, over-specific tests, and unreliable environments. | `wp.env-remediation`; `ap.benchmark-validity-drift`; `cf.environment-is-logic` |
| [s.swe-agent] | A purpose-built agent-computer interface materially improves an LM agent's ability to edit, navigate, and run tests. | `sp.interface-shaping`; `sp.execute-observe-repair`; `wp.comprehension-report` |
| [s.agentless] | A three-phase localization → repair → validation pipeline, with hierarchical file → class/function → edit-site narrowing. | `sp.hierarchical-localization`; `cf.shape-is-topology` |
| [s.autocoderover] | Code search over an AST program representation, sharpened by spectrum-based fault localization when a suite exists. | `sp.structured-code-search`; `sp.spectrum-localization` |
| [s.react] | Interleaved reasoning traces and actions: thoughts guide action selection, observations inform subsequent reasoning. | `sp.execute-observe-repair` |
| [s.reflexion] | Verbal feedback is summarized into episodic memory and added as context for the next episode, without weight updates. | `sp.verbal-retrospective` |
| [s.codet] | Generated tests plus dual execution agreement select among sampled solutions, raising HumanEval pass@1 to 65.8%. | `sp.generate-and-filter`; `wp.oracle-construction` |
| [s.mast] | 14 failure modes over 1,600+ traces in three categories: system design, inter-agent misalignment, task verification (kappa = 0.88). | `ax.failure-mode`; six `ap.*` entries |
| [s.mixed-initiative] | Principles for coupling automated services with direct manipulation rather than choosing between them. | `sp.escalation-checkpoint` |
| [s.grounded-copilot] | Interaction is bimodal — acceleration (knows the next step) versus exploration (surveys options, validates more). | `sp.budgeted-exploration` |
| [s.swe-at-google-lsc] | An LSC is a logically related change set that cannot practically be submitted atomically; max atomic size falls as scale grows. | `wp.mechanical-sweep`; `ap.atomicity-overreach`; `sp.rule-then-sweep` |
| [s.parallel-change] | Expand, migrate, contract: keep the old design while adding the new, so the code is never broken. | `wp.interface-migration`; `sp.expand-migrate-contract`; `cf.repair-is-restructure` |
| [s.strangler-fig] | Gradual replacement: wrap the old system, redirect incrementally, retire the host. | `sp.facade-and-drain`; `wp.interface-migration` |
| [s.git-bisect] | Binary search over revisions for any property change, explicitly including benchmark-performance movement. | `sp.history-bisection`; `wp.perf-under-invariant` |

## Sources

All entries accessed **2026-08-10**. Verification granularity is metadata and abstract, per
the limit stated above.

- **[s.workflow-patterns]** W. M. P. van der Aalst, A. H. M. ter Hofstede, B. Kiepuszewski, A. P. Barros. "Workflow Patterns." *Distributed and Parallel Databases* 14, 5–51 (2003). <https://doi.org/10.1023/A:1022883727209> · catalogue: <http://www.workflowpatterns.com/documentation/>
- **[s.delta-debugging]** A. Zeller, R. Hildebrandt. "Simplifying and Isolating Failure-Inducing Input." *IEEE TSE* 28(2), 183–200 (2002). <https://doi.org/10.1109/32.988498>
- **[s.oracle-problem]** E. T. Barr, M. Harman, P. McMinn, M. Shahbaz, S. Yoo. "The Oracle Problem in Software Testing: A Survey." *IEEE TSE* 41(5), 507–525 (2015). <https://doi.org/10.1109/TSE.2014.2372785> · open access: <https://discovery.ucl.ac.uk/1471263/>
- **[s.metamorphic-survey]** S. Segura, G. Fraser, A. B. Sánchez, A. Ruiz-Cortés. "A Survey on Metamorphic Testing." *IEEE TSE* 42(9), 805–824 (2016). <https://doi.org/10.1109/TSE.2016.2532875>
- **[s.fault-localization-survey]** W. E. Wong, R. Gao, Y. Li, R. Abreu, F. Wotawa. "A Survey on Software Fault Localization." *IEEE TSE* 42(8), 707–740 (2016). <https://doi.org/10.1109/TSE.2016.2521368>
- **[s.apr-bibliography]** M. Monperrus. "Automatic Software Repair: A Bibliography." *ACM Computing Surveys* 51(1), Article 17 (2018). <https://doi.org/10.1145/3105906>
- **[s.apr-overfitting]** E. K. Smith, E. T. Barr, C. Le Goues, Y. Brun. "Is the cure worse than the disease? Overfitting in automated program repair." *ESEC/FSE 2015*, 532–543. <https://doi.org/10.1145/2786805.2786825>
- **[s.plastic-surgery]** E. T. Barr, Y. Brun, P. Devanbu, M. Harman, F. Sarro. "The Plastic Surgery Hypothesis." *FSE 2014*, 306–317. <https://doi.org/10.1145/2635868.2635898>
- **[s.who-should-fix]** J. Anvik, L. Hiew, G. C. Murphy. "Who Should Fix This Bug?" *ICSE 2006*, 361–370. <https://doi.org/10.1145/1134285.1134336>
- **[s.swe-bench]** C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, K. Narasimhan. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" *ICLR 2024*. <https://arxiv.org/abs/2310.06770>
- **[s.swe-bench-verified]** OpenAI. "Introducing SWE-bench Verified." 13 August 2024. <https://openai.com/index/introducing-swe-bench-verified/>
- **[s.swe-agent]** J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, O. Press. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." *NeurIPS 2024*. <https://arxiv.org/abs/2405.15793>
- **[s.agentless]** C. S. Xia, Y. Deng, S. Dunn, L. Zhang. "Agentless: Demystifying LLM-based Software Engineering Agents." arXiv:2407.01489 (2024). <https://arxiv.org/abs/2407.01489>
- **[s.autocoderover]** Y. Zhang, H. Ruan, Z. Fan, A. Roychoudhury. "AutoCodeRover: Autonomous Program Improvement." *ISSTA 2024*. <https://arxiv.org/abs/2404.05427>
- **[s.react]** S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, Y. Cao. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*. <https://arxiv.org/abs/2210.03629>
- **[s.reflexion]** N. Shinn, F. Cassano, B. Labash, A. Gopinath, K. Narasimhan, S. Yao. "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*. <https://arxiv.org/abs/2303.11366>
- **[s.codet]** B. Chen, F. Zhang, A. Nguyen, D. Zan, Z. Lin, J.-G. Lou, W. Chen. "CodeT: Code Generation with Generated Tests." *ICLR 2023*. <https://arxiv.org/abs/2207.10397>
- **[s.mast]** M. Cemri, M. Z. Pan, S. Yang, et al. "Why Do Multi-Agent LLM Systems Fail?" arXiv:2503.13657 (2025). <https://arxiv.org/abs/2503.13657>
- **[s.mixed-initiative]** E. Horvitz. "Principles of Mixed-Initiative User Interfaces." *CHI 1999*, 159–166. <https://doi.org/10.1145/302979.303030>
- **[s.grounded-copilot]** S. Barke, M. B. James, N. Polikarpova. "Grounded Copilot: How Programmers Interact with Code-Generating Models." *PACMPL* 7(OOPSLA1) (2023). <https://doi.org/10.1145/3586030>
- **[s.swe-at-google-lsc]** T. Winters, T. Manshreck, H. Wright. *Software Engineering at Google*, Chapter 22: "Large-Scale Changes." O'Reilly, 2020. <https://abseil.io/resources/swe-book/html/ch22.html>
- **[s.parallel-change]** D. Sato. "Parallel Change." martinfowler.com bliki, 13 May 2014. <https://martinfowler.com/bliki/ParallelChange.html>
- **[s.strangler-fig]** M. Fowler. "Strangler Fig Application." martinfowler.com bliki. <https://martinfowler.com/bliki/StranglerFigApplication.html>
- **[s.git-bisect]** Git project. "git-bisect(1)." <https://git-scm.com/docs/git-bisect>

Twenty-four sources; the definition of done required at least twelve.

## Machine-readable companion and its determinism witness

[`coding-workload-vocabulary.json`](coding-workload-vocabulary.json) carries every axis,
pattern, subpattern, mechanism, antipattern, conflation, and source above under schema
`fak.workload-vocabulary.v1`. There is no report generator to run twice; instead the
artifact is **canonical**, so re-serialization is a fixed point and any editor that
round-trips it must produce the same bytes:

```sh
python3 - <<'PY'
import json
p = "docs/research/coding-workload-vocabulary.json"
raw = open(p, encoding="utf-8").read()
canon = json.dumps(json.loads(raw), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
print("canonical:", raw.replace("\r\n", "\n") == canon)
PY
```

Counts are asserted by the same artifact (`counts` block) and match this page: 4 axes,
11 patterns, 22 subpatterns, 10 mechanisms, 9 antipatterns, 6 conflations, 24 sources.

## Promotion, demotion, and invalidating assumptions

**Promotion evidence — what would move this toward `gen/now`.** Two independent annotators
classify a sample of this repository's own trajectories using only this page's
inclusion/exclusion criteria and reach substantial agreement on `ax.workload-shape` without
consulting each other, *and* at least one downstream consumer (a dispatch classifier, a
report schema, or an issue template) reads the `wp.*`/`sp.*` identifiers rather than
re-deriving its own names. Vocabulary that nothing consumes stays a hypothesis.

**Demotion or retirement evidence.** Retire an element when annotators cannot separate it
from a neighbour without inventing a criterion absent from this page; demote the whole
proposal if a survey published after 2026-08-10 supplies an equivalent axis split with
broader evidence, in which case borrow it and keep only the `new-synthesis` entries that
survive.

**Invalidating assumptions.**

1. **The axes are assumed independent.** If, across real trajectories, workload shape
   predicts orchestration topology strongly enough that topology carries no additional
   information, then `ax.orchestration-topology` is a derived attribute and the four-axis
   claim is wrong.
2. **The four `new-synthesis` elements are assumed genuinely unnamed.** `wp.env-remediation`,
   `wp.comprehension-report`, `sp.blast-radius-scoping`, and `sp.citation-binding` rest on
   a corpus verified at abstract granularity by one worker on one day; a full-text sweep, or
   a body of literature outside the surveyed streams, could show any of them is a rename.
3. **Sources are assumed to say what their abstracts say.** Verification did not reach full
   text (`WebFetch` is refused at this host's capability floor). Any attribution above that
   the full paper contradicts invalidates that row of the evidence table.
