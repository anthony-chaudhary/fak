# Harness domain boundary audit — 2026-08-15

## Verdict

The harness work has the right production center but the wrong apparent center.

**Keep and finish the explicit compiler path:** authored profile/layer selection, typed asset
composition, component dependency resolution, immutable locks, and fail-closed launch. Those
are reusable product contracts. **Do not make contextual domain classification a production
default.** The current `legal` / `coding` / `integrated` classifier is a useful deterministic
prove-out and authoring aid, but its vocabulary and authored corpus do not establish a
portable product ontology.

The immediate problem is not that the classifier is unsafe in isolation. It already abstains
on ambiguity. The problem is lifecycle conflation: `classify`, `select`, `compose`, `resolve`,
preview, and generated lock launch are presented as adjacent harness features even though only
some belong on a production request path. The production-shaped pieces are also split across
three intermediate manifest types and are not yet one lock compiler and validator.

Issue [#6938](https://github.com/anthony-chaudhary/fak/issues/6938) now owns that P0 boundary:
one explicit manifest/profile-to-lock compiler, lock-only production launch, and classifier
opt-in rather than default.

## Value frame and problem fit

- **Centrality:** Enabling. This does not itself improve model quality; it makes the kernel's
  policy, routing, and context controls safely reusable by product builders.
- **For:** builders shipping a differentiated agent product on the fak kernel.
- **Problem:** they need deterministic product composition without inheriting a guessed task
  ontology or stitching together several partially overlapping schemas.
- **Today:** explicit layers, typed assets, dependencies, and lock-shaped outputs exist, but
  classifier inference can feed selection and no one compiler/lock contract owns launch.
- **Better because:** explicit product intent becomes reproducible and reviewable; optional
  inference can help authors without becoming production authority.
- **Witness:** compile two explicit stacks to distinct immutable locks, launch only those locks,
  reject seeded invalid locks before adapters run, and observe zero classifier calls by default.

Against the real next-best alternative—host-native checked-in profiles plus ordinary config
layering—the win can only come from a stronger compile-time contract and explanation. Automatic
classification is extra maintenance and debugging cost unless a specific product opts in and
measures it. The existing crossover study correctly found no universal win
(`docs/benchmarks/harness-contextual-crossover/witness.json`).

### P1–P4 check

| Problem | Audit result |
|---|---|
| P1 managed context | Typed instruction/tool/memory assets can bound what enters a product, but only after all selected layers flow through one compiler. Classifier evidence must not become resident production context by default. |
| P2 net-true efficiency | Deterministic lock reuse can remove repeated setup. Classification has only a microbenchmark and authored-corpus result; neither proves lower operator work than explicit profiles. |
| P3 bounded adaptation | Explicit selectors plus fail-closed composition are bounded. Token/path inference is not a safe adaptation authority; keep it advisory and opt-in. |
| P4 integrated operations | A versioned lock, provenance, validation, and launch receipt fit deployment and audit operations. Separate phase schemas and a generated lock that is not compiler-derived leave the operational chain incomplete. |

## What exists now

Line counts below are a complexity signal, not a quality metric. They count the named
implementation, CLI, docs, tests, and corpus paths at the audited trunk tip.

| Area | Approx. lines | Current role | Production disposition |
|---|---:|---|---|
| Classification | 592 | Closed three-domain token/extension scorer, scoped remembered choices, abstention, authored corpus | **Optional authoring/validation.** Keep deterministic and inspectable; no default production invocation and no ontology expansion without a product contract. |
| Selection | 465 | Precedence layering for person/company/project/domain/task profiles; can call classification when tags are absent | **Compiler input phase.** Explicit tags/layers are core; inferred tags must move behind an explicit opt-in boundary. |
| Typed composition | 620 | Per-kind add/replace/remove checks for instructions, tools, memory, policy, routes, secrets, workflows, and UI | **Production core.** Preserve and strengthen as the asset merge phase of one compiler. |
| Dependency resolution | 816 | Component roots, capability providers, version ranges, conflicts, cycles, compatibility, budgets, evidence, digests, explanations | **Production core.** Preserve and integrate with composition and lock validation. |
| Risk preview | 706 | Diffs current and candidate locks and asks on novelty/conflict/widening | **Optional control-plane UX.** Useful before deployment or interactively; not a prerequisite in unattended production. |

The classifier is therefore not the largest subsystem, but it is over-prominent relative to
its evidence. Its implementation hard-codes three domains and English token/extension weights
in `internal/harnessclassify/harnessclassify.go`; its corpus has twelve authored cases in
`internal/harnessclassify/testdata/domain-corpus.json`. The docs accurately disclaim a
population estimate, yet `cmd/fak/harness_select.go` automatically invokes it when explicit
tags are absent and task/domain inputs are present. That wiring makes a development hypothesis
look like a product-selection layer.

## Production boundary

Use this lifecycle split until #6938 replaces it with a versioned public contract.

| Surface | Lifecycle | Allowed authority |
|---|---|---|
| `harness discover` | Development/authoring | Suggest candidate local inputs; never grant capabilities or launch. |
| `harness classify` | Experimental authoring/validation | Suggest a domain with evidence or abstain. Only an explicit product opt-in may consume the suggestion. |
| `harness preview` | Validation/control-plane UX | Explain a candidate lock change and collect approval where an operator exists. Headless deployment uses policy, not prompts. |
| Explicit profile/layer selection | Production compiler input | State operator/product intent. Inputs must be schema-checked and provenance-bearing. |
| Typed asset composition | Production compiler phase | Merge by asset-specific rules; reject widening, cross-boundary memory, illegal removal, and ambiguity. |
| Component dependency resolution | Production compiler phase | Solve providers/ranges/conflicts/compatibility/budgets deterministically and explain every inclusion. |
| Immutable lock validation | Production gate | Verify schema, environment, provenance, and referenced content before any adapter or model starts. |
| Lock launch | Production runtime | Execute exactly the validated lock; perform no discovery, inference, or implicit recomposition. |

A product may deliberately package a classifier—for example, an IDE that asks whether a mixed
repository task is documentation or code—but that is product policy above the compiler. It is
not a universal fak default and cannot widen the lock's capability floor.

## Overbuilt or prematurely productized

1. **Closed-domain inference is ahead of a stable selector contract.** `legal`, `coding`, and
   `integrated` are demonstration categories, not a versioned extensible vocabulary. Adding more
   weights, domains, model inference, or learning would deepen the wrong abstraction.
2. **Remembered classifier choices are polished before lock consumption is unified.** Scope,
   TTL, reason, and reversible files are sensible validation mechanics, but they are secondary
   to making explicit lock compilation the sole production path.
3. **Preview UX is broader than the deployment seam it previews.** The risk-only interaction is
   useful, but its value depends on a canonical candidate lock. It should not create another
   selection/composition path.
4. **Three public-looking phase commands expose implementation decomposition.** `select`,
   `compose`, and `resolve` are useful test seams. Treating each schema as an independent product
   contract multiplies migrations and permits information loss between phases.

These are not recommendations to delete the code. They are recommendations to freeze their
scope, mark lifecycle honestly, and spend the next work on the production spine.

## Underspecified or disconnected production areas

| Gap | Current evidence | Required contract |
|---|---|---|
| One compiler and schema chain | `internal/harnessselect`, `internal/harnesscompose`, and `internal/harnessresolve` define separate manifests/results; the CLI requires users to move data between phases. | One canonical input model, explicit phase mapping, unknown-field rejection throughout, and one versioned lock output. |
| Launch is not lock-derived end to end | `internal/harnessinit` generates and consumes its own `harness.lock.json`; it does not invoke the resolver/composer that produce `fak.harness-product-lock/v1alpha1`. | Generated and first-party launch must validate and execute the compiler lock, not a parallel lock shape. |
| Lock validation and freshness | Resolution computes an ID from selected content, but launch-side validation of schema, digest references, environment, and stale inputs is not the shared gate. | A reusable validator must fail before adapters execute on schema drift, digest drift, incompatible environment, missing provenance, or unknown fields. |
| Selector-to-asset/component mapping | Selection returns ordered profile fragments while composition and resolution expect different layer/component structures. | Specify lossless translation: selected layer identity, source, precedence, assets, roots, constraints, and provenance survive into the lock. |
| Compatibility evolution | The resolver accepts a narrow contract string/range syntax, while issue #6805 remains open for semantic negotiation and upgrades. | Version negotiation, diagnostics, migration, and rollback must be explicit before calling the lock format stable. |
| External conformance | Internal tests witness individual phases; issue #6793 remains open. | A clean-room fixture and certificate must exercise compile, validate, launch, explain, and seeded failures through public surfaces. |
| Subtractive absence | Composition rejects removal of locked/mandatory assets, but issue #6791 remains open for witnessed capability absence. | A production lock must prove both what is present and what selected profiles intentionally exclude. |

Several originally requested resolver mechanics are already present and should not be
reimplemented: missing/ambiguous providers, basic version constraints, dependency cycles,
component conflicts, compatibility checks, resource budgets, evidence requirements, sorted
output, content digests, and typed asset safety checks all have code and tests. The gap is the
cross-phase contract and launch authority, not another solver rewrite.

## Sequence

1. **Freeze classifier and preview fan-out.** Documentation and CLI help should call both
   optional; no new domain vocabulary or inference backend until a product-specific need has an
   external corpus and net-true comparison.
2. **Land #6938.** Collapse explicit selection → typed composition → dependency resolution →
   lock validation into one canonical compile/launch spine, preserving individual packages as
   internal testable phases where useful.
3. **Close production contract dependencies.** Drive #6791 (subtractive absence), #6793
   (external conformance), and #6805 (compatibility negotiation) against that same lock rather
   than parallel formats.
4. **Only then evaluate optional classifiers.** Require product-owned labels, corpus,
   false-selection/abstention costs, and an opt-in deployment decision. A negative result should
   leave explicit profiles as the default.

## Decision rules

- An explicit manifest/profile selection always beats discovery or classification.
- Missing explicit selection uses a declared safe default or fails with a recovery action; it
  does not silently guess a domain.
- Classification output is advisory data, never a capability grant.
- Production runtime consumes a validated immutable lock and does no merging or inference.
- Every effective asset and component has source, reason, version/digest where applicable, and
  a path back to the selected layer.
- A production claim requires a clean-room compile-and-launch witness, not only package tests or
  an authored classification corpus.

## Evidence inspected

- `cmd/fak/harness_select.go`, `cmd/fak/harness_classify.go`, and `cmd/fak/harness_stack.go`
- `internal/harnessselect`, `internal/harnessclassify`, `internal/harnesscompose`,
  `internal/harnessresolve`, `internal/harnesspreview`, and `internal/harnessinit`
- `docs/harness-init.md` and `docs/integrations/harness-{classification,composition,resolution,preview,lock-launch}.md`
- Issues #6777, #6791, #6792, #6793, #6805, #6900, #6902, #6903, and #6904
- Commits `da5a97e1f2`, `09bfc954bf`, `a86eaf050e`, and `d87991c55d`

This is a source/contract audit, not a runtime performance claim. It intentionally does not
promote classifier microbenchmarks or twelve authored cases into production-readiness evidence.
