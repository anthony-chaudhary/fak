# Generation Assumption Ledger

This ledger closes #1643. It gives each generation stream an explicit assumption,
confidence, owner, review date, promotion evidence, and demotion or retirement
path so later agents can continue without rereading the parent epic.

Generation remains orthogonal to priority, shared trunk, and runtime feature
gates. Priority still answers urgency. Shared trunk still means every stream
lands on `main` by explicit path with the same witness rules. Runtime gates still
decide exposure. This ledger only records which horizon owns an assumption and
what evidence can move it.

## Ledger

| Stream | Assumption | Confidence | Owner | Review Date | Promotion Evidence | Demotion / Retirement Evidence |
|---|---|---:|---|---|---|---|
| `gen/now` | Current-product work has a direct witness and no dependency on an unproven generation mechanism. | High | issue resolver | 2026-07-14 | Focused test, captured command output, or witnessed commit proves the issue improves an active operator/product path. | Witness regresses, code is only reachable behind a future gate, or the issue no longer maps to a current path. |
| `gen/next` | Near-term foundation can become agent-runnable after a gate, schema, handoff, or dogfood run lands. | Medium | generation gardener | 2026-07-21 | A worker or operator uses the artifact through `fak index`, dispatch, handoff, milestone, or release tooling with a passing focused witness. | No dogfood caller appears, the gate remains unspecified, or repeat workers cannot produce a committed artifact. |
| `gen/second-next` | The architecture option is worth carrying because a simulation, compatibility policy, or dependency edge could promote it. | Medium-low | architecture owner | 2026-08-04 | A simulation, compatibility rule, or dependency graph shows the option can be decomposed into `gen/next` foundation work. | Compatibility fails, a simpler shipped path supersedes it, or the dependency cost exceeds expected value. |
| `gen/future` | Long-horizon research or narrative preserves option value without pretending to be on the current release train. | Low | research owner | 2026-08-18 | The memo changes a roadmap decision, scorecard, market narrative, benchmark target, or second-next option. | The market/standards premise fails, evidence is stale, or the note has no owner, recheck date, or decision it can influence. |
| `unclassified` | The issue lacks enough evidence to assign a horizon safely. | Unknown | triage owner | 2026-07-07 | Labels, milestone, issue body, and witness expectation are updated to exactly one generation stream. | If classification cannot be made cheaply, keep `needs-triage` and do not dispatch as generation work. |

## Issue Body Contract

Generation issues should carry these fields in the body or closeout comment:

- generation stream and matching `gen/*` label;
- matching Generation G* milestone;
- assumption being tested;
- confidence and owner;
- review date;
- promotion evidence required;
- demotion or retirement evidence;
- at least one invalidating assumption.

## Promotion Rules

Promotion is evidence-driven, not label-driven:

- `future -> second-next`: the research note names a concrete option, dependency,
  or compatibility question.
- `second-next -> next`: simulation, policy, or dependency evidence makes the
  option agent-runnable with a bounded surface.
- `next -> now`: a gate, schema, handoff, dogfood run, or default-exposure proof
  lands with a focused witness.
- `unclassified -> any stream`: labels, milestone, and issue body agree.

Demotion or retirement uses the same ledger. If an assumption fails, update the
issue with the evidence and move it farther from `now`, park it, or retire it.
Do not silently change the label.

## Invalidating Assumption

This ledger assumes owners and review dates stay visible in issue bodies,
operator notes, or closeout comments. If those fields are not maintained, the
ledger becomes decorative and should be replaced by a machine-checked issue
field or `fak index generation` extension before it is used for dispatch.

