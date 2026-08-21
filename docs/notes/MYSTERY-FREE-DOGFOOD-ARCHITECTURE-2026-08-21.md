---
title: "Mystery-free dogfood: repository architecture"
description: "A captured use of fak's five-field learning method against the live architecture graph, including the counterfactual and defects it exposed."
---

# Mystery-free dogfood: repository architecture — 2026-08-21

## Result

The five-field method found a real simplification candidate in fak's live architecture:
`learningdebt` imports `windowgate` directly even though `windowgate` is already reachable
through `dogfoodissues`. The method was useful, but not fully mystery-free: the architecture
report identifies the package edge, not the source file/import that owns it. That product
defect is filed as [#8452](https://github.com/anthony-chaudhary/fak/issues/8452).

This is a dogfood readout, not a claim that the direct edge should be removed blindly.
Redundant reachability means “candidate for transitive reduction,” not “behaviorally safe to
delete.” The public API and dependency direction still need review before a code change.

## Live question

Can a maintainer use the atlas to understand one real architecture result and name the
smallest controlled adjustment without guessing?

The selected leaf was `internal/learningdebt`: it is the subsystem that turns learning
scorecard defects into backlog work, so it is directly relevant to the learning surface
being dogfooded.

## Five-field run

### Outcome

Run from the repository root:

```powershell
fak architecture --leaf learningdebt
```

Observed live result:

```text
architecture: 721 leaves, 0 upward violation(s), max tier distance 0
learningdebt declared=composer(4) floor=composer(4) gap=0
  deps=[dogfoodissues windowgate] dependency-reach=5 dependency-depth=2 blast-radius=0
  redundant dependency edges:
    windowgate alternate=learningdebt -> dogfoodissues -> windowgate
```

The leaf is tier-correct and has no upward violation. The narrower observed outcome is one
transitive-reduction candidate: the direct `learningdebt -> windowgate` edge adds no new
reachability because an alternate path already exists.

### Decider

The deciding evidence is the `redundant dependency edges` row. The architecture analyzer
removes the tested direct edge and searches for another deterministic path to the same
destination. Here that path is:

```text
learningdebt -> dogfoodissues -> windowgate
```

The authoritative implementation is `redundantDependencies` in
[`internal/archreport/report.go`](../../internal/archreport/report.go). This is not an
upward-tier failure; the report explicitly says there are zero upward violations.

### Owned input

The direct package edge comes from this import in
[`internal/learningdebt/learningdebt.go`](../../internal/learningdebt/learningdebt.go):

```go
"github.com/anthony-chaudhary/fak/internal/windowgate"
```

and its use at `windowgate.ConfigureBackgroundCommand(cmd)`. The alternate path exists
because [`internal/dogfoodissues`](../../internal/dogfoodissues) also imports `windowgate`.
The owned input is therefore the direct import/API call, not the tier declaration and not
an unrelated architecture threshold.

Finding that source fact required a separate repository search. The report itself did not
name it; [#8452](https://github.com/anthony-chaudhary/fak/issues/8452) tracks deterministic
source provenance for reported edges.

### Counterfactual

The atlas's architecture row asks for an isolated source change, but this edge is a useful
case where “one variable” still has a behavioral precondition: removing an import also
removes the `ConfigureBackgroundCommand` call. The predicted graph result is that deleting
the direct import would remove the redundant-edge row while preserving the alternate
package path. The safe behavioral experiment first needs a `dogfoodissues`-owned seam and
its affected tests, so it is filed rather than performed inside this docs-only dogfood run.

A separate, immediately runnable counterfactual exercised the same one-input discipline
against this spine's live fan-out planner. It is read-only: ask the fan-out planner for`r`nthe same spine with
its area filter narrowed to only `dogfood`:

```powershell
fak-dev issue fanout --title "Mystery-free learning and adjustment" `
  --leaf docs --spine 20fb142ebd --parent "#7642" --parent-issue 7642 `
  --parent-baseline-points 100 `
  --paths "LEARNING-PATH.md,docs/fak/mystery-free-adjustment-atlas.md,internal/adjudicator" `
  --areas dogfood --max 3 --completion-standard integrated `
  --target-envelope "New learner can explain and safely adjust core fak processes from repository docs" `
  --witnessed-envelope "Offline PowerShell policy lab and cross-process atlas" --json
```

Prediction: narrowing one input, `--areas`, removes the QA candidates and leaves only the
two dogfood candidates. Observed result: the command refused because two candidates are
below the fan-out floor of three and told the operator to widen or drop the filter. The
changed outcome was caused by the area filter; the refusal was caused by the independent
minimum-fan-out invariant. No GitHub issue was created because `--live` was absent.


### Source

- Method: [`docs/fak/mystery-free-adjustment-atlas.md`](../fak/mystery-free-adjustment-atlas.md)
- Architecture graph and redundant-edge algorithm:
  [`internal/archreport/report.go`](../../internal/archreport/report.go)
- Direct edge: [`internal/learningdebt/learningdebt.go`](../../internal/learningdebt/learningdebt.go)
- Alternate edge: [`internal/dogfoodissues/dogfoodissues.go`](../../internal/dogfoodissues/dogfoodissues.go)
- Original redundant-edge capability: [#6369](https://github.com/anthony-chaudhary/fak/issues/6369)

## Teach-back

In plain language: `learningdebt` directly depends on `windowgate`, but it can already
reach that package through `dogfoodissues`. The graph therefore calls the direct edge
redundant. That does **not** prove the import can be deleted, because reachability and API
ownership are different questions. The next safe move is to decide whether
`dogfoodissues` should own the background-command configuration seam, then rerun the same
leaf report and affected tests.

## Defects and disposition

| Observation | Disposition |
|---|---|
| The report did not identify the source file/import for its direct edge. | Filed [#8452](https://github.com/anthony-chaudhary/fak/issues/8452). |
| The atlas said the report names the concrete import edge, which could imply source-level provenance. | Corrected in the atlas alongside this readout: the report names package edges; use source provenance when present, otherwise search the named package. |
| Redundant graph reachability is easy to misread as permission to delete behavior. | Added the reachability-vs-API warning to the architecture atlas row and this teach-back. |

No other defect surfaced in this bounded run. QA coverage remains tracked separately in
#8448, #8449, and #8450.
