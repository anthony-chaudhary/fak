# Architecture report dogfood — 2026-08-09

## Verdict

`fak architecture` is not yet usable against the repository state that ships it.
Both the full graph and a scoped `--leaf archreport` query stop at one unrelated stale
tier declaration instead of returning the healthy portion of the graph.

## Reproduction

Source snapshot: committed `origin/main@7a746b82c4aa` exported with `git archive` to a
clean temporary directory. A tiny Go caller inside that exported module invoked the
production `internal/archreport.Analyze` seam, avoiding the peer-dirty checkout.

Full report:

```text
Analyze(".", "")
panic: inspect declared leaf "docrender": declared package directory internal\docrender does not exist; create the package or remove its stale tier declaration
```

Scoped report:

```text
Analyze(".", "archreport")
panic: inspect declared leaf "docrender": declared package directory internal\docrender does not exist; create the package or remove its stale tier declaration
```

The snapshot contains a `"docrender": 2` entry in
`internal/architest/architest_test.go`, but no `internal/docrender` tree.

## Defect ledger

- #6084 — `archreport-dogfood-defect: stale-declaration-global-outage`: represent stale
  declarations as diagnostics so one bad row cannot suppress all healthy full/scoped output.

No other defects are claimed: this first failure prevents the report from reaching later
leaves, so later graph quality remains unobserved until #6084 lands.

## Post-fix candidate readout (#6084)

The #6084 candidate was overlaid onto a fresh `origin/main` archive and the same production
`Analyze` seam completed instead of aborting:

```text
schema:       fak-architecture/1
tiers:       6
healthy leaves: 552
diagnostics: 1
upward violations: 0
top hotspot: abi (direct fan-in 78)
```

The one diagnostic is `stale-tier-declaration` for `docrender`, with recovery
`create the package or remove its stale tier declaration`. This confirms the report can now
expose the defect and the rest of the committed architecture graph simultaneously. Final
ship evidence remains the origin-backed #6084 commit, not this uncommitted candidate run.
