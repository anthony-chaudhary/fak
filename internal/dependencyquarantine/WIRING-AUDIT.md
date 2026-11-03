# Wiring audit — debtwave2-internal-dependencyquarantine

Read-only production-wiring audit required by flow ticket
`debtwave2-internal-dependencyquarantine` (work item 2), run 2026-11-03 from the
ticket worktree at base commit `6acf4518a1d178d868798a007e552e41240b98d6`.

## Verdict: ABSENT (disconnected — never invoked by a production call path)

`Check(root)` / `NestedModules(root)` are linked into no production binary or verb.
Presence of the package (and its tests) is not invokability; the import scan proves
no caller exists.

## Evidence (re-run these to reproduce)

```sh
# 1. Import scan, all Go files (test and non-test): 0 hits
rg -n 'github.com/anthony-chaudhary/fak/internal/dependencyquarantine' --glob '*.go'

# 2. Every other mention in the tree (all declarative or test-surface, none invocable):
rg -n 'dependencyquarantine' dos.toml .github/ internal/architest/
```

| Hit | Nature |
|---|---|
| `dos.toml:446` | leaf-name vocabulary list (declarative) |
| `dos.toml:1499` | `dependencyquarantine = ["internal/dependencyquarantine/**"]` lane-tree mapping (declarative) |
| `.github/workflows/release-artifacts.yml:685` | `go test ./internal/dependencyquarantine` — release-CI **test** gate, not a production call path |
| `internal/architest/architest_test.go:598` | architest budget-table entry: `"dependencyquarantine": 0 // #5947: ... imports no sibling leaf` — a test fixture asserting zero internal imports |

The debt-lanes reader agrees: `fak debt-lanes --lane dependencyquarantine` reports
`integrated: false` (unreachable from the `cmd/` import graph via
`internal/debtlane/scan.go:scanReachableFromCmd`).

## Why no wiring was forced

The ticket boundary restricts this lane to edits under `internal/dependencyquarantine/`.
Any real production edge requires editing a sibling package, so the minimal connection
does not fit the allowed surface; per work item 2 the finding is recorded ABSENT-with-reason
instead of forcing new wiring.

Smallest plausible seam for a future, separately-scoped ticket: `cmd/fak-dev`
`ci-preflight` (`case "ci-preflight"` in `cmd/fak-dev/main.go`, "check committed
repository tip in isolation") — one file, would call `Check(root)` on the isolated tip.
Runner-up: standalone `fak validate` (`cmd/fak-validate/main.go`).

## Next checkable step

A follow-on ticket may wire `Check` into `fak-dev ci-preflight` (1-file seam above);
that flips `integrated: true`, unclamps maturity past the tested ceiling (5.0), and
retires the remaining 3.0-point gap on this lane.
