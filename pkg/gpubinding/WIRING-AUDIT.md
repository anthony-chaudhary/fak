# Wiring audit — pkg/gpubinding (lane `gpubinding`)

Read-only audit required by flow ticket `debtwave2-pkg-gpubinding` (work item 2):
establish whether a production call path actually invokes this package
(presence != invokability). Audited 2026-11-03 at base commit `6acf4518a1`.

## Commands and results

| Check | Command | Result |
|---|---|---|
| All Go references | `rg -n "gpubinding" --glob '*.go'` | Only the package itself and its two `*_test.go` files |
| Non-test importers | `rg -n "pkg/gpubinding" --glob '*.go' -g '!*_test.go'` | **No matches** (also none under the full module path `github.com/anthony-chaudhary/fak/pkg/gpubinding`) |
| Verb/registry surface | `rg -n "gpubinding\|DiagnoseGPUBinding\|BlackwellFloor\|VerdictReady" cmd/ --glob '*.go'` | No match in `cmd/fak` (production binary), `main.go`, or `verbs_gen.go` |
| Upstream logic home | `rg -n "DiagnoseGPUBinding\|NewCUDAPCIeTopologyDiscovery" --glob '*.go' -g '!*_test.go' internal/` | Owned by `internal/compute` (`cuda_gpu_binding.go`, `cuda_gpudirect_topology.go`); sole caller chain is `cmd/fak-dev` -> `internal/devcmd` (`cuda_gpudirect_binding.go`) -> `internal/compute` **directly**, bypassing this seam |
| Lane declaration | `rg -n "gpubinding" dos.toml` | `dos.toml:772: gpubinding = ["pkg/gpubinding/**"]` |

## Verdict: ABSENT-with-reason

No production call path in this repository invokes `pkg/gpubinding`; the debt-lanes
reader's `integrated=false` is confirmed by the import graph. The one live consumer of
the underlying doctor is the dev-only tool `fak-dev cuda-gpudirect binding`, which
calls `internal/compute` directly. Inside this repo the package is consumed only by
its own tests; its purpose is the public seam for companion repositories (per the
package doc), which is outside this lane's tree.

## Why no wiring landed

A minimal production connection would fit in 2 files (a new `cmd/fak/compute_cudabinding.go`
subverb plus one dispatch case in `cmd/fak/compute_amddirect.go`), but the ticket's hard
boundary restricts edits to `pkg/gpubinding/`. Per work item 2, the audit evidence lands
instead of forced wiring. If the operator later wants the seam wired into `fak compute`,
the 2-file change above is the shape to take (separate ticket, wider boundary).
