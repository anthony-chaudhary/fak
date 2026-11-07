# Wiring audit — pkg/gpubinding (lane `gpubinding`)

## 2026-11-06 update: the seam is now the implementation home

The GPU binding doctor moved OUT of `internal/compute` INTO this package. The
seam is self-contained: stdlib plus `github.com/anthony-chaudhary/fak/pkg/sysproc`
only, no `internal/*` imports, so a companion repository can import
`github.com/anthony-chaudhary/fak/pkg/gpubinding` without pulling private code.

| Check | Result |
|---|---|
| Imports | stdlib + `pkg/sysproc` only (`rg -n "anthony-chaudhary/fak/internal" pkg/gpubinding/*.go` → no matches) |
| Consumer inside this repo | `internal/compute` keeps the historical `compute.*` names as thin aliases/wrappers into this package (`cuda_gpu_binding.go`, `cuda_gpu_binding_topology.go`, `cuda_gpu_binding_package.go`); the live call chain is still `cmd/fak-dev` -> `internal/devcmd` -> `internal/compute` -> **this package** |
| Tests | The doctor's full suite lives here (`binding_test.go`, `topology_test.go`); `internal/compute/cuda_gpu_binding_delegation_test.go` pins the alias identity and legacy entry points |
| Defect cured | The transitive `internal/compute` dependency made the "public seam" unimportable by companion repos; that dependency no longer exists |

The 2026-11-03 audit below is kept as history: its ABSENT-with-reason verdict
described the pre-move state where this package was an empty mirror of logic
that lived in `internal/compute`.

## 2026-11-03 audit (history): ABSENT-with-reason

Read-only audit required by flow ticket `debtwave2-pkg-gpubinding` (work item 2):
establish whether a production call path actually invokes this package
(presence != invokability). Audited 2026-11-03 at base commit `6acf4518a1`.

### Commands and results

| Check | Command | Result |
|---|---|---|
| All Go references | `rg -n "gpubinding" --glob '*.go'` | Only the package itself and its two `*_test.go` files |
| Non-test importers | `rg -n "pkg/gpubinding" --glob '*.go' -g '!*_test.go'` | **No matches** (also none under the full module path `github.com/anthony-chaudhary/fak/pkg/gpubinding`) |
| Verb/registry surface | `rg -n "gpubinding\|DiagnoseGPUBinding\|BlackwellFloor\|VerdictReady" cmd/ --glob '*.go'` | No match in `cmd/fak` (production binary), `main.go`, or `verbs_gen.go` |
| Upstream logic home | `rg -n "DiagnoseGPUBinding\|NewCUDAPCIeTopologyDiscovery" --glob '*.go' -g '!*_test.go' internal/` | Owned by `internal/compute` (`cuda_gpu_binding.go`, `cuda_gpudirect_topology.go`); sole caller chain is `cmd/fak-dev` -> `internal/devcmd` (`cuda_gpudirect_binding.go`) -> `internal/compute` **directly**, bypassing this seam |
| Lane declaration | `rg -n "gpubinding" dos.toml` | `dos.toml:772: gpubinding = ["pkg/gpubinding/**"]` |

### Verdict: ABSENT-with-reason

No production call path in this repository invoked `pkg/gpubinding`; the debt-lanes
reader's `integrated=false` was confirmed by the import graph. The one live consumer of
the underlying doctor was the dev-only tool `fak-dev cuda-gpudirect binding`, which
called `internal/compute` directly. Inside this repo the package was consumed only by
its own tests; its purpose was the public seam for companion repositories (per the
package doc), which is outside this lane's tree.

### Why no wiring landed

A minimal production connection would fit in 2 files (a new `cmd/fak/compute_cudabinding.go`
subverb plus one dispatch case in `cmd/fak/compute_amddirect.go`), but the ticket's hard
boundary restricted edits to `pkg/gpubinding/`. Per work item 2, the audit evidence landed
instead of forced wiring. If the operator later wants the seam wired into `fak compute`,
the 2-file change above is the shape to take (separate ticket, wider boundary).
