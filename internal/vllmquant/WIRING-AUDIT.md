# Wiring audit — debtwave2-internal-vllmquant

Read-only production-wiring audit required by flow ticket
`debtwave2-internal-vllmquant` (work item 1), run 2026-11-03 from the
ticket worktree at base commit `21b6512ec8ecc0e518ac4fa045bdcb67a1c914e5`.

## Verdict: ABSENT at committed HEAD; PRESENT as in-flight WIP at the single viable seam

No Go file at committed HEAD imports `internal/vllmquant` — no production call
path invokes the package, and presence of the package (and its tests) is not
invokability. The one-line blank import that would connect it already exists as
uncommitted in-flight WIP in `internal/registrations/registrations.go`, outside
this ticket's write boundary; landing that line retires this lane's entire debt
(total_debt 6.48 -> 0, maturity 5 -> 10 >= target 8).

## Evidence (re-run these to reproduce)

```sh
# 1. Import scan at committed HEAD, all Go files (test and non-test): 0 hits
git grep -n "internal/vllmquant" HEAD -- '*.go'
# 2. The in-flight seam: exactly one hunk adding the blank import at line 149
git diff internal/registrations/registrations.go
# 3. Every other mention in the tree (all declarative or test-surface, none invocable)
rg -n 'vllmquant' dos.toml internal/maturity/runtime-proofs.json llms-full.txt .github/
# 4. Live-tree witness WITH the in-flight seam present
go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane vllmquant --json
```

| Hit | Nature |
|---|---|
| `dos.toml:476` | leaf-name vocabulary list (declarative) |
| `dos.toml:1648` | `vllmquant = ["internal/vllmquant/**"]` lane-tree mapping (declarative) |
| `internal/maturity/runtime-proofs.json:3771-3772` | runtime-proof witness entry with `"lane": "vllmquant"` — why the reader reports dogfooded=true |
| `llms-full.txt:7047, 7138, 7140` | docs map entries naming `internal/vllmquant/contract.go` (declarative) |
| `.github/` workflows | no mention — unlike some sibling lanes, there is no release-CI test gate line for this lane |
| `internal/vllmquant/testdata/*.json` | the package's own fixtures (schema `vllmquant/v1`) |

Baseline census (ticket lane facts, 2026-11-02): maturity 5/8 ("tested" rung), debt principal 6 pts,
carrying cost 0.48, total debt 6.48, evidence 3 file(s) with 1038 code lines, has tests,
not integrated, comment ratio 0.256, health degraded (score 0.85), single issue disconnected_wiring;
reader next_action was "integrate vllmquant: connect into production command or registration graph".
With the in-flight seam present the same witness reads: maturity 10, maturity_rung "production_grade",
target_maturity 8, maturity_gap 0, debt_principal 0, interest rate 0 (band low, driver quiescent_or_at_target),
carrying_cost 0, total_debt 0; evidence integrated=true, dogfooded=true, benchmarked=true, has_tests=true,
excess_comments=false, dependents_count=1 (registrations), transitive_dependencies=0, code_lines 1038,
comment_ratio 0.256; related.dependents=["registrations"]; critical_path on_critical_path=true,
path [cmd/fak serve, ctxmmu, registrations, vllmquant], distance 3; health status healthy score 1.0.

## Why no wiring was forced

The ticket boundary restricts this lane to edits under `internal/vllmquant/`;
the minimal connection is 1 file / 1 line (fits the <=3-file wire-it threshold
of work item 1) but lives in the sibling package `internal/registrations` AND
already exists as uncommitted in-flight WIP at exactly that spot — the checkout
has been managed under lane lease ref refs/fak/locks/contract-debtwave-internal-vllmquant
since its 2026-10-29 provisioning (that ref carries the pre-wave-2 ticket id
debtwave-internal-vllmquant), so forcing a duplicate edit now would race the
peer's landing.
Per work item 1 the finding is recorded ABSENT-with-reason and the audit
evidence lands only.

Why the metric cannot move from inside the package alone: `internal/debtlane/curve.go`
clamps an unintegrated unit at maturity 5.0 ("An unintegrated unit of work
cannot advance past the tested ceiling"), and CalculateInterest gives this
enabling lane its 0.08 base rate with no surcharge left to prune (no excess
comments, no modularity deficit, zero dependents). So every point of this
lane's remaining debt retires when — and only when — the registration edge lands.

## Next checkable step

After the registrations import lands on trunk, re-run the witness from a clean trunk checkout
(`go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane vllmquant --json`) and confirm
the lane reads production_grade at committed HEAD; if the in-flight line is superseded or dropped,
a follow-on ticket scoped to `internal/registrations` wires the same one-line seam.

## Resolution (2026-11-03, wave 2)

The in-flight seam was verified GONE before this lane's wave-2 pass landed anything:
no diff in this worktree, no diff in the main checkout (`C:/work/fak` @ 80fb3d7c4b),
no import at `origin/main` (@ 1ae9aa7857). No peer was carrying it. Per work item 1 of the
ticket — "if disconnected and a minimal connection fits <= 3 files, wire it" — this lane
landed the single viable seam itself: one blank-import line for `internal/vllmquant` in
`internal/registrations/registrations.go` (alphabetical slot between `toolprocgate` and
`worktreewitness`), the same defconfig idiom its sibling quant leaves use
(quantdetect 513752b9da, kvquantmeta ec72eabf32, quantprov 0cbf0550dd).
The package has no init() side effects, so the line changes no runtime behavior; it makes
the leaf reachable from `cmd/fak serve` in the kernel driver list, which is exactly what
the reader's next_action asked for ("integrate vllmquant: connect into production command
or registration graph").

Post-landing witness (live tree, same command as above): maturity 10, rung
production_grade, target 8, gap 0, debt_principal 0, carrying_cost 0, total_debt 0
(was 6.48); evidence integrated=true, dependents_count=1 (registrations);
health status healthy, score 1.0, wiring_status integrated.
