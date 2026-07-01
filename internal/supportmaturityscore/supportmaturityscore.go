// Package supportmaturityscore grades the covmatrix support rungs as a scorecard.
//
// growth_debt is intentionally narrow: only silently undefined cells are hard debt.
// This card answers "how much of the declared model x backend grid is below the rung its
// regime honestly expects?" Each cell's CURRENT rung is read from the closed M0-M7 ladder
// (internal/supportmaturity.FromSupport); its TARGET rung from the committed declared-target
// table here (#1247, C4 of epic #1243). support_maturity_debt = sum(target - current) over
// cells below their target -- so an honestly-FENCED accelerated cell whose declared target
// IS the fence contributes 0, not debt. That is the anti-Goodhart fence: the score names
// exactly the cells whose regime expects more, instead of demanding SUPPORTED everywhere.
package supportmaturityscore

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	// Schema is the control-pane schema id for this scorecard.
	Schema = "fak-support-maturity-scorecard/1"
	// DebtKey is the headline integer the control pane folds.
	DebtKey = "support_maturity_debt"
)

// cpuBackend is the topology-aware scalar reference backend; every other backend is an
// accelerated hot-path copy (covmatrix.accelerated).
const cpuBackend = "cpu"

// TargetRule is one committed row of the declared-target table: the rung a cell's REGIME is
// honestly expected to reach. Rules are evaluated in order and the FIRST whose Match fires
// sets the cell's target, so the table reads top-to-bottom as "most-specific regime first".
// Targets are capped at M4Correct on purpose: M5-M7 (optimized / SOTA-parity / beyond-SOTA)
// are witnessed by benches and the parity tracks, not by the covmatrix present-and-honest
// axis this card folds -- demanding them here would be the "SOTA everywhere" Goodhart the
// fence exists to prevent (epic #1243; those rungs are the C5/C7+ children's job).
type TargetRule struct {
	// Name is the regime label shown in the corpus / rationale.
	Name string
	// Target is the rung this regime is honestly expected to reach.
	Target supportmaturity.Rung
	// Why documents, in one line, why this regime's honest ceiling is Target.
	Why string
	// Match reports whether cell c belongs to this regime. It decides from facts already on
	// the Cell (Backend + Topology), so the table cannot drift from the grid it scores.
	Match func(c covmatrix.Cell) bool `json:"-"`
}

// DeclaredTargets is the committed declared-target table (#1247, C4), read back per cell by
// declaredTarget. The last rule matches every cell, so declaredTarget is total -- no cell is
// left without a target. This IS the anti-Goodhart fence: the accelerated-fenced regime
// declares M1 (its fence), not M4, so a correctly-fenced cell is zero debt.
var DeclaredTargets = []TargetRule{
	{
		Name:   "accelerated-fenced",
		Target: supportmaturity.M1Fenced,
		Why:    "a non-PreNorm family on an accelerated backend honestly terminates at its fence (requirePreNorm / requireGLMDsaSession); the fence IS the regime ceiling, so un-fencing is not expected -- target M1.",
		Match:  func(c covmatrix.Cell) bool { return c.Backend != cpuBackend && c.Topology != string(covmatrix.PreNorm) },
	},
	{
		Name:   "accelerated-prenorm",
		Target: supportmaturity.M4Correct,
		Why:    "a PreNorm family runs on the accelerated hot path; its regime expects a correct, CI-witnessed result -- target M4.",
		Match:  func(c covmatrix.Cell) bool { return c.Backend != cpuBackend },
	},
	{
		Name:   "cpu-reference",
		Target: supportmaturity.M4Correct,
		Why:    "the cpu path is the topology-aware reference; its regime is correctness, so every cell targets a CI numeric oracle -- target M4.",
		Match:  func(c covmatrix.Cell) bool { return true },
	},
}

// declaredTarget resolves cell c to its declared target rung by reading the committed
// DeclaredTargets table top-to-bottom. The final catch-all rule guarantees a result; the
// fallback floors to M4Correct so a future edit that drops the catch-all fails loud (as
// debt) rather than silently as zero.
func declaredTarget(c covmatrix.Cell) supportmaturity.Rung {
	for _, r := range DeclaredTargets {
		if r.Match(c) {
			return r.Target
		}
	}
	return supportmaturity.M4Correct
}

// rungStepNeed names the witness a cell must earn to climb ONE rung to r -- the per-step
// action behind one unit of support_maturity_debt (epic #1243 Plane A binds a distinct
// witness to each rung). Summed over a cell's missing rungs it IS the (target - current)
// shortfall, which is why Build emits one defect per missing rung.
func rungStepNeed(r supportmaturity.Rung) string {
	switch r {
	case supportmaturity.M1Fenced:
		return "install an honest fence (M0 none -> M1 fenced)"
	case supportmaturity.M2Loads:
		return "pass preflight / load (M1 -> M2 loads)"
	case supportmaturity.M3Runs:
		return "run on the reference proof path (M2 -> M3 runs)"
	case supportmaturity.M4Correct:
		return "add a CI numeric oracle (M3 -> M4 correct)"
	case supportmaturity.M5Optimized:
		return "optimize + commit a bench (M4 -> M5 optimized)"
	case supportmaturity.M6Parity:
		return "reach SOTA-local parity (M5 -> M6 parity)"
	case supportmaturity.M7BeyondSOTA:
		return "beat the SOTA-local baseline (M6 -> M7 beyond-sota)"
	default:
		return "advance one rung"
	}
}

// Build folds the live coverage matrix into a support-maturity scorecard payload. The
// headline support_maturity_debt is sum(target - current) over cells below their declared
// target: each cell's current rung is read from the closed M0-M7 ladder
// (supportmaturity.FromSupport) and its target from the committed DeclaredTargets table.
// One defect is emitted per missing rung-step so the kernel's count fold (Fold sums
// len(Defects)) equals the rung-weighted shortfall, and each defect names the witness that
// one step needs.
func Build() scorecard.Payload {
	cells := covmatrix.Grid()
	counts := countBy(cells)
	supported := counts[covmatrix.Supported]
	total := len(cells)

	var defects []string
	belowTarget := 0
	for _, c := range cells {
		cur := supportmaturity.FromSupport(c.Support)
		tgt := declaredTarget(c)
		if !cur.Less(tgt) {
			continue // at or above its declared target: contributes 0 (the anti-Goodhart fence)
		}
		belowTarget++
		for r := cur + 1; r <= tgt; r++ {
			defects = append(defects, fmt.Sprintf("%s x %s: %s(%s) -> target %s(%s): %s",
				c.Family, c.Backend, cur, cur.Label(), tgt, tgt.Label(), rungStepNeed(r)))
		}
	}
	atOrAbove := total - belowTarget

	kpi := scorecard.KPI{
		Key:   "cells_at_or_above_declared_target",
		Group: "maturity",
		Score: pct(atOrAbove, total),
		Detail: fmt.Sprintf("%d/%d cells at or above declared target; shortfall sum(target-current) = %d rung(s) over %d cell(s) below target",
			atOrAbove, total, len(defects), belowTarget),
		Defects: defects,
	}

	return scorecard.Fold(Schema, []scorecard.KPI{kpi}, DebtKey, nil, scorecard.Messages{
		Finding: fmt.Sprintf("support_maturity_debt %d = sum(target-current) over %d cell(s) whose regime expects a higher rung",
			len(defects), belowTarget),
		FindingClean:    fmt.Sprintf("all %d model x backend cells are at or above their declared target rung", total),
		NextAction:      "retire the shortfall named in each defect (climb the cell to its declared target with the witness that rung binds); a FENCED cell whose target IS the fence is already 0",
		NextActionClean: "hold the line: re-run the support-maturity scorecard on every model/backend change; the declared-target table is the anti-Goodhart fence",
		ExtraCorpus: map[string]any{
			"families":                 len(covmatrix.Families),
			"backends":                 len(covmatrix.Backends),
			"cells":                    total,
			"supported":                supported,
			"proof_path_only":          counts[covmatrix.ProofPathOnly],
			"fenced":                   counts[covmatrix.Fenced],
			"undefined":                counts[covmatrix.Undefined],
			"support_coverage_pct":     scorecard.Round1(pct(supported, total)),
			"target_coverage_pct":      scorecard.Round1(pct(atOrAbove, total)),
			"cells_below_target":       belowTarget,
			"cells_at_or_above_target": atOrAbove,
		},
	})
}

func countBy(cells []covmatrix.Cell) map[covmatrix.Support]int {
	m := map[covmatrix.Support]int{}
	for _, c := range cells {
		m[c.Support]++
	}
	return m
}

func pct(n, total int) float64 {
	if total == 0 {
		return 100
	}
	return 100 * float64(n) / float64(total)
}

// The generated support-maturity matrix view embedded in docs/HARDWARE-MATRIX.md (#1255,
// E6 of epic #1243). The matrix cells are GENERATED from the same covmatrix.Grid() the
// scorecard folds, never hand-typed: `fak support-maturity-scorecard --write-doc` splices
// MatrixBlock() between the two markers, and --check-doc (plus the cmd/fak freshness test)
// reds the trunk when a committed cell drifts from the live grid. That is the "view of the
// instrument, kept honest by the same freshness check as the other scorecards" the epic
// asks for: a stale cell becomes a failing gate, not a silent prose drift.
const (
	// MatrixBegin and MatrixEnd bound the generated block inside docs/HARDWARE-MATRIX.md.
	MatrixBegin = "<!-- BEGIN support-maturity-matrix (generated by `fak support-maturity-scorecard --write-doc`; do not hand-edit) -->"
	MatrixEnd   = "<!-- END support-maturity-matrix -->"
)

// matrixLegend is the prose under the header line; one paragraph defining each rung so the
// generated table reads on its own. ASCII-only, like the rest of the block.
const matrixLegend = "Generated from `internal/covmatrix` (the kernel's own model x backend grid) and " +
	"folded by `internal/supportmaturityscore` -- not hand-typed. Each cell is the support rung " +
	"of one (model family x backend): SUPPORTED (runs with a CI witness on this path), " +
	"PROOF-PATH-ONLY (runs on the cpu scalar path, no CI numeric oracle), FENCED (the accelerated " +
	"path refuses honestly rather than diverge), UNDEFINED (a silently-reachable wrong result -- " +
	"the debt this view exists to catch). The headline support_maturity_debt is the declared-TARGET " +
	"shortfall (#1247): each cell declares the rung its regime honestly expects (a non-PreNorm " +
	"accelerated cell targets its FENCE, not SUPPORTED), and debt = sum(target - current) over cells " +
	"below target -- so an honestly-fenced cell is 0, not debt.\n\n"

// MatrixBlock renders the generated support-maturity matrix block (markers inclusive),
// deterministically from covmatrix.Grid() + the folded scorecard. Columns are derived from
// covmatrix.Backends and rows from Grid()'s deterministic family order, so adding a backend
// or family changes the doc as a diff. The block is ASCII-only so the committed bytes are
// byte-identical on every platform (no em-dash / middot to mojibake on a Windows checkout).
func MatrixBlock() string {
	p := Build()
	c := p.Corpus
	cells := covmatrix.Grid()
	backends := covmatrix.Backends

	byFamily := map[string]map[string]string{}
	topo := map[string]string{}
	var familyOrder []string
	for _, cell := range cells {
		if _, ok := byFamily[cell.Family]; !ok {
			byFamily[cell.Family] = map[string]string{}
			topo[cell.Family] = cell.Topology
			familyOrder = append(familyOrder, cell.Family)
		}
		byFamily[cell.Family][cell.Backend] = string(cell.Support)
	}

	var b strings.Builder
	b.WriteString(MatrixBegin)
	b.WriteString("\n\n")
	fmt.Fprintf(&b, "**Grade %v** - value %v - legacy score %v - support_maturity_debt **%v** (sum(target-current) rungs over %v cell(s) below their declared target) - %v/%v cells SUPPORTED\n\n",
		c["grade"], c["value"], c["score"], c[DebtKey], c["cells_below_target"], c["supported"], c["cells"])
	b.WriteString(matrixLegend)

	b.WriteString("| Model family | Topology |")
	for _, bk := range backends {
		b.WriteString(" " + bk + " |")
	}
	b.WriteString("\n|---|---|")
	for range backends {
		b.WriteString("---|")
	}
	b.WriteString("\n")
	for _, fam := range familyOrder {
		row := byFamily[fam]
		fmt.Fprintf(&b, "| %s | %s |", fam, topo[fam])
		for _, bk := range backends {
			b.WriteString(" " + row[bk] + " |")
		}
		b.WriteString("\n")
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "Counts: %v SUPPORTED, %v PROOF-PATH-ONLY, %v FENCED, %v UNDEFINED across %v cells (%v families x %v backends).\n\n",
		c["supported"], c["proof_path_only"], c["fenced"], c["undefined"], c["cells"], c["families"], c["backends"])
	b.WriteString(MatrixEnd)
	return b.String()
}

// ExtractMatrixBlock returns the generated block (markers inclusive) embedded in doc, or
// ("", false) when the two markers are not both present in order. The freshness check
// compares this against MatrixBlock().
func ExtractMatrixBlock(doc string) (string, bool) {
	i := strings.Index(doc, MatrixBegin)
	if i < 0 {
		return "", false
	}
	j := strings.Index(doc[i:], MatrixEnd)
	if j < 0 {
		return "", false
	}
	return doc[i : i+j+len(MatrixEnd)], true
}

// SpliceMatrixBlock replaces the block between the markers in doc with a freshly generated
// MatrixBlock() and returns the new doc. It errors when the markers are absent so the
// generator never guesses where to write.
func SpliceMatrixBlock(doc string) (string, error) {
	i := strings.Index(doc, MatrixBegin)
	if i < 0 {
		return "", fmt.Errorf("begin marker not found: %s", MatrixBegin)
	}
	j := strings.Index(doc[i:], MatrixEnd)
	if j < 0 {
		return "", fmt.Errorf("end marker not found after begin marker: %s", MatrixEnd)
	}
	end := i + j + len(MatrixEnd)
	return doc[:i] + MatrixBlock() + doc[end:], nil
}
