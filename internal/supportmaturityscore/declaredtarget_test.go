package supportmaturityscore

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
)

// findCell returns the live grid cell for (family, backend), failing the test if absent.
func findCell(t *testing.T, family, backend string) covmatrix.Cell {
	t.Helper()
	for _, c := range covmatrix.Grid() {
		if c.Family == family && c.Backend == backend {
			return c
		}
	}
	t.Fatalf("cell %s x %s not found in covmatrix.Grid()", family, backend)
	return covmatrix.Cell{}
}

// TestDeclaredTargetIsRegimeAware reads the committed declared-target table back per cell and
// asserts the anti-Goodhart fence (#1247): a non-PreNorm family on an accelerated backend
// declares its FENCE (M1), not SUPPORTED, while the cpu reference path and the accelerated
// PreNorm hot path declare correctness (M4). This is the "the table is committed and read
// back" witness named in the issue.
func TestDeclaredTargetIsRegimeAware(t *testing.T) {
	cases := []struct {
		family, backend string
		want            supportmaturity.Rung
	}{
		{"GPT-NeoX", "cuda", supportmaturity.M1Fenced},      // accelerated non-PreNorm -> the fence IS the ceiling
		{"DeepSeek-MLA", "metal", supportmaturity.M1Fenced}, // accelerated SparseAttn -> fenced, not debt
		{"Gemma2/3", "cpu", supportmaturity.M4Correct},      // cpu reference path -> correctness
		{"Llama", "cuda", supportmaturity.M4Correct},        // accelerated PreNorm hot path -> correctness
		{"Qwen2/3.x", "vulkan", supportmaturity.M4Correct},
	}
	for _, tc := range cases {
		if got := declaredTarget(findCell(t, tc.family, tc.backend)); got != tc.want {
			t.Errorf("declaredTarget(%s x %s) = %s, want %s", tc.family, tc.backend, got, tc.want)
		}
	}
}

// TestAtOrAboveTargetIsZeroBelowIsGap is the #1247 debt witness, checked per cell over the live
// grid: a cell at-or-above its declared target contributes 0 defect(s); a cell below contributes
// exactly (target - current). It also asserts the fence actually fires -- at least one FENCED
// accelerated cell is at its target and so contributes nothing, which is precisely the cell the
// old count-below-SUPPORTED debt wrongly charged.
func TestAtOrAboveTargetIsZeroBelowIsGap(t *testing.T) {
	defects := Build().KPIs[0].Defects
	fencedAtTarget := 0
	for _, c := range covmatrix.Grid() {
		cur := supportmaturity.FromSupport(c.Support)
		tgt := declaredTarget(c)
		prefix := c.Family + " x " + c.Backend + ":"
		got := 0
		for _, d := range defects {
			if strings.HasPrefix(d, prefix) {
				got++
			}
		}
		want := 0
		if cur.Less(tgt) {
			want = int(tgt - cur)
		}
		if got != want {
			t.Errorf("%s contributed %d defect(s), want %d (current %s, target %s)", prefix, got, want, cur, tgt)
		}
		if c.Backend != "cpu" && c.Support == covmatrix.Fenced && !cur.Less(tgt) {
			fencedAtTarget++
		}
	}
	if fencedAtTarget == 0 {
		t.Fatal("no FENCED accelerated cell was at-or-above its declared target; the anti-Goodhart fence is not exercised")
	}
}
