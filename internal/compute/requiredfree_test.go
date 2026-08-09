package compute

import "testing"

// RequiredFreeBytes is only worth having if it is the EXACT boundary the fit check draws. An
// approximation would be worse than no helper at all: a pre-flight gate that rounds one byte the
// wrong way either refuses a run the in-process check would have admitted, or admits one it then
// refuses after the ranks are already spawned — the second reads as a code regression rather than
// as the environmental refusal it is, which is how #4952 came to be filed against the load plan.
//
// So the test is stated against fitsWithinReportedMemory, the function the load-time checks
// actually call, rather than against a restatement of the formula.
func TestRequiredFreeBytesIsExactFitBoundary(t *testing.T) {
	const gib = int64(1) << 30

	for _, headroom := range []float64{0.05, 0.15, 0.5, 0.9, 0.01} {
		for _, want := range []int64{
			1, 2, 3, 7, 999, 1000, 1001,
			gib, 73*gib + 235*gib/1000, 81*gib + 330*gib/1000,
			1 << 40, 1<<40 + 1,
		} {
			need := RequiredFreeBytes(want, headroom)

			if got := BudgetAfterHeadroom(need, headroom); got < want {
				t.Fatalf("RequiredFreeBytes(%d, %g) = %d, but BudgetAfterHeadroom(%d, %g) = %d < %d: the reported requirement is not actually enough",
					want, headroom, need, need, headroom, got, want)
			}
			if got := BudgetAfterHeadroom(need-1, headroom); got >= want {
				t.Fatalf("RequiredFreeBytes(%d, %g) = %d, but %d free already yields budget %d >= %d: the requirement overshoots the boundary by at least a byte",
					want, headroom, need, need-1, got, want)
			}

			// The same statement through the function the load-time fit checks call, so the helper
			// is pinned to the gate rather than to a second copy of the arithmetic.
			if v, _ := fitsWithinReportedMemory(need*2, need, true, want, headroom); v != FitOK {
				t.Fatalf("want=%d headroom=%g: %d free (the reported requirement) gives %v, want FitOK", want, headroom, need, v)
			}
			if v, _ := fitsWithinReportedMemory(need*2, need-1, true, want, headroom); v != FitTooBig {
				t.Fatalf("want=%d headroom=%g: %d free (one byte under the requirement) gives %v, want FitTooBig", want, headroom, need-1, v)
			}
		}
	}
}

// The degenerate cases have to mirror BudgetAfterHeadroom's, or the two disagree at the edges
// where a caller is least likely to be looking.
func TestRequiredFreeBytesMirrorsBudgetAfterHeadroomDegenerateCases(t *testing.T) {
	for _, tc := range []struct {
		name     string
		want     int64
		headroom float64
		req      int64
	}{
		{"nothing wanted needs nothing", 0, 0.15, 0},
		{"a negative demand needs nothing", -5, 0.15, 0},
		{"zero headroom is the identity", 1000, 0, 1000},
		{"a headroom of 1 or more is ignored, as in BudgetAfterHeadroom", 1000, 1.5, 1000},
		{"a negative headroom is ignored", 1000, -0.2, 1000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := RequiredFreeBytes(tc.want, tc.headroom); got != tc.req {
				t.Fatalf("RequiredFreeBytes(%d, %g) = %d, want %d", tc.want, tc.headroom, got, tc.req)
			}
		})
	}
}

// The numbers experiments/glm-gpu-witness/glm52-ep-preflight-guard-selfcheck-2026-07-15.json
// publishes for GLM-5.2 EP, pinned here so the witness cannot drift away from the code it names.
// 8 ranks fit an 80 GiB-class card with room to spare; 7 ranks do not fit one at all.
func TestRequiredFreeBytesMatchesThePublishedGLM52EPPreflightNumbers(t *testing.T) {
	const gib = float64(1 << 30)
	const headroom = 0.05

	for _, tc := range []struct {
		ranks    int
		planGiB  float64
		wantFree float64 // GiB, to one decimal place — the precision the witness publishes
	}{
		{ranks: 8, planGiB: 73.23, wantFree: 77.1},
		{ranks: 7, planGiB: 81.33, wantFree: 85.6},
	} {
		got := float64(RequiredFreeBytes(int64(tc.planGiB*gib), headroom)) / gib
		if rounded := float64(int64(got*10+0.5)) / 10; rounded != tc.wantFree {
			t.Fatalf("ranks=%d plan=%.2f GiB: required free = %.4f GiB (rounds to %.1f), witness publishes %.1f GiB",
				tc.ranks, tc.planGiB, got, rounded, tc.wantFree)
		}
	}
}
