package turnbench

import (
	"context"
	"encoding/json"
	"testing"
)

// TestDivergenceHistogram_DistributionAndExactFraction is the #501(b) proof: a corpus of
// traces × candidate policies rolls up into a first_divergence distribution and an
// exact-cell fraction. Using the spine corpus (4 arms over a 5-call trace, reference
// "recorded"), the three NON-reference candidates are: equivalent-on-trace (exact),
// strict-no-book (bounded@3), permissive-plus (bounded@1). So one trace yields 3 cells:
// 1 exact + 2 bounded with first_divergence indices {1, 3}.
func TestDivergenceHistogram_DistributionAndExactFraction(t *testing.T) {
	ctx := context.Background()
	corpus := []DivHistInput{
		{Trace: spineTrace(t), Arms: spineArms(), RefName: "recorded"},
	}

	rep, err := RunDivergenceHistogram(ctx, corpus, DefaultCostModel())
	if err != nil {
		t.Fatalf("RunDivergenceHistogram: %v", err)
	}

	if rep.Traces != 1 {
		t.Errorf("traces: want 1, got %d", rep.Traces)
	}
	// 4 arms, 1 is the reference => 3 candidate cells.
	if rep.Cells != 3 {
		t.Fatalf("cells: want 3 (4 arms minus the reference), got %d", rep.Cells)
	}
	if rep.ExactCells != 1 || rep.BoundedCells != 2 {
		t.Errorf("want 1 exact + 2 bounded cells, got exact=%d bounded=%d", rep.ExactCells, rep.BoundedCells)
	}
	// exact fraction = 1/3.
	if got, want := rep.ExactFraction, 1.0/3.0; got != want {
		t.Errorf("exact fraction: want %v, got %v", want, got)
	}

	// The first_divergence distribution: one bounded cell at index 1 (permissive-plus),
	// one at index 3 (strict-no-book). Bins sum to BoundedCells.
	gotBins := map[int]int{}
	sum := 0
	for _, b := range rep.FirstDivergence {
		gotBins[b.Index] = b.Cells
		sum += b.Cells
	}
	if sum != rep.BoundedCells {
		t.Errorf("histogram bins (%d) must sum to bounded cells (%d)", sum, rep.BoundedCells)
	}
	if gotBins[1] != 1 || gotBins[3] != 1 {
		t.Errorf("first_divergence distribution: want {1:1, 3:1}, got %+v", gotBins)
	}
	// The bins must be sorted ascending by index (stable artifact ordering).
	for i := 1; i < len(rep.FirstDivergence); i++ {
		if rep.FirstDivergence[i-1].Index >= rep.FirstDivergence[i].Index {
			t.Errorf("histogram bins not sorted ascending: %+v", rep.FirstDivergence)
		}
	}
}

// TestDivergenceHistogram_NoDivergenceControlIs100Exact is the load-bearing control: a
// corpus whose candidate arms are all IDENTICAL to the reference must read exactly 100%
// exact — the witness cannot manufacture a divergence where no policy changed, so the
// exact-cell fraction is the honest baseline the headline multiple is quoted against.
func TestDivergenceHistogram_NoDivergenceControlIs100Exact(t *testing.T) {
	ctx := context.Background()
	ref := spineArms()[0] // "recorded"
	noDiv := []PolicyArm{
		ref,
		{Name: "ref-copy", Policy: ref.Policy},
		{Name: "ref-copy-2", Policy: ref.Policy},
	}
	corpus := []DivHistInput{
		{Trace: spineTrace(t), Arms: noDiv, RefName: "recorded"},
		// A second no-divergence trace, to prove the control holds across the corpus.
		{Trace: redactTrace(t), Arms: []PolicyArm{
			{Name: "served-raw", Policy: spineArms()[0].Policy},
			{Name: "served-raw-copy", Policy: spineArms()[0].Policy},
		}, RefName: "served-raw"},
	}

	rep, err := RunDivergenceHistogram(ctx, corpus, DefaultCostModel())
	if err != nil {
		t.Fatalf("RunDivergenceHistogram: %v", err)
	}

	if rep.BoundedCells != 0 {
		t.Errorf("no-divergence control must have ZERO bounded cells, got %d", rep.BoundedCells)
	}
	if rep.ExactFraction != 1.0 {
		t.Fatalf("no-divergence control must read 100%% exact, got %v (exact=%d/cells=%d)",
			rep.ExactFraction, rep.ExactCells, rep.Cells)
	}
	if len(rep.FirstDivergence) != 0 {
		t.Errorf("no-divergence control must have an empty first_divergence histogram, got %+v", rep.FirstDivergence)
	}
	if rep.ExactCells != rep.Cells {
		t.Errorf("every cell must be exact: exact=%d cells=%d", rep.ExactCells, rep.Cells)
	}
}

// TestDivergenceHistogram_Deterministic asserts the report is a regenerable artifact: a
// fixed corpus yields byte-identical JSON across runs (modulo the host-dependent
// provenance fields), so the distribution and exact fraction are reproducible numbers,
// not a sample.
func TestDivergenceHistogram_Deterministic(t *testing.T) {
	ctx := context.Background()
	mk := func() []DivHistInput {
		return []DivHistInput{{Trace: spineTrace(t), Arms: spineArms(), RefName: "recorded"}}
	}

	a, err := RunDivergenceHistogram(ctx, mk(), DefaultCostModel())
	if err != nil {
		t.Fatalf("run a: %v", err)
	}
	b, err := RunDivergenceHistogram(ctx, mk(), DefaultCostModel())
	if err != nil {
		t.Fatalf("run b: %v", err)
	}

	// Zero the host-dependent provenance so the comparison is over the measured surface.
	a.Provenance = Provenance{}
	b.Provenance = Provenance{}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Errorf("histogram report drifted across runs:\n a=%s\n b=%s", ja, jb)
	}
}
