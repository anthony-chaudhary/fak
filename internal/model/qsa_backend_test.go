package model

import (
	"errors"
	"math"
	"testing"
)

// qsaTestFixture is the fixed tile fixture the QSA oracle runs over: two query
// rows, four KV tiles, head dim 4. Query row 0 aligns with key tile 0 and row
// 1 with key tile 1, so the top-2 selection is [0,1] and [1,0] respectively.
func qsaTestFixture() (QSAParams, QSAFixture) {
	params := QSAParams{TopK: 2, HeadDim: 4, SoftmaxScale: 1}
	fixture := QSAFixture{
		QueryTiles: [][]float32{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
		},
		KeyTiles: [][]float32{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		},
		ValueTiles: [][]float32{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		},
	}
	return params, fixture
}

// qsaReversedBackend wraps the oracle but advertises the core-attention phase
// before the indexer-selection phase, so RunQSA must refuse it.
type qsaReversedBackend struct{ *QSAOracle }

func (b *qsaReversedBackend) Plan(params QSAParams, fixture QSAFixture) (QSAPlan, error) {
	plan, err := b.QSAOracle.Plan(params, fixture)
	if err != nil {
		return plan, err
	}
	plan.Phases = []QSAPhase{QSAPhaseCoreAttention, QSAPhaseIndexerSelection}
	return plan, nil
}

// TestQSABackend pins the Qwen3.8 QSA pluggable sparse-attention backend
// contract: the params/plan object shape, the
// indexer-selection-then-core-attention phase order, pluggable
// registration/swap, and the fail-closed refusal on an unregistered backend.
func TestQSABackend(t *testing.T) {
	t.Run("params/plan object shape", func(t *testing.T) {
		oracle := NewQSAOracle()
		params, fixture := qsaTestFixture()
		plan, err := oracle.Plan(params, fixture)
		if err != nil {
			t.Fatalf("Plan error = %v", err)
		}
		if plan.QueryRows != 2 {
			t.Fatalf("plan.QueryRows = %d, want 2", plan.QueryRows)
		}
		if plan.TileCount != 4 {
			t.Fatalf("plan.TileCount = %d, want 4", plan.TileCount)
		}
		if plan.Params != params {
			t.Fatalf("plan.Params = %+v, want %+v", plan.Params, params)
		}
		if !plan.valid() {
			t.Fatalf("plan.valid() = false, want true for a positive-budget plan")
		}
		// A zero-sized budget must fail closed at the params seam.
		bad := params
		bad.TopK = 0
		if _, err := oracle.Plan(bad, fixture); !errors.Is(err, ErrQSAParams) {
			t.Fatalf("Plan(top_k=0) error = %v, want ErrQSAParams", err)
		}
	})

	t.Run("indexer-selection then core-attention phase order", func(t *testing.T) {
		oracle := NewQSAOracle()
		params, fixture := qsaTestFixture()
		plan, err := oracle.Plan(params, fixture)
		if err != nil {
			t.Fatalf("Plan error = %v", err)
		}
		want := []QSAPhase{QSAPhaseIndexerSelection, QSAPhaseCoreAttention}
		if len(plan.Phases) != len(want) {
			t.Fatalf("plan.Phases = %v, want %v", plan.Phases, want)
		}
		for i, p := range plan.Phases {
			if p != want[i] {
				t.Fatalf("plan.Phases[%d] = %v, want %v", i, p, want[i])
			}
		}
		// A reversed program must be refused before any phase runs.
		reversed := &qsaReversedBackend{QSAOracle: oracle}
		if _, err := RunQSA(reversed, params, fixture); !errors.Is(err, ErrQSAPhaseOrder) {
			t.Fatalf("RunQSA(reversed) error = %v, want ErrQSAPhaseOrder", err)
		}
	})

	t.Run("pluggable registration and swap", func(t *testing.T) {
		reg := NewQSARegistry()
		oracle := NewQSAOracle()
		if err := reg.Register(oracle); err != nil {
			t.Fatalf("Register error = %v", err)
		}
		got, err := reg.Lookup(QSAPluggableBackendPath)
		if err != nil {
			t.Fatalf("Lookup error = %v", err)
		}
		if got != oracle {
			t.Fatalf("Lookup = %p, want the registered oracle %p", got, oracle)
		}
		// Run the two-phase program through the looked-up backend.
		params, fixture := qsaTestFixture()
		if _, err := RunQSA(got, params, fixture); err != nil {
			t.Fatalf("RunQSA(registered) error = %v", err)
		}
		// Swap in a second implementation under the same path; the lookup must
		// now return the replacement.
		second := NewQSAOracle()
		if err := reg.Swap(QSAPluggableBackendPath, second); err != nil {
			t.Fatalf("Swap error = %v", err)
		}
		after, err := reg.Lookup(QSAPluggableBackendPath)
		if err != nil {
			t.Fatalf("Lookup after swap error = %v", err)
		}
		if after != second {
			t.Fatalf("Lookup after swap = %p, want the replacement %p", after, second)
		}
		// A swap to a path with no current registration must fail closed.
		if err := reg.Swap("qwen38/qsa-pluggable-backend-v9", second); !errors.Is(err, ErrQSAUnregisteredBackend) {
			t.Fatalf("Swap(unregistered) error = %v, want ErrQSAUnregisteredBackend", err)
		}
	})

	t.Run("fail-closed refusal on unregistered backend", func(t *testing.T) {
		reg := NewQSARegistry()
		_, err := reg.Lookup(QSAPluggableBackendPath)
		if !errors.Is(err, ErrQSAUnregisteredBackend) {
			t.Fatalf("Lookup(unregistered) error = %v, want ErrQSAUnregisteredBackend", err)
		}
		// A nil backend and an empty path must both be refused at registration.
		if err := reg.Register(nil); !errors.Is(err, ErrQSAParams) {
			t.Fatalf("Register(nil) error = %v, want ErrQSAParams", err)
		}
	})

	t.Run("closed oracle over fixed tile fixture", func(t *testing.T) {
		oracle := NewQSAOracle()
		params, fixture := qsaTestFixture()
		plan, err := oracle.Plan(params, fixture)
		if err != nil {
			t.Fatalf("Plan error = %v", err)
		}
		sel, err := oracle.SelectIndexer(plan, fixture)
		if err != nil {
			t.Fatalf("SelectIndexer error = %v", err)
		}
		if len(sel) != 2 {
			t.Fatalf("selection rows = %d, want 2", len(sel))
		}
		wantSel := [][]int{{0, 1}, {1, 0}}
		for row := range sel {
			if len(sel[row]) != 2 {
				t.Fatalf("selection row %d = %v, want 2 tiles", row, sel[row])
			}
			for i, want := range wantSel[row] {
				if sel[row][i] != want {
					t.Fatalf("selection row %d[%d] = %d, want %d", row, i, sel[row][i], want)
				}
			}
		}
		out, err := oracle.RunCore(plan, sel, fixture)
		if err != nil {
			t.Fatalf("RunCore error = %v", err)
		}
		if len(out) != 2 {
			t.Fatalf("output rows = %d, want 2", len(out))
		}
		for row := range out {
			if len(out[row]) != params.HeadDim {
				t.Fatalf("output row %d dim = %d, want %d", row, len(out[row]), params.HeadDim)
			}
		}
		// Row 0 attends more to tile 0 than tile 1; the two selected value
		// tiles span dims 0 and 1, so their weights sum to 1 and the
		// unselected tiles (dims 2,3) contribute nothing.
		if out[0][0] <= out[0][1] {
			t.Fatalf("out[0] = %v, want dim0 > dim1 (tile 0 dominates)", out[0])
		}
		if got := out[0][0] + out[0][1]; math.Abs(float64(got)-1.0) > 1e-5 {
			t.Fatalf("out[0][0]+out[0][1] = %v, want 1 (softmax weights sum)", got)
		}
		if out[0][2] != 0 || out[0][3] != 0 {
			t.Fatalf("out[0] = %v, want unselected tiles to contribute nothing", out[0])
		}
		if out[1][1] <= out[1][0] {
			t.Fatalf("out[1] = %v, want dim1 > dim0 (tile 1 dominates)", out[1])
		}
	})
}
