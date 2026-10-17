package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The backend publication is preallocated to isolate host pick materialization.
type v41PickSlabBackend struct {
	compute.Backend
	picks compute.V41RouterPicks
}

func (*v41PickSlabBackend) V41RouterTopKAvailable() bool { return true }
func (b *v41PickSlabBackend) V41RouterTopK(_, _ compute.Tensor, _, _, _ int, _ float32, _, _ bool) (compute.V41RouterPicks, error) {
	return b.picks, nil
}

// fak-test:runtime fast est=10ms lane=default
func TestV41DeviceRoutePickSlabOwnershipAndAllocations(t *testing.T) {
	const rows = v41ProjPanelRowChunk
	cfg := v41DefaultRouterConfig()
	b := &v41PickSlabBackend{picks: compute.V41RouterPicks{
		Experts: make([]int32, rows*cfg.TopK), Weights: make([]float32, rows*cfg.TopK),
	}}
	for i := range b.picks.Experts {
		b.picks.Experts[i] = int32(i % cfg.Experts)
		b.picks.Weights[i] = float32(i+1) / 1024
	}
	var got [][]routePick
	allocs := testing.AllocsPerRun(100, func() {
		var err error
		var completed bool
		got, completed, err = v41DeviceRouteCore(b, compute.Tensor{}, compute.Tensor{}, rows, cfg, true, true)
		if err != nil || !completed {
			panic("device route did not complete")
		}
	})
	for row, picks := range got {
		if len(picks) != cfg.TopK || cap(picks) != cfg.TopK {
			t.Fatalf("row %d has len/cap %d/%d, want %d/%d", row, len(picks), cap(picks), cfg.TopK, cfg.TopK)
		}
		for slot, pick := range picks {
			i := row*cfg.TopK + slot
			if pick.expert != int(b.picks.Experts[i]) || pick.weight != b.picks.Weights[i] {
				t.Fatalf("row %d slot %d changed: %v", row, slot, pick)
			}
		}
	}
	next := got[1][0]
	extended := append(got[0], routePick{expert: -1})
	if got[1][0] != next || extended[cfg.TopK].expert != -1 {
		t.Fatal("appending a row changed the adjacent row")
	}
	b.picks.Experts[0], b.picks.Weights[0] = 99, 99
	if got[0][0].expert != 0 || got[0][0].weight != 1.0/1024 {
		t.Fatal("backend publication aliases retained picks")
	}
	// Two output buffers plus the existing error-classification storage.
	if allocs > 3 {
		t.Fatalf("pick materialization allocated %g objects for %d rows, want at most 3", allocs, rows)
	}
}
