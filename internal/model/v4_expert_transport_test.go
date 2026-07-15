package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV4LoopbackExpertTransportMatchesResidentOracle(t *testing.T) {
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	t.Setenv(v4ExpertWorldEnv, "384")
	t.Setenv(v4ExpertRankEnv, "2")
	dir, weights := writeV4RuntimeFixture(t)
	be := compute.Default()
	r, err := newV4ExpertRuntime(dir, pinnedV4RuntimeConfig(), be, 16384, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	r.transport = v4LoopbackExpertTransport{}

	picks := []routePick{{expert: 5, weight: .1}, {expert: 0, weight: .2}, {expert: 3, weight: .3}, {expert: 2, weight: .4}, {expert: 1, weight: .5}, {expert: 4, weight: .6}}
	xHost := make([]float32, 32)
	for i := range xHost {
		xHost[i] = float32((i%5)-2) / 2
	}
	x := be.Upload(compute.NewF32(be, []int{32}, xHost), compute.F32)
	defer be.Free(x)
	got, err := r.forwardSelected(3, picks, x)
	if err != nil {
		t.Fatal(err)
	}
	want := runtimeResidentOracle(3, picks, xHost, weights, 10)
	if !closeSlice(got, want, 2e-5) {
		t.Fatalf("loopback=%v want resident oracle=%v", got, want)
	}
	stats := r.Stats()
	if stats.TransportDispatches != 1 || stats.TransportPartials != 6 {
		t.Fatalf("transport counters=%+v", stats)
	}
	if stats.LocalSelected != 1 || stats.RemoteSelected != 5 {
		t.Fatalf("placement counters=%+v", stats)
	}
}

func TestV4LoopbackExpertTransportStableRankOrder(t *testing.T) {
	dispatch := map[int][]V4ExpertDispatch{
		0: {{Rank: 0, Expert: 0, Weight: 1}},
		1: {{Rank: 1, Expert: 64, Weight: 1}},
		2: {{Rank: 2, Expert: 128, Weight: 1}},
	}
	var order []int
	got, partials, err := (v4LoopbackExpertTransport{}).Forward(dispatch, func(picks []routePick) ([]float32, error) {
		order = append(order, picks[0].expert)
		return []float32{float32(picks[0].expert + 1)}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if partials != 3 || len(got) != 1 || got[0] != 195 || len(order) != 3 || order[0] != 0 || order[1] != 64 || order[2] != 128 {
		t.Fatalf("output=%v partials=%d order=%v", got, partials, order)
	}
}
