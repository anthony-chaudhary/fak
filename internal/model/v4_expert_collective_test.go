package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV4CollectiveExpertTransportMatchesResidentOracle(t *testing.T) {
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
	transport, err := newV4CollectiveExpertTransport(be, r.placement)
	if err != nil {
		t.Fatal(err)
	}
	r.transport = transport

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
		t.Fatalf("collective=%v want resident oracle=%v", got, want)
	}
	stats := r.Stats()
	if stats.TransportDispatches != 1 || stats.TransportPartials != 6 {
		t.Fatalf("transport counters=%+v", stats)
	}
	if stats.LocalSelected != 1 || stats.RemoteSelected != 5 {
		t.Fatalf("placement counters=%+v", stats)
	}
}

func TestV4CollectiveExpertTransportFailsClosedOnMalformedDispatch(t *testing.T) {
	be := compute.Default()
	placement, err := NewV4ExpertPlacement(pinnedV4RuntimeConfig(), 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	transport, err := newV4CollectiveExpertTransport(be, placement)
	if err != nil {
		t.Fatal(err)
	}
	bad := map[int][]V4ExpertDispatch{
		0: {{Rank: 0, Expert: 0, Weight: 1}},
		1: {{Rank: 0, Expert: 192, Weight: 1}},
	}
	_, _, err = transport.Forward(bad, func([]routePick) ([]float32, error) { return []float32{1}, nil })
	if !errors.Is(err, ErrV4ExpertPlacement) {
		t.Fatalf("rank-mismatch err=%v", err)
	}

	widthMismatch := map[int][]V4ExpertDispatch{
		0: {{Rank: 0, Expert: 0, Weight: 1}},
		1: {{Rank: 1, Expert: 192, Weight: 1}},
	}
	_, _, err = transport.Forward(widthMismatch, func(picks []routePick) ([]float32, error) {
		if picks[0].expert == 0 {
			return []float32{1}, nil
		}
		return []float32{1, 2}, nil
	})
	if !errors.Is(err, ErrV4ExpertPlacement) {
		t.Fatalf("width-mismatch err=%v", err)
	}
}

func TestV4CollectiveExpertTransportSingleRankIdentity(t *testing.T) {
	be := compute.Default()
	placement, err := NewV4ExpertPlacement(pinnedV4RuntimeConfig(), 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	transport, err := newV4CollectiveExpertTransport(be, placement)
	if err != nil {
		t.Fatal(err)
	}
	got, partials, err := transport.Forward(map[int][]V4ExpertDispatch{0: {{Rank: 0, Expert: 7, Weight: .5}}}, func(picks []routePick) ([]float32, error) {
		return []float32{float32(picks[0].expert), picks[0].weight}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if partials != 1 || len(got) != 2 || got[0] != 7 || got[1] != .5 {
		t.Fatalf("got=%v partials=%d", got, partials)
	}
}
