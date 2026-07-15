package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV4ExpertPlacementCoversPinnedNamespaceExactlyOnce(t *testing.T) {
	cfg := pinnedV4RuntimeConfig()
	for _, world := range []int{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384} {
		seen := make([]int, cfg.NumExperts)
		for rank := 0; rank < world; rank++ {
			p, err := NewV4ExpertPlacement(cfg, world, rank)
			if err != nil {
				t.Fatalf("world=%d rank=%d: %v", world, rank, err)
			}
			for expert := 0; expert < cfg.NumExperts; expert++ {
				owner, err := p.Owner(expert)
				if err != nil {
					t.Fatal(err)
				}
				if owner == rank {
					seen[expert]++
				}
			}
		}
		for expert, count := range seen {
			if count != 1 {
				t.Fatalf("world=%d expert=%d ownership count=%d", world, expert, count)
			}
		}
	}
}

func TestV4ExpertPlacementDispatchPreservesCompositionOrder(t *testing.T) {
	p, err := NewV4ExpertPlacement(pinnedV4RuntimeConfig(), 6, 2)
	if err != nil {
		t.Fatal(err)
	}
	picks := []routePick{{expert: 11, weight: .1}, {expert: 0, weight: .2}, {expert: 9, weight: .3}, {expert: 8, weight: .4}, {expert: 7, weight: .5}, {expert: 10, weight: .6}}
	dispatch, err := p.Dispatch(picks)
	if err != nil {
		t.Fatal(err)
	}
	var recombined []routePick
	positions := make(map[int]routePick)
	for rank, work := range dispatch {
		for _, item := range work {
			if item.Rank != rank {
				t.Fatalf("dispatch rank mismatch: %+v", item)
			}
			positions[item.Position] = routePick{expert: item.Expert, weight: item.Weight}
		}
	}
	for i := range picks {
		recombined = append(recombined, positions[i])
	}
	if !reflect.DeepEqual(recombined, picks) {
		t.Fatalf("recombined=%+v want %+v", recombined, picks)
	}
	if len(dispatch[0]) != 6 {
		t.Fatalf("rank-local group=%+v", dispatch[0])
	}
	_, weights := writeV4RuntimeFixture(t)
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32((i%5)-2) / 2
	}
	want := runtimeResidentOracle(3, picks, x, weights, 10)
	got := make([]float32, len(want))
	for _, work := range dispatch {
		rankPicks := make([]routePick, len(work))
		for i, item := range work {
			rankPicks[i] = routePick{expert: item.Expert, weight: item.Weight}
		}
		partial := runtimeResidentOracle(3, rankPicks, x, weights, 10)
		for i := range got {
			got[i] += partial[i]
		}
	}
	if !closeSlice(got, want, 2e-5) {
		t.Fatalf("rank recombine=%v want resident oracle=%v", got, want)
	}
}

func TestV4ExpertPlacementFailsClosed(t *testing.T) {
	cfg := pinnedV4RuntimeConfig()
	for _, tc := range []struct{ world, rank int }{{0, 0}, {5, 0}, {385, 0}, {6, -1}, {6, 6}} {
		if _, err := NewV4ExpertPlacement(cfg, tc.world, tc.rank); !errors.Is(err, ErrV4ExpertPlacement) {
			t.Fatalf("world=%d rank=%d err=%v", tc.world, tc.rank, err)
		}
	}
	bad := cfg
	bad.NumExperts = 383
	if _, err := NewV4ExpertPlacement(bad, 1, 0); !errors.Is(err, ErrV4ExpertPlacement) {
		t.Fatalf("bad shape err=%v", err)
	}
}

func TestV4ExpertRuntimePlacementRefusesRemoteBeforePayloadRead(t *testing.T) {
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	t.Setenv(v4ExpertWorldEnv, "6")
	t.Setenv(v4ExpertRankEnv, "5")
	dir, _ := writeV4RuntimeFixture(t)
	be := compute.Default()
	r, err := newV4ExpertRuntime(dir, pinnedV4RuntimeConfig(), be, 16384, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	logits := make([]float32, 384)
	for i := range logits {
		logits[i] = -10
	}
	for i := 0; i < 6; i++ {
		logits[i] = float32(6 - i)
	}
	xHost := make([]float32, 32)
	x := be.Upload(compute.NewF32(be, []int{32}, xHost), compute.F32)
	defer be.Free(x)
	if _, err := r.forwardScored(3, logits, nil, x); !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("remote dispatch err=%v", err)
	}
	stats := r.Stats()
	if stats.WorldSize != 6 || stats.Rank != 5 || stats.LocalSelected != 0 || stats.RemoteSelected != 6 {
		t.Fatalf("placement counters=%+v", stats)
	}
	if stats.SourceReads != 0 || stats.ExpertReadCount != 0 {
		t.Fatalf("remote refusal opened payloads: %+v", stats)
	}
}
