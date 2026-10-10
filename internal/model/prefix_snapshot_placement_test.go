package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
// Exercises the public capture/clone/restore route on CPU-owned storage. The
// serving planner reapplies its own placement; this is the direct API contract.
func TestPrefixSnapshotClonePreservesLayerPlacement(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name         string
		dense, alias int
	}{
		{"primary", 1, 0},
		{"alias", 0, 2},
		{"both", 2, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{NumLayers: 3, NumHeads: 1, NumKVHeads: 1, HeadDim: 1}
			m := &Model{Cfg: cfg}
			be := compute.Pick("cpu-ref")
			newSession := func() *Session {
				return &Session{
					M: m, Cache: NewKVCache(cfg), Backend: be,
					halKV: be.NewKV(compute.KVConfig{NumLayers: 3, NumKVHeads: 1, HeadDim: 1}),
				}
			}
			source := newSession()
			defer source.Close()
			source.DenseGPULayers, source.GPULayers = tc.dense, tc.alias
			captured, err := source.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer captured.Close()
			for _, explicitTarget := range []bool{false, true} {
				clone, err := captured.Clone()
				if err != nil {
					t.Fatal(err)
				}
				defer clone.Close()
				if clone.DenseGPULayers != tc.dense || clone.GPULayers != tc.alias {
					t.Fatalf("clone placement=%d/%d, want %d/%d", clone.DenseGPULayers, clone.GPULayers, tc.dense, tc.alias)
				}
				target := newSession()
				defer target.Close()
				wantDense, wantAlias := tc.dense, tc.alias
				if explicitTarget {
					target.DenseGPULayers, target.GPULayers = 3, 3
					wantDense, wantAlias = 3, 3
				}
				if err := clone.Restore(target); err != nil {
					t.Fatal(err)
				}
				if target.DenseGPULayers != wantDense || target.GPULayers != wantAlias {
					t.Fatalf("restore placement=%d/%d, want %d/%d", target.DenseGPULayers, target.GPULayers, wantDense, wantAlias)
				}
				_, split := target.validateDenseGPULayers()
				if split == explicitTarget {
					t.Fatal("restored placement selected the wrong partial/all-backend mode")
				}
			}
			if captured.Cache == nil || captured.DenseGPULayers != tc.dense || captured.GPULayers != tc.alias {
				t.Fatal("cloned restores consumed or changed the retained snapshot")
			}
		})
	}
}
