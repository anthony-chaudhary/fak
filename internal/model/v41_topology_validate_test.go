package model

import (
	"errors"
	"reflect"
	"testing"
)

func v41TopologyValidateFixture(ratios ...int) Config {
	if len(ratios) == 0 {
		ratios = []int{0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	}
	return Config{ModelType: "deepseek41", NumLayers: len(ratios), HiddenSize: 32, NumHeads: 1, NumKVHeads: 1,
		CompressRatios: ratios, HCMult: 4, HCSinkhornIters: 20, HCEps: 1e-6, SwigluLimit: 10, NormTopKProb: true}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestDeepSeekV41GGUFConfigRefusesIndexSegmentCrossingKVSource(t *testing.T) {
	for _, tc := range []struct {
		name    string
		cfg     Config
		topo    V41GGUFTopology
		refused bool
	}{
		{"index segments inside kv segments", v41TopologyValidateFixture(), V41GGUFTopology{KVSourceLayerIDs: []int{2, 8}, IndexSourceLayerIDs: []int{2, 8}}, false},
		{"index finer than kv", v41TopologyValidateFixture(), V41GGUFTopology{KVSourceLayerIDs: []int{2}, IndexSourceLayerIDs: []int{2, 5, 9}}, false},
		{"ratio-zero layer between source and reader", v41TopologyValidateFixture(0, 0, 2, 0, 2, 2), V41GGUFTopology{KVSourceLayerIDs: []int{2}, IndexSourceLayerIDs: []int{2}}, false},
		{"unobserved topology", v41TopologyValidateFixture(), V41GGUFTopology{}, false},
		{"no kv source", v41TopologyValidateFixture(), V41GGUFTopology{KVSourceLayerIDs: []int{}, IndexSourceLayerIDs: []int{1}}, false},
		{"unimplemented ratio left to the forward", v41TopologyValidateFixture(0, 0, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2), V41GGUFTopology{KVSourceLayerIDs: []int{2, 8}, IndexSourceLayerIDs: []int{2, 8}}, false},
		{"index segment crosses a later kv source", v41TopologyValidateFixture(), V41GGUFTopology{KVSourceLayerIDs: []int{2, 8}, IndexSourceLayerIDs: []int{2, 6}}, true},
		{"kv source without its own index source", v41TopologyValidateFixture(), V41GGUFTopology{KVSourceLayerIDs: []int{2, 8}, IndexSourceLayerIDs: []int{2}}, true},
		{"selection reused before any kv source", v41TopologyValidateFixture(), V41GGUFTopology{KVSourceLayerIDs: []int{4}, IndexSourceLayerIDs: []int{2, 4}}, true},
	} {
		d41, err := NewDeepSeekV41GGUFConfig(tc.cfg, tc.topo)
		if tc.refused {
			if !errors.Is(err, ErrV41ConfigAdmission) || !errors.Is(err, errV41TopologyIndexCrossesKV) || d41 != nil {
				t.Errorf("%s: (cfg=%v, err=%v), want (nil, ErrV41ConfigAdmission+errV41TopologyIndexCrossesKV)", tc.name, d41 != nil, err)
			}
			continue
		}
		if err != nil || d41 == nil {
			t.Errorf("%s: (cfg=%v, err=%v), want admitted", tc.name, d41 != nil, err)
			continue
		}
		if tc.topo.KVSourceLayerIDs != nil && (!v41EqualInts(d41.KVSourceLayerIDs, tc.topo.KVSourceLayerIDs) || !v41EqualInts(d41.IndexSourceLayerIDs, tc.topo.IndexSourceLayerIDs)) {
			t.Errorf("%s: sources kv=%v index=%v, want %v %v", tc.name, d41.KVSourceLayerIDs, d41.IndexSourceLayerIDs, tc.topo.KVSourceLayerIDs, tc.topo.IndexSourceLayerIDs)
		}
	}

	published := v41TopologyValidateFixture(v41PublishedCompressRatios...)
	published.NumLayers = 40
	d41, err := NewDeepSeekV41GGUFConfig(published, V41GGUFTopology{
		KVSourceLayerIDs: []int{2, 8, 14, 20}, IndexSourceLayerIDs: []int{2, 8, 14, 20, 24, 28, 32, 36},
	})
	if err != nil || d41 == nil {
		t.Fatalf("published schedule: err = %v", err)
	}
	if got := []int{d41.CandidateSourceLayerID, d41.CandidateTopKBlocks, d41.CandidateBlockSize}; !reflect.DeepEqual(got, []int{20, 2048, 8}) {
		t.Fatalf("published candidate = %v, want [20 2048 8]", got)
	}
	if err := v41ValidateSharedTopology(published, d41); err != nil {
		t.Fatalf("published topology fails the DeepInfer refinement rule: %v", err)
	}
}
