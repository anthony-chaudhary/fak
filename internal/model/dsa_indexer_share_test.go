package model

import (
	"reflect"
	"sync"
	"testing"
)

// dsa_indexer_share_test.go — witnesses the layer-group-scoped DSA light-indexer
// selection cache (dsa_indexer_share.go). The contract under test is the mlx-lm
// GLM-5.2 cross-layer indexer sharing port: aliasing layers in a group share ONE
// top-k selection per step, and a layer outside the group never shares it.
//
// The "must fail if each layer recomputes independently" property is pinned by
// counting computations: a per-layer recompute would drive Computes() to the layer
// count, not 1.

// dsaShareFixture is a small, unambiguous indexer input: distinct scores per key
// position so the top-k selection is deterministic and has no unintended ties.
func dsaShareFixture() (scores [][]float64, queryPositions, keyPositions []int, topK int) {
	keyPositions = []int{0, 1, 2, 3, 4, 5}
	queryPositions = []int{5}
	// Key 3 scores highest, then 1, then 5; the rest are lower. topK=3 selects
	// {3, 1, 5} in score order.
	scores = [][]float64{{10, 40, 5, 90, 20, 30}}
	topK = 3
	return scores, queryPositions, keyPositions, topK
}

func TestDsaIndexerShareAliasingLayersGetIdenticalSelection(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	// Four layers alias the same light indexer in group "layers-0-3" at step 7.
	const group = "layers-0-3"
	const step = 7
	var first, broadcast [][]int
	for layer := 0; layer < 4; layer++ {
		sel, ok := cache.Select(group, step, scores, qPos, kPos, topK)
		if !ok {
			t.Fatalf("layer %d: group selection failed", layer)
		}
		if layer == 0 {
			first = sel
			if want := [][]int{{3, 1, 5}}; !reflect.DeepEqual(sel, want) {
				t.Fatalf("group selection = %v, want %v", sel, want)
			}
			continue
		}
		if !reflect.DeepEqual(sel, first) {
			t.Fatalf("aliasing layer %d got %v, want identical to layer 0 %v", layer, sel, first)
		}
		broadcast = sel
	}

	// The broadcast must be a clone: mutating one layer's view cannot corrupt a
	// sibling's or the cached entry.
	first[0][0] = 999
	again, ok := cache.Select(group, step, scores, qPos, kPos, topK)
	if !ok {
		t.Fatal("re-select after mutation failed")
	}
	if again[0][0] == 999 {
		t.Fatalf("cached selection aliases a caller's mutable slice: %v", again)
	}

	// The compute path and the broadcast path clone separately, so a view an
	// aliasing layer received from the cache needs its own mutation check.
	broadcast[0][0] = 999
	after, ok := cache.Select(group, step, scores, qPos, kPos, topK)
	if !ok {
		t.Fatal("re-select after mutating a broadcast view failed")
	}
	if want := [][]int{{3, 1, 5}}; !reflect.DeepEqual(after, want) {
		t.Fatalf("mutating a broadcast view corrupted the group selection: got %v, want %v", after, want)
	}
}

func TestDsaIndexerShareComputesOncePerGroupPerStep(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	const group = "layers-0-3"
	const step = 7
	for layer := 0; layer < 8; layer++ {
		if _, ok := cache.Select(group, step, scores, qPos, kPos, topK); !ok {
			t.Fatalf("layer %d selection failed", layer)
		}
	}
	// Eight aliasing layers, one group, one step => exactly one computation. A
	// per-layer recompute would report 8 here.
	if got := cache.Computes(); got != 1 {
		t.Fatalf("computes = %d, want 1 (aliasing layers must share one selection)", got)
	}
}

func TestDsaIndexerShareNewStepRecomputes(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	const group = "layers-0-3"
	if _, ok := cache.Select(group, 7, scores, qPos, kPos, topK); !ok {
		t.Fatal("step 7 selection failed")
	}
	if got := cache.Computes(); got != 1 {
		t.Fatalf("computes after step 7 = %d, want 1", got)
	}
	// Same group, new step: the cached entry is stale, so it recomputes.
	if _, ok := cache.Select(group, 8, scores, qPos, kPos, topK); !ok {
		t.Fatal("step 8 selection failed")
	}
	if got := cache.Computes(); got != 2 {
		t.Fatalf("computes after new step = %d, want 2 (new step must recompute)", got)
	}
	// Re-selecting the SAME new step is a hit again.
	if _, ok := cache.Select(group, 8, scores, qPos, kPos, topK); !ok {
		t.Fatal("step 8 re-selection failed")
	}
	if got := cache.Computes(); got != 2 {
		t.Fatalf("computes after re-selecting step 8 = %d, want 2", got)
	}
}

func TestDsaIndexerShareChangedGroupKeyRecomputes(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	if _, ok := cache.Select("layers-0-3", 7, scores, qPos, kPos, topK); !ok {
		t.Fatal("group A selection failed")
	}
	// A different group id is a different key: it must compute its own selection,
	// not inherit group A's.
	if _, ok := cache.Select("layers-4-7", 7, scores, qPos, kPos, topK); !ok {
		t.Fatal("group B selection failed")
	}
	if got := cache.Computes(); got != 2 {
		t.Fatalf("computes across two groups = %d, want 2 (changed group key must recompute)", got)
	}
}

func TestDsaIndexerShareLayerOutsideGroupDoesNotShare(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	const step = 7
	// Group A: two aliasing layers share one selection.
	if _, ok := cache.Select("layers-0-3", step, scores, qPos, kPos, topK); !ok {
		t.Fatal("group A layer 0 failed")
	}
	if _, ok := cache.Select("layers-0-3", step, scores, qPos, kPos, topK); !ok {
		t.Fatal("group A layer 1 failed")
	}
	if got := cache.Computes(); got != 1 {
		t.Fatalf("group A computes = %d, want 1", got)
	}

	// A layer outside the group presents a different group id. It must NOT be
	// served group A's cached selection: it computes its own (a second compute),
	// and its digest is its own entry, not group A's.
	// Its indexer scores differ, so being handed group A's selection is visible
	// in the result and not only in the compute counter.
	outsideScores := [][]float64{{90, 40, 5, 10, 20, 30}}
	outside, ok := cache.Select("layers-8-11", step, outsideScores, qPos, kPos, topK)
	if !ok {
		t.Fatal("outside-group selection failed")
	}
	if got := cache.Computes(); got != 2 {
		t.Fatalf("computes after outside-group layer = %d, want 2 (outside layer must not share)", got)
	}
	if cache.Digest("layers-8-11", step) == "" {
		t.Fatal("outside group has no cached digest of its own")
	}
	if cache.Digest("layers-0-3", step) == cache.Digest("layers-8-11", step) {
		t.Fatal("outside group shares group A's digest; it must hold its own selection")
	}
	if want := [][]int{{0, 1, 5}}; !reflect.DeepEqual(outside, want) {
		t.Fatalf("outside-group selection = %v, want its own %v", outside, want)
	}
	// Group A's entry is untouched by the outside layer's compute.
	groupA, ok := cache.Select("layers-0-3", step, scores, qPos, kPos, topK)
	if !ok || !reflect.DeepEqual(groupA, [][]int{{3, 1, 5}}) {
		t.Fatalf("group A selection after outside compute = %v (ok=%v), want [[3 1 5]]", groupA, ok)
	}
	if got := cache.Computes(); got != 2 {
		t.Fatalf("computes after re-reading group A = %d, want 2", got)
	}
}

func TestDsaIndexerShareInvalidateForcesRecompute(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	const group = "layers-0-3"
	const step = 7
	if _, ok := cache.Select(group, step, scores, qPos, kPos, topK); !ok {
		t.Fatal("initial selection failed")
	}
	if got := cache.Computes(); got != 1 {
		t.Fatalf("computes = %d, want 1", got)
	}
	cache.Invalidate(group)
	if cache.Digest(group, step) != "" {
		t.Fatal("digest survived invalidation")
	}
	if _, ok := cache.Select(group, step, scores, qPos, kPos, topK); !ok {
		t.Fatal("post-invalidate selection failed")
	}
	if got := cache.Computes(); got != 2 {
		t.Fatalf("computes after invalidate = %d, want 2", got)
	}
}

func TestDsaIndexerShareFailsClosedOnBadInput(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	if _, ok := cache.Select("", 7, scores, qPos, kPos, topK); ok {
		t.Fatal("empty group id should fail closed")
	}
	if _, ok := cache.Select("layers-0-3", 7, scores, qPos, kPos, 0); ok {
		t.Fatal("non-positive topK should fail closed")
	}
	if got := cache.Computes(); got != 0 {
		t.Fatalf("computes after rejected inputs = %d, want 0", got)
	}
}

func TestDsaIndexerShareConcurrentAliasingLayersComputeOnce(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	cache := newDSAIndexerGroupCache()

	const group = "layers-0-3"
	const step = 7
	const layers = 16
	var wg sync.WaitGroup
	results := make([][][]int, layers)
	oks := make([]bool, layers)
	for layer := 0; layer < layers; layer++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], oks[i] = cache.Select(group, step, scores, qPos, kPos, topK)
		}(layer)
	}
	wg.Wait()

	for i := 0; i < layers; i++ {
		if !oks[i] {
			t.Fatalf("concurrent layer %d selection failed", i)
		}
		if !reflect.DeepEqual(results[i], results[0]) {
			t.Fatalf("concurrent layer %d got %v, want identical to layer 0 %v", i, results[i], results[0])
		}
	}
	if got := cache.Computes(); got != 1 {
		t.Fatalf("concurrent computes = %d, want 1 (compute-once must hold under contention)", got)
	}
}

func TestDsaIndexerShareGroupFromArchDescriptor(t *testing.T) {
	scores, qPos, kPos, topK := dsaShareFixture()
	const step = 3

	// Layers 1 and 2 alias layer 0's indexer; layer 3 runs its own, layer 4
	// aliases that one.
	cfg := Config{NumLayers: 5, IndexerTypes: []string{"full", "shared", "shared", "full", "shared"}}
	wantGroups := []string{"indexer-layer-0", "indexer-layer-0", "indexer-layer-0", "indexer-layer-3", "indexer-layer-3"}
	for layer, want := range wantGroups {
		if got, ok := dsaIndexerGroupForLayer(cfg, layer); !ok || got != want {
			t.Fatalf("layer %d group = %q, %v; want %q", layer, got, ok, want)
		}
	}
	cache := newDSAIndexerGroupCache()
	otherScores := [][]float64{{90, 40, 5, 10, 20, 30}}
	for layer := range wantGroups {
		layerScores, want := scores, [][]int{{3, 1, 5}}
		if layer >= 3 {
			layerScores, want = otherScores, [][]int{{0, 1, 5}}
		}
		got, ok := cache.SelectForLayer(cfg, layer, step, layerScores, qPos, kPos, topK)
		if !ok || !reflect.DeepEqual(got, want) {
			t.Fatalf("layer %d selection = %v, %v; want %v", layer, got, ok, want)
		}
	}
	// Five layers, two indexer groups: the aliasing layers consumed the group
	// selection and only the two full layers computed.
	if got := cache.Computes(); got != 2 {
		t.Fatalf("computes = %d, want 2", got)
	}

	// Every layout the descriptor cannot satisfy refuses and caches nothing.
	for name, tc := range map[string]struct {
		cfg   Config
		layer int
	}{
		"leading shared layer has no full source":  {Config{NumLayers: 3, IndexerTypes: []string{"shared", "full", "shared"}}, 0},
		"dense layer publishes no selection":       {Config{NumLayers: 3, IndexerTypes: []string{"dense", "full", "shared"}}, 0},
		"dense layer clears the share source":      {Config{NumLayers: 3, IndexerTypes: []string{"full", "dense", "shared"}}, 2},
		"unknown indexer kind":                     {Config{NumLayers: 2, IndexerTypes: []string{"full", "sparse-ish"}}, 1},
		"unknown kind behind a shared layer":       {Config{NumLayers: 2, IndexerTypes: []string{"sparse-ish", "shared"}}, 1},
		"legal layer of a plan that starts shared": {Config{NumLayers: 3, IndexerTypes: []string{"shared", "full", "shared"}}, 2},
		"legal layer before an unknown kind":       {Config{NumLayers: 3, IndexerTypes: []string{"full", "shared", "sparse-ish"}}, 1},
		"no layer count":                           {Config{IndexerTypes: []string{"full", "shared"}}, 1},
		"negative layer":                           {cfg, -1},
		"layer past the model":                     {cfg, 5},
	} {
		refused := newDSAIndexerGroupCache()
		if group, ok := dsaIndexerGroupForLayer(tc.cfg, tc.layer); ok {
			t.Fatalf("%s: resolved group %q", name, group)
		}
		if sel, ok := refused.SelectForLayer(tc.cfg, tc.layer, step, scores, qPos, kPos, topK); ok || sel != nil {
			t.Fatalf("%s: SelectForLayer = %v, %v; want refusal", name, sel, ok)
		}
		if refused.Computes() != 0 {
			t.Fatalf("%s: a refused layer computed a selection", name)
		}
	}
	// A dense head is a legal plan: the layers after it still resolve, and a
	// layer the plan does not list defaults to its own full indexer.
	densePlan := Config{NumLayers: 4, IndexerTypes: []string{"dense", "full", "shared"}}
	for layer, want := range map[int]string{1: "indexer-layer-1", 2: "indexer-layer-1", 3: "indexer-layer-3"} {
		if got, ok := dsaIndexerGroupForLayer(densePlan, layer); !ok || got != want {
			t.Fatalf("dense-head plan layer %d = %q, %v; want %q", layer, got, ok, want)
		}
	}
}
