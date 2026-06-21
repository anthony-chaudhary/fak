package model

import (
	"reflect"
	"testing"
)

// msa_index_test.go — host-tractable witnesses for MiniMax-M3 MiniMax Sparse
// Attention (MSA) block selection (msa_index.go). These prove the cache-relevant
// numeric/control properties of MSA with no HF download and no real checkpoint:
// the block max-pool, top-k-plus-always-on-local selection under block-level
// causality, the broadcast-to-keys step, and that the resulting sparse attention
// equals a dense softmax over exactly the admitted keys. They are the MSA analogue
// of the GLM-DSA witnesses in dsa_index_test.go. The shared helpers
// denseMaskedAttentionReference / assert3DClose / clone3D / float64Close live there.

func TestMSABlockScoresMaxPoolsPerKeyScores(t *testing.T) {
	keyScores := []float64{1, 5, 2, 9, 3}
	keyPositions := []int{0, 1, 2, 3, 4}

	// Full causal range (queryPos=4): block0={pos0,1}=max(1,5)=5,
	// block1={pos2,3}=max(2,9)=9, block2={pos4}=max(3)=3.
	scores, ok := msaBlockScores(keyScores, keyPositions, 4, 2)
	if !ok {
		t.Fatal("msaBlockScores failed")
	}
	if want := map[int]float64{0: 5, 1: 9, 2: 3}; !reflect.DeepEqual(scores, want) {
		t.Fatalf("block scores = %v, want %v", scores, want)
	}

	// Causal pooling at queryPos=2: only pos0,1,2 contribute, so block0=max(1,5)=5,
	// block1={pos2}=2; the future pos3/pos4 do not raise any block score.
	causal, ok := msaBlockScores(keyScores, keyPositions, 2, 2)
	if !ok {
		t.Fatal("msaBlockScores (causal) failed")
	}
	if want := map[int]float64{0: 5, 1: 2}; !reflect.DeepEqual(causal, want) {
		t.Fatalf("causal block scores = %v, want %v", causal, want)
	}
}

func TestMSASelectBlocksTopKPlusLocal(t *testing.T) {
	scores := map[int]float64{0: 5, 1: 9, 2: 3}

	// top-1 by score is block1 (9); the always-on local-1 window is the most-recent
	// candidate block, block2. Union, ascending = [1,2].
	got, ok := msaSelectBlocks(scores, 4, 2, 1, 1)
	if !ok {
		t.Fatal("msaSelectBlocks failed")
	}
	if want := []int{1, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("topK=1 local=1 -> %v, want %v", got, want)
	}

	// local=0: only the single top block survives.
	if got, _ := msaSelectBlocks(scores, 4, 2, 1, 0); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("topK=1 local=0 -> %v, want [1]", got)
	}
	// topK=0: only the most-recent 2 blocks (the always-on local window).
	if got, _ := msaSelectBlocks(scores, 4, 2, 0, 2); !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("topK=0 local=2 -> %v, want [1 2]", got)
	}
	// topK over-large clamps to all candidate blocks.
	if got, _ := msaSelectBlocks(scores, 4, 2, 99, 0); !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Fatalf("topK=99 -> %v, want [0 1 2]", got)
	}
	// Ties break on the smaller block index for determinism.
	if got, _ := msaSelectBlocks(map[int]float64{0: 5, 1: 5, 2: 1}, 4, 2, 1, 0); !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("tie topK=1 -> %v, want [0]", got)
	}
}

// TestMSASelectBlocksLocalWindowAlwaysOn proves the local window is selected even
// when its block carries the LOWEST indexer score — the "always-on" guarantee that
// keeps the recent context in view regardless of what the indexer prefers.
func TestMSASelectBlocksLocalWindowAlwaysOn(t *testing.T) {
	// blockSize=2, queryPos=5: blocks 0={0,1}, 1={2,3}, 2={4,5}(the query's own block).
	// block2 has the lowest score but is the local window; block0 wins top-1; block1
	// is neither top-1 nor local and must be excluded.
	scores := map[int]float64{0: 100, 1: 50, 2: 1}
	got, ok := msaSelectBlocks(scores, 5, 2, 1, 1)
	if !ok {
		t.Fatal("msaSelectBlocks failed")
	}
	if want := []int{0, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("always-on local -> %v, want %v (low-score local block must stay, mid block must drop)", got, want)
	}
}

// TestMSABlockCausalityIgnoresFutureBlocks proves a query never selects a block that
// lies entirely in its future, even when those future blocks carry huge scores, and
// that a query's block decision is identical whether or not the future keys exist
// (prefix stability — the property a reusable block-index cache relies on).
func TestMSABlockCausalityIgnoresFutureBlocks(t *testing.T) {
	keyPositions := []int{0, 1, 2, 3, 4, 5}
	keyScores := []float64{1, 2, 900, 900, 900, 900}

	full, ok := msaBlockScores(keyScores, keyPositions, 1, 2)
	if !ok {
		t.Fatal("msaBlockScores failed")
	}
	if want := map[int]float64{0: 2}; !reflect.DeepEqual(full, want) {
		t.Fatalf("query@1 block scores = %v, want only causal block0 (future 900s ignored)", full)
	}
	blocks, ok := msaSelectBlocks(full, 1, 2, 2, 2)
	if !ok {
		t.Fatal("msaSelectBlocks failed")
	}
	if want := []int{0}; !reflect.DeepEqual(blocks, want) {
		t.Fatalf("query@1 selected %v, want only [0] (no future block selectable)", blocks)
	}

	// Same prefix, no future keys present: identical decision.
	prefix, ok := msaBlockScores(keyScores[:2], keyPositions[:2], 1, 2)
	if !ok {
		t.Fatal("msaBlockScores (prefix) failed")
	}
	if !reflect.DeepEqual(prefix, full) {
		t.Fatalf("block decision changed when suffix absent: prefix=%v full=%v", prefix, full)
	}
}

func TestMSASelectedKeyPositionsBroadcast(t *testing.T) {
	keyPositions := []int{0, 1, 2, 3, 4}

	// blockSize=2, queryPos=4. block1 -> pos2,3.
	if got, _ := msaSelectedKeyPositions(keyPositions, 4, 2, []int{1}); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Fatalf("block1 broadcast -> %v, want [2 3]", got)
	}
	// blocks 0,2 -> pos0,1 (block0) + pos4 (block2).
	if got, _ := msaSelectedKeyPositions(keyPositions, 4, 2, []int{0, 2}); !reflect.DeepEqual(got, []int{0, 1, 4}) {
		t.Fatalf("blocks0,2 broadcast -> %v, want [0 1 4]", got)
	}
	// Causality: at queryPos=2, block1's pos3 is in the future and is dropped.
	if got, _ := msaSelectedKeyPositions(keyPositions, 2, 2, []int{1}); !reflect.DeepEqual(got, []int{2}) {
		t.Fatalf("block1 broadcast@queryPos2 -> %v, want [2] (pos3 is future)", got)
	}
}

// TestMSAAttentionMatchesDenseMaskedReference proves the end-to-end MSA path
// (score -> max-pool -> select blocks -> broadcast -> sparse softmax) equals a dense
// softmax over EXACTLY the admitted keys, and that an unselected causal block has no
// effect on the output (changing its values is a no-op).
func TestMSAAttentionMatchesDenseMaskedReference(t *testing.T) {
	queryPositions := []int{1, 3}
	keyPositions := []int{0, 1, 2, 3}
	const blockSize, topK, local = 2, 1, 0
	keyScores := [][]float64{
		{3, 1, 0, 0},  // query@1: only block0 causal -> selects block0 (keys 0,1)
		{10, 1, 2, 0}, // query@3: block0=10 > block1=2 -> top-1 block0 (keys 0,1); block1 excluded
	}
	query := [][][]float64{
		{{1, 0}, {0, 1}},
		{{0.5, 1}, {1, 0.5}},
	}
	key := [][][]float64{
		{{1, 0}, {0, 1}},
		{{0, 1}, {1, 0}},
		{{1, 1}, {1, 1}},
		{{2, 2}, {2, 2}},
	}
	value := [][][]float64{
		{{1, 10}, {2, 20}},
		{{3, 30}, {4, 40}},
		{{5, 50}, {6, 60}}, // pos2 (block1) excluded for both queries
		{{7, 70}, {8, 80}}, // pos3 (block1) excluded; also future for query@1
	}

	// Recompute the per-query admitted key positions through the public helpers so the
	// reference is masked by the same selection MSA produced.
	sel := make([][]int, len(queryPositions))
	for qi := range queryPositions {
		bs, ok := msaBlockScores(keyScores[qi], keyPositions, queryPositions[qi], blockSize)
		if !ok {
			t.Fatalf("msaBlockScores q%d failed", qi)
		}
		blocks, ok := msaSelectBlocks(bs, queryPositions[qi], blockSize, topK, local)
		if !ok {
			t.Fatalf("msaSelectBlocks q%d failed", qi)
		}
		keys, ok := msaSelectedKeyPositions(keyPositions, queryPositions[qi], blockSize, blocks)
		if !ok {
			t.Fatalf("msaSelectedKeyPositions q%d failed", qi)
		}
		sel[qi] = keys
	}
	if want := [][]int{{0, 1}, {0, 1}}; !reflect.DeepEqual(sel, want) {
		t.Fatalf("MSA admitted keys = %v, want %v", sel, want)
	}

	got, ok := msaAttention(query, key, value, keyScores, queryPositions, keyPositions, blockSize, topK, local, 0.5)
	if !ok {
		t.Fatal("msaAttention failed")
	}
	want, ok := denseMaskedAttentionReference(query, key, value, queryPositions, keyPositions, sel, 0.5)
	if !ok {
		t.Fatal("dense masked reference failed")
	}
	assert3DClose(t, got, want, 1e-12)

	// Excluded-block invariance: block1 (pos2,3) is in neither query's selection, so
	// rewriting its values must not change the MSA output.
	changed := clone3D(value)
	changed[2][0] = []float64{99999, -99999}
	changed[2][1] = []float64{-99999, 99999}
	changed[3][0] = []float64{12345, 67890}
	changed[3][1] = []float64{-12345, -67890}
	gotChanged, ok := msaAttention(query, key, changed, keyScores, queryPositions, keyPositions, blockSize, topK, local, 0.5)
	if !ok {
		t.Fatal("msaAttention failed after mutating excluded block")
	}
	assert3DClose(t, gotChanged, got, 1e-12)
}

// TestMSAAttentionSupersetEqualsDenseCausal proves that when the block budget covers
// every causal block (top-k >= number of blocks), MSA degenerates to full dense
// causal attention — the honesty check that sparsity is a strict subset, not a
// different computation.
func TestMSAAttentionSupersetEqualsDenseCausal(t *testing.T) {
	queryPositions := []int{0, 1, 2, 3}
	keyPositions := []int{0, 1, 2, 3}
	const blockSize = 2
	keyScores := [][]float64{
		{1, 0, 0, 0},
		{2, 5, 0, 0},
		{6, 1, 4, 0},
		{1, 8, 3, 9},
	}
	query := [][][]float64{
		{{1, 0}},
		{{0, 1}},
		{{1, 1}},
		{{0.5, 0.5}},
	}
	key := [][][]float64{
		{{1, 0}},
		{{0, 1}},
		{{1, 1}},
		{{2, 1}},
	}
	value := [][][]float64{
		{{1, 2}},
		{{3, 4}},
		{{5, 6}},
		{{7, 8}},
	}

	// Full causal allow-set per query (every key at position <= queryPos).
	allCausal := make([][]int, len(queryPositions))
	for qi, qp := range queryPositions {
		for _, kp := range keyPositions {
			if kp <= qp {
				allCausal[qi] = append(allCausal[qi], kp)
			}
		}
	}

	got, ok := msaAttention(query, key, value, keyScores, queryPositions, keyPositions, blockSize, 99, 0, 0.5)
	if !ok {
		t.Fatal("msaAttention (superset) failed")
	}
	want, ok := denseMaskedAttentionReference(query, key, value, queryPositions, keyPositions, allCausal, 0.5)
	if !ok {
		t.Fatal("dense causal reference failed")
	}
	assert3DClose(t, got, want, 1e-12)
}

func TestMSARejectsMalformedInput(t *testing.T) {
	keyPositions := []int{0, 1}
	if _, ok := msaBlockScores([]float64{1, 2}, keyPositions, 1, 0); ok {
		t.Fatal("blockSize=0 should fail")
	}
	if _, ok := msaBlockScores([]float64{1}, keyPositions, 1, 2); ok {
		t.Fatal("mismatched score/position lengths should fail")
	}
	if _, ok := msaSelectBlocks(map[int]float64{}, 1, 2, 1, 0); ok {
		t.Fatal("empty block scores should fail")
	}
	if _, ok := msaSelectBlocks(map[int]float64{0: 1}, 1, 2, -1, 0); ok {
		t.Fatal("negative topK should fail")
	}
	if _, ok := msaSelectedKeyPositions(keyPositions, 1, 2, nil); ok {
		t.Fatal("empty block selection should fail")
	}
	// Duplicate cache position fails closed.
	if _, ok := msaSelectedKeyPositions([]int{0, 0}, 1, 2, []int{0}); ok {
		t.Fatal("duplicate cache position should fail")
	}
	// Length mismatch at the batch entry point.
	query := [][][]float64{{{1, 0}}}
	key := [][][]float64{{{1, 0}}, {{0, 1}}}
	value := [][][]float64{{{1}}, {{2}}}
	if _, ok := msaAttention(query, key, value, [][]float64{{1, 2}}, []int{0, 1}, []int{0, 1}, 2, 1, 0, 1); ok {
		t.Fatal("query/queryPositions length mismatch should fail")
	}
}
