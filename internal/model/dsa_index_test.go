package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func TestDSAIndexScoresUsesProjectedWeightsAndRelu(t *testing.T) {
	scores, ok := dsaIndexScores(
		[][][]float64{
			{
				{2, 0},
				{0, 3},
			},
		},
		[][]float64{
			{4, 0},
			{0, -2},
		},
		[][]float64{{0.25, 2}},
		0.5,
	)
	if !ok {
		t.Fatal("DSA projected index scores failed")
	}
	// key 0: 0.25 * relu(0.5 * dot([2,0],[4,0])) = 1.
	// key 1: both heads score negative/zero after ReLU, so it contributes 0.
	if !float64Close(scores[0][0], 1, 1e-12) || !float64Close(scores[0][1], 0, 1e-12) {
		t.Fatalf("projected DSA scores = %v, want [[1 0]]", scores)
	}
}

func TestDSATopKIndicesAreCausalAndPrefixReusable(t *testing.T) {
	keyPositions := []int{0, 1, 2, 3, 4}
	queryPositions := []int{0, 1, 2, 3, 4}
	// Future keys deliberately carry huge scores. A causal DSA indexer must still
	// ignore them, so prefix decisions for queries 0..2 are identical whether the
	// suffix keys 3..4 exist or not.
	fullScores := [][]float64{
		{1, 900, 800, 700, 600},
		{2, 5, 800, 700, 600},
		{6, 1, 4, 700, 600},
		{1, 8, 3, 9, 600},
		{7, 5, 4, 3, 2},
	}
	full, ok := dsaTopKIndices(fullScores, queryPositions, keyPositions, 2)
	if !ok {
		t.Fatal("full DSA top-k failed")
	}
	for qi, row := range full {
		for _, key := range row {
			if key > queryPositions[qi] {
				t.Fatalf("query %d selected future key %d from %v", qi, key, row)
			}
		}
	}

	prefixScores := [][]float64{
		fullScores[0][:3],
		fullScores[1][:3],
		fullScores[2][:3],
	}
	prefix, ok := dsaTopKIndices(prefixScores, queryPositions[:3], keyPositions[:3], 2)
	if !ok {
		t.Fatal("prefix DSA top-k failed")
	}
	if !reflect.DeepEqual(full[:3], prefix) {
		t.Fatalf("same-prefix DSA decisions changed after suffix was present:\nfull prefix=%v\nprefix=%v",
			full[:3], prefix)
	}
	if got, want := prefix, [][]int{{0}, {1, 0}, {0, 2}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected causal top-k decisions = %v, want %v", got, want)
	}
}

func TestDSASparseAttentionMatchesDenseMaskedReference(t *testing.T) {
	queryPositions := []int{0, 1, 2}
	keyPositions := []int{0, 1, 2, 3}
	indexScores, ok := dsaIndexScores(
		[][][]float64{
			{{1, 0}, {0, 1}},
			{{0, 1}, {1, 1}},
			{{1, 1}, {1, 0}},
		},
		[][]float64{
			{1, 0},
			{0, 1},
			{1, 1},
			{9, 9},
		},
		[][]float64{
			{1, 1},
			{1, 1},
			{1, 1},
		},
		1,
	)
	if !ok {
		t.Fatal("DSA projected index scores failed")
	}
	topK, ok := dsaTopKIndices(indexScores, queryPositions, keyPositions, 2)
	if !ok {
		t.Fatal("DSA top-k failed")
	}
	if want := [][]int{{0}, {1, 0}, {2, 0}}; !reflect.DeepEqual(topK, want) {
		t.Fatalf("top-k = %v, want %v", topK, want)
	}

	query := [][][]float64{
		{{1, 0}, {0, 1}},
		{{0.5, 1}, {1, 0.5}},
		{{1, 1}, {0.25, 1}},
	}
	key := [][][]float64{
		{{1, 0}, {0, 1}},
		{{0, 1}, {1, 0}},
		{{1, 1}, {1, 1}},
		{{5, 5}, {5, 5}},
	}
	value := [][][]float64{
		{{1, 10}, {2, 20}},
		{{3, 30}, {4, 40}},
		{{5, 50}, {6, 60}},
		{{1000, 1000}, {2000, 2000}},
	}
	got, ok := dsaSparseAttention(query, key, value, queryPositions, keyPositions, topK, 0.5)
	if !ok {
		t.Fatal("DSA sparse attention failed")
	}
	want, ok := denseMaskedAttentionReference(query, key, value, queryPositions, keyPositions, topK, 0.5)
	if !ok {
		t.Fatal("dense masked attention reference failed")
	}
	assert3DClose(t, got, want, 1e-12)

	changedFuture := clone3D(value)
	changedFuture[3][0] = []float64{999999, -999999}
	changedFuture[3][1] = []float64{-999999, 999999}
	gotAfterFutureChange, ok := dsaSparseAttention(query, key, changedFuture, queryPositions, keyPositions, topK, 0.5)
	if !ok {
		t.Fatal("DSA sparse attention failed after changing an excluded future value")
	}
	assert3DClose(t, gotAfterFutureChange, got, 1e-12)
}

func TestDSASparseAttentionRejectsInvalidSelections(t *testing.T) {
	query := [][][]float64{{{1, 0}}}
	key := [][][]float64{{{1, 0}}, {{0, 1}}}
	value := [][][]float64{{{1}}, {{2}}}
	if _, ok := dsaSparseAttention(query, key, value, []int{0}, []int{0, 1}, [][]int{{1}}, 1); ok {
		t.Fatal("non-causal DSA selection should fail")
	}
	if _, ok := dsaSparseAttention(query, key, value, []int{1}, []int{0, 1}, [][]int{{2}}, 1); ok {
		t.Fatal("unknown DSA key position should fail")
	}
	if _, ok := dsaSparseAttention(query, key, value, []int{1}, []int{0, 1}, [][]int{{0, 0}}, 1); ok {
		t.Fatal("duplicate DSA key selection should fail")
	}
}

func TestDSAIndexShareReusesPreviousFullIndexer(t *testing.T) {
	first, ok := dsaTopKIndices(
		[][]float64{{3, 99}, {1, 4}},
		[]int{0, 1},
		[]int{0, 1},
		2,
	)
	if !ok {
		t.Fatal("first full indexer failed")
	}
	second, ok := dsaTopKIndices(
		[][]float64{{9, 99}, {8, 7}},
		[]int{0, 1},
		[]int{0, 1},
		2,
	)
	if !ok {
		t.Fatal("second full indexer failed")
	}

	shared, ok := dsaIndexShare(
		[]string{"full", "shared", "shared", "shared", "full", "shared"},
		map[int][][]int{0: first, 4: second},
	)
	if !ok {
		t.Fatal("IndexShare expansion failed")
	}
	for _, layer := range []int{0, 1, 2, 3} {
		if !reflect.DeepEqual(shared[layer], first) {
			t.Fatalf("layer %d did not reuse first full indexer: got %v want %v", layer, shared[layer], first)
		}
	}
	for _, layer := range []int{4, 5} {
		if !reflect.DeepEqual(shared[layer], second) {
			t.Fatalf("layer %d did not reuse second full indexer: got %v want %v", layer, shared[layer], second)
		}
	}

	shared[1][0][0] = 99
	if shared[0][0][0] == 99 || first[0][0] == 99 {
		t.Fatalf("IndexShare output aliases mutable decisions: shared=%v first=%v", shared, first)
	}
}

func TestDSAIndexShareRejectsMissingFullIndexer(t *testing.T) {
	if _, ok := dsaIndexShare([]string{"shared"}, nil); ok {
		t.Fatal("shared layer without previous full indexer should fail")
	}
	if _, ok := dsaIndexShare([]string{"full"}, nil); ok {
		t.Fatal("full layer without supplied decisions should fail")
	}
	if _, ok := dsaIndexShare([]string{"full", "bogus"}, map[int][][]int{0: {{0}}}); ok {
		t.Fatal("unknown indexer layer type should fail")
	}
}

func TestDSAIndexDecisionFeedsAttentionIndexCachemeta(t *testing.T) {
	decision, ok := dsaTopKIndices(
		[][]float64{
			{1, 9, 8, 7},
			{2, 5, 8, 7},
			{6, 1, 4, 7},
			{1, 8, 3, 9},
		},
		[]int{0, 1, 2, 3},
		[]int{0, 1, 2, 3},
		2,
	)
	if !ok {
		t.Fatal("DSA top-k failed")
	}
	kv := cachemeta.FromKVPrefix(cachemeta.KVPrefix{
		Tokens:      []int{10, 20, 30, 40},
		ModelID:     "glm-5.2",
		TokenizerID: "glm-tokenizer",
		Owner:       "radixkv",
	})
	entry := cachemeta.FromAttentionIndex(cachemeta.AttentionIndex{
		Tokens:           []int{10, 20, 30, 40},
		ModelID:          "glm-5.2",
		TokenizerID:      "glm-tokenizer",
		IndexerID:        "glm52-dsa-indexer:v1",
		LayerGroup:       "layers-0-3",
		Layers:           []int{0, 1, 2, 3},
		DecisionDigest:   dsaIndexDigest(decision),
		ParentKV:         kv.ID,
		Owner:            "glm-dsa",
		Causal:           true,
		CausalityWitness: "synthetic-dsa-causal-topk",
	})
	req := cachemeta.AttentionIndexRequest{
		Tokens:         []int{10, 20, 30, 40},
		ModelID:        "glm-5.2",
		TokenizerID:    "glm-tokenizer",
		IndexerID:      "glm52-dsa-indexer:v1",
		LayerGroup:     "layers-0-3",
		DecisionDigest: dsaIndexDigest(decision),
		ParentKV:       kv.ID,
	}
	v := cachemeta.AttentionIndexLookup(req, cachemeta.AttentionIndex{
		Tokens:         []int{10, 20, 30, 40},
		ModelID:        "glm-5.2",
		TokenizerID:    "glm-tokenizer",
		IndexerID:      "glm52-dsa-indexer:v1",
		LayerGroup:     "layers-0-3",
		Layers:         []int{0, 1, 2, 3},
		DecisionDigest: dsaIndexDigest(decision),
		ParentKV:       kv.ID,
		Owner:          "glm-dsa",
		Causal:         true,
	})
	if v.Kind != cachemeta.LookupHit || !cachemeta.AttentionIndexReferences(entry, kv.ID) {
		t.Fatalf("DSA decision did not produce reusable cachemeta index: verdict=%+v entry=%+v", v, entry)
	}
}

func denseMaskedAttentionReference(query, key, value [][][]float64, queryPositions, keyPositions []int, topK [][]int, scale float64) ([][][]float64, bool) {
	if len(query) == 0 || len(query) != len(queryPositions) || len(key) != len(value) || len(key) != len(keyPositions) || len(topK) != len(query) {
		return nil, false
	}
	out := make([][][]float64, len(query))
	for qi := range query {
		allowed := map[int]struct{}{}
		for _, pos := range topK[qi] {
			allowed[pos] = struct{}{}
		}
		out[qi] = make([][]float64, len(query[qi]))
		for h := range query[qi] {
			scores := make([]float64, 0, len(key))
			keyIdx := make([]int, 0, len(key))
			maxScore := math.Inf(-1)
			for ki, pos := range keyPositions {
				if pos > queryPositions[qi] {
					continue
				}
				if _, ok := allowed[pos]; !ok {
					continue
				}
				score := dot64(query[qi][h], key[ki][h]) * scale
				scores = append(scores, score)
				keyIdx = append(keyIdx, ki)
				if score > maxScore {
					maxScore = score
				}
			}
			if len(scores) == 0 {
				return nil, false
			}
			var denom float64
			for i := range scores {
				scores[i] = math.Exp(scores[i] - maxScore)
				denom += scores[i]
			}
			out[qi][h] = make([]float64, len(value[keyIdx[0]][h]))
			for i, ki := range keyIdx {
				w := scores[i] / denom
				for d := range out[qi][h] {
					out[qi][h][d] += w * value[ki][h][d]
				}
			}
		}
	}
	return out, true
}

func clone3D(in [][][]float64) [][][]float64 {
	out := make([][][]float64, len(in))
	for i := range in {
		out[i] = make([][]float64, len(in[i]))
		for j := range in[i] {
			out[i][j] = append([]float64(nil), in[i][j]...)
		}
	}
	return out
}

func assert3DClose(t *testing.T, got, want [][][]float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("rank-3 length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("rank-3[%d] length = %d, want %d", i, len(got[i]), len(want[i]))
		}
		for j := range got[i] {
			if len(got[i][j]) != len(want[i][j]) {
				t.Fatalf("rank-3[%d][%d] length = %d, want %d", i, j, len(got[i][j]), len(want[i][j]))
			}
			for k := range got[i][j] {
				if !float64Close(got[i][j][k], want[i][j][k], tol) {
					t.Fatalf("value[%d][%d][%d] = %.17g, want %.17g\n got=%v\nwant=%v",
						i, j, k, got[i][j][k], want[i][j][k], got, want)
				}
			}
		}
	}
}

func float64Close(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}
