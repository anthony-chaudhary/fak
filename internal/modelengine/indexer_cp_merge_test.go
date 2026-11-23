package modelengine

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// indexerCPOracle is the unsharded reference: k rounds of a linear scan that
// picks the highest unpicked score, preferring the lower block id on a tie.
func indexerCPOracle(scores []float32, k int) []IndexerCandidate {
	picked := make([]bool, len(scores))
	out := []IndexerCandidate{}
	for len(out) < k {
		best := -1
		for id, s := range scores {
			if picked[id] {
				continue
			}
			if best == -1 || s > scores[best] {
				best = id
			}
		}
		if best == -1 {
			break
		}
		picked[best] = true
		out = append(out, IndexerCandidate{BlockID: best, Score: scores[best]})
	}
	return out
}

func indexerCPFixtures() map[string][]float32 {
	rng := rand.New(rand.NewSource(7))
	tieHeavy := make([]float32, 37)
	for i := range tieHeavy {
		tieHeavy[i] = float32(rng.Intn(4)) - 1.5
	}
	wide := make([]float32, 101)
	for i := range wide {
		wide[i] = float32(rng.NormFloat64())
	}
	return map[string][]float32{
		"empty":  {},
		"single": {0.25},
		// Equal scores straddle every shard boundary for 2, 3 and 4 ranks.
		"cross_rank_ties": {1, 5, 5, 2, 5, 5, 0, 5, 5, 3, 5, 5},
		"all_equal":       {2, 2, 2, 2, 2, 2, 2},
		"infinities":      {float32(math.Inf(-1)), 0, float32(math.Inf(1)), -1, float32(math.Inf(1)), float32(math.Inf(-1))},
		"signed_zero":     {float32(math.Copysign(0, -1)), 0, -1, float32(math.Copysign(0, -1))},
		"tie_heavy":       tieHeavy,
		"wide":            wide,
	}
}

func TestIndexerCPMergeEqualsSingleRankTopK(t *testing.T) {
	for name, scores := range indexerCPFixtures() {
		n := len(scores)
		for _, k := range []int{0, 1, 3, 16, n, n + 5} {
			want := indexerCPOracle(scores, k)
			if len(want) != min(k, n) {
				t.Fatalf("%s k=%d: oracle returned %d candidates", name, k, len(want))
			}
			// ranks beyond n leave trailing ranks with empty shards.
			for _, ranks := range []int{1, 2, 3, 4, 5, 8, n + 3} {
				got, err := IndexerCPTopK(scores, ranks, k)
				if err != nil {
					t.Fatalf("%s ranks=%d k=%d: %v", name, ranks, k, err)
				}
				if !indexerCPSameCandidates(got, want) {
					t.Fatalf("%s ranks=%d k=%d:\n got  %v\n want %v", name, ranks, k, got, want)
				}
			}
		}
	}
}

// indexerCPSameCandidates compares bit patterns so -0 and +0 are not conflated.
func indexerCPSameCandidates(a, b []IndexerCandidate) bool {
	return slices.EqualFunc(a, b, func(x, y IndexerCandidate) bool {
		return x.BlockID == y.BlockID && math.Float32bits(x.Score) == math.Float32bits(y.Score)
	})
}

func TestIndexerCPMergePartitionCoversBlocksOnce(t *testing.T) {
	for _, tc := range []struct{ blocks, ranks int }{{0, 1}, {0, 4}, {3, 8}, {10, 4}, {12, 4}, {13, 8}, {1, 1}} {
		shards, err := IndexerCPPartition(tc.blocks, tc.ranks)
		if err != nil {
			t.Fatalf("%+v: %v", tc, err)
		}
		if len(shards) != tc.ranks {
			t.Fatalf("%+v: %d shards", tc, len(shards))
		}
		next, minSize, maxSize := 0, tc.blocks+1, -1
		for r, s := range shards {
			if s.Rank != r || s.Start != next || s.End < s.Start {
				t.Fatalf("%+v: shard %d = %+v, want start %d", tc, r, s, next)
			}
			size := s.End - s.Start
			minSize, maxSize = min(minSize, size), max(maxSize, size)
			next = s.End
		}
		if next != tc.blocks {
			t.Fatalf("%+v: shards end at %d", tc, next)
		}
		if maxSize-minSize > 1 {
			t.Fatalf("%+v: shard sizes span %d..%d", tc, minSize, maxSize)
		}
	}
}

func TestIndexerCPMergeIsDeterministicAcrossRankAndCandidateOrder(t *testing.T) {
	scores := indexerCPFixtures()["cross_rank_ties"]
	const k = 6
	shards, err := IndexerCPPartition(len(scores), 4)
	if err != nil {
		t.Fatal(err)
	}
	perRank := make([][]IndexerCandidate, len(shards))
	for i, s := range shards {
		if perRank[i], err = IndexerCPLocalTopK(scores, s, k); err != nil {
			t.Fatal(err)
		}
	}
	want := indexerCPOracle(scores, k)
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 50; trial++ {
		shuffled := make([][]IndexerCandidate, len(perRank))
		for i, p := range rng.Perm(len(perRank)) {
			shuffled[i] = slices.Clone(perRank[p])
			rng.Shuffle(len(shuffled[i]), func(a, b int) { shuffled[i][a], shuffled[i][b] = shuffled[i][b], shuffled[i][a] })
		}
		got, err := IndexerCPMerge(shuffled, k)
		if err != nil {
			t.Fatal(err)
		}
		if !indexerCPSameCandidates(got, want) {
			t.Fatalf("trial %d: got %v want %v", trial, got, want)
		}
	}
}

func TestIndexerCPMergeTieBreakPrefersLowerBlockAcrossRanks(t *testing.T) {
	perRank := [][]IndexerCandidate{
		{{BlockID: 9, Score: 4}, {BlockID: 8, Score: 4}},
		{},
		{{BlockID: 2, Score: 4}, {BlockID: 30, Score: 7}},
		{{BlockID: 5, Score: 4}},
	}
	got, err := IndexerCPMerge(perRank, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []IndexerCandidate{{BlockID: 30, Score: 7}, {BlockID: 2, Score: 4}, {BlockID: 5, Score: 4}}
	if !indexerCPSameCandidates(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestIndexerCPMergeAllRanksEmpty(t *testing.T) {
	got, err := IndexerCPMerge([][]IndexerCandidate{{}, nil, {}}, 4)
	if err != nil || len(got) != 0 {
		t.Fatalf("got %v, %v; want empty", got, err)
	}
}

func TestIndexerCPMergeSingleRankIsUntouched(t *testing.T) {
	// Deliberately not in merge order: the non-CP path must not re-sort.
	only := []IndexerCandidate{{BlockID: 7, Score: 1}, {BlockID: 3, Score: 9}, {BlockID: 7, Score: 1}}
	got, err := IndexerCPMerge([][]IndexerCandidate{only}, 16)
	if err != nil {
		t.Fatal(err)
	}
	if !indexerCPSameCandidates(got, only) || &got[0] != &only[0] {
		t.Fatalf("single rank altered: got %v want %v", got, only)
	}
	got, err = IndexerCPMerge([][]IndexerCandidate{only}, 2)
	if err != nil || !indexerCPSameCandidates(got, only[:2]) {
		t.Fatalf("single rank truncation: got %v, %v", got, err)
	}
}

func TestIndexerCPMergeRejectsUnorderableInput(t *testing.T) {
	nan := float32(math.NaN())
	if _, err := IndexerCPTopK([]float32{1, nan, 2}, 2, 2); !errors.Is(err, errIndexerCPNaN) {
		t.Fatalf("NaN score in scan: err = %v", err)
	}
	if _, err := IndexerCPMerge([][]IndexerCandidate{{{BlockID: 0, Score: 1}}, {{BlockID: 1, Score: nan}}}, 2); !errors.Is(err, errIndexerCPNaN) {
		t.Fatalf("NaN score in merge: err = %v", err)
	}
	if _, err := IndexerCPMerge([][]IndexerCandidate{{{BlockID: 4, Score: 1}}, {{BlockID: 4, Score: 1}}}, 2); err == nil {
		t.Fatal("block nominated by two ranks was accepted")
	}
	if _, err := IndexerCPMerge(nil, 1); !errors.Is(err, errIndexerCPRanks) {
		t.Fatalf("zero ranks: err = %v", err)
	}
	if _, err := IndexerCPTopK([]float32{1}, 0, 1); !errors.Is(err, errIndexerCPRanks) {
		t.Fatalf("zero ranks in pipeline: err = %v", err)
	}
	if _, err := IndexerCPTopK([]float32{1}, 2, -1); !errors.Is(err, errIndexerCPNegativeK) {
		t.Fatalf("negative k: err = %v", err)
	}
	if _, err := IndexerCPLocalTopK([]float32{1, 2}, IndexerCPShard{Start: 1, End: 3}, 1); err == nil {
		t.Fatal("shard past the score slice was accepted")
	}
}
