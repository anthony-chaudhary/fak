package gateway

import (
	"fmt"
	"reflect"
	"sync"
	"testing"
)

var _ ResidencyView = (*PrefixResidencyScorer)(nil)

func prsByWorker(t *testing.T, scores []PrefixResidencyScore) map[string]PrefixResidencyScore {
	t.Helper()
	out := make(map[string]PrefixResidencyScore, len(scores))
	for _, s := range scores {
		out[s.Worker] = s
	}
	return out
}

func TestPrefixResidencyScoreSharedPrefixEvictionKeepsOtherHolderVisible(t *testing.T) {
	s := NewPrefixResidencyScorer()
	chain := []string{"h1", "h2", "h3"}
	s.ApplyStored("A", "", chain)
	s.ApplyStored("B", "", chain)
	s.ApplyRemoved("A", chain)

	if got := s.NodeCount(); got != 3 {
		t.Fatalf("shared nodes must survive while B holds them: NodeCount=%d, want 3", got)
	}
	query := []string{"h1", "h2", "h3", "h4"}
	got := prsByWorker(t, s.Score(query, []string{"A", "B", "C"}))
	if got["B"].MatchedBlocks != 3 || got["B"].Score != 0.75 {
		t.Fatalf("B = %+v, want depth 3 score 0.75", got["B"])
	}
	if got["A"].MatchedBlocks != 0 || got["A"].Score != 0 {
		t.Fatalf("A evicted its copy, still scored %+v", got["A"])
	}
	if !(got["C"].Score < got["B"].Score) {
		t.Fatalf("cold C %+v must score strictly below holder B %+v", got["C"], got["B"])
	}
	if best := RankPrefixResidencyScores(s.Score(query, []string{"C", "A", "B"}))[0].Worker; best != "B" {
		t.Fatalf("best = %s, want B", best)
	}
}

func TestPrefixResidencyScoreChainUnderAnotherWorkersBlockSurvivesItsEviction(t *testing.T) {
	s := NewPrefixResidencyScorer()
	s.ApplyStored("A", "", []string{"p10"})
	s.ApplyStored("B", "p10", []string{"p11", "p12", "p13"})
	query := []string{"p10", "p11", "p12", "p13", "p14"}

	if d := s.Depths(query); d["A"] != 1 || d["B"] != 4 {
		t.Fatalf("depths = %v, want A=1 B=4", d)
	}
	s.ApplyRemoved("A", []string{"p10"})
	d := s.Depths(query)
	if _, ok := d["A"]; ok {
		t.Fatalf("A must vanish after evicting its only block: %v", d)
	}
	if d["B"] != 4 {
		t.Fatalf("B depth = %d after A evicted the parent, want 4", d["B"])
	}
	got := prsByWorker(t, s.Score(query, []string{"B", "C"}))
	if !(got["B"].Score > got["C"].Score) || got["C"].MatchedBlocks != 0 {
		t.Fatalf("B %+v must beat cold C %+v", got["B"], got["C"])
	}
}

func TestPrefixResidencyScoreContinuationUnderOwnParentChains(t *testing.T) {
	s := NewPrefixResidencyScorer()
	s.ApplyStored("A", "", []string{"x1", "x2"})
	s.ApplyStored("A", "x2", []string{"x3"})
	if n := s.NodeCount(); n != 3 {
		t.Fatalf("NodeCount=%d, want 3 (continuation hangs under own node)", n)
	}
	if d := s.Depths([]string{"x1", "x2", "x3"}); d["A"] != 3 {
		t.Fatalf("depth = %d, want 3", d["A"])
	}
}

func TestPrefixResidencyScoreLastHolderEvictionPrunes(t *testing.T) {
	s := NewPrefixResidencyScorer()
	s.ApplyStored("A", "", []string{"a", "b", "c"})
	s.ApplyRemoved("A", []string{"c"})
	if n := s.NodeCount(); n != 2 {
		t.Fatalf("leaf eviction by its only holder: NodeCount=%d, want 2", n)
	}
	s.ApplyRemoved("A", []string{"a", "b"})
	if n := s.NodeCount(); n != 0 {
		t.Fatalf("NodeCount=%d after last holder evicted all, want 0", n)
	}
	if len(s.byHash) != 0 {
		t.Fatalf("reverse index not emptied: %v", s.byHash)
	}
}

func TestPrefixResidencyScoreMidChainEvictionTruncatesOnlyThatWorker(t *testing.T) {
	s := NewPrefixResidencyScorer()
	chain := []string{"1", "2", "3", "4"}
	s.ApplyStored("deep", "", chain)
	s.ApplyStored("holed", "", chain)
	s.ApplyStored("shallow", "", chain[:2])
	s.ApplyRemoved("holed", []string{"2"})

	want := map[string]int{"deep": 4, "holed": 1, "shallow": 2}
	if d := s.Depths(chain); !reflect.DeepEqual(d, want) {
		t.Fatalf("depths = %v, want %v", d, want)
	}
	if n := s.NodeCount(); n != 4 {
		t.Fatalf("NodeCount=%d, want 4", n)
	}

	solo := NewPrefixResidencyScorer()
	solo.ApplyStored("A", "", chain)
	solo.ApplyRemoved("A", []string{"2"})
	if n := solo.NodeCount(); n != 4 {
		t.Fatalf("interior node with held descendants must stay: NodeCount=%d", n)
	}
	if d := solo.Depths(chain); d["A"] != 1 {
		t.Fatalf("depth = %d, want 1 (stops at the hole)", d["A"])
	}
}

func TestPrefixResidencyScoreIdempotentStoreAndUnknownEvict(t *testing.T) {
	s := NewPrefixResidencyScorer()
	chain := []string{"k1", "k2"}
	s.ApplyStored("A", "", chain)
	s.ApplyStored("A", "", chain)
	if n := s.NodeCount(); n != 2 {
		t.Fatalf("duplicate store: NodeCount=%d, want 2", n)
	}
	before := s.Depths(chain)
	s.ApplyRemoved("A", []string{"nope"})
	s.ApplyRemoved("Z", chain)
	s.ApplyRemoved("", chain)
	s.ApplyStored("", "", chain)
	s.ApplyStored("A", "", nil)
	s.ApplyAllBlocksCleared("Z")
	if n := s.NodeCount(); n != 2 {
		t.Fatalf("no-op events changed NodeCount=%d", n)
	}
	if after := s.Depths(chain); !reflect.DeepEqual(before, after) {
		t.Fatalf("no-op events changed depths %v -> %v", before, after)
	}
	s.ApplyRemoved("A", []string{"k2"})
	s.ApplyRemoved("A", []string{"k2"})
	if d := s.Depths(chain); d["A"] != 1 || s.NodeCount() != 1 {
		t.Fatalf("repeated evict: depths=%v NodeCount=%d", d, s.NodeCount())
	}
}

func TestPrefixResidencyScoreWorkerResetClearsOnlyThatWorker(t *testing.T) {
	s := NewPrefixResidencyScorer()
	s.ApplyStored("A", "", []string{"r1", "r2", "r3"})
	s.ApplyStored("B", "", []string{"r1", "r2", "r4"})
	s.ApplyAllBlocksCleared("A")

	if n := s.NodeCount(); n != 3 {
		t.Fatalf("NodeCount=%d, want 3 (A's exclusive r3 pruned)", n)
	}
	if d := s.Depths([]string{"r1", "r2", "r3"}); !reflect.DeepEqual(d, map[string]int{"B": 2}) {
		t.Fatalf("depths = %v, want only B=2", d)
	}
	if d := s.Depths([]string{"r1", "r2", "r4"}); d["B"] != 3 {
		t.Fatalf("B depth = %d, want 3", d["B"])
	}
}

func TestPrefixResidencyScoreDeterministicTieBreak(t *testing.T) {
	s := NewPrefixResidencyScorer()
	chain := []string{"t1", "t2"}
	for _, w := range []string{"w3", "w1", "w2"} {
		s.ApplyStored(w, "", chain)
	}
	want := []string{"w1", "w2", "w3", "cold"}
	for _, order := range [][]string{{"w3", "cold", "w1", "w2"}, {"cold", "w2", "w3", "w1"}} {
		for i := 0; i < 5; i++ {
			ranked := RankPrefixResidencyScores(s.Score(chain, order))
			var got []string
			for _, r := range ranked {
				got = append(got, r.Worker)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("ranked = %v, want %v", got, want)
			}
		}
	}
	rw := s.ResidentWorkers(chain)
	if len(rw) != 3 || rw[0].Worker != "w1" || rw[0].Overlap != 2 {
		t.Fatalf("ResidentWorkers = %+v", rw)
	}
}

func TestPrefixResidencyScoreEmptyChainAndNoCandidates(t *testing.T) {
	s := NewPrefixResidencyScorer()
	s.ApplyStored("A", "", []string{"e1"})
	got := s.Score(nil, []string{"A", "B"})
	want := []PrefixResidencyScore{{Worker: "A"}, {Worker: "B"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("empty chain = %+v, want flat zeros", got)
	}
	if got := s.Score([]string{"e1"}, nil); len(got) != 0 {
		t.Fatalf("no candidates = %+v, want empty", got)
	}
	if got := s.ResidentWorkers(nil); got != nil {
		t.Fatalf("ResidentWorkers(nil) = %+v", got)
	}
	if got := s.Overlap("A", []string{"e1", "e2"}); got != 1 {
		t.Fatalf("Overlap = %d, want 1", got)
	}
}

func TestPrefixResidencyScoreExternalBatchVocabulary(t *testing.T) {
	s := NewPrefixResidencyScorer()
	n := s.ApplyExternalBatch(ExternalKVCacheEventBatch{Worker: "A", Events: []ExternalKVCacheEvent{
		{Type: "BlockStored", BlockHashes: []string{"b1", "b2"}},
		{Type: "unknown", BlockHashes: []string{"b9"}},
		{Type: "BlockRemoved", BlockHashes: []string{"b2"}},
		{Type: "BlockStored", BlockHashes: []string{" ", "b3"}},
		{Type: "BlockStored"},
		{Type: "BlockRemoved", BlockHashes: []string{"", "  "}},
	}})
	if n != 2 {
		t.Fatalf("applied = %d, want 2", n)
	}
	if d := s.Depths([]string{"b1", "b2"}); d["A"] != 1 {
		t.Fatalf("depth = %d, want 1", d["A"])
	}
	s.ApplyExternalBatch(ExternalKVCacheEventBatch{Worker: "A", Events: []ExternalKVCacheEvent{{Type: "AllBlocksCleared"}}})
	if s.NodeCount() != 0 {
		t.Fatalf("NodeCount=%d after clear, want 0", s.NodeCount())
	}
}

func TestPrefixResidencyScoreConcurrentApplyAndScore(t *testing.T) {
	s := NewPrefixResidencyScorer()
	s.ApplyStored("anchor", "", []string{"c0", "c1"})
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			w := fmt.Sprintf("w%d", g)
			for i := 0; i < 200; i++ {
				chain := []string{"c0", "c1", fmt.Sprintf("c2-%d", i%4)}
				s.ApplyStored(w, "", chain)
				_ = s.Score(chain, []string{"anchor", w})
				s.ApplyRemoved(w, chain[2:])
				if i%50 == 0 {
					s.ApplyAllBlocksCleared(w)
				}
			}
			s.ApplyAllBlocksCleared(w)
		}(g)
	}
	wg.Wait()
	if n := s.NodeCount(); n != 2 {
		t.Fatalf("NodeCount=%d after all workers cleared, want 2", n)
	}
	if d := s.Depths([]string{"c0", "c1"}); !reflect.DeepEqual(d, map[string]int{"anchor": 2}) {
		t.Fatalf("depths = %v, want anchor=2 only", d)
	}
}
