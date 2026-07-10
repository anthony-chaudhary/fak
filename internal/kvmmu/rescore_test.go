package kvmmu_test

import (
	"math"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// rescore_test.go — acceptance for issue #2626 (cheap query re-score over Kraw).
//
// Two gates ship here:
//
//  1. THE ORACLE (TestReScoreOracleTopKVsFullReAttend): the CacheBlend question — does
//     the cheap layer-0 re-score's top-k agree with a FULL re-attend's top-k? The full
//     re-attend is the expensive ground truth: re-prefill the same candidates in a
//     fresh session, prefill the probe with the rung-1 attention observer installed,
//     and rank candidates by witnessed all-layer post-softmax mass (Tier 2b). The
//     cheap path must rank with one layer-0 QK^T read over re-rotated Kraw (Tier 2a).
//     This test ships REGARDLESS of the answer — its assertions encode the honestly
//     observed agreement on this fixture, not a hoped-for one.
//
//  2. NO MUTATION (TestReScoreLeavesCacheBitIdentical): re-rotation writes to scratch;
//     the cached Kraw/K/V bytes and the position count are bit-identical before/after.
//
// The fixture is the same deterministic fixed-seed synthetic model the other bridge
// witnesses use (synthCfg — 2 layers, 4 heads, GQA 2 KV heads), so the observed
// rankings are stable across runs and platforms.

// reScoreFixtureSpans are the candidate spans appended to the session cache: four
// spans with distinct token content, so a probe repeating one span's token has an
// unambiguous lexical target.
var reScoreFixtureSpans = []struct {
	id     string
	tokens []int
}{
	{"A", []int{7, 7, 7, 7}},
	{"B", []int{13, 13, 13, 13}},
	{"C", []int{21, 21, 21, 21}},
	{"D", []int{5, 29, 5, 29}},
}

// reScoreProbe repeats span B's token — the probe whose relevance target is known.
var reScoreProbe = []int{13, 13, 13}

// newReScoreCtx builds a fresh synthetic session + kvmmu context with the fixture
// candidate spans appended (KV-resident), returning both handles.
func newReScoreCtx(t *testing.T) (*model.Model, *model.Session, *kvmmu.Context) {
	t.Helper()
	m := model.NewSynthetic(synthCfg())
	s := m.NewSession()
	c := kvmmu.New(s)
	for _, sp := range reScoreFixtureSpans {
		c.Append(sp.id, "tool"+sp.id, sp.tokens)
	}
	return m, s, c
}

// candidateIDs returns the fixture span ids in append order.
func candidateIDs() []string {
	ids := make([]string, len(reScoreFixtureSpans))
	for i, sp := range reScoreFixtureSpans {
		ids[i] = sp.id
	}
	return ids
}

// rankDesc returns ids sorted by descending score (ties broken by id for determinism).
func rankDesc(scores map[string]float64) []string {
	ids := make([]string, 0, len(scores))
	for id := range scores {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		if scores[ids[i]] != scores[ids[j]] {
			return scores[ids[i]] > scores[ids[j]]
		}
		return ids[i] < ids[j]
	})
	return ids
}

// fullReAttendRank runs the Tier-2b ground truth: a FRESH session over the same model,
// the same candidate spans re-prefilled (same order, positions [0,T) — the same layout
// the cheap path scores at), then the probe prefilled with the attention observer
// installed. Candidates are ranked by the witnessed all-layer post-softmax mass the
// probe turn placed on each span.
func fullReAttendRank(t *testing.T, m *model.Model) map[string]float64 {
	t.Helper()
	s2 := m.NewSession()
	c2 := kvmmu.New(s2)
	for _, sp := range reScoreFixtureSpans {
		c2.Append(sp.id, "tool"+sp.id, sp.tokens)
	}
	// Observe ONLY the probe's forward pass: the candidates' own build-up attention is
	// the old-query mass this issue exists to distrust.
	m.SetAttnObserver(c2.AttentionObserver())
	defer m.SetAttnObserver(nil)
	c2.Append("probe", "probe", reScoreProbe)

	mass := make(map[string]float64)
	for _, seg := range c2.Segments() {
		if seg.ID == "probe" {
			continue // the probe's self-attention is not a candidate score
		}
		mass[seg.ID] = seg.Attended
	}
	return mass
}

// TestReScoreOracleTopKVsFullReAttend is the shipped oracle: cheap layer-0 re-score
// top-k vs full re-attend top-k on the deterministic fixture.
func TestReScoreOracleTopKVsFullReAttend(t *testing.T) {
	m, _, c := newReScoreCtx(t)

	cheap, err := c.ReScore(reScoreProbe, candidateIDs())
	if err != nil {
		t.Fatalf("ReScore: %v", err)
	}
	cheapScores := make(map[string]float64, len(cheap))
	var sum float64
	for _, sc := range cheap {
		cheapScores[sc.ID] = sc.Score
		sum += sc.Score
		if math.IsNaN(sc.Score) || math.IsInf(sc.Score, 0) || sc.Score < 0 {
			t.Fatalf("ReScore %s = %v, want finite non-negative", sc.ID, sc.Score)
		}
	}
	// Share semantics: softmax over the candidate keys only, so the spans partition
	// the probe's candidate-directed mass.
	if d := math.Abs(sum - 1.0); d > 1e-6 {
		t.Errorf("Σ cheap scores = %v, want 1.0 (Δ=%v)", sum, d)
	}

	full := fullReAttendRank(t, m)

	cheapRank := rankDesc(cheapScores)
	fullRank := rankDesc(full)
	t.Logf("cheap (layer-0 Kraw re-score): %v  scores=%v", cheapRank, cheapScores)
	t.Logf("full  (re-attend, all layers): %v  mass=%v", fullRank, full)

	// THE ORACLE'S ADJUDICATED ANSWER on this fixture (shipped, not hidden — the
	// CacheBlend question): the cheap layer-0 re-score recovers the full re-attend's
	// top-2 SET exactly (observed 2/2 overlap), but the 1-vs-2 ORDER of the two
	// near-tied leaders FLIPS between tiers (observed: cheap [D C A B] with D and C
	// 0.05% apart, full [C D A B] with C and D 2% apart). Both tiers agree on the
	// bottom of the ranking. So the honest mode label, carried on the ReScore API
	// docs: Tier 2a is a NARROWING signal — trust the top-k SET, not the total
	// order; ordering WITHIN the selected subset needs the full re-attend. This
	// test therefore asserts set-level agreement and deliberately does NOT assert
	// rank-level agreement (it is known not to hold). If the set assertion ever
	// fails, the oracle is doing its job — re-adjudicate the mode label, do not
	// silence the test.
	top2 := func(r []string) map[string]bool { return map[string]bool{r[0]: true, r[1]: true} }
	ct, ft := top2(cheapRank), top2(fullRank)
	overlap := 0
	for id := range ct {
		if ft[id] {
			overlap++
		}
	}
	if overlap != 2 {
		t.Errorf("top-2 set agreement lost: cheap %v vs full %v (overlap %d/2) — the layer-0 narrowing claim no longer holds on this fixture; re-adjudicate the ReScore mode label", cheapRank[:2], fullRank[:2], overlap)
	}
	if cheapRank[0] != fullRank[0] {
		t.Logf("top-1 order differs (known, near-tied): cheap %q vs full %q — set-level agreement is the shipped claim", cheapRank[0], fullRank[0])
	}
}

// TestReScoreLeavesCacheBitIdentical is the no-mutation acceptance: a re-score call
// leaves every cached Kraw/K/V byte and the position count exactly as it found them
// (re-rotation targets scratch, never the cache).
func TestReScoreLeavesCacheBitIdentical(t *testing.T) {
	_, s, c := newReScoreCtx(t)

	snap := func() [][]uint32 {
		var out [][]uint32
		for _, planes := range [][][]float32{s.Cache.K, s.Cache.Kraw, s.Cache.V} {
			for _, rows := range planes {
				bits := make([]uint32, len(rows))
				for i, v := range rows {
					bits[i] = math.Float32bits(v)
				}
				out = append(out, bits)
			}
		}
		return out
	}
	before := snap()
	lenBefore := s.Cache.Len()

	if _, err := c.ReScore(reScoreProbe, candidateIDs()); err != nil {
		t.Fatalf("ReScore: %v", err)
	}

	after := snap()
	if s.Cache.Len() != lenBefore {
		t.Fatalf("cache Len changed: %d -> %d", lenBefore, s.Cache.Len())
	}
	if len(before) != len(after) {
		t.Fatalf("cache plane count changed: %d -> %d", len(before), len(after))
	}
	for p := range before {
		if len(before[p]) != len(after[p]) {
			t.Fatalf("plane %d length changed: %d -> %d", p, len(before[p]), len(after[p]))
		}
		for i := range before[p] {
			if before[p][i] != after[p][i] {
				t.Fatalf("plane %d word %d mutated: %08x -> %08x — re-score wrote into the cache", p, i, before[p][i], after[p][i])
			}
		}
	}
}

// TestReScoreFailsClosedOnBadCandidates: an unknown id and an evicted id each fail
// the whole call (no silent survivor-subset ranking).
func TestReScoreFailsClosedOnBadCandidates(t *testing.T) {
	_, _, c := newReScoreCtx(t)

	if _, err := c.ReScore(reScoreProbe, []string{"A", "nope"}); err == nil {
		t.Fatalf("ReScore with unknown candidate id: want error, got nil")
	}
	if _, ok := c.Quarantine("C"); !ok {
		t.Fatalf("Quarantine(C) failed to evict")
	}
	if _, err := c.ReScore(reScoreProbe, []string{"A", "C"}); err == nil {
		t.Fatalf("ReScore with evicted candidate: want error, got nil")
	}
	// The surviving spans still score cleanly after the eviction renumbered them.
	got, err := c.ReScore(reScoreProbe, []string{"A", "B", "D"})
	if err != nil {
		t.Fatalf("ReScore after eviction: %v", err)
	}
	var sum float64
	for _, sc := range got {
		sum += sc.Score
	}
	if d := math.Abs(sum - 1.0); d > 1e-6 {
		t.Errorf("post-eviction Σ scores = %v, want 1.0", sum)
	}
}
