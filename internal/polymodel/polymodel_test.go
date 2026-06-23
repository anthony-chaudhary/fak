package polymodel

import (
	"errors"
	"testing"
)

// ---------------------------------------------------------------------------
// Residency pool.
// ---------------------------------------------------------------------------

func TestPoolAdmitWithinBudget(t *testing.T) {
	p := NewPool(100)
	for _, m := range []Model{
		{ID: "a", WeightBytes: 30},
		{ID: "b", WeightBytes: 30},
		{ID: "c", WeightBytes: 30},
	} {
		evicted, err := p.Admit(m)
		if err != nil {
			t.Fatalf("admit %s: %v", m.ID, err)
		}
		if len(evicted) != 0 {
			t.Fatalf("admit %s evicted %v under budget", m.ID, evicted)
		}
	}
	if p.Used() != 90 {
		t.Fatalf("used = %d, want 90", p.Used())
	}
	if p.Len() != 3 {
		t.Fatalf("len = %d, want 3", p.Len())
	}
}

func TestPoolLRUEvictsColdest(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 40})
	mustAdmit(t, p, Model{ID: "b", WeightBytes: 40})
	// Touch a so b becomes the coldest; admitting c (40) must evict b, not a.
	p.Touch("a")
	evicted, err := p.Admit(Model{ID: "c", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit c: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "b" {
		t.Fatalf("evicted = %v, want [b] (the coldest after touching a)", evicted)
	}
	if !p.Has("a") || !p.Has("c") || p.Has("b") {
		t.Fatalf("residency = %v, want a,c not b", p.Resident())
	}
	if p.Used() > p.Budget() {
		t.Fatalf("used %d exceeds budget %d", p.Used(), p.Budget())
	}
}

func TestPoolPinnedNeverEvicted(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "draft", WeightBytes: 60, Pinned: true})
	mustAdmit(t, p, Model{ID: "x", WeightBytes: 30})
	// A 60-byte model fits only by dropping the pinned draft → refused, unchanged.
	used, ln := p.Used(), p.Len()
	_, err := p.Admit(Model{ID: "big", WeightBytes: 60})
	if !errors.Is(err, ErrPinnedNoRoom) {
		t.Fatalf("admit big: err = %v, want ErrPinnedNoRoom", err)
	}
	if p.Used() != used || p.Len() != ln {
		t.Fatalf("pool mutated on refused admit: used %d->%d len %d->%d", used, p.Used(), ln, p.Len())
	}
	// A 40-byte model fits by evicting unpinned x only; the pinned draft survives.
	evicted, err := p.Admit(Model{ID: "ok", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit ok: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "x" {
		t.Fatalf("evicted = %v, want [x]", evicted)
	}
	if !p.Has("draft") {
		t.Fatal("pinned draft was evicted")
	}
}

func TestPoolTooLargeRefusedUnchanged(t *testing.T) {
	p := NewPool(50)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 20})
	used := p.Used()
	_, err := p.Admit(Model{ID: "huge", WeightBytes: 60})
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v, want ErrTooLarge", err)
	}
	if p.Used() != used || p.Has("huge") {
		t.Fatal("pool mutated on too-large admit")
	}
}

func TestPoolReadmitIsTouch(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 40})
	mustAdmit(t, p, Model{ID: "b", WeightBytes: 40})
	// Re-admit a (now the older of the two) → it becomes most-recent, so the next
	// eviction targets b. Used must not double-count a's bytes.
	if _, err := p.Admit(Model{ID: "a", WeightBytes: 40}); err != nil {
		t.Fatalf("re-admit a: %v", err)
	}
	if p.Used() != 80 {
		t.Fatalf("used = %d after re-admit, want 80 (no double count)", p.Used())
	}
	evicted, _ := p.Admit(Model{ID: "c", WeightBytes: 40})
	if len(evicted) != 1 || evicted[0] != "b" {
		t.Fatalf("evicted = %v, want [b] (re-admit refreshed a's recency)", evicted)
	}
}

// TestPoolBudgetNeverExceeded drives an arbitrary deterministic admit/touch
// sequence and asserts the core invariant after every step: Used() <= Budget().
func TestPoolBudgetNeverExceeded(t *testing.T) {
	p := NewPool(256)
	// A fixed pseudo-sequence (no rand import: reproducible by construction).
	sizes := []int64{50, 70, 90, 30, 110, 40, 200, 60, 20, 80, 130, 10, 170, 25, 95}
	for i, sz := range sizes {
		id := ModelID(string(rune('a' + i)))
		_, err := p.Admit(Model{ID: id, WeightBytes: sz})
		if err != nil && !errors.Is(err, ErrTooLarge) && !errors.Is(err, ErrPinnedNoRoom) {
			t.Fatalf("admit %s(%d): unexpected err %v", id, sz, err)
		}
		if p.Used() > p.Budget() {
			t.Fatalf("after admit %s(%d): used %d exceeds budget %d", id, sz, p.Used(), p.Budget())
		}
		if i%3 == 0 && p.Len() > 0 {
			p.Touch(p.Resident()[0])
		}
	}
}

// ---------------------------------------------------------------------------
// Decode lane.
// ---------------------------------------------------------------------------

func TestNextDecoderPolicy(t *testing.T) {
	reqs := []Request{
		{Model: "low", Decode: 5, Priority: 1, Seq: 1},
		{Model: "hi-late", Decode: 5, Priority: 9, Seq: 7},
		{Model: "hi-early", Decode: 5, Priority: 9, Seq: 2},
		{Model: "done", Decode: 0, Priority: 99, Seq: 0},
	}
	// Highest priority, then lowest Seq → "hi-early". "done" has no decode work.
	if got := NextDecoder(reqs, nil); reqs[got].Model != "hi-early" {
		t.Fatalf("NextDecoder = %s, want hi-early", reqs[got].Model)
	}
	// Residency filter: only "low" is warm → it is chosen despite lower priority.
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "low", WeightBytes: 10})
	if got := NextDecoder(reqs, p); reqs[got].Model != "low" {
		t.Fatalf("NextDecoder(resident=low) = %s, want low", reqs[got].Model)
	}
	// Nothing eligible → -1.
	if got := NextDecoder([]Request{{Model: "x", Decode: 0}}, nil); got != -1 {
		t.Fatalf("NextDecoder(no work) = %d, want -1", got)
	}
}

func TestScheduleSerialDecodeAndConservation(t *testing.T) {
	reqs := []Request{
		{Model: "a", Prefill: 100, Decode: 5, Priority: 5, Seq: 1},
		{Model: "b", Prefill: 200, Decode: 3, Priority: 5, Seq: 2},
	}
	steps, st := Schedule(reqs, 2)

	if st.MaxConcurrentDecode != 1 {
		t.Fatalf("MaxConcurrentDecode = %d, want 1 (the serial-lane invariant)", st.MaxConcurrentDecode)
	}
	if st.PrefillTokens != 300 || st.DecodeTokens != 8 {
		t.Fatalf("tokens prefill=%d decode=%d, want 300/8", st.PrefillTokens, st.DecodeTokens)
	}

	// Every prefill emitted exactly once; decode tokens conserved per model.
	prefillCount := map[ModelID]int{}
	decodeTokens := map[ModelID]int{}
	for _, s := range steps {
		switch s.Phase {
		case Prefill:
			prefillCount[s.Model]++
		case Decode:
			if s.Tokens <= 0 || s.Tokens > 2 {
				t.Fatalf("decode step tokens=%d, want 1..quantum(2)", s.Tokens)
			}
			decodeTokens[s.Model] += s.Tokens
		}
	}
	if prefillCount["a"] != 1 || prefillCount["b"] != 1 {
		t.Fatalf("prefill counts = %v, want one each", prefillCount)
	}
	if decodeTokens["a"] != 5 || decodeTokens["b"] != 3 {
		t.Fatalf("decode tokens = %v, want a=5 b=3", decodeTokens)
	}

	// The lane interleaves: with quantum 2, a and b alternate while both have work,
	// so the decode sub-sequence is not all-a-then-all-b.
	var decodeOrder []ModelID
	for _, s := range steps {
		if s.Phase == Decode {
			decodeOrder = append(decodeOrder, s.Model)
		}
	}
	if len(decodeOrder) < 2 || decodeOrder[0] != "a" || decodeOrder[1] != "b" {
		t.Fatalf("decode order = %v, want interleaved starting a,b", decodeOrder)
	}
}

func TestScheduleEmptyAndDefaults(t *testing.T) {
	steps, st := Schedule(nil, 0)
	if len(steps) != 0 || st.DecodeSteps != 0 || st.MaxConcurrentDecode != 0 {
		t.Fatalf("empty schedule = %v / %+v", steps, st)
	}
	// quantum<=0 defaults to 1: a 3-token decode yields 3 single-token steps.
	steps, st = Schedule([]Request{{Model: "a", Decode: 3}}, 0)
	if st.DecodeSteps != 3 {
		t.Fatalf("DecodeSteps = %d, want 3 (quantum defaulted to 1)", st.DecodeSteps)
	}
	_ = steps
}

func TestDecodeBandwidthAccounting(t *testing.T) {
	steps := []Step{
		{Model: "big", Phase: Prefill, Tokens: 1000}, // prefill is NOT bandwidth-counted
		{Model: "big", Phase: Decode, Tokens: 4},
		{Model: "small", Phase: Decode, Tokens: 10},
		{Model: "unknown", Phase: Decode, Tokens: 100}, // missing weight → 0
	}
	weights := map[ModelID]int64{"big": 1_000_000, "small": 10_000}
	got := DecodeBandwidthBytes(steps, weights)
	want := int64(4)*1_000_000 + int64(10)*10_000
	if got != want {
		t.Fatalf("bandwidth = %d, want %d", got, want)
	}
}

// ---------------------------------------------------------------------------
// Cache-led MTP: speculative accept.
// ---------------------------------------------------------------------------

func TestAcceptGreedy(t *testing.T) {
	cases := []struct {
		name          string
		draft, target []int
		accepted, adv int
		keep, evict   int
	}{
		// All 3 accepted + a bonus 4th from the verify pass → advance 4, evict 0.
		{"all+bonus", []int{1, 2, 3}, []int{1, 2, 3, 4}, 3, 4, 3, 0},
		// All 3 accepted, no bonus position → advance 3, evict 0.
		{"all-no-bonus", []int{1, 2, 3}, []int{1, 2, 3}, 3, 3, 3, 0},
		// Diverge at index 1: keep 1, correct to target → advance 2, evict 2.
		{"partial", []int{1, 9, 9}, []int{1, 2, 3}, 1, 2, 1, 2},
		// Reject at index 0 → advance 1 (the correction), evict all 3.
		{"all-rejected", []int{9, 9, 9}, []int{1, 2, 3}, 0, 1, 0, 3},
		// No draft → a plain decode: advance 1, nothing to keep/evict.
		{"empty-draft", nil, []int{7}, 0, 1, 0, 0},
	}
	for _, c := range cases {
		r := AcceptGreedy(c.draft, c.target)
		if r.Accepted != c.accepted || r.Advance != c.adv || r.KeepKV != c.keep || r.EvictKV != c.evict {
			t.Fatalf("%s: got %+v, want accepted=%d advance=%d keep=%d evict=%d",
				c.name, r, c.accepted, c.adv, c.keep, c.evict)
		}
		// Invariant: every drafted position is either kept or evicted, never lost.
		if r.KeepKV+r.EvictKV != len(c.draft) {
			t.Fatalf("%s: keep+evict=%d != draft len %d", c.name, r.KeepKV+r.EvictKV, len(c.draft))
		}
	}
}

func TestAcceptTree(t *testing.T) {
	// A LINEAR chain must reduce exactly to AcceptGreedy.
	chain := SpecTree{Nodes: []TreeNode{
		{TargetArgmax: 1, Children: []int{1}}, // root: predicts 1
		{Token: 1, TargetArgmax: 2, Children: []int{2}},
		{Token: 2, TargetArgmax: 3, Children: []int{3}},
		{Token: 3, TargetArgmax: 4},
	}}
	tr := AcceptTree(chain)
	gr := AcceptGreedy([]int{1, 2, 3}, []int{1, 2, 3, 4})
	if tr.Advance != gr.Advance || tr.KeepKV != gr.KeepKV || tr.EvictKV != gr.EvictKV {
		t.Fatalf("chain tree %+v != AcceptGreedy %+v", tr, gr)
	}
	if len(tr.Path) != 3 || tr.Path[0] != 1 || tr.Path[2] != 3 {
		t.Fatalf("chain path = %v, want [1 2 3]", tr.Path)
	}

	// A BRANCH: the accepted path descends a non-first sibling at each level.
	branch := SpecTree{Nodes: []TreeNode{
		{TargetArgmax: 5, Children: []int{1, 2}}, // root predicts 5 → node 2 (Token 5)
		{Token: 9},                               // rejected sibling
		{Token: 5, TargetArgmax: 7, Children: []int{3, 4}},
		{Token: 1},                  // rejected sibling (Token 1 != 7)
		{Token: 7, TargetArgmax: 0}, // matches 7 → accepted; predicts 0, no children → stop
	}}
	br := AcceptTree(branch)
	if len(br.Path) != 2 || br.Path[0] != 2 || br.Path[1] != 4 {
		t.Fatalf("branch path = %v, want [2 4]", br.Path)
	}
	if br.Advance != 3 || br.KeepKV != 2 || br.EvictKV != 2 {
		t.Fatalf("branch result = %+v, want advance=3 keep=2 evict=2", br)
	}

	// ALL REJECTED at the root: nothing matches the target's argmax.
	none := SpecTree{Nodes: []TreeNode{
		{TargetArgmax: 100, Children: []int{1, 2}},
		{Token: 1}, {Token: 2},
	}}
	nr := AcceptTree(none)
	if len(nr.Path) != 0 || nr.Advance != 1 || nr.KeepKV != 0 || nr.EvictKV != 2 {
		t.Fatalf("all-rejected = %+v, want path=[] advance=1 keep=0 evict=2", nr)
	}

	// Invariant across all trees: KEEP + EVICT == number of speculative nodes.
	for name, tree := range map[string]SpecTree{"chain": chain, "branch": branch, "none": none} {
		r := AcceptTree(tree)
		if r.KeepKV+r.EvictKV != len(tree.Nodes)-1 {
			t.Fatalf("%s: keep+evict=%d != speculative nodes %d", name, r.KeepKV+r.EvictKV, len(tree.Nodes)-1)
		}
	}

	// Empty tree is a no-op.
	if e := AcceptTree(SpecTree{}); e.Advance != 0 || len(e.Path) != 0 {
		t.Fatalf("empty tree = %+v, want zero", e)
	}
}

func TestPickDrafterCheapestSameFamily(t *testing.T) {
	p := NewPool(1000)
	mustAdmit(t, p, Model{ID: "target", Family: "qwen", WeightBytes: 500})
	mustAdmit(t, p, Model{ID: "mid", Family: "qwen", WeightBytes: 200})
	mustAdmit(t, p, Model{ID: "tiny", Family: "qwen", WeightBytes: 50})
	mustAdmit(t, p, Model{ID: "alien", Family: "llama", WeightBytes: 10})

	// Cheapest same-family peer (not the target itself) → "tiny".
	if d := PickDrafter("target", p); d != "tiny" {
		t.Fatalf("PickDrafter(target) = %q, want tiny", d)
	}
	// A model with no same-family peer → no drafter.
	if d := PickDrafter("alien", p); d != "" {
		t.Fatalf("PickDrafter(alien) = %q, want \"\" (no same-family peer)", d)
	}
	// Unique family ("") → never drafts.
	mustAdmit(t, p, Model{ID: "solo", Family: "", WeightBytes: 5})
	if d := PickDrafter("solo", p); d != "" {
		t.Fatalf("PickDrafter(solo) = %q, want \"\" (empty family)", d)
	}
	// Non-resident active → no drafter.
	if d := PickDrafter("ghost", p); d != "" {
		t.Fatalf("PickDrafter(ghost) = %q, want \"\"", d)
	}
}

func TestCanShare(t *testing.T) {
	base := Model{ID: "base", Family: "qwen", PrefixDigest: "sha-AAA"}
	twin := Model{ID: "twin", Family: "qwen", PrefixDigest: "sha-AAA"} // same family + weights band
	fork := Model{ID: "fork", Family: "qwen", PrefixDigest: "sha-BBB"} // same family, DIFFERENT weights
	alien := Model{ID: "alien", Family: "llama", PrefixDigest: "sha-AAA"}
	bare := Model{ID: "bare", Family: "qwen"} // no declared shareable band

	cases := []struct {
		name string
		a, b Model
		want bool
	}{
		{"identical band shares", base, twin, true},
		{"self always shares", base, base, true},
		{"different weights do NOT share (KV would differ)", base, fork, false},
		{"different family does NOT share", base, alien, false},
		{"empty digest never shares", base, bare, false},
		{"empty digest never shares (reverse)", bare, twin, false},
	}
	for _, c := range cases {
		if got := CanShare(c.a, c.b); got != c.want {
			t.Fatalf("%s: CanShare(%s,%s)=%v, want %v", c.name, c.a.ID, c.b.ID, got, c.want)
		}
	}
}

func TestEffectiveTokensPerVerify(t *testing.T) {
	const eps = 1e-9
	check := func(k int, a, want float64) {
		if got := EffectiveTokensPerVerify(k, a); got < want-eps || got > want+eps {
			t.Fatalf("E(k=%d,a=%g) = %g, want %g", k, a, got, want)
		}
	}
	check(0, 0.9, 1)    // no draft → plain decode
	check(3, 0, 1)      // never accept → 1 real token/verify
	check(2, 1, 3)      // always accept → k+1
	check(1, 0.5, 1.5)  // 1 + 0.5
	check(2, 0.5, 1.75) // 1 + 0.5 + 0.25
	check(4, 1, 5)      // k+1

	// Monotone in acceptance and in draft length (the speedup levers).
	if EffectiveTokensPerVerify(3, 0.8) <= EffectiveTokensPerVerify(3, 0.4) {
		t.Fatal("E must increase with acceptance probability")
	}
	if EffectiveTokensPerVerify(8, 0.8) <= EffectiveTokensPerVerify(3, 0.8) {
		t.Fatal("E must increase with draft length")
	}
}

func mustAdmit(t *testing.T, p *Pool, m Model) {
	t.Helper()
	if _, err := p.Admit(m); err != nil {
		t.Fatalf("admit %s: %v", m.ID, err)
	}
}
