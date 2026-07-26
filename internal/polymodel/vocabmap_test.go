package polymodel

// vocabmap_test.go — the heterogeneous-vocabulary (TLI) drafting witness (#4208).
//
// The load-bearing test is TestSpecDecodeVocabBridgedLossless: it proves the whole bridged
// path — context translation, constrained proposal, id translation, accept, rollback — emits
// a token stream BIT-EXACT to no-drafter greedy decode of the same target, across four
// drafter qualities, while a real bridged drafter still buys mean acceptance > 1. Both
// halves matter: a bridge that never proposes anything would pass the losslessness half
// vacuously.

import (
	"math"
	"math/rand"
	"reflect"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// A heterogeneous tokenizer pair, built so the two id spaces are genuinely disjoint.
// ---------------------------------------------------------------------------
//
// Shared surface tokens "t0".."t39" carry COMPLETELY different ids in the two vocabularies
// (target: 0..39 ascending; drafter: 100..139 REVERSED), so any test that passes by
// accidentally treating a drafter id as a target id fails loudly. On top of the intersection
// each side keeps private tokens the other cannot express:
//
//	target-only "u0".."u9"  → target ids 40..49  (the target can emit what no draft can propose)
//	drafter-only "d0".."d9" → drafter ids 200..209 (the drafter can propose what no target has)

const (
	sharedTokens   = 40
	targetOnly     = 10
	drafterOnly    = 10
	targetVocabLen = sharedTokens + targetOnly // 50
)

func testVocabs() (drafterVocab, targetVocab map[string]int) {
	drafterVocab = map[string]int{}
	targetVocab = map[string]int{}
	for i := 0; i < sharedTokens; i++ {
		s := "t" + itoa(i)
		targetVocab[s] = i                             // 0..39 ascending
		drafterVocab[s] = 100 + (sharedTokens - 1 - i) // 100..139 reversed
	}
	for j := 0; j < targetOnly; j++ {
		targetVocab["u"+itoa(j)] = sharedTokens + j // 40..49
	}
	for j := 0; j < drafterOnly; j++ {
		drafterVocab["d"+itoa(j)] = 200 + j // 200..209
	}
	return drafterVocab, targetVocab
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

// targetOracle is the deterministic target model: its argmax depends only on the last
// committed token, and it ranges over the FULL target vocabulary — so it regularly emits a
// target-only token (40..49) that no bridged draft can ever propose. That is the honest hard
// case: those positions always cost a correction round, and must still decode identically.
//
// The multiplier/increment are a FULL-PERIOD affine map mod 50 (a-1 divisible by 50's prime
// factors 2 and 5; c coprime to 50), so the target walks all 50 ids before repeating. That
// matters for the throughput half of the witness: a short-cycle oracle would park the target
// on a handful of ids and, if any were target-only, silence the drafter almost every round —
// making the acceptance bar measure the fixture's cycle rather than the bridge. Here the
// drafter meets shared runs averaging ~4 tokens (80% of the vocabulary is shared), which is
// what a real bridged drafter faces.
func targetOracle(ctx []int) int {
	if len(ctx) == 0 {
		return 0
	}
	last := ctx[len(ctx)-1]
	return (last*11 + 13) % targetVocabLen
}

// testVerifier is the Verifier contract over targetOracle: index 0 is the target's next
// token after `committed` with NO draft applied; index i>0 is its next token after
// committed + draft[:i]. len == len(draft)+1.
func testVerifier(committed, draft []int) []int {
	out := make([]int, 0, len(draft)+1)
	ctx := append([]int(nil), committed...)
	out = append(out, targetOracle(ctx))
	for i := 0; i < len(draft); i++ {
		ctx = append(ctx, draft[i])
		out = append(out, targetOracle(ctx))
	}
	return out
}

// ---------------------------------------------------------------------------
// VocabMap construction + translation.
// ---------------------------------------------------------------------------

func TestVocabMapIntersectsAndTranslatesBothWays(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)

	if got, want := m.Len(), sharedTokens; got != want {
		t.Fatalf("Len = %d, want %d (the shared surface tokens)", got, want)
	}
	// Round-trip every shared token through both directions, and check the ids really are
	// the reversed/offset ones (not an accidental identity mapping).
	for i := 0; i < sharedTokens; i++ {
		did := 100 + (sharedTokens - 1 - i)
		tid := i
		if got, ok := m.ToTarget(did); !ok || got != tid {
			t.Fatalf("ToTarget(%d) = (%d,%v), want (%d,true)", did, got, ok, tid)
		}
		if got, ok := m.ToDrafter(tid); !ok || got != did {
			t.Fatalf("ToDrafter(%d) = (%d,%v), want (%d,true)", tid, got, ok, did)
		}
	}
	// Private tokens on each side are outside the intersection.
	for j := 0; j < drafterOnly; j++ {
		if _, ok := m.ToTarget(200 + j); ok {
			t.Fatalf("ToTarget(%d) mapped a drafter-only token; want !ok", 200+j)
		}
	}
	for j := 0; j < targetOnly; j++ {
		if _, ok := m.ToDrafter(sharedTokens + j); ok {
			t.Fatalf("ToDrafter(%d) mapped a target-only token; want !ok", sharedTokens+j)
		}
	}
	// Coverage is the honest acceptance-cost predictor: 40/50 either way here.
	if got, want := m.DrafterCoverage(), float64(sharedTokens)/float64(sharedTokens+drafterOnly); math.Abs(got-want) > 1e-9 {
		t.Errorf("DrafterCoverage = %v, want %v", got, want)
	}
	if got, want := m.TargetCoverage(), float64(sharedTokens)/float64(targetVocabLen); math.Abs(got-want) > 1e-9 {
		t.Errorf("TargetCoverage = %v, want %v", got, want)
	}
	// SharedDrafterIDs is sorted and complete.
	ids := m.SharedDrafterIDs()
	if len(ids) != sharedTokens {
		t.Fatalf("SharedDrafterIDs len = %d, want %d", len(ids), sharedTokens)
	}
	for i := 1; i < len(ids); i++ {
		if ids[i-1] >= ids[i] {
			t.Fatalf("SharedDrafterIDs not sorted ascending at %d: %v", i, ids[:i+1])
		}
	}
}

func TestVocabMapEmptyAndNilAreSafe(t *testing.T) {
	if got := NewVocabMap(nil, nil).Len(); got != 0 {
		t.Errorf("NewVocabMap(nil,nil).Len() = %d, want 0", got)
	}
	dv, _ := testVocabs()
	if got := NewVocabMap(dv, map[string]int{"zz": 1}).Len(); got != 0 {
		t.Errorf("disjoint vocabularies: Len = %d, want 0", got)
	}
	// A nil *VocabMap answers every query safely — the "no bridge" case is not a panic.
	var nilMap *VocabMap
	if nilMap.Len() != 0 || nilMap.DrafterCoverage() != 0 || nilMap.TargetCoverage() != 0 {
		t.Error("nil VocabMap: want zero Len/coverage")
	}
	if _, ok := nilMap.ToTarget(1); ok {
		t.Error("nil VocabMap ToTarget: want !ok")
	}
	if got, trunc := nilMap.MapDraft([]int{1, 2}); len(got) != 0 || !trunc {
		t.Errorf("nil VocabMap MapDraft = (%v,%v), want (empty,true)", got, trunc)
	}
	if got := nilMap.BridgeDrafter(nil)([]int{1}); got != nil {
		t.Errorf("nil VocabMap BridgeDrafter proposed %v, want nil", got)
	}
}

// TestVocabMapMapDraftTruncatesAtFirstUnmappable pins the positional-integrity choice: a
// draft is verified POSITIONALLY, so an unmappable proposal truncates (keeping the prefix's
// meaning) rather than being dropped (which would silently re-index every later proposal).
func TestVocabMapMapDraftTruncatesAtFirstUnmappable(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)

	// drafter ids: 139 → t0 → target 0; 138 → t1 → target 1; 200 is drafter-only; 137 → t2.
	got, truncated := m.MapDraft([]int{139, 138, 200, 137})
	if !truncated {
		t.Error("MapDraft: truncated = false, want true (a drafter-only token is present)")
	}
	if want := []int{0, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MapDraft = %v, want %v (truncate at the unmappable token, NOT drop it)", got, want)
	}
	// A fully-mappable draft is untouched and reports no truncation.
	got, truncated = m.MapDraft([]int{139, 138, 137})
	if truncated {
		t.Error("MapDraft: truncated = true on a fully-shared draft, want false")
	}
	if want := []int{0, 1, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MapDraft = %v, want %v", got, want)
	}
	// An unmappable FIRST token yields an empty draft: that round is a plain decode.
	got, truncated = m.MapDraft([]int{200, 139})
	if len(got) != 0 || !truncated {
		t.Fatalf("MapDraft = (%v,%v), want (empty,true)", got, truncated)
	}
}

// TestVocabMapMapContextDropsUnmappable pins the OPPOSITE choice for the drafter's input:
// the context is never verified, so dropping keeps the drafter maximally informed where
// truncating would discard the most recent (most informative) history.
func TestVocabMapMapContextDropsUnmappable(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)

	// target ids: 0 → t0 → drafter 139; 45 is target-only; 1 → t1 → drafter 138.
	got, dropped := m.MapContext([]int{0, 45, 1})
	if dropped != 1 {
		t.Errorf("MapContext dropped = %d, want 1", dropped)
	}
	if want := []int{139, 138}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MapContext = %v, want %v (drop the unmappable token, keep the tail)", got, want)
	}
}

// TestVocabMapConstrainLogitsMasksToIntersection proves the distribution-level half of TLI:
// after constraining, a drafter's own argmax can never land outside the shared vocabulary,
// so it never wastes a draft slot on a proposal that would truncate to nothing.
func TestVocabMapConstrainLogitsMasksToIntersection(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)

	// A logit vector over the drafter's id space, with the GLOBAL max deliberately parked on
	// a drafter-only token (id 205) that the target has no id for.
	logits := make([]float32, 210)
	for i := range logits {
		logits[i] = float32(i % 7)
	}
	logits[205] = 1000 // the unconstrained argmax — outside the intersection
	logits[137] = 500  // the best SHARED token (drafter id 137 → "t2" → target id 2)

	masked := m.ConstrainLogits(logits)
	if want := len(logits) - sharedTokens; masked != want {
		t.Errorf("ConstrainLogits masked = %d, want %d", masked, want)
	}
	if !math.IsInf(float64(logits[205]), -1) {
		t.Errorf("logits[205] = %v, want -Inf (a drafter-only token must be masked)", logits[205])
	}
	// The constrained argmax is the best SHARED token, and it translates into the target space.
	best := 0
	for i := range logits {
		if logits[i] > logits[best] {
			best = i
		}
	}
	if best != 137 {
		t.Fatalf("constrained argmax = %d, want 137 (the best shared token)", best)
	}
	if got, ok := m.ToTarget(best); !ok || got != 2 {
		t.Fatalf("ToTarget(constrained argmax) = (%d,%v), want (2,true)", got, ok)
	}
	// Every surviving (non -Inf) entry is a shared token.
	for i := range logits {
		if !math.IsInf(float64(logits[i]), -1) {
			if _, ok := m.ToTarget(i); !ok {
				t.Fatalf("logits[%d] survived masking but is not a shared token", i)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// The per-tokenizer-pair cache.
// ---------------------------------------------------------------------------

func TestVocabCacheBuildsOncePerOrderedPair(t *testing.T) {
	dv, tv := testVocabs()
	c := NewVocabCache()
	builds := 0
	build := func() *VocabMap { builds++; return NewVocabMap(dv, tv) }

	first := c.Get("drafter-digest", "target-digest", build)
	for i := 0; i < 5; i++ {
		if got := c.Get("drafter-digest", "target-digest", build); got != first {
			t.Fatal("Get returned a different *VocabMap for the same pair; want the cached one")
		}
	}
	if builds != 1 {
		t.Errorf("build called %d times, want 1 (the intersection is built once per pair)", builds)
	}
	// The pair is ORDERED: the reverse direction is a different bridge.
	if rev := c.Get("target-digest", "drafter-digest", build); rev == first {
		t.Error("the reversed pair returned the same map; the ordered key must distinguish it")
	}
	if builds != 2 {
		t.Errorf("build called %d times after the reversed pair, want 2", builds)
	}
	if got := c.Len(); got != 2 {
		t.Errorf("cache Len = %d, want 2", got)
	}
}

func TestVocabCacheConcurrentGetsBuildOnce(t *testing.T) {
	dv, tv := testVocabs()
	c := NewVocabCache()
	var mu sync.Mutex
	builds := 0

	var wg sync.WaitGroup
	got := make([]*VocabMap, 16)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got[i] = c.Get("d", "t", func() *VocabMap {
				mu.Lock()
				builds++
				mu.Unlock()
				return NewVocabMap(dv, tv)
			})
		}(i)
	}
	wg.Wait()
	if builds != 1 {
		t.Errorf("build called %d times under concurrent Get, want 1", builds)
	}
	for i := range got {
		if got[i] != got[0] {
			t.Fatalf("concurrent Get %d returned a different map; want one shared build", i)
		}
	}
}

// ---------------------------------------------------------------------------
// Eligibility — the relaxation, and what it must NOT change.
// ---------------------------------------------------------------------------

// TestBridgeRolesRefusesCrossVocabPair is the FAILURE CLASS the issue reports, pinned as it
// stands today: a cross-vocabulary drafter is structurally refused, however good it is. This
// test documents the old behavior (BridgeRoles is unchanged and stays strict) — it is the
// "before" half of the fix, and BridgeRolesVocab below is the "after".
func TestBridgeRolesRefusesCrossVocabPair(t *testing.T) {
	p := NewPool(1000)
	mustAdmit(t, p, Model{ID: "tiny", Family: "tinyfam", WeightBytes: 10, PrefixDigest: "a"})
	mustAdmit(t, p, Model{ID: "big", Family: "bigfam", WeightBytes: 500, PrefixDigest: "b"})

	if _, err := BridgeRoles("tiny", "big", p); err != ErrNotCoResident {
		t.Fatalf("BridgeRoles(cross-vocab) err = %v, want ErrNotCoResident (the restriction #4208 lifts)", err)
	}
	// And a unique-Family target refuses every drafter outright.
	if got := PickDrafter("big", p); got != "" {
		t.Fatalf("PickDrafter(cross-vocab target) = %q, want \"\"", got)
	}
	p2 := NewPool(1000)
	mustAdmit(t, p2, Model{ID: "tiny", Family: "tinyfam", WeightBytes: 10})
	mustAdmit(t, p2, Model{ID: "uniq", Family: "", WeightBytes: 500})
	if got := PickDrafter("uniq", p2); got != "" {
		t.Fatalf("PickDrafter(Family==\"\") = %q, want \"\" (no peer may draft for it)", got)
	}
}

// TestBridgeRolesVocabAdmitsCrossVocabPair is the "after": the same pair BridgeRoles refuses
// resolves through a VocabMap, and is honestly labeled VocabBridged.
func TestBridgeRolesVocabAdmitsCrossVocabPair(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)
	p := NewPool(1000)
	mustAdmit(t, p, Model{ID: "tiny", Family: "tinyfam", WeightBytes: 10, PrefixDigest: "a"})
	mustAdmit(t, p, Model{ID: "big", Family: "bigfam", WeightBytes: 500, PrefixDigest: "b"})

	w, err := BridgeRolesVocab("tiny", "big", p, m)
	if err != nil {
		t.Fatalf("BridgeRolesVocab err = %v, want nil", err)
	}
	if w.Drafter != "tiny" || w.Verifier != "big" {
		t.Errorf("wiring roles = (%q,%q), want (tiny,big)", w.Drafter, w.Verifier)
	}
	if !w.VocabBridged {
		t.Error("VocabBridged = false, want true (the tokenizers differ)")
	}
	// No same-Family peer is warm, so the bridged drafter IS the best available pick.
	if w.PreferredDrafter != "tiny" || !w.DrafterIsPreferred {
		t.Errorf("preferred = (%q,%v), want (tiny,true) — nothing lossless is warm to prefer",
			w.PreferredDrafter, w.DrafterIsPreferred)
	}
	// The accept rule is identical: by Accept time the draft is already in the target's space.
	if got := w.Accept([]int{1, 2, 3}, []int{1, 2, 9, 4}); got.Accepted != 2 || got.EvictKV != 1 {
		t.Errorf("Accept = %+v, want Accepted=2 EvictKV=1", got)
	}
}

func TestBridgeRolesVocabSurfacesAPreferredSameFamilyPeer(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)
	p := NewPool(2000)
	mustAdmit(t, p, Model{ID: "cross", Family: "other", WeightBytes: 10, PrefixDigest: "x"})
	mustAdmit(t, p, Model{ID: "sibling", Family: "bigfam", WeightBytes: 50, PrefixDigest: "b"})
	mustAdmit(t, p, Model{ID: "big", Family: "bigfam", WeightBytes: 500, PrefixDigest: "b"})

	w, err := BridgeRolesVocab("cross", "big", p, m)
	if err != nil {
		t.Fatalf("BridgeRolesVocab err = %v, want nil", err)
	}
	// A same-Family peer is warm and needs no translation, so the bridged pick is surfaced
	// as NOT preferred rather than silently accepted.
	if w.PreferredDrafter != "sibling" || w.DrafterIsPreferred {
		t.Errorf("preferred = (%q,%v), want (sibling,false) — the lossless peer wins",
			w.PreferredDrafter, w.DrafterIsPreferred)
	}
}

func TestBridgeRolesVocabRefusals(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)
	p := NewPool(1000)
	mustAdmit(t, p, Model{ID: "tiny", Family: "tinyfam", WeightBytes: 10})
	mustAdmit(t, p, Model{ID: "big", Family: "bigfam", WeightBytes: 500})

	for _, tc := range []struct {
		name              string
		drafter, verifier ModelID
		pool              *Pool
		m                 *VocabMap
		want              error
	}{
		{"empty drafter", "", "big", p, m, ErrEmptyRole},
		{"empty verifier", "tiny", "", p, m, ErrEmptyRole},
		{"self draft", "big", "big", p, m, ErrNotCoResident},
		{"nil pool", "tiny", "big", nil, m, ErrNotCoResident},
		{"drafter evicted", "ghost", "big", p, m, ErrNotCoResident},
		{"verifier evicted", "tiny", "ghost", p, m, ErrNotCoResident},
		{"nil map", "tiny", "big", p, nil, ErrNoVocabOverlap},
		{"zero overlap", "tiny", "big", p, NewVocabMap(dv, map[string]int{"zz": 0}), ErrNoVocabOverlap},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := BridgeRolesVocab(tc.drafter, tc.verifier, tc.pool, tc.m); err != tc.want {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestPickDrafterBridgedNilBridgeEqualsPickDrafter proves the relaxation is STRICTLY
// ADDITIVE: with no bridge, the new pick is byte-identical to the old one over a randomized
// pool, so nothing that worked before can change.
func TestPickDrafterBridgedNilBridgeEqualsPickDrafter(t *testing.T) {
	rng := rand.New(rand.NewSource(4208))
	families := []string{"", "a", "b", "c"}
	for trial := 0; trial < 300; trial++ {
		p := NewPool(1 << 20)
		n := 1 + rng.Intn(6)
		for i := 0; i < n; i++ {
			mustAdmit(t, p, Model{
				ID:          ModelID("m" + itoa(i)),
				Family:      families[rng.Intn(len(families))],
				WeightBytes: int64(1 + rng.Intn(100)),
			})
		}
		for _, id := range p.Resident() {
			old := PickDrafter(id, p)
			got := PickDrafterBridged(id, p, nil)
			if old != got {
				t.Fatalf("trial %d, active %q: PickDrafterBridged(nil bridge) = %q, PickDrafter = %q",
					trial, id, got, old)
			}
		}
		// A non-resident active model picks nothing either way.
		if got := PickDrafterBridged("absent", p, func(_, _ Model) bool { return true }); got != "" {
			t.Fatalf("trial %d: non-resident active picked %q, want \"\"", trial, got)
		}
	}
}

// TestPickDrafterBridgedAdmitsWhatFamilyRefused is the eligibility half of the fix: a target
// whose Family is unique ("" — "no peer may draft for it") now gets a bridged drafter.
func TestPickDrafterBridgedAdmitsWhatFamilyRefused(t *testing.T) {
	p := NewPool(1000)
	mustAdmit(t, p, Model{ID: "tiny", Family: "tinyfam", WeightBytes: 10})
	mustAdmit(t, p, Model{ID: "mid", Family: "midfam", WeightBytes: 90})
	mustAdmit(t, p, Model{ID: "uniq", Family: "", WeightBytes: 500})

	if got := PickDrafter("uniq", p); got != "" {
		t.Fatalf("precondition: PickDrafter = %q, want \"\"", got)
	}
	all := func(_, _ Model) bool { return true }
	if got := PickDrafterBridged("uniq", p, all); got != "tiny" {
		t.Fatalf("PickDrafterBridged = %q, want \"tiny\" (bridged, and the cheapest)", got)
	}
	// The bridge predicate is honored: refusing "tiny" falls through to the next cheapest.
	notTiny := func(d, _ Model) bool { return d.ID != "tiny" }
	if got := PickDrafterBridged("uniq", p, notTiny); got != "mid" {
		t.Fatalf("PickDrafterBridged = %q, want \"mid\" (the bridge refused tiny)", got)
	}
	// A bridge that admits nothing leaves the target self-decoding.
	none := func(_, _ Model) bool { return false }
	if got := PickDrafterBridged("uniq", p, none); got != "" {
		t.Fatalf("PickDrafterBridged = %q, want \"\" (no bridge → self-decode)", got)
	}
}

// TestPickDrafterBridgedPrefersSameFamilyOverCheaperBridged pins the tier order: a bridge can
// only SUBTRACT acceptance, so a lossless same-Family drafter wins even when a bridged
// candidate is far cheaper.
func TestPickDrafterBridgedPrefersSameFamilyOverCheaperBridged(t *testing.T) {
	p := NewPool(2000)
	mustAdmit(t, p, Model{ID: "featherweight", Family: "other", WeightBytes: 1})
	mustAdmit(t, p, Model{ID: "sibling", Family: "fam", WeightBytes: 400})
	mustAdmit(t, p, Model{ID: "target", Family: "fam", WeightBytes: 900})

	all := func(_, _ Model) bool { return true }
	if got := PickDrafterBridged("target", p, all); got != "sibling" {
		t.Fatalf("PickDrafterBridged = %q, want \"sibling\" (same-Family beats a 400x cheaper bridged peer)", got)
	}
	// Drop the same-Family peer and the bridged featherweight takes over.
	p.Evict("sibling")
	if got := PickDrafterBridged("target", p, all); got != "featherweight" {
		t.Fatalf("PickDrafterBridged = %q, want \"featherweight\"", got)
	}
}

// TestCanShareStaysStrictUnderVocabBridge guards the line the fix deliberately does NOT
// cross: a vocabulary bridge maps token IDS and grants nothing about KV bit-identity, so the
// prefill-share gate must be exactly as strict as it was.
func TestCanShareStaysStrictUnderVocabBridge(t *testing.T) {
	tiny := Model{ID: "tiny", Family: "tinyfam", WeightBytes: 10, PrefixDigest: "a"}
	big := Model{ID: "big", Family: "bigfam", WeightBytes: 500, PrefixDigest: "b"}
	if CanShare(big, tiny) {
		t.Error("CanShare(cross-vocab pair) = true; a vocab bridge must NOT unlock prefill share")
	}
	// Even a bridged, spec-decodable pair shares no prefill.
	dv, tv := testVocabs()
	p := NewPool(1000)
	mustAdmit(t, p, tiny)
	mustAdmit(t, p, big)
	if _, err := BridgeRolesVocab("tiny", "big", p, NewVocabMap(dv, tv)); err != nil {
		t.Fatalf("BridgeRolesVocab err = %v, want nil", err)
	}
	if CanShare(big, tiny) {
		t.Error("CanShare = true after a successful vocab bridge; the two gates must stay separate")
	}
}

// ---------------------------------------------------------------------------
// THE WITNESS: losslessness of the whole bridged path.
// ---------------------------------------------------------------------------

// TestSpecDecodeVocabBridgedLossless is the #4208 done-condition. A drafter running in a
// FOREIGN id space is bridged into the target's space and driven through the real SpecDecode
// loop; the emitted stream must be token-identical to no-drafter greedy decode of the same
// target, for every drafter quality — including drafters that propose tokens the target has
// no id for, and against a target that regularly emits tokens no draft can propose.
//
// The second assertion is what keeps the first from being vacuous: the good bridged drafter
// must also buy mean acceptance > 1. A bridge that is "lossless" because it never proposes
// anything is not a fix.
func TestSpecDecodeVocabBridgedLossless(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)
	prompt := []int{3, 7}
	const budget = 60

	// The baseline: plain greedy decode of the target, no drafter at all.
	base, err := SpecDecode(prompt, nil, testVerifier, SpecDecodeConfig{MaxNewTokens: budget})
	if err != nil {
		t.Fatalf("baseline SpecDecode err = %v", err)
	}
	if len(base.Output) != budget {
		t.Fatalf("baseline emitted %d tokens, want %d", len(base.Output), budget)
	}
	if base.MeanAcceptanceLength != 1 {
		t.Fatalf("baseline MeanAcceptanceLength = %v, want exactly 1 (no drafting)", base.MeanAcceptanceLength)
	}

	// An "oracle" drafter expressed in the DRAFTER's id space: it maps its context back up to
	// the target space, runs the target's own rule, and maps the answer back down. Note it is
	// only as informed as MapContext left it — every target-only token was dropped from its
	// view, so it is genuinely misinformed right after one. That is the honest case: the
	// output must STILL be bit-exact.
	oracleDrafter := func(k int) Drafter {
		return func(dctx []int) []int {
			tctx := make([]int, 0, len(dctx))
			for _, d := range dctx {
				if tid, ok := m.ToTarget(d); ok {
					tctx = append(tctx, tid)
				}
			}
			out := make([]int, 0, k)
			for i := 0; i < k; i++ {
				tid := targetOracle(tctx)
				tctx = append(tctx, tid)
				did, ok := m.ToDrafter(tid)
				if !ok {
					break // a target-only token: this drafter genuinely cannot express it
				}
				out = append(out, did)
			}
			return out
		}
	}

	cases := []struct {
		name    string
		drafter Drafter
		// wantAccept asserts the throughput the drafting bought, so a bridge that is safe
		// only because it never proposes is caught.
		wantAcceptOver float64
	}{
		{
			name:           "oracle drafter (high acceptance)",
			drafter:        m.BridgeDrafter(oracleDrafter(4)),
			wantAcceptOver: 1.5,
		},
		{
			name: "noisy drafter (alternates oracle and garbage)",
			drafter: m.BridgeDrafter(func() Drafter {
				n := 0
				oracle := oracleDrafter(4)
				return func(dctx []int) []int {
					n++
					if n%2 == 0 {
						return []int{100, 101, 102, 103} // shared ids, but wrong
					}
					return oracle(dctx)
				}
			}()),
			wantAcceptOver: 1.0,
		},
		{
			name:           "garbage drafter (always wrong, always mappable)",
			drafter:        m.BridgeDrafter(func([]int) []int { return []int{139, 138, 137, 136} }),
			wantAcceptOver: 0, // may accept nothing; correctness must not care
		},
		{
			name:           "unmappable drafter (proposes only drafter-only tokens)",
			drafter:        m.BridgeDrafter(func([]int) []int { return []int{200, 201, 202, 203} }),
			wantAcceptOver: 0,
		},
		{
			name:           "silent drafter (proposes nothing)",
			drafter:        m.BridgeDrafter(func([]int) []int { return nil }),
			wantAcceptOver: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			evicted := 0
			run, err := SpecDecode(prompt, tc.drafter, testVerifier, SpecDecodeConfig{
				MaxNewTokens: budget,
				MaxDraft:     4,
				Rollback:     func(n int) { evicted += n },
			})
			if err != nil {
				t.Fatalf("SpecDecode err = %v", err)
			}
			// THE losslessness assertion: bit-exact to no-drafter greedy decode.
			if !reflect.DeepEqual(run.Output, base.Output) {
				t.Fatalf("bridged output diverged from plain greedy decode\n got: %v\nwant: %v", run.Output, base.Output)
			}
			// The rollback accounting must reconcile: every drafted token was either accepted
			// or rolled back.
			if run.DraftedTokens != run.AcceptedDrafts+run.EvictKV {
				t.Errorf("drafted %d != accepted %d + evicted %d", run.DraftedTokens, run.AcceptedDrafts, run.EvictKV)
			}
			if evicted != run.EvictKV {
				t.Errorf("Rollback saw %d evictions, run reported %d", evicted, run.EvictKV)
			}
			if run.MeanAcceptanceLength < 1 {
				t.Errorf("MeanAcceptanceLength = %v, want >= 1 (a round always commits a real token)",
					run.MeanAcceptanceLength)
			}
			if tc.wantAcceptOver > 0 && run.MeanAcceptanceLength <= tc.wantAcceptOver {
				t.Errorf("MeanAcceptanceLength = %v, want > %v — the bridge must buy real throughput, "+
					"not merely be safe by never proposing", run.MeanAcceptanceLength, tc.wantAcceptOver)
			}
			t.Logf("%s: rounds=%d drafted=%d accepted=%d evicted=%d mean-acceptance=%.3f",
				tc.name, run.Rounds, run.DraftedTokens, run.AcceptedDrafts, run.EvictKV, run.MeanAcceptanceLength)
		})
	}
}

// TestSpecDecodeVocabBridgedLosslessAcrossBudgets re-runs the losslessness check across a
// spread of draft depths and decode budgets: the bridged stream must equal plain greedy
// decode at every (K, budget), so no depth can smuggle in a divergence.
func TestSpecDecodeVocabBridgedLosslessAcrossBudgets(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)
	prompt := []int{11, 2, 39}

	rng := rand.New(rand.NewSource(38174))
	// A random drafter in the DRAFTER's id space, freely mixing shared and drafter-only ids
	// so both the mappable and the truncating path fire.
	randomDrafter := m.BridgeDrafter(func(_ []int) []int {
		n := rng.Intn(5)
		out := make([]int, n)
		for i := range out {
			if rng.Intn(4) == 0 {
				out[i] = 200 + rng.Intn(drafterOnly) // unmappable
			} else {
				out[i] = 100 + rng.Intn(sharedTokens)
			}
		}
		return out
	})

	for _, budget := range []int{1, 2, 5, 17, 40} {
		for _, k := range []int{0, 1, 2, 3, 8} {
			base, err := SpecDecode(prompt, nil, testVerifier, SpecDecodeConfig{MaxNewTokens: budget})
			if err != nil {
				t.Fatalf("baseline err = %v", err)
			}
			run, err := SpecDecode(prompt, randomDrafter, testVerifier, SpecDecodeConfig{
				MaxNewTokens: budget, MaxDraft: k,
			})
			if err != nil {
				t.Fatalf("bridged err = %v", err)
			}
			if !reflect.DeepEqual(run.Output, base.Output) {
				t.Fatalf("budget=%d k=%d: bridged output diverged\n got: %v\nwant: %v",
					budget, k, run.Output, base.Output)
			}
			if len(run.Output) != budget {
				t.Fatalf("budget=%d k=%d: emitted %d tokens, want %d", budget, k, len(run.Output), budget)
			}
		}
	}
}

// TestSpecDecodeVocabBridgedHonorsStopToken checks the bridged path against the StopToken
// path too: an EOS committed through a bridged draft must end the run exactly where plain
// greedy decode ends it.
func TestSpecDecodeVocabBridgedHonorsStopToken(t *testing.T) {
	dv, tv := testVocabs()
	m := NewVocabMap(dv, tv)
	prompt := []int{3}

	// Find a token the target actually emits, so the stop is reachable.
	probe, err := SpecDecode(prompt, nil, testVerifier, SpecDecodeConfig{MaxNewTokens: 30})
	if err != nil {
		t.Fatalf("probe err = %v", err)
	}
	stop := probe.Output[5]

	cfg := SpecDecodeConfig{MaxNewTokens: 30, MaxDraft: 4, StopToken: stop, StopEnabled: true}
	base, err := SpecDecode(prompt, nil, testVerifier, cfg)
	if err != nil {
		t.Fatalf("baseline err = %v", err)
	}
	oracle := m.BridgeDrafter(func(dctx []int) []int {
		tctx := make([]int, 0, len(dctx))
		for _, d := range dctx {
			if tid, ok := m.ToTarget(d); ok {
				tctx = append(tctx, tid)
			}
		}
		out := make([]int, 0, 4)
		for i := 0; i < 4; i++ {
			tid := targetOracle(tctx)
			tctx = append(tctx, tid)
			did, ok := m.ToDrafter(tid)
			if !ok {
				break
			}
			out = append(out, did)
		}
		return out
	})
	run, err := SpecDecode(prompt, oracle, testVerifier, cfg)
	if err != nil {
		t.Fatalf("bridged err = %v", err)
	}
	if !reflect.DeepEqual(run.Output, base.Output) {
		t.Fatalf("bridged stop-token run diverged\n got: %v\nwant: %v", run.Output, base.Output)
	}
	if last := run.Output[len(run.Output)-1]; last != stop {
		t.Errorf("run ended on %d, want the stop token %d", last, stop)
	}
}
