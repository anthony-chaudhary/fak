package polymodel

// vocabmap.go — heterogeneous-vocabulary (TLI) speculative drafting (#4208, epic #4207).
//
// THE RESTRICTION THIS LIFTS. Until now a drafter was eligible for a target only when the
// two declared the same non-empty Family — "models that share a tokenizer + embedding (and
// therefore a valid draft-token vocabulary)". A cross-vocabulary drafter (a tiny generic
// model drafting for a big hosted target that tokenizes differently) was therefore
// STRUCTURALLY refused by PickDrafter and BridgeRoles, however good a proposer it was. That
// is exactly the limitation vLLM's #38174 removes with Token-Level Intersection (ICML 2025,
// arXiv:2502.05202): intersect the two vocabularies, constrain the draft distribution to the
// shared tokens, and translate the proposed ids into the target's id space before the
// accept pass.
//
// Prior-art note: the algorithm is TLI (arXiv:2502.05202); the shape of the seam
// (intersection map + id translation + a per-tokenizer-pair cache) follows vLLM's
// vocab_mapping.py / llm_base_proposer.py. Clean-room Go — no bytes vendored.
//
// WHY THIS IS LOSSLESS, AND WHY THAT IS NOT A NEW ARGUMENT. specdecode.go already states the
// invariant this rides on: greedy speculative decoding is "LOSSLESS BY CONSTRUCTION, FOR ANY
// DRAFTER" — a drafted token is committed ONLY when it equals the target's own argmax at
// that position, and the first divergence is replaced by the target's own token. The
// verifier gates every token, so correctness never depends on the draft being right; only
// speed does. A vocabulary-bridged drafter is, after translation, just another drafter
// proposing target-space ids. So the bridge cannot cost correctness — it can only cost
// ACCEPTANCE RATE. That asymmetry is what makes every approximation below safe:
//
//   - an unmappable PROPOSED token truncates the draft (MapDraft) — a shorter draft, never a
//     wrong one;
//   - an unmappable token in the drafter's CONTEXT is dropped (MapContext) — a worse-informed
//     drafter, never a wrong commit;
//   - a zero-overlap pair proposes nothing — SpecDecode degrades to plain decode.
//
// Each is a throughput knob with a correctness floor of "identical to no drafting at all".
// The SpecDecodeVocabBridgedLossless witness proves the whole path token-identical to
// no-drafter greedy decode, AND that a real bridged drafter still buys mean acceptance > 1
// (a bridge that is merely safe because it never proposes anything is not a fix).
//
// WHAT STAYS REFUSED: PREFILL SHARE. CanShare is deliberately NOT relaxed. It gates reuse of
// another model's already-computed prefix KV, which is only sound when the prefill-relevant
// weights are byte-identical (same Family AND same PrefixDigest) — the reused KV must be
// BIT-identical, not merely meaningful. A vocabulary bridge maps token IDS; it does not make
// two different models' KV interchangeable, so it grants nothing here. Speculation
// eligibility and prefill-share eligibility were conflated in BridgeRoles (it required
// CanShare for a spec pair); this file separates them: a bridged pair speculates without
// sharing prefill.

import (
	"errors"
	"math"
	"sort"
	"sync"
)

// ErrNoVocabOverlap is returned when a drafter/target pair is bridged by a VocabMap with an
// EMPTY shared vocabulary. Such a pair can never propose a verifiable token, so it is not a
// speculation pair at all — refuse it rather than wire a drafter that always proposes
// nothing (which would be "lossless" only in the vacuous sense of never drafting).
var ErrNoVocabOverlap = errors.New("polymodel: drafter and target vocabularies share no tokens")

// VocabMap is the token-level intersection of a drafter's and a target's vocabulary: the
// shared surface tokens, plus the id↔id translation between the two tokenizers' id spaces.
// It is built ONCE per tokenizer pair (VocabCache) because the intersection is a pure
// function of the two vocabularies and costs O(|drafter vocab|) to compute.
//
// The two vocabularies cross the seam as plain map[string]int (surface token → id) — this
// leaf imports nothing internal and owns no tokenizer, exactly the way Drafter/Verifier
// cross it as closures. A token is SHARED iff the identical surface string is present in
// both; the map is immutable and safe for concurrent readers once built.
type VocabMap struct {
	draftToTarget map[int]int
	targetToDraft map[int]int
	drafterSize   int
	targetSize    int
}

// NewVocabMap intersects the two vocabularies (surface token → id) and builds the id↔id
// translation both ways. Ids present in only one vocabulary are absent from the map — the
// callers above treat "absent" as the constraint boundary (mask / truncate / drop), never as
// an error. Either vocabulary may be nil or empty, yielding an empty map (Len()==0), which
// BridgeRolesVocab refuses as ErrNoVocabOverlap.
//
// It iterates the SMALLER vocabulary for the lookup, so the cost is O(min(|d|,|t|)) map
// probes — a ~150k-entry pair builds in a few milliseconds, once, and is then cached.
func NewVocabMap(drafterVocab, targetVocab map[string]int) *VocabMap {
	m := &VocabMap{
		draftToTarget: make(map[int]int),
		targetToDraft: make(map[int]int),
		drafterSize:   len(drafterVocab),
		targetSize:    len(targetVocab),
	}
	// Probe the smaller side against the larger; the intersection is symmetric.
	if len(drafterVocab) <= len(targetVocab) {
		for surface, did := range drafterVocab {
			if tid, ok := targetVocab[surface]; ok {
				m.draftToTarget[did] = tid
				m.targetToDraft[tid] = did
			}
		}
	} else {
		for surface, tid := range targetVocab {
			if did, ok := drafterVocab[surface]; ok {
				m.draftToTarget[did] = tid
				m.targetToDraft[tid] = did
			}
		}
	}
	return m
}

// Len returns the number of SHARED tokens — the size of the intersection. It is the honest
// measure of how much of a drafter's expressive range survives the bridge: a pair with a
// large Len drafts nearly as well as a same-Family pair; a pair with Len==0 cannot draft at
// all (ErrNoVocabOverlap).
func (m *VocabMap) Len() int {
	if m == nil {
		return 0
	}
	return len(m.draftToTarget)
}

// DrafterCoverage is the fraction of the DRAFTER's vocabulary that maps into the target
// (Len / |drafter vocab|), in [0,1]. It predicts the acceptance-rate cost of the bridge: the
// drafter can only ever propose within this fraction of its own range, so a low coverage
// means the drafter is frequently silenced (a shorter draft), never wrong. Returns 0 for an
// empty/nil drafter vocabulary.
func (m *VocabMap) DrafterCoverage() float64 {
	if m == nil || m.drafterSize == 0 {
		return 0
	}
	return float64(len(m.draftToTarget)) / float64(m.drafterSize)
}

// TargetCoverage is the fraction of the TARGET's vocabulary reachable from the drafter
// (Len / |target vocab|), in [0,1]. It bounds what fraction of the target's own output
// distribution a bridged draft can ever hit: target tokens outside the intersection are
// never proposed, so they always cost a correction round. Returns 0 for an empty/nil target
// vocabulary.
func (m *VocabMap) TargetCoverage() float64 {
	if m == nil || m.targetSize == 0 {
		return 0
	}
	return float64(len(m.draftToTarget)) / float64(m.targetSize)
}

// ToTarget translates one DRAFTER id into the target's id space. ok is false when the token
// is outside the intersection (the drafter can say something the target has no id for).
func (m *VocabMap) ToTarget(drafterID int) (int, bool) {
	if m == nil {
		return 0, false
	}
	id, ok := m.draftToTarget[drafterID]
	return id, ok
}

// ToDrafter translates one TARGET id into the drafter's id space. ok is false when the token
// is outside the intersection.
func (m *VocabMap) ToDrafter(targetID int) (int, bool) {
	if m == nil {
		return 0, false
	}
	id, ok := m.targetToDraft[targetID]
	return id, ok
}

// SharedDrafterIDs returns the drafter-space ids of every shared token, sorted ascending
// (deterministic regardless of map iteration order). It is the intersection made
// enumerable — for a host that builds its own constrained sampler rather than using
// ConstrainLogits.
func (m *VocabMap) SharedDrafterIDs() []int {
	if m == nil {
		return nil
	}
	out := make([]int, 0, len(m.draftToTarget))
	for id := range m.draftToTarget {
		out = append(out, id)
	}
	sort.Ints(out)
	return out
}

// MapDraft translates a proposal from the drafter's id space into the target's, TRUNCATING
// at the first unmappable token, and reports whether a truncation occurred.
//
// Truncate — do not drop. Speculative decoding verifies a draft POSITIONALLY: index i is a
// proposal for "the token after committed + draft[:i]". Dropping an unmappable token
// mid-draft would silently re-index every later proposal onto a position it was never
// proposed for — the drafts would then be verified against the wrong positions, which costs
// acceptance (never correctness, since the verifier still gates) but is pure waste.
// Truncating keeps the surviving prefix meaning exactly what it meant. A draft whose FIRST
// token is unmappable maps to an empty draft: that round is a plain single-token decode.
func (m *VocabMap) MapDraft(draft []int) ([]int, bool) {
	if m == nil {
		return nil, len(draft) > 0
	}
	out := make([]int, 0, len(draft))
	for _, d := range draft {
		tid, ok := m.draftToTarget[d]
		if !ok {
			return out, true // truncate at the first unmappable proposal
		}
		out = append(out, tid)
	}
	return out, false
}

// MapContext translates the committed history from the TARGET's id space into the drafter's,
// DROPPING unmappable tokens, and reports how many were dropped.
//
// Drop — do not truncate. This is the opposite choice from MapDraft, for the opposite
// reason. The context is the drafter's INPUT: it conditions the proposal but is never
// verified, so a gap costs only proposal quality. Truncating here would throw away the whole
// recent history after the first exotic token — usually the most informative part —
// whereas dropping keeps the drafter maximally informed. This is a genuine approximation:
// the drafter sees a slightly different string than the target did (upstream vLLM
// re-tokenizes the text to avoid it, which needs a detokenizer this leaf does not own). It
// cannot affect the emitted stream, because the target's verify pass regenerates its own
// argmax from ITS OWN untouched context every round — the drafter's view feeds nothing but
// the proposal. Dropped tokens therefore show up as a lower acceptance rate and nothing else.
func (m *VocabMap) MapContext(committed []int) ([]int, int) {
	if m == nil {
		return nil, len(committed)
	}
	out := make([]int, 0, len(committed))
	dropped := 0
	for _, t := range committed {
		did, ok := m.targetToDraft[t]
		if !ok {
			dropped++
			continue
		}
		out = append(out, did)
	}
	return out, dropped
}

// ConstrainLogits masks a drafter's logit vector to the SHARED vocabulary in place, setting
// every non-shared drafter id to -Inf, and returns how many entries it masked. It is the
// distribution-level half of TLI: a host that samples/argmaxes the drafter's own logits
// calls this first, so the drafter can only ever propose a token the target has an id for.
//
// This is a THROUGHPUT lever, not a correctness one — MapDraft already makes an unmappable
// proposal harmless by truncating. Constraining first means the drafter spends its draft
// slots on tokens that can actually be verified instead of burning a round on a proposal
// that truncates to nothing. logits is indexed by drafter id; entries at indices outside the
// intersection (including any index past the drafter's vocabulary) are masked.
func (m *VocabMap) ConstrainLogits(logits []float32) int {
	if m == nil {
		return 0
	}
	neg := float32(math.Inf(-1))
	masked := 0
	for i := range logits {
		if _, ok := m.draftToTarget[i]; !ok {
			logits[i] = neg
			masked++
		}
	}
	return masked
}

// BridgeDrafter wraps a drafter that runs in its OWN vocabulary into a Drafter speaking the
// TARGET's id space, so a cross-vocabulary proposer drops straight into SpecDecode with no
// other change:
//
//	run, _ := polymodel.SpecDecode(prompt, vm.BridgeDrafter(tinyDrafter), targetVerify, cfg)
//
// Per round it maps the committed history down into the drafter's space (MapContext), calls
// the inner drafter, and maps the proposal back up into the target's space (MapDraft). Both
// directions are total: no error path, because every mapping failure degrades to a shorter
// draft or a thinner context, which the verifier absorbs. A nil inner drafter (or a nil
// VocabMap) yields a drafter that proposes nothing — a plain decode, still correct.
//
// The inner drafter keeps the specdecode.go contract in its own space: it owns its context
// and session, and receives only token ids.
func (m *VocabMap) BridgeDrafter(inner Drafter) Drafter {
	if m == nil || inner == nil {
		return func([]int) []int { return nil }
	}
	return func(committed []int) []int {
		ctx, _ := m.MapContext(committed)
		proposal := inner(ctx)
		mapped, _ := m.MapDraft(proposal)
		return mapped
	}
}

// ---------------------------------------------------------------------------
// The per-tokenizer-pair cache.
// ---------------------------------------------------------------------------

// VocabCache memoizes VocabMaps per ORDERED tokenizer pair (drafter, target). The
// intersection is a pure function of the two vocabularies but costs a full pass over the
// smaller one, so a live pool that re-picks a drafter every round must not rebuild it every
// round — build once, reuse for the process lifetime. The key is the caller's tokenizer
// identity (a digest or name), not the vocabulary contents: this leaf cannot hash a
// tokenizer it does not own, so a caller that reuses a key for a DIFFERENT vocabulary gets
// the stale map. Key by a content digest, never a mutable label.
//
// The zero value is not usable; call NewVocabCache. It is safe for concurrent use.
type VocabCache struct {
	mu sync.Mutex
	m  map[[2]string]*VocabMap
}

// NewVocabCache returns an empty VocabMap cache.
func NewVocabCache() *VocabCache {
	return &VocabCache{m: map[[2]string]*VocabMap{}}
}

// Get returns the cached VocabMap for the ordered (drafterKey, targetKey) pair, calling
// build exactly once per pair to construct it on a miss. Concurrent Gets for the same pair
// build once and share the result; a build returning nil is cached as nil (a pair with no
// bridge stays a miss-free negative). The pair is ORDERED — (a,b) and (b,a) are different
// bridges, since the map's directions are not interchangeable.
func (c *VocabCache) Get(drafterKey, targetKey string, build func() *VocabMap) *VocabMap {
	key := [2]string{drafterKey, targetKey}
	c.mu.Lock()
	defer c.mu.Unlock()
	if m, ok := c.m[key]; ok {
		return m
	}
	var m *VocabMap
	if build != nil {
		m = build()
	}
	c.m[key] = m
	return m
}

// Len returns the number of cached pairs.
func (c *VocabCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.m)
}

// ---------------------------------------------------------------------------
// Eligibility — the relaxed drafter pick and the bridged role wiring.
// ---------------------------------------------------------------------------

// VocabBridge reports whether a cross-vocabulary bridge exists that lets drafter propose for
// target — typically "a VocabMap for this pair has a non-empty intersection". It crosses the
// seam as a predicate closure (like Drafter/Verifier) because this leaf owns no tokenizers
// and cannot answer the question itself. It is consulted ONLY for pairs the same-Family fast
// path already rejected.
type VocabBridge func(drafter, target Model) bool

// PickDrafterBridged is PickDrafter with the shared-Family restriction LIFTED (#4208). It
// picks the best speculator for the active decoder among the other resident models, in two
// tiers:
//
//  1. SAME-FAMILY (the fast, no-map path): a different model in the same non-empty Family.
//     Its ids are already the target's ids — no intersection, no translation, no coverage
//     loss. A same-Family drafter ALWAYS beats a bridged one, however cheap the bridged one
//     is, because the bridge can only subtract acceptance.
//  2. VOCAB-BRIDGED: any other resident model the bridge predicate admits. This is the tier
//     that did not exist before — including for a target with Family=="" ("unique"), which
//     PickDrafter refuses outright and which a bridge can now legitimately draft for.
//
// Within a tier it prefers the cheapest (smallest WeightBytes, so drafting is fast),
// tie-broken by ID — the same order PickDrafter uses. Returns "" when the active model is
// not resident or nothing is eligible, in which case the target self-decodes (no
// speculation, always correct).
//
// bridge == nil makes this EXACTLY PickDrafter (tier 2 is empty), which the witness asserts
// over a randomized pool — the relaxation is strictly additive and cannot change a pick that
// PickDrafter already made.
func PickDrafterBridged(active ModelID, pool *Pool, bridge VocabBridge) ModelID {
	if pool == nil {
		return ""
	}
	want, ok := pool.Get(active)
	if !ok {
		return ""
	}
	best := ModelID("")
	var bestBytes int64
	bestBridged := true                  // tier flag: a same-Family (false) candidate outranks a bridged (true) one
	for _, id := range pool.Resident() { // sorted → deterministic
		if id == active {
			continue
		}
		m, _ := pool.Get(id)
		bridged := true
		switch {
		case want.Family != "" && m.Family == want.Family:
			bridged = false // tier 1: same tokenizer, no map needed
		case bridge != nil && bridge(m, want):
			bridged = true // tier 2: cross-vocabulary, bridged by a VocabMap
		default:
			continue // ineligible: neither same-Family nor bridgeable
		}
		if best == "" || (bestBridged && !bridged) || (bestBridged == bridged && m.WeightBytes < bestBytes) {
			best, bestBytes, bestBridged = id, m.WeightBytes, bridged
		}
	}
	return best
}

// BridgeRolesVocab is BridgeRoles for a HETEROGENEOUS-VOCABULARY pair (#4208): it resolves a
// declared drafter/verifier pair whose tokenizers DIFFER into spec-decode wiring, bridged by
// m. It is the eligibility relaxation the TLI borrow is for — BridgeRoles refuses such a
// pair with ErrNotCoResident, however good the drafter is.
//
// The preconditions it keeps, and the two it drops:
//
//  1. both ids non-empty (else ErrEmptyRole) — kept;
//  2. drafter != verifier (else ErrNotCoResident) — kept: a model cannot draft for itself;
//  3. both resident in pool (else ErrNotCoResident) — kept: you cannot draft with weights
//     you evicted;
//  4. m has a non-empty intersection (else ErrNoVocabOverlap) — NEW: this replaces the
//     Family check as the "the drafter's tokens are meaningful for the target" gate.
//
// DROPPED: the same-Family requirement (that is the whole point), and the CanShare
// requirement. BridgeRoles demanded CanShare — byte-identical prefill weights — for a
// SPECULATION pair, which conflated two different questions. CanShare gates reuse of
// another model's prefix KV, which must be BIT-identical to be sound. Speculation needs
// nothing of the kind: the target computes its own KV and its own argmax, and merely checks
// the drafter's guesses against them. A bridged pair therefore speculates WITHOUT sharing
// prefill, and CanShare stays exactly as strict as it was.
//
// PreferredDrafter resolves to the same-Family drafter residency policy would pick
// (PickDrafter — the free, no-map path), so a bridged drafter chosen while a same-Family
// peer is warm is SURFACED via DrafterIsPreferred=false rather than hidden. When no
// same-Family peer is warm, the bridged drafter IS the best available and
// DrafterIsPreferred is true. The resolved wiring carries VocabBridged=true.
//
// Like BridgeRoles it is PURE over a fixed (drafter, verifier, Pool, m) and does not consult
// Enabled(); a live-path caller gates FAK_POLYMODEL separately.
func BridgeRolesVocab(drafter, verifier ModelID, pool *Pool, m *VocabMap) (SpecWiring, error) {
	if drafter == "" || verifier == "" {
		return SpecWiring{}, ErrEmptyRole
	}
	if drafter == verifier {
		return SpecWiring{}, ErrNotCoResident
	}
	if pool == nil {
		return SpecWiring{}, ErrNotCoResident
	}
	if _, ok := pool.Get(drafter); !ok {
		return SpecWiring{}, ErrNotCoResident
	}
	if _, ok := pool.Get(verifier); !ok {
		return SpecWiring{}, ErrNotCoResident
	}
	if m.Len() == 0 {
		return SpecWiring{}, ErrNoVocabOverlap
	}
	preferred := PickDrafter(verifier, pool)
	if preferred == "" {
		// No same-Family peer is warm, so nothing cheaper-and-lossless exists to prefer:
		// the bridged drafter is the best available pick.
		preferred = drafter
	}
	return SpecWiring{
		Drafter:            drafter,
		Verifier:           verifier,
		PreferredDrafter:   preferred,
		DrafterIsPreferred: preferred == drafter,
		VocabBridged:       true,
	}, nil
}
