package spec

// drafters.go — the production Drafter implementations for the speculative-decode
// lane (epic #529 / native spec-decode #23). SpeculativeGreedy is drafter-agnostic;
// this file supplies the three drafters a poly-model serving lane actually runs:
//
//   - ModelDrafter — a real co-resident draft MODEL (the classic draft/target pair):
//     a cheaper model whose argmax proposes the chain, threading its own KV and
//     rolling back its own rejected span with the same bit-exact Evict.
//   - LookupDrafter — prompt-lookup / n-gram MULTI-TOKEN PREDICTION: no second model
//     at all. The committed context (and optional reference texts) IS the drafter —
//     the longest n-gram suffix of the context is found earlier in the corpus and
//     the tokens that followed it are proposed. Free to draft (no forward pass), so
//     any acceptance at all is pure win; a miss degrades to plain greedy.
//   - MultiDrafter — MULTI-DRAFT-MODEL support: an ensemble of named drafters with
//     per-member measured acceptance (AcceptanceMeter, #284) and deterministic
//     routing — warm every member up, then exploit the best-measured member with a
//     periodic probe so a member whose acceptance shifts is re-discovered. This is
//     the REALIZED-acceptance complement of polymodel.PickDrafter, which picks a
//     drafter a priori by static cost (cheapest same-Family co-resident); the
//     ensemble routes by what each drafter's acceptance actually turned out to be.
//
// All three are pure CPU bookkeeping over the frozen Drafter seam — no GPU, no new
// verify path. Losslessness is untouched by construction: a Drafter only PROPOSES;
// the target's verify (AcceptGreedy + the bit-exact squash) never trusts it, so even
// an adversarial member costs only rolled-back KV (the witnesses in drafters_test.go
// pin token-identity for every drafter here).

import "github.com/anthony-chaudhary/fak/internal/model"

// ---------------------------------------------------------------------------
// ModelDrafter — a real co-resident draft model.
// ---------------------------------------------------------------------------

// ModelDrafter is a Drafter backed by a real (typically cheaper) draft model: Draft
// proposes its own greedy argmax chain, threading its own KV cache so each proposal
// conditions on the previous one; Commit rolls back the drafter's OWN speculative
// span with the bit-exact model.KVCache.Evict and re-threads the tokens the target
// actually committed, so the next Draft continues from the true shared context. The
// draft model's token ids must be valid for the target (same tokenizer — the same
// Family constraint polymodel.PickDrafter enforces when it picks one).
type ModelDrafter struct {
	s      *model.Session
	logits []float32
	// specFrom is the drafter's KV length at the start of the OPEN draft, or -1 when
	// no draft is open — so a Commit on a round this member did not draft (the
	// MultiDrafter fans Commit to every member) never evicts committed context.
	specFrom int
}

// NewModelDrafter prefills a fresh session of m on prompt and returns the drafter
// positioned to propose the first continuation token.
func NewModelDrafter(m *model.Model, prompt []int) *ModelDrafter {
	s := m.NewSession()
	return &ModelDrafter{s: s, logits: s.Prefill(prompt), specFrom: -1}
}

// Draft proposes up to k tokens by greedy self-decode on the draft model, appending
// k speculative positions to the drafter's own KV (rolled back in Commit).
func (d *ModelDrafter) Draft(k int) []int {
	d.specFrom = d.s.Cache.Len()
	if k <= 0 {
		return nil
	}
	drafts := make([]int, 0, k)
	l := d.logits
	for j := 0; j < k; j++ {
		t := argmax(l)
		drafts = append(drafts, t)
		l = d.s.Step(t)
	}
	return drafts
}

// Commit drops the open speculative draft span (bit-exact Evict), then advances the
// drafter by the truly committed tokens so its next Draft continues from the real
// context. With no open draft (this member sat the round out) it only re-threads.
func (d *ModelDrafter) Commit(committed []int) {
	if d.specFrom >= 0 {
		if grew := d.s.Cache.Len() - d.specFrom; grew > 0 {
			d.s.Cache.Evict(d.specFrom, grew)
		}
		d.specFrom = -1
	}
	for _, t := range committed {
		d.logits = d.s.Step(t)
	}
}

// ---------------------------------------------------------------------------
// LookupDrafter — prompt-lookup / n-gram multi-token prediction (no draft model).
// ---------------------------------------------------------------------------

// LookupDrafter is model-free multi-token prediction by prompt lookup (the n-gram /
// "prompt lookup decoding" method; vLLM's `ngram` speculative method is the same
// idea): the committed context and any reference texts are the draft model. Draft
// finds the longest suffix n-gram of the current context (maxN down to minN) at an
// earlier position in the context or a reference, and proposes the tokens that
// followed that occurrence. Drafting costs no forward pass, so any nonzero
// acceptance is pure decode win — the regime it shines in is exactly fak's: agent
// transcripts and code, where the model re-emits spans it has already seen (a diff
// echoing the file, a repeated tool-call shape, a retrieved reference).
//
// The zero value is unusable; construct with NewLookupDrafter. Not safe for
// concurrent use (one decode lane, like every Drafter).
type LookupDrafter struct {
	seq  []int   // prompt + committed tokens — the true context, grown by Commit
	refs [][]int // optional reference corpora (retrieval-primed lookup)
	minN int
	maxN int
}

// NewLookupDrafter builds a prompt-lookup drafter over prompt. minN..maxN is the
// suffix n-gram window tried longest-first (maxN cap 8; minN floor 1; a maxN below
// minN collapses to minN). refs are optional reference token sequences also searched
// for continuations — the retrieval-primed form (e.g. a prior generation, a source
// file the answer will quote); pass none for pure prompt lookup.
func NewLookupDrafter(prompt []int, minN, maxN int, refs ...[]int) *LookupDrafter {
	if minN < 1 {
		minN = 1
	}
	if maxN > 8 {
		maxN = 8
	}
	if maxN < minN {
		maxN = minN
	}
	seq := make([]int, len(prompt))
	copy(seq, prompt)
	return &LookupDrafter{seq: seq, refs: refs, minN: minN, maxN: maxN}
}

// Draft proposes up to k tokens: the longest suffix n-gram of the context found at
// an earlier position (context first, then each reference, most recent occurrence
// first) supplies the tokens that followed it. No match returns nil — the round
// degrades to plain greedy, which is always correct.
func (d *LookupDrafter) Draft(k int) []int {
	if k <= 0 || len(d.seq) == 0 {
		return nil
	}
	for n := d.maxN; n >= d.minN; n-- {
		if n > len(d.seq) {
			continue
		}
		suffix := d.seq[len(d.seq)-n:]
		// Self-lookup: most recent earlier occurrence with at least one follower.
		if c := continuationAfter(d.seq, suffix, len(d.seq)-n, k); c != nil {
			return c
		}
		for _, ref := range d.refs {
			if c := continuationAfter(ref, suffix, len(ref), k); c != nil {
				return c
			}
		}
	}
	return nil
}

// Commit appends the committed tokens to the context; the next Draft's suffix (and
// its lookup index) always reflects the true decoded sequence.
func (d *LookupDrafter) Commit(committed []int) {
	d.seq = append(d.seq, committed...)
}

// continuationAfter scans corpus for the most recent occurrence of pat starting
// strictly before limit and returns up to k tokens following it (nil when no
// occurrence has a follower). Linear scan — corpora here are one context window.
func continuationAfter(corpus, pat []int, limit, k int) []int {
	if len(pat) == 0 || len(pat) > len(corpus) {
		return nil
	}
	if limit > len(corpus)-len(pat) {
		limit = len(corpus) - len(pat)
	}
scan:
	for i := limit - 1; i >= 0; i-- {
		for j, p := range pat {
			if corpus[i+j] != p {
				continue scan
			}
		}
		from := i + len(pat)
		if from >= len(corpus) {
			continue // matched at the very end: nothing follows
		}
		to := from + k
		if to > len(corpus) {
			to = len(corpus)
		}
		out := make([]int, to-from)
		copy(out, corpus[from:to])
		return out
	}
	return nil
}

// ---------------------------------------------------------------------------
// MultiDrafter — the multi-draft-model ensemble with measured-acceptance routing.
// ---------------------------------------------------------------------------

// NamedDrafter is one ensemble member: a Drafter plus the name its measured
// acceptance is reported under (a model id, "lookup", …).
type NamedDrafter struct {
	Name string
	D    Drafter
}

// MultiDrafter is the multi-draft-model ensemble: it holds N named drafters, meters
// each member's REALIZED acceptance per round it drafts (AcceptanceMeter, #284), and
// routes each round deterministically — first a warmup round per member, then the
// best-measured member, with every probeEvery-th round given round-robin to the
// ensemble so a member whose acceptance shifts mid-stream is re-discovered (the
// context can move into and out of a drafter's regime: a lookup drafter is superb in
// repetitive spans and useless in novel ones). Commit fans the committed tokens to
// EVERY member, so all drafters stay aligned with the true context and any of them
// can take the next round.
//
// Losslessness is inherited, not asserted: members only propose; the target's verify
// discards wrong drafts identically whoever drafted them. Routing therefore affects
// only the acceptance rate (the speed), never the output.
type MultiDrafter struct {
	members    []NamedDrafter
	meters     []AcceptanceMeter
	routed     []int
	probeEvery int // every probeEvery-th round is a round-robin probe; <=0 → 8
	round      int
	last       int   // member that drafted the open round; -1 when none open
	lastDraft  []int // its (clamped) proposal, for deriving accepted in Commit
}

// NewMultiDrafter builds the ensemble over members in the given order (the order is
// the warmup order and the deterministic tie-break). probeEvery <= 0 defaults to 8.
// At least one member is required; a nil member Drafter panics on first use, which
// is the honest failure (a silent skip would misreport the ensemble).
func NewMultiDrafter(probeEvery int, members ...NamedDrafter) *MultiDrafter {
	if probeEvery <= 0 {
		probeEvery = 8
	}
	return &MultiDrafter{
		members:    members,
		meters:     make([]AcceptanceMeter, len(members)),
		routed:     make([]int, len(members)),
		probeEvery: probeEvery,
		last:       -1,
	}
}

// pick chooses the member for this round: (1) warmup — the first member never yet
// routed; (2) probe — every probeEvery-th round cycles the ensemble round-robin;
// (3) exploit — the member with the highest measured acceptance rate (ties to the
// earlier member). Pure function of the ensemble's own deterministic state.
func (md *MultiDrafter) pick() int {
	for i := range md.members {
		if md.routed[i] == 0 {
			return i
		}
	}
	if md.round%md.probeEvery == md.probeEvery-1 {
		return (md.round / md.probeEvery) % len(md.members)
	}
	best := 0
	for i := 1; i < len(md.members); i++ {
		if md.meters[i].AcceptanceRate() > md.meters[best].AcceptanceRate() {
			best = i
		}
	}
	return best
}

// Draft routes the round to the picked member and proposes its draft (clamped to k
// so the per-member meters count exactly what the target verifies).
func (md *MultiDrafter) Draft(k int) []int {
	i := md.pick()
	drafts := md.members[i].D.Draft(k)
	if k < 0 {
		k = 0
	}
	if len(drafts) > k {
		drafts = drafts[:k]
	}
	md.last, md.lastDraft = i, drafts
	md.routed[i]++
	md.round++
	return drafts
}

// Commit closes the round: the accepted count is derived as the matching prefix of
// (proposal, committed) — exact under the greedy accept rule, because the correction
// token at the divergence position differs from the rejected draft by construction —
// metered against the member that drafted, then the committed tokens fan to EVERY
// member so all drafters stay aligned on the true context.
func (md *MultiDrafter) Commit(committed []int) {
	if md.last >= 0 {
		accepted := 0
		for accepted < len(md.lastDraft) && accepted < len(committed) &&
			md.lastDraft[accepted] == committed[accepted] {
			accepted++
		}
		md.meters[md.last].Observe(len(md.lastDraft), accepted, len(committed))
		md.last, md.lastDraft = -1, nil
	}
	for _, m := range md.members {
		m.D.Commit(committed)
	}
}

// MemberStats is one ensemble member's measured record: how many rounds were routed
// to it and its realized acceptance (an AcceptanceStats snapshot, #284).
type MemberStats struct {
	Name         string          `json:"name"`
	RoundsRouted int             `json:"rounds_routed"`
	Stats        AcceptanceStats `json:"stats"`
}

// Stats snapshots every member's measured acceptance, in member order.
func (md *MultiDrafter) Stats() []MemberStats {
	out := make([]MemberStats, len(md.members))
	for i, m := range md.members {
		out[i] = MemberStats{Name: m.Name, RoundsRouted: md.routed[i], Stats: md.meters[i].Snapshot()}
	}
	return out
}

// Best returns the name of the member the exploit rule would pick right now — the
// highest measured acceptance rate, ties to the earlier member. Empty ensemble
// returns "".
func (md *MultiDrafter) Best() string {
	if len(md.members) == 0 {
		return ""
	}
	best := 0
	for i := 1; i < len(md.members); i++ {
		if md.meters[i].AcceptanceRate() > md.meters[best].AcceptanceRate() {
			best = i
		}
	}
	return md.members[best].Name
}
