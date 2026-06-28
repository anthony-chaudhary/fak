package ctxplan

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strconv"
)

// plancache.go — the prepared-statement / plan cache (issue #561). Re-planning a
// stable forecast over an append-only store is wasted work: the same (forecast,
// candidate set) yields a BYTE-IDENTICAL Plan every turn (Optimize is deterministic),
// so the second compute the index already attacks is paid twice for nothing. This is
// the ctxplan analogue of a query planner's prepared statement: prepare the plan once,
// keyed by a deterministic FINGERPRINT of its inputs, and RE-EXECUTE the cached plan
// on the next turn when the inputs have not changed in a way that would alter the
// selection — a re-execution rather than a re-plan.
//
// The reuse contract is fail-closed and OBSERVABLY EQUIVALENT to a fresh plan
// (issue #561's witness): a HIT yields exactly the Plan a re-plan would, and the
// cache REFUSES (forcing a recompute) on any change that could alter the resident
// view — a changed forecast, OR a store change that is not a pure append of spans
// that cannot enter the view. A pure append of a higher-benefit LIVE span misses,
// so the recompute folds it in; only appends of inert (sealed/tombstoned) spans —
// which can never be resident — keep the cached plan valid.
//
// It binds to cachemeta's plan-template plane through the SHARED fingerprint:
// PlanFingerprint is the deterministic reuse key a higher-tier adapter lowers into
// cachemeta.PlanTemplate (as the ParamsDigest of the forecast + the StateWitness of
// the store version), so a reused ctxplan plan is a first-class plan-template cache
// entry, not a private side cache.

// PlanFingerprint is the deterministic reuse key for a planned view: the forecast's
// fingerprint (the planner's "query") bound to the store's selection-relevant version
// (the data the query ran over). Two turns share a fingerprint iff a re-plan would
// produce the SAME Plan, so equality is exactly the condition under which the cached
// plan may be re-executed instead of recomputed.
type PlanFingerprint struct {
	Forecast string `json:"forecast"` // ForecastFingerprint — canonical over intents/horizon/pins/weights
	Store    string `json:"store"`    // StoreVersion — canonical over the SELECTION-RELEVANT span metadata
	Budget   int    `json:"budget"`   // the token cap (a different budget is a different plan)
}

// ForecastFingerprint is the deterministic identity of a forecast's PLANNING inputs.
// It is canonical: the SAME forecast (up to intent order, pin order, and a zero-vs-
// default weight vector) always hashes to the SAME value, and any change that would
// alter scoring — a new intent, a different horizon, a changed pin set, retuned
// weights — changes it. Intents are reduced to their DISTINCT content-token set (the
// exact tokens relevance() matches, sorted) so a reordering or a stopword edit that
// cannot change selection does not spuriously miss; pins are sorted; weights are
// taken through orDefault so a zero literal and an explicit DefaultWeights agree.
func ForecastFingerprint(f Forecast) string {
	h := sha256.New()
	writeField := func(s string) {
		_, _ = h.Write([]byte(s))
		_, _ = h.Write([]byte{0})
	}
	writeFloat := func(x float64) {
		writeField(strconv.FormatFloat(x, 'g', -1, 64))
	}

	// Intents -> the distinct content-token set relevance() actually scores against,
	// sorted, so reordering or a sub-token edit that cannot change a match is stable.
	toks := make([]string, 0)
	for t := range tokenSet(joinIntents(f.Intents)) {
		toks = append(toks, t)
	}
	sort.Strings(toks)
	writeField("intents")
	for _, t := range toks {
		writeField(t)
	}

	writeField("horizon")
	writeField(strconv.Itoa(f.Horizon))

	// Pins are a SET (order-independent); a duplicate pin id cannot change selection,
	// so dedup + sort before hashing.
	pins := dedupSorted(f.Pins)
	writeField("pins")
	for _, p := range pins {
		writeField(p)
	}

	w := f.Weights.orDefault()
	writeField("weights")
	writeFloat(w.Relevance)
	writeFloat(w.Utility)
	writeFloat(w.Durability)
	writeFloat(w.Recency)
	writeFloat(w.Primacy)

	return hex.EncodeToString(h.Sum(nil))
}

// StoreVersion is the deterministic identity of the SELECTION-RELEVANT state of a
// store: a fingerprint over every span's id, step, role, descriptor, digest, byte
// size, durability, and the two trust-gate bits — the exact fields Candidates/Optimize
// read to score and place a span. Spans are taken in id order so the version is
// independent of scan order. Two stores with this version equal plan IDENTICALLY under
// any forecast; any change to a scored field — a new span, a removed span, an edited
// descriptor, a flipped seal bit — changes the version and forces a recompute.
//
// It deliberately covers EVERY span (live and inert): a flip of a span's seal bit IS a
// selection change (an unsealed span becomes a candidate), so the version must move on
// it. Append-detection (appendedInert) is a SEPARATE, finer check layered on top.
func StoreVersion(spans []Span) string {
	idx := make([]int, len(spans))
	for i := range spans {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return spans[idx[a]].ID < spans[idx[b]].ID })

	h := sha256.New()
	field := func(s string) {
		_, _ = h.Write([]byte(s))
		_, _ = h.Write([]byte{0})
	}
	for _, i := range idx {
		s := spans[i]
		field(s.ID)
		field(strconv.Itoa(s.Step))
		field(s.Role)
		field(s.Descriptor)
		field(s.Digest)
		field(strconv.FormatInt(s.Bytes, 10))
		field(s.Durability)
		field(boolField(s.Sealed))
		field(boolField(s.Tombstoned))
		field(spanUtilityAttr(s)) // Attrs["utility"] is the only Attr Benefit reads
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Fingerprint computes the full reuse key for planning `spans` against `f` under
// `budget` — the value PlanCache is keyed on.
func Fingerprint(spans []Span, f Forecast, budget Budget) PlanFingerprint {
	return PlanFingerprint{
		Forecast: ForecastFingerprint(f),
		Store:    StoreVersion(spans),
		Budget:   normBudget(budget),
	}
}

// PlanCacheVerdict is the typed outcome of a plan-cache lookup: HIT re-executes the
// cached plan (no recompute); MISS recomputes. Reason names WHY a lookup missed so a
// caller can EXPLAIN the cache the way Optimize explains a plan.
type PlanCacheVerdict struct {
	Hit    bool   `json:"hit"`
	Reason string `json:"reason"` // "" on hit; one of the PlanCacheMiss* reasons on miss
	Plan   Plan   `json:"plan"`   // the reused plan on HIT; the zero Plan on MISS
}

// Plan-cache miss reasons.
const (
	PlanCacheMissCold     = "cold"             // nothing cached yet
	PlanCacheMissForecast = "forecast_changed" // the forecast fingerprint changed
	PlanCacheMissBudget   = "budget_changed"   // the token budget changed
	PlanCacheMissStore    = "store_not_append" // the store changed by more than an inert append
)

// PlanCache is the prepared-statement cache: one cached (fingerprint, plan, spans)
// entry the planner re-executes across stable turns. It is NOT an LRU — a single
// entry is the whole mechanism, because the win is turn-to-turn stability (the next
// turn's forecast is usually the previous turn's, lightly revised). A higher-tier
// caller that wants more than one prepared plan keys a map by PlanFingerprint; this
// type is the minimal, deterministic core.
//
// The zero value is a usable empty cache (every Lookup is a cold MISS). PlanCache is
// not safe for concurrent use; a caller serializes turns (the planner is per-session).
type PlanCache struct {
	primed bool
	key    PlanFingerprint
	plan   Plan
	spans  []Span // the span set the cached plan was built over (for append-detection)
}

// Lookup asks whether the cached plan may be re-executed for planning `spans` against
// `f` under `budget`. It returns a HIT (carrying the cached plan) iff the forecast and
// budget are unchanged AND the store changed by at most an INERT append (zero or more
// appended spans, none of which can enter the resident view) — the only store delta
// that provably cannot alter the selection. Otherwise it returns a MISS with the
// reason, and the caller recomputes (then calls Store to prime the cache).
//
// A HIT is observably equivalent to a fresh plan: under these conditions Optimize
// would return the byte-identical Plan, so re-execution is a sound substitute for a
// re-plan — the issue's correctness witness.
func (c *PlanCache) Lookup(spans []Span, f Forecast, budget Budget) PlanCacheVerdict {
	if !c.primed {
		return PlanCacheVerdict{Reason: PlanCacheMissCold}
	}
	if got := ForecastFingerprint(f); got != c.key.Forecast {
		return PlanCacheVerdict{Reason: PlanCacheMissForecast}
	}
	if normBudget(budget) != c.key.Budget {
		return PlanCacheVerdict{Reason: PlanCacheMissBudget}
	}
	// Store unchanged (selection-relevant version equal) OR grew only by inert appends
	// (spans that can never be resident). Either way the cached plan is still optimal.
	if StoreVersion(spans) == c.key.Store || appendedInert(c.spans, spans) {
		return PlanCacheVerdict{Hit: true, Plan: c.plan}
	}
	return PlanCacheVerdict{Reason: PlanCacheMissStore}
}

// Store primes (or replaces) the cache with a freshly-computed plan and the inputs it
// was computed over. A caller invokes it after a MISS recompute, so the next stable
// turn HITs.
func (c *PlanCache) Store(spans []Span, f Forecast, budget Budget, p Plan) {
	c.primed = true
	c.key = Fingerprint(spans, f, budget)
	c.plan = p
	// Copy the span set so a caller mutating its slice cannot corrupt append-detection.
	c.spans = append([]Span(nil), spans...)
}

// PlanWithCache is the cached planning entry point: a drop-in for PlanCells that
// consults the cache first. On a HIT it returns the cached plan and hit=true WITHOUT
// re-scoring; on a MISS it computes a fresh plan via PlanCells, primes the cache, and
// returns hit=false. The returned plan is byte-identical to PlanCells(spans,f,budget,
// cost) in BOTH cases — the cache changes only whether the work was done this turn.
func (c *PlanCache) PlanWithCache(spans []Span, f Forecast, budget Budget, cost CostModel) (Plan, bool) {
	if v := c.Lookup(spans, f, budget); v.Hit {
		return v.Plan, true
	}
	p := PlanCells(spans, f, budget, cost)
	c.Store(spans, f, budget, p)
	return p, false
}

// appendedInert reports whether `cur` is `prev` with zero or more spans APPENDED, where
// every appended span is INERT — sealed or tombstoned, so it can never enter the
// resident view (Optimize elides sealed/tombstoned spans up front). Under that delta
// the candidate set the planner would SELECT from is unchanged, so the cached plan is
// still optimal. It is the conservative, fail-closed append rule:
//
//   - every span in `prev` (by id) must be present in `cur` and BYTE-IDENTICAL in its
//     selection-relevant fields (a modified pre-existing span is NOT an append);
//   - every span in `cur` NOT in `prev` must be inert (an appended LIVE span — a
//     possibly-higher-benefit candidate — is refused, so the recompute folds it in);
//   - no `prev` span may be missing from `cur` (a removal is not an append).
//
// A span is matched by ID; a duplicate id in either set fails closed (returns false),
// since the recovery-handle uniqueness contract (storeaudit.go) is then violated.
func appendedInert(prev, cur []Span) bool {
	prevByID := make(map[string]Span, len(prev))
	for _, s := range prev {
		if _, dup := prevByID[s.ID]; dup {
			return false // ambiguous prior — fail closed
		}
		prevByID[s.ID] = s
	}
	curByID := make(map[string]Span, len(cur))
	for _, s := range cur {
		if _, dup := curByID[s.ID]; dup {
			return false // ambiguous current — fail closed
		}
		curByID[s.ID] = s
	}
	// Every prior span must survive UNCHANGED in its selection-relevant fields.
	for id, ps := range prevByID {
		cs, ok := curByID[id]
		if !ok {
			return false // a prior span was removed — not an append
		}
		if !selectionEqual(ps, cs) {
			return false // a prior span was edited — not an append
		}
	}
	// Every NEW span must be inert (cannot be resident); an appended live span misses.
	for id, cs := range curByID {
		if _, existed := prevByID[id]; existed {
			continue
		}
		if !cs.Sealed && !cs.Tombstoned {
			return false // an appended live candidate — refuse, force a recompute
		}
	}
	return true
}

// selectionEqual reports whether two spans are equal in every field the planner reads
// to score and place a span — the same fields StoreVersion hashes. It is the per-span
// "would this change the plan?" predicate appendedInert uses to reject an edited span.
func selectionEqual(a, b Span) bool {
	return a.ID == b.ID &&
		a.Step == b.Step &&
		a.Role == b.Role &&
		a.Descriptor == b.Descriptor &&
		a.Digest == b.Digest &&
		a.Bytes == b.Bytes &&
		a.Durability == b.Durability &&
		a.Sealed == b.Sealed &&
		a.Tombstoned == b.Tombstoned &&
		spanUtilityAttr(a) == spanUtilityAttr(b)
}

// dedupSorted returns the sorted, de-duplicated copy of ids — a SET, since pin order
// and duplicate pins cannot change selection.
func dedupSorted(ids []string) []string {
	if len(ids) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(ids))
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// spanUtilityAttr extracts the one Attr the benefit model reads (Attrs["utility"]) as a
// plain string, so a span with no Attrs map and a span with an absent key fingerprint
// identically. Any other Attrs entry is provenance metadata the planner never scores,
// so it is deliberately excluded from the selection identity.
func spanUtilityAttr(s Span) string {
	if s.Attrs == nil {
		return ""
	}
	return s.Attrs["utility"]
}

// boolField renders a bool as a stable one-byte field for hashing.
func boolField(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

// normBudget clamps a budget the way Optimize does (a negative budget plans as 0), so
// two budgets that plan identically fingerprint identically.
func normBudget(b Budget) int {
	if b.Tokens < 0 {
		return 0
	}
	return b.Tokens
}
