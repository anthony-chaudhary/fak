package ctxplan

import "sort"

// layout.go — the flexible O(1) context layout profile. ProbeOptions exposes the
// original index knobs (one recency window + one candidate cap). Layout lifts those knobs
// into the four operator-facing regions the live prompt actually has:
//
//   - current: the newest entry/entries, usually exact and forced resident.
//   - recent: the last N entries before current, planned as near-verbatim history.
//   - deep: old history reached by relevance/durability, usually planned or pointer-only.
//   - base: structural prompt material such as system/developer/task pins.
//
// Each area has its own N, optional token-size cap, and precision. The total per-turn
// compute is still bounded by the sum of those area limits (and optionally
// Layout.MaxCandidates), so making the view more flexible does not give up the O(1)
// performance claim.

// Area names carried into Selection/Elision metadata for EXPLAIN and audits.
const (
	AreaBase    = "base"
	AreaCurrent = "current"
	AreaRecent  = "recent"
	AreaDeep    = "deep"
)

// Area precision controls what the planner may do with spans reached through that area.
const (
	// PrecisionExact means exact bytes are forced resident by pinning the span. The trust
	// gate still wins: a sealed/tombstoned span is elided, never rendered.
	PrecisionExact = "exact"
	// PrecisionPlanned means the span is a normal candidate: it may be selected exactly or
	// elided recoverably depending on benefit/cost and budget.
	PrecisionPlanned = "planned"
	// PrecisionPointer means the span is represented as a cold, recoverable pointer in
	// this turn. It is never selected until an explicit demand-page/expand operation asks
	// for it.
	PrecisionPointer = "pointer"
)

const (
	DefaultBaseSpans    = 8
	DefaultCurrentSpans = 1
	DefaultDeepSpans    = 64
)

// AreaPolicy configures one area of the reconstructed context. MaxSpans is the N for the
// area; 0 means the default for that area, and a negative value disables the area.
// MaxTokens, when positive, caps that area's estimated candidate tokens before the global
// Budget is optimized. Precision defaults per area (base/current exact, recent/deep
// planned).
type AreaPolicy struct {
	MaxSpans  int    `json:"max_spans,omitempty"`
	MaxTokens int    `json:"max_tokens,omitempty"`
	Precision string `json:"precision,omitempty"`
}

// Layout is the caller-controlled four-area context profile. It controls the candidate
// access path only; the Budget still caps resident tokens globally, so a layout can widen
// or narrow what is considered without making the rendered view unbounded.
type Layout struct {
	Base    AreaPolicy `json:"base,omitempty"`
	Current AreaPolicy `json:"current,omitempty"`
	Recent  AreaPolicy `json:"recent,omitempty"`
	Deep    AreaPolicy `json:"deep,omitempty"`

	// IncludeDurability selects which durability classes the deep area may reach through
	// the durability access path. nil defaults to {durable, bounded}.
	IncludeDurability []string `json:"include_durability,omitempty"`
	// MaxCandidates is a final hard cap across all areas. 0 means DefaultMaxCandidates;
	// negative means no extra global cap beyond the per-area limits.
	MaxCandidates int `json:"max_candidates,omitempty"`
}

// DefaultLayout is the seed four-area profile: exact base pins, one exact current entry,
// a recent tail of DefaultRecencyWindow spans, and a bounded deep-history probe.
func DefaultLayout() Layout {
	return Layout{
		Base:              AreaPolicy{MaxSpans: DefaultBaseSpans, Precision: PrecisionExact},
		Current:           AreaPolicy{MaxSpans: DefaultCurrentSpans, Precision: PrecisionExact},
		Recent:            AreaPolicy{MaxSpans: DefaultRecencyWindow, Precision: PrecisionPlanned},
		Deep:              AreaPolicy{MaxSpans: DefaultDeepSpans, Precision: PrecisionPlanned},
		IncludeDurability: []string{DurabilityDurable, DurabilityBounded},
		MaxCandidates:     DefaultMaxCandidates,
	}
}

// AreaSpan is one span reached by a Layout probe, annotated with the area and precision
// that reached it.
type AreaSpan struct {
	Span
	Area      string
	Precision string
	priority  int
}

func (l Layout) withDefaults() Layout {
	def := DefaultLayout()
	l.Base = areaWithDefault(l.Base, def.Base)
	l.Current = areaWithDefault(l.Current, def.Current)
	l.Recent = areaWithDefault(l.Recent, def.Recent)
	l.Deep = areaWithDefault(l.Deep, def.Deep)
	if l.IncludeDurability == nil {
		l.IncludeDurability = def.IncludeDurability
	}
	if l.MaxCandidates == 0 {
		l.MaxCandidates = def.MaxCandidates
	}
	return l
}

func areaWithDefault(p, def AreaPolicy) AreaPolicy {
	if p.MaxSpans == 0 {
		p.MaxSpans = def.MaxSpans
	}
	if p.Precision == "" {
		p.Precision = def.Precision
	}
	p.Precision = normPrecision(p.Precision)
	return p
}

func normPrecision(p string) string {
	switch p {
	case PrecisionExact, PrecisionPlanned, PrecisionPointer:
		return p
	default:
		return PrecisionPlanned
	}
}

// ProbeLayout returns the bounded candidate set reached by the four-area layout, with
// area/precision metadata attached. A span reached by multiple areas is assigned to the
// highest-priority one: current, base, recent, then deep. The final order is render order.
func (ix *Index) ProbeLayout(f Forecast, layout Layout, cost CostModel) []AreaSpan {
	layout = layout.withDefaults()
	if cost == nil {
		cost = TokenCost
	}

	type hit struct {
		idx       int
		area      string
		precision string
		priority  int
	}
	hits := map[int]hit{}
	add := func(idx int, area string, p AreaPolicy, priority int) bool {
		if p.MaxSpans < 0 || idx < 0 || idx >= len(ix.spans) {
			return false
		}
		h := hit{idx: idx, area: area, precision: normPrecision(p.Precision), priority: priority}
		if old, ok := hits[idx]; !ok || h.priority < old.priority {
			hits[idx] = h
			return true
		}
		return false
	}
	spanCost := func(idx int) int {
		c := cost(ix.spans[idx])
		if c < 0 {
			return 0
		}
		return c
	}
	fits := func(used *int, idx int, p AreaPolicy) bool {
		if p.MaxTokens <= 0 {
			return true
		}
		c := spanCost(idx)
		if *used+c > p.MaxTokens {
			return false
		}
		*used += c
		return true
	}
	addWithin := func(used *int, idx int, area string, p AreaPolicy, priority int) bool {
		if old, ok := hits[idx]; ok && old.priority <= priority {
			return false
		}
		if !fits(used, idx, p) {
			return false
		}
		return add(idx, area, p, priority)
	}

	// Priority 0 — current: the newest entry/entries are the most specific region.
	if layout.Current.MaxSpans > 0 {
		n := layout.Current.MaxSpans
		used := 0
		for i := len(ix.spans) - 1; i >= 0 && n > 0; i-- {
			if addWithin(&used, i, AreaCurrent, layout.Current, 0) {
				n--
			}
		}
	}

	// Priority 1 — base: caller pins (system/developer/task prompt or other structural
	// anchors). Pins are IDs, so this is independent of where the base material lives.
	if layout.Base.MaxSpans > 0 {
		n := layout.Base.MaxSpans
		used := 0
		for _, id := range f.Pins {
			if n <= 0 {
				break
			}
			if i, ok := ix.byID[id]; ok {
				if addWithin(&used, i, AreaBase, layout.Base, 1) {
					n--
				}
			}
		}
	}

	// Priority 2 — recent: the N entries immediately before the current area. Duplicates
	// with current/base are ignored by the priority map above.
	if layout.Recent.MaxSpans > 0 {
		skipCurrent := layout.Current.MaxSpans
		if skipCurrent < 0 {
			skipCurrent = 0
		}
		hi := len(ix.spans) - skipCurrent - 1
		n := layout.Recent.MaxSpans
		used := 0
		for i := hi; i >= 0 && n > 0; i-- {
			if addWithin(&used, i, AreaRecent, layout.Recent, 2) {
				n--
			}
		}
	}

	// Priority 3 — deep: relevance postings plus the configured durable classes. The
	// collection order is deterministic before the area cap is applied.
	if layout.Deep.MaxSpans > 0 {
		deep := map[int]bool{}
		for t := range tokenSet(joinIntents(f.Intents)) {
			for _, j := range ix.posting[t] {
				deep[j] = true
			}
		}
		admit := durabilityAdmitSet(layout.IncludeDurability)
		for _, i := range ix.durable {
			if admit[NormDurability(ix.spans[i].Durability)] {
				deep[i] = true
			}
		}
		order := make([]int, 0, len(deep))
		for i := range deep {
			if _, already := hits[i]; already {
				continue
			}
			order = append(order, i)
		}
		sort.Slice(order, func(a, b int) bool {
			sa, sb := ix.spans[order[a]], ix.spans[order[b]]
			if sa.Step != sb.Step {
				return sa.Step > sb.Step
			}
			return sa.ID < sb.ID
		})
		if len(order) > layout.Deep.MaxSpans {
			order = order[:layout.Deep.MaxSpans]
		}
		used := 0
		for _, i := range order {
			addWithin(&used, i, AreaDeep, layout.Deep, 3)
		}
	}

	out := make([]AreaSpan, 0, len(hits))
	for _, h := range hits {
		out = append(out, AreaSpan{
			Span: ix.spans[h.idx], Area: h.area, Precision: h.precision, priority: h.priority,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].priority != out[j].priority {
			return out[i].priority < out[j].priority
		}
		if out[i].Step != out[j].Step {
			return out[i].Step > out[j].Step
		}
		return out[i].ID < out[j].ID
	})
	if layout.MaxCandidates > 0 && len(out) > layout.MaxCandidates {
		out = out[:layout.MaxCandidates]
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Step != out[j].Step {
			return out[i].Step < out[j].Step
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// PlanLayout plans an O(1) view using the flexible four-area layout. PrecisionExact spans
// are added to the hard pin set; PrecisionPlanned spans go through the normal optimizer;
// PrecisionPointer spans are intentionally elided as recoverable pointers.
func (ix *Index) PlanLayout(f Forecast, b Budget, cost CostModel, layout Layout) Plan {
	probe := ix.ProbeLayout(f, layout, cost)
	spans := make([]Span, len(probe))
	for i, s := range probe {
		spans[i] = s.Span
	}
	cands := Candidates(spans, f, cost)
	for i := range cands {
		cands[i].Area = probe[i].Area
		cands[i].Precision = probe[i].Precision
	}

	pins := pinSet(f.Pins)
	if pins == nil {
		pins = map[string]bool{}
	}
	var planned []Candidate
	var pointer []Elision
	for _, c := range cands {
		switch c.Precision {
		case PrecisionExact:
			pins[c.Cell.ID] = true
			planned = append(planned, c)
		case PrecisionPointer:
			pointer = append(pointer, pointerElision(c))
		default:
			planned = append(planned, c)
		}
	}

	p := Optimize(planned, b, pins, ObjGreedy)
	p.Candidates += len(pointer)
	p.Elided = append(p.Elided, pointer...)
	p.Horizon = f.Horizon
	finalize(&p, p.CostUsed)
	return p
}

func pointerElision(c Candidate) Elision {
	switch {
	case c.Cell.Sealed:
		return elisionOf(c, ElideSealed)
	case c.Cell.Tombstoned:
		return elisionOf(c, ElideTombstoned)
	default:
		return elisionOf(c, ElidePointer)
	}
}
