package fitplan

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// T04 joint model-fit search (`fak serve --fit auto`). Plan prices ONE pinned boot
// config; Search enumerates the ctx x KV-dtype x slots grid, scores every row with
// the same T01 roofline predictions Plan uses, gates each against the stated
// workload + SLO, and picks the max-throughput row at B* = min(B_mem, B_slo).
// Explicit sizing pins win over the search: a pin fixes its axis exactly and is
// never silently overridden — an infeasible pin refuses typed (pin_infeasible),
// mirroring llama.cpp's "an explicit flag is never overridden by auto-sizing".
//
// Borrowed shapes, pinned in docs/tickets/model-fit-auto-planner-20261023/TICKET-04.md:
//   - vLLM: the largest context that fits a fixed slot count is found by binary
//     search over the monotone per-stream charge (max-model-length shape), not a
//     linear scan.
//   - TensorRT-LLM: the chunked-prefill token budget is batch x (1 + ISL/OSL),
//     clamped to [2048, 32768]; TTFT is priced as the sum of prefill chunks.
//   - NVIDIA aiconfigurator: SLA filtering drops configs that miss the latency
//     SLO before ranking by throughput.
//
// The search is DETERMINISTIC: no clock reads, no allocation order; ties break on
// closed keys (aggregate desc, context desc, slots asc, then enumeration order).

// DefaultOutputTokens is the stated-workload default for output tokens when the
// caller gives none: one typical agent-turn completion. It pairs with the T01
// DefaultPromptTokens to bound the chunked-prefill budget (ISL/OSL ratio).
const DefaultOutputTokens = 512

const (
	maxSearchSlots  = 64 // enumeration cap on the slots ladder (a target/pinned value above it still gets its own row)
	maxAlternatives = 3  // top-N besides the chosen row reported in SearchReport.Alternatives
)

// Refusal reasons added by the T04 joint search (additive to the T01 vocabulary).
const (
	ReasonFitsOnlyAtCtxLT       RefusalReason = "fits_only_at_ctx_lt"
	ReasonNeedsRing             RefusalReason = "needs_ring"
	ReasonBelowSLO              RefusalReason = "below_slo"
	ReasonPinInfeasible         RefusalReason = "pin_infeasible"
	ReasonNoRankableCalibration RefusalReason = "no_rankable_calibration"
)

var (
	ErrFitsOnlyAtCtxLT       = errors.New("fitplan: fits_only_at_ctx_lt")
	ErrNeedsRing             = errors.New("fitplan: needs_ring")
	ErrBelowSLO              = errors.New("fitplan: below_slo")
	ErrPinInfeasible         = errors.New("fitplan: pin_infeasible")
	ErrNoRankableCalibration = errors.New("fitplan: no_rankable_calibration")
)

// rankable reports whether the calibration may RANK rows against each other: SOL
// rows carry no engine efficiency and drift rows are demoted out of ranking, so
// neither can pick a config (T04 NO_RANKABLE_CALIBRATION). Empty/unknown tiers
// are equally unrankable.
func (c Calibration) rankable() bool {
	switch c.Tier {
	case TierEmpirical, TierCalibrated, TierSilicon:
		return c.State != RowDrift
	default:
		return false
	}
}

// SearchWorkload is the stated serving load the search gates against: prompt and
// output token counts (the ISL/OSL pair) and the target concurrency (slots).
type SearchWorkload struct {
	PromptTokens int `json:"prompt_tokens"`
	OutputTokens int `json:"output_tokens"`
	Concurrency  int `json:"concurrency"`
}

// SearchSLO is the optional latency SLO the search filters against (aiconfigurator
// shape): TPOT and TTFT ceilings in milliseconds. Zero means unconstrained.
type SearchSLO struct {
	TPOTMS float64 `json:"tpot_ms,omitempty"`
	TTFTMS float64 `json:"ttft_ms,omitempty"`
}

// SearchPins are the explicit sizing flags the operator set; each nonzero pin
// fixes its axis EXACTLY (never overridden, never silently changed). A pin that
// cannot be honored refuses typed instead of falling back to a nearby row.
type SearchPins struct {
	ContextTokens int    `json:"context_tokens,omitempty"`
	KVPrecision   string `json:"kv_precision,omitempty"`
	Slots         int    `json:"slots,omitempty"`
	Placement     string `json:"placement,omitempty"` // PlacementDeviceResident refuses rather than offload
}

// SearchRow is one scored point of the ctx x KV-dtype x slots grid.
type SearchRow struct {
	KVPrecision   string  `json:"kv_precision"`
	ContextTokens int     `json:"context_tokens"`
	Slots         int     `json:"slots"`
	BStar         int     `json:"b_star"`                    // min(B_mem, B_slo, slots)
	AggregateTokS float64 `json:"aggregate_tok_s,omitempty"` // B* x per-seq decode rate
	TTFTSeconds   float64 `json:"ttft_seconds,omitempty"`    // chunked-prefill estimate at B*
	Current       bool    `json:"current,omitempty"`
}

// SearchReport is the T04 joint-search result: the chosen row, the top-N
// alternatives, the evidence the rows were priced against, and the typed refusal
// when no row is eligible. StatusUnknown keeps the T01 fail-open semantics: the
// rows may still be present (Chosen non-nil) but the pool or host capacity is
// unprobeable, so nothing was refused.
type SearchReport struct {
	Status       Status         `json:"status"`
	Chosen       *SearchRow     `json:"chosen,omitempty"`
	Alternatives []SearchRow    `json:"alternatives"`
	Refusal      *Refusal       `json:"refusal,omitempty"`
	Workload     SearchWorkload `json:"workload"`
	SLO          SearchSLO      `json:"slo"`
	Pins         SearchPins     `json:"pins"`
	Evidence     Evidence       `json:"evidence"`
	Omitted      []Omission     `json:"omitted,omitempty"`
	Model        *Model         `json:"model,omitempty"`
	// Placement is the dense-core-first offload candidate when the weights exceed the pool.
	Placement *TieredPlacement `json:"placement,omitempty"`
}

// SearchInput is everything Search needs: the T01 plan inputs plus the stated
// workload, SLO, pins, and the resolved context (the auto-sized boot context,
// which anchors the context ladder). CurrentKVPrecision marks the row matching
// today's boot config (for the report's current-row flag, not for selection).
type SearchInput struct {
	Model           Model
	Weights         compute.MemoryPlan
	Hardware        Hardware
	Calibration     Calibration
	Tiers           []KVTier
	ResolvedContext int
	CurrentKV       string
	Workload        SearchWorkload
	SLO             SearchSLO
	Pins            SearchPins
}

// searchScored is one priced grid row plus its eligibility facts. The row's
// Slots is the enumerated operating point; BStar is what the box actually
// serves there: min(B_mem, slots, B_slo).
type searchScored struct {
	row     SearchRow
	ctxOK   bool    // ctx >= ISL
	ttftOK  bool    // TTFT within SLO (true when unconstrained)
	bMem    int     // memory-only concurrent streams at this ctx
	tpot1MS float64 // predicted TPOT at batch 1, ms; 0 = unpriceable
}

// Search runs the T04 joint search. It reuses Plan's pricing byte-for-byte (the
// shared planner); only the enumeration, gating, and selection are new.
func Search(in SearchInput) SearchReport {
	isl := in.Workload.PromptTokens
	if isl <= 0 {
		isl = DefaultPromptTokens
	}
	osl := in.Workload.OutputTokens
	if osl <= 0 {
		osl = DefaultOutputTokens
	}
	target := in.Workload.Concurrency
	if target <= 0 {
		target = 1
	}
	rep := SearchReport{
		Status:       StatusFits,
		Workload:     SearchWorkload{PromptTokens: isl, OutputTokens: osl, Concurrency: target},
		SLO:          in.SLO,
		Pins:         in.Pins,
		Alternatives: []SearchRow{},
	}

	p := newPlanner(Input{
		Model:       in.Model,
		Weights:     in.Weights,
		Hardware:    in.Hardware,
		Workload:    Workload{PromptTokens: isl, Concurrency: target},
		Calibration: in.Calibration,
	})
	rep.Evidence = p.evidence()
	rep.Model = &in.Model

	// Hard memory facts first: they hold regardless of calibration quality. Oversize weights
	// are a cost, not a refusal, unless the operator pinned the all-resident placement.
	if searchTiered(p, &rep, in, isl) {
		return rep
	}
	if p.hostShort {
		refuseSearch(&rep, ReasonWeightsExceedPool, "", 0, 0, p.hostWeights-p.in.Hardware.HostPoolUsableBytes, true)
		return rep
	}
	if !in.Calibration.rankable() {
		refuseSearch(&rep, ReasonNoRankableCalibration, "", 0, 0, 0, false)
		return rep
	}
	if p.gate != "" {
		rep.Status = StatusUnknown
		rep.Omitted = []Omission{p.gate}
		return rep
	}
	if p.avail <= 0 {
		rep.Status = StatusUnknown
		rep.Omitted = append(rep.Omitted, OmitPoolCapacityUnknown)
		return rep
	}

	tiers := dedupeTiers(in.Tiers)
	if len(tiers) == 0 {
		rep.Status = StatusUnknown
		rep.Omitted = append(rep.Omitted, OmitKVGeometryUnknown)
		return rep
	}

	ctxs := contextLadder(in.ResolvedContext, in.Model.MaxContextTokens, in.Pins.ContextTokens)
	if len(ctxs) == 0 {
		rep.Status = StatusUnknown
		rep.Omitted = append(rep.Omitted, OmitContextUnknown)
		return rep
	}
	if in.Pins.ContextTokens > 0 {
		ctxs = []int{in.Pins.ContextTokens} // a pin fixes the axis exactly
	}
	slotsList := slotLadder(target, in.Pins.Slots)
	if in.Pins.Slots > 0 {
		slotsList = []int{in.Pins.Slots}
	}
	if pin := strings.ToLower(strings.TrimSpace(in.Pins.KVPrecision)); pin != "" {
		var filtered []KVTier
		for _, t := range tiers {
			if strings.ToLower(t.Precision) == pin {
				filtered = append(filtered, t)
			}
		}
		if len(filtered) == 0 {
			refuseSearch(&rep, ReasonPinInfeasible, pin, 0, 0, 0, false)
			rep.Refusal.Pin = "kv_precision=" + pin
			return rep
		}
		tiers = filtered
	}

	var rows []searchScored
	for _, tier := range tiers {
		c := Candidate{KVPrecision: tier.Precision, Context: tier.Context}
		if !kvSizable(c.Context) {
			continue
		}
		for _, ctx := range ctxs {
			c.ContextTokens = ctx
			lkv := min(isl, ctx)
			rate1 := p.stepRateAt(c, 1, lkv)
			for _, slots := range slotsList {
				q := p.atConcurrency(slots)
				pr := q.predict(c)
				bMem := pr.MaxConcurrency
				if bMem < 1 {
					continue // not even one stream fits at this ctx
				}
				bStar := min(bMem, slots)
				if in.SLO.TPOTMS > 0 && bStar > 0 {
					bStar = maxBatchAtRate(p, c, lkv, bStar, 1000.0/in.SLO.TPOTMS)
				}
				ttft := 0.0
				if bStar > 0 {
					ttft = p.chunkedTTFT(c, lkv, bStar, osl)
				}
				rows = append(rows, searchScored{
					row: SearchRow{
						KVPrecision:   tier.Precision,
						ContextTokens: ctx,
						Slots:         slots,
						BStar:         bStar,
						AggregateTokS: aggregateAt(p, c, lkv, bStar),
						TTFTSeconds:   ttft,
						Current:       in.CurrentKV != "" && strings.EqualFold(in.CurrentKV, tier.Precision) && ctx == in.ResolvedContext && slots == target,
					},
					ctxOK:   ctx >= isl,
					ttftOK:  in.SLO.TTFTMS <= 0 || ttft*1000 <= in.SLO.TTFTMS+1e-9,
					bMem:    bMem,
					tpot1MS: tpotMS(rate1),
				})
			}
		}
	}

	// Rows sharing (tier, ctx, B*) are the same achievable configuration priced
	// at different enumerated operating points; keep the tightest one (rows are
	// generated in ascending slot order, so the first occurrence wins).
	deduped := rows[:0]
	seenKey := map[string]bool{}
	for _, s := range rows {
		key := fmt.Sprintf("%s|%d|%d", s.row.KVPrecision, s.row.ContextTokens, s.row.BStar)
		if seenKey[key] {
			continue
		}
		seenKey[key] = true
		deduped = append(deduped, s)
	}
	rows = deduped

	// A row serves the stated workload iff its context covers the prompt, its
	// TTFT holds the SLO, and B* reaches the target concurrency.
	eligible := make([]searchScored, 0, len(rows))
	for _, s := range rows {
		if s.ctxOK && s.ttftOK && s.row.BStar >= target {
			eligible = append(eligible, s)
		}
	}
	sort.SliceStable(eligible, func(i, j int) bool {
		a, b := eligible[i].row, eligible[j].row
		if a.AggregateTokS != b.AggregateTokS {
			return a.AggregateTokS > b.AggregateTokS
		}
		if a.ContextTokens != b.ContextTokens {
			return a.ContextTokens > b.ContextTokens
		}
		return a.Slots < b.Slots
	})
	if len(eligible) > 0 {
		chosen := eligible[0]
		rep.Chosen = &chosen.row
		alt := eligible[1:]
		if len(alt) > maxAlternatives {
			alt = alt[:maxAlternatives]
		}
		for _, a := range alt {
			rep.Alternatives = append(rep.Alternatives, a.row)
		}
		if p.hostUnknown {
			rep.Status = StatusUnknown
			rep.Omitted = append(rep.Omitted, OmitHostPoolCapacityUnknown)
		}
		return rep
	}

	// No eligible row: classify into the typed refusal vocabulary.
	if anyPinSet(in.Pins) {
		classifyPin(&rep, rows, in, isl, target)
		return rep
	}
	if subSat(p.avail, p.weights) < 0 {
		if in.Model.NumExperts > 0 && in.Model.RoutedExpertBytes > 0 {
			refuseSearch(&rep, ReasonNeedsRing, "", 0, 0, subSat(p.weights, p.avail), false)
		} else {
			refuseSearch(&rep, ReasonWeightsExceedPool, "", 0, 0, subSat(p.weights, p.avail), false)
		}
		return rep
	}
	desired := in.ResolvedContext
	if desired < isl {
		desired = isl
	}
	maxOne, kvOne := p.maxSlotContextFor(tiers, 1)
	maxTarget, _ := p.maxSlotContextFor(tiers, target)
	switch {
	case maxOne == 0:
		refuseSearch(&rep, ReasonContextExceedsPool, bestTierPrecision(tiers), 0, 0, p.shortfallOneStream(tiers, isl), false)
	case maxTarget < desired:
		if maxOne < desired {
			refuseSearch(&rep, ReasonFitsOnlyAtCtxLT, kvOne, maxOne, 0, 0, false)
			rep.Refusal.RequestedContext = desired
		} else {
			refuseSearch(&rep, ReasonConcurrencyExceedsPool, "", 0, maxConcAt(rows, desired), 0, false)
		}
	default:
		refuseSLO(&rep, rows, in.SLO, target)
	}
	return rep
}

// searchTiered answers an unpinned weights overflow with the tiered placement, priced on the
// pinned (else first) KV tier at the pinned (else resolved) context.
func searchTiered(p *planner, rep *SearchReport, in SearchInput, isl int) bool {
	if !p.weightsOverflow() || pinnedResident(in.Pins.Placement) {
		return false
	}
	var kv *Candidate
	pin := strings.ToLower(strings.TrimSpace(in.Pins.KVPrecision))
	for _, t := range dedupeTiers(in.Tiers) {
		if pin == "" || strings.ToLower(t.Precision) == pin {
			ctx := in.Pins.ContextTokens
			if ctx <= 0 {
				ctx = in.ResolvedContext
			}
			if ctx <= 0 {
				ctx = isl
			}
			kv = &Candidate{KVPrecision: t.Precision, Context: t.Context, ContextTokens: ctx}
			break
		}
	}
	tp := p.placeTiered(kv)
	if tp == nil {
		return false
	}
	rep.Status, rep.Placement = StatusTiered, tp
	return true
}

// aggregateAt is B x per-seq decode rate, 0 for B < 1.
func aggregateAt(p *planner, c Candidate, lkv, b int) float64 {
	if b < 1 {
		return 0
	}
	return float64(b) * p.stepRateAt(c, b, lkv)
}

func tpotMS(rate float64) float64 {
	if rate <= 0 {
		return 0
	}
	return 1000 / rate
}

// classifyPin turns a failed pinned search into a typed pin_infeasible refusal:
// the pin is honored verbatim in the report (never silently changed) and the
// missing thing is named precisely. The verdict is judged on the canonical
// row — the tier with the most headroom at the TARGET concurrency (always in
// the enumerated slot ladder) — because that is the config the operator asked for.
func classifyPin(rep *SearchReport, rows []searchScored, in SearchInput, isl, target int) {
	pin := pinString(in.Pins)
	rep.Status = StatusRefused
	rep.Refusal = &Refusal{Reason: ReasonPinInfeasible, Pin: pin}
	var canon *searchScored
	for i := range rows {
		s := &rows[i]
		if s.row.Slots != target {
			continue
		}
		if canon == nil || s.bMem > canon.bMem {
			canon = s
		}
	}
	if canon == nil {
		// Not even one stream fit at the pinned combo: name the memory shortfall.
		rep.Refusal.ShortfallBytes = pinShortfall(in, isl, target)
		return
	}
	switch {
	case canon.bMem < target:
		rep.Refusal.ShortfallBytes = pinShortfall(in, isl, target)
	case !canon.ctxOK:
		rep.Refusal.RequestedContext = isl
	case !canon.ttftOK:
		rep.Refusal.SLOBound = "ttft"
		rep.Refusal.SLOLimitMS = in.SLO.TTFTMS
		rep.Refusal.PredictedMS = canon.row.TTFTSeconds * 1000
	default:
		// Memory and window hold, but the TPOT SLO caps B* below the target.
		rep.Refusal.SLOBound = "tpot"
		rep.Refusal.SLOLimitMS = in.SLO.TPOTMS
		rep.Refusal.PredictedMS = canon.tpot1MS
	}
}

// pinShortfall names the bytes short of serving `target` streams at the pinned
// context, on the tier closest to fitting.
func pinShortfall(in SearchInput, isl, target int) int64 {
	p := newPlanner(Input{
		Model:       in.Model,
		Weights:     in.Weights,
		Hardware:    in.Hardware,
		Workload:    Workload{PromptTokens: isl, Concurrency: target},
		Calibration: in.Calibration,
	})
	best := int64(0)
	found := false
	for _, tier := range dedupeTiers(in.Tiers) {
		c := Candidate{KVPrecision: tier.Precision, Context: tier.Context}
		if !kvSizable(c.Context) {
			continue
		}
		ctx := in.Pins.ContextTokens
		if ctx <= 0 {
			ctx = isl
		}
		c.ContextTokens = ctx
		slots := in.Pins.Slots
		if slots <= 0 {
			slots = target
		}
		short := subSat(p.needAt(c, slots), p.avail)
		if short > 0 && (!found || short < best) {
			best, found = short, true
		}
	}
	return best
}

func pinString(p SearchPins) string {
	var parts []string
	if p.ContextTokens > 0 {
		parts = append(parts, fmt.Sprintf("context_tokens=%d", p.ContextTokens))
	}
	if s := strings.TrimSpace(p.KVPrecision); s != "" {
		parts = append(parts, "kv_precision="+s)
	}
	if p.Slots > 0 {
		parts = append(parts, fmt.Sprintf("slots=%d", p.Slots))
	}
	return strings.Join(parts, "+")
}

func anyPinSet(p SearchPins) bool {
	return p.ContextTokens > 0 || strings.TrimSpace(p.KVPrecision) != "" || p.Slots > 0
}

// refuseSLO names the SLO bound every memory-eligible row misses: TTFT when all
// miss the TTFT ceiling, else TPOT (with the best predicted batch-1 TPOT).
func refuseSLO(rep *SearchReport, rows []searchScored, slo SearchSLO, target int) {
	type mem struct {
		ttftOK  bool
		ttftMS  float64
		tpot1MS float64
	}
	var mems []mem
	allTTFTFail := true
	for _, s := range rows {
		if s.bMem < target || !s.ctxOK {
			continue
		}
		mems = append(mems, mem{ttftOK: s.ttftOK, ttftMS: s.row.TTFTSeconds * 1000, tpot1MS: s.tpot1MS})
		if s.ttftOK {
			allTTFTFail = false
		}
	}
	rep.Status = StatusRefused
	if len(mems) == 0 {
		// Defensive: maxTarget >= desired guarantees a memory-eligible row.
		rep.Refusal = &Refusal{Reason: ReasonBelowSLO}
		return
	}
	if allTTFTFail && slo.TTFTMS > 0 {
		pred := math.Inf(1)
		for _, m := range mems {
			if m.ttftMS > 0 && m.ttftMS < pred {
				pred = m.ttftMS
			}
		}
		rep.Refusal = &Refusal{Reason: ReasonBelowSLO, SLOBound: "ttft", SLOLimitMS: slo.TTFTMS, PredictedMS: finiteOrZero(pred)}
		return
	}
	pred := math.Inf(1)
	for _, m := range mems {
		if m.tpot1MS > 0 && m.tpot1MS < pred {
			pred = m.tpot1MS
		}
	}
	rep.Refusal = &Refusal{Reason: ReasonBelowSLO, SLOBound: "tpot", SLOLimitMS: slo.TPOTMS, PredictedMS: finiteOrZero(pred)}
}

// maxConcAt is the largest concurrent-stream count any tier holds at an
// enumerated context at or above `desired` (the concurrency fact for
// concurrency_exceeds_pool).
func maxConcAt(rows []searchScored, desired int) int {
	best := 0
	for _, s := range rows {
		if s.row.ContextTokens >= desired && s.bMem > best {
			best = s.bMem
		}
	}
	return best
}

// dedupeTiers collapses duplicate precision entries (first wins), keeping input
// order so tier-order tie-breaks are stable.
func dedupeTiers(tiers []KVTier) []KVTier {
	out := make([]KVTier, 0, len(tiers))
	seen := map[string]bool{}
	for _, t := range tiers {
		key := strings.ToLower(strings.TrimSpace(t.Precision))
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, t)
	}
	return out
}

// contextLadder is the enumerated context axis: the powers-of-two run from the
// minimum window up to the model max, plus the anchor (resolved/boot context)
// and any pinned context. Ascending, deduped.
func contextLadder(anchor, modelMax, pinned int) []int {
	set := map[int]bool{}
	for c := 4096; c <= modelMax; c <<= 1 {
		set[c] = true
	}
	if modelMax > 0 {
		set[modelMax] = true
	}
	for _, extra := range []int{anchor, pinned} {
		if extra > 0 {
			set[extra] = true
		}
	}
	out := make([]int, 0, len(set))
	for c := range set {
		out = append(out, c)
	}
	sort.Ints(out)
	return out
}

// slotLadder is the enumerated slots axis: the powers-of-two run capped at
// maxSearchSlots, plus the target concurrency and any pinned value.
func slotLadder(target, pinned int) []int {
	set := map[int]bool{}
	for s := 1; s <= maxSearchSlots; s <<= 1 {
		set[s] = true
	}
	for _, extra := range []int{target, pinned} {
		if extra > 0 {
			set[extra] = true
		}
	}
	out := make([]int, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	sort.Ints(out)
	return out
}

// TokenBudget is the TRT-LLM chunked-prefill token budget: batch x (1 + ISL/OSL),
// clamped to [2048, 32768]. One prefill step admits at most this many tokens.
func TokenBudget(batch, isl, osl int) int {
	if batch <= 0 {
		batch = 1
	}
	if osl <= 0 {
		osl = 1
	}
	v := float64(batch) * (1 + float64(isl)/float64(osl))
	const lo, hi = 2048, 32768
	if v < lo {
		v = lo
	}
	if v > hi {
		v = hi
	}
	return int(v)
}

// chunkedTTFT prices time-to-first-token as the SUM of prefill-chunk stage costs
// covering the attended span, each chunk bounded by TokenBudget(batch, ISL, OSL).
// With a budget at least as large as the span this reduces to the single
// unchunked prefill stage Plan prices, so the two agree on short prompts.
func (p *planner) chunkedTTFT(c Candidate, lkv, b, osl int) float64 {
	if lkv <= 0 {
		return 0
	}
	chunk := TokenBudget(b, p.wl.PromptTokens, osl)
	if chunk <= 0 {
		chunk = lkv
	}
	active := nonneg(p.in.Model.ActiveParamsPerToken)
	head := math.Min(nonneg(p.in.Model.HeadParams), active)
	rho := p.rho
	bw, f := p.effectiveRates()
	var total float64
	prefix := 0
	for prefix < lkv {
		n := chunk
		if n > lkv-prefix {
			n = lkv - prefix
		}
		gemm := 2 * (active - head) * float64(n)
		if prefix+n == lkv {
			gemm += 2 * head // the LM head runs once, on the final chunk
		}
		attnInc := 2 * p.attnWidth() * (float64(prefix+n)*float64(prefix+n) - float64(prefix)*float64(prefix))
		bytes := rho*float64(p.in.Model.StreamBytes(n)) + float64(subSat(c.Context.KVBytes(prefix+n), c.Context.KVBytes(prefix)))
		total += finiteOrZero(compute.StageCost{Name: "prefill-chunk", FLOPs: sat(gemm + attnInc), WeightBytes: sat(bytes)}.RooflineSeconds(f, bw))
		prefix += n
	}
	return total
}

// effectiveRates is the calibrated (bandwidth, flops) pair every roofline price
// in this package uses — factored out of priceThroughput so the T04 search
// prices rows with the SAME numbers Plan does.
func (p *planner) effectiveRates() (bw, f float64) {
	cal, peaks, m := p.in.Calibration, p.in.Hardware.Peaks, p.in.Model
	peakBW := peaks.BytesPerSec
	peakF := 0.0
	if m.ActiveParamsPerToken > 0 && positiveFinite(peaks.FLOPsPerSec) {
		peakF = peaks.FLOPsPerSec
	}
	return cal.effectiveBandwidth(peakBW), cal.ComputeEfficiency * peakF
}

// decodeKVRead is the per-step KV read a decode step pays: the attended span
// (min(ISL, ctx)) plus the session-state working set.
func (p *planner) decodeKVRead(c Candidate, lkv int) int64 {
	return addSat(attendedKVBytes(c.Context, lkv), c.Context.SessionState.Total())
}

// stepRateAt is the per-sequence decode rate (tok/s) at batch b — the same
// roofline price priceThroughput applies, exposed for the B_slo search.
func (p *planner) stepRateAt(c Candidate, b, lkv int) float64 {
	if b < 1 {
		return 0
	}
	bw, f := p.effectiveRates()
	r := p.decodeStep(b, lkv, p.decodeKVRead(c, lkv))
	return finiteOrZero(withOverhead(r.RooflineTokPerSecCeiling(f, bw), p.in.Calibration.OverheadSeconds))
}

// maxBatchAtRate is the LARGEST batch whose per-sequence decode rate stays at or
// above need tok/s. stepRateAt is monotonically non-increasing in batch (more
// tokens per step under the same roofline), so a binary search is exact.
func maxBatchAtRate(p *planner, c Candidate, lkv, capB int, need float64) int {
	if capB < 1 {
		return 0
	}
	if p.stepRateAt(c, capB, lkv) >= need {
		return capB
	}
	if p.stepRateAt(c, 1, lkv) < need {
		return 0
	}
	lo, hi := 1, capB // invariant: rate(lo) >= need > rate(hi)
	for lo+1 < hi {
		mid := (lo + hi) / 2
		if p.stepRateAt(c, mid, lkv) >= need {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo
}

// maxSlotContextFor is the largest context ANY tier serves `slots` streams at,
// by binary search over the monotone per-stream charge (vLLM max-model-length
// shape), with the precision of the tier that achieves it. Returns 0 when even
// one stream at the smallest context does not fit.
func (p *planner) maxSlotContextFor(tiers []KVTier, slots int) (int, string) {
	best, kv := 0, ""
	for _, tier := range tiers {
		c := Candidate{KVPrecision: tier.Precision, Context: tier.Context}
		if !kvSizable(c.Context) {
			continue
		}
		if got := p.maxSlotContext(c, slots); got > best {
			best, kv = got, tier.Precision
		}
	}
	return best, kv
}

// maxSlotContext is the largest context `slots` streams of c's tier fit, by
// binary search over the monotone per-stream charge at a fixed slot count —
// O(log MaxContext) instead of a linear scan down from the window.
func (p *planner) maxSlotContext(c Candidate, slots int) int {
	csc := c.Context
	if p.avail <= 0 || !kvSizable(csc) || slots < 1 {
		return 0
	}
	room := p.room(c)
	if room <= 0 {
		return 0
	}
	per := func(tokens int) int64 { return perStreamBytes(c, tokens) }
	if mulSat(int64(slots), per(1)) > room {
		return 0
	}
	// Largest i in [1, MaxContext] with slots*per(i) <= room.
	return sort.Search(csc.MaxContext, func(i int) bool {
		return mulSat(int64(slots), per(i+1)) > room
	})
}

// atConcurrency returns a planner copy judging against `concurrency` streams
// (batched decode whenever more than one), leaving the original untouched.
func (p *planner) atConcurrency(concurrency int) planner {
	q := *p
	q.wl = Workload{PromptTokens: p.wl.PromptTokens, Concurrency: concurrency, BatchedDecode: concurrency > 1}
	return q
}

// evidence builds the SearchReport evidence block exactly as Plan does.
func (p *planner) evidence() Evidence {
	in := p.in
	ev := Evidence{
		Calibration:     in.Calibration.finite(in.Hardware.Peaks.BytesPerSec),
		Peaks:           in.Hardware.Peaks.finite(),
		Topology:        in.Hardware.Topology,
		PoolUsableBytes: max(p.avail, 0),
	}
	if !in.Hardware.Topology.sharesHostPool() {
		ev.HostPoolUsableBytes = max(in.Hardware.HostPoolUsableBytes, 0)
	}
	if p.rhoOK {
		ev.ResidentScale = finiteOrZero(p.rho)
	}
	return ev
}

// needAt is the pool bytes `slots` streams of c at its context require.
func (p *planner) needAt(c Candidate, slots int) int64 {
	return addSat(addSat(p.weights, scratchBytes(c)), mulSat(int64(slots), perStreamBytes(c, c.ContextTokens)))
}

// shortfallOneStream names the bytes short of one stream at the smallest
// context, on the tier closest to fitting (context_exceeds_pool detail).
func (p *planner) shortfallOneStream(tiers []KVTier, isl int) int64 {
	best, found := int64(0), false
	for _, tier := range tiers {
		c := Candidate{KVPrecision: tier.Precision, Context: tier.Context}
		if !kvSizable(c.Context) {
			continue
		}
		c.ContextTokens = isl
		short := subSat(p.needAt(c, 1), p.avail)
		if short > 0 && (!found || short < best) {
			best, found = short, true
		}
	}
	return best
}

func bestTierPrecision(tiers []KVTier) string {
	if len(tiers) == 0 {
		return ""
	}
	return tiers[0].Precision
}

// refuseSearch fills rep with a typed refusal (StatusRefused + sentinel-carrying
// Refusal). Additive T04 detail fields stay zero unless the caller sets them.
func refuseSearch(rep *SearchReport, reason RefusalReason, kv string, maxCtx, maxConc int, shortfall int64, hostPool bool) {
	rep.Status = StatusRefused
	rep.Refusal = &Refusal{
		Reason:           reason,
		KVPrecision:      kv,
		MaxContextTokens: maxCtx,
		MaxConcurrency:   maxConc,
		ShortfallBytes:   shortfall,
		HostPool:         hostPool,
	}
}

// newPlanner factors Plan's planner construction out so the T04 search prices
// rows with the identical state Plan initializes.
func newPlanner(in Input) *planner {
	p := &planner{in: in, wl: in.Workload.normalized(), avail: in.Hardware.PoolUsableBytes}
	p.weights = weightsInPool(in.Hardware.Topology, in.Weights)
	discrete := !in.Hardware.Topology.sharesHostPool()
	if discrete {
		p.hostWeights = in.Weights.HostTotal()
		hostAvail := in.Hardware.HostPoolUsableBytes
		p.hostShort = p.hostWeights > 0 && hostAvail > 0 && p.hostWeights > hostAvail
		p.hostUnknown = p.hostWeights > 0 && hostAvail <= 0
	}
	p.rho, p.rhoOK = residentScale(in.Model, in.Weights)
	p.gate, p.gateBound = p.throughputGate()
	return p
}
