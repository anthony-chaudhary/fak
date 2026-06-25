package ctxplan

// CONTEXT SIGNAL-TO-NOISE (#563 frontier; the metric the cache-hit number cannot give).
//
// Provider cache-hit % is a DENOMINATOR ARTIFACT, not a quality signal. It is
// cache_read / (cache_read + fresh_input); as a session appends to a stable prefix,
// the cached prefix grows every turn while fresh input stays ~turn-sized, so the
// ratio approaches 1.0 MECHANICALLY — independent of whether the resident context is
// any good. Measured on fak's own 247-session corpus: cache-hit climbs 0.88 (short)
// -> 0.99 (long) with turn count (Pearson r≈0.48), and two sessions both at ~99%
// cache-hit differ 10x in context density. A high cache-hit on a bloated window is
// "efficiently re-reading the wrong thing" — caching garbage cheaply.
//
// The quantity actually worth maximizing is whether the RESIDENT window equals the
// DESIRED window: |resident| == |desired|. ctxplan already records the ground truth
// for that, per turn, in an Outcome:
//
//	Hits   — resident spans the turn REFERENCED        (signal: pulled weight)
//	Wasted — resident spans the turn never touched     (noise: carried, paid for, idle)
//	Faults — ELIDED spans the turn demand-paged back   (under-resident: too lean)
//
// SignalNoise turns that into the headline ratio. It is TOKEN-WEIGHTED (it sums each
// span's planned Cost, the same unit the Budget caps), so a single large stale blob
// counts as the noise it is, not as one span among many. Crucially it is INVARIANT to
// caching and to session length: re-reading a Wasted span cheaply (cached) does not
// make it signal — it is still resident-but-untouched. So a session can report
// cache_hit 0.99 AND ratio 0.30 simultaneously, which is the exact pathology the
// cache-hit number hides.
//
// Faults (under-resident) are the OPPOSITE failure — a window trimmed too lean — and
// are reported on their OWN axis, never folded into the ratio, so driving S/N up by
// dropping needed spans shows up as rising faults, not as a better score.

// SignalNoise is the per-turn context signal-to-noise breakdown over a Plan's resident
// set, judged against a witnessed Outcome. All token counts are the planner's own Cost
// unit (≈ bytes/4), summed from the resident Selections — so the ratio is the fraction
// of the window the turn PAID for that actually pulled weight.
type SignalNoise struct {
	// SignalTokens is the resident cost of spans the turn referenced (Outcome.Hits) plus
	// every pinned span (a pin is signal by construction — the turn cannot proceed without
	// it, so it is never counted as idle even if a given turn did not cite it).
	SignalTokens int `json:"signal_tokens"`
	// NoiseTokens is the resident cost of spans the turn never touched (Outcome.Wasted) and
	// that are not pinned — the budget spent on context that idled this turn.
	NoiseTokens int `json:"noise_tokens"`
	// UnaccountedTokens is resident cost neither hit, wasted, nor pinned — a span the
	// Outcome did not label (e.g. an outcome recorded before the span entered the view).
	// Surfaced separately so it neither inflates signal nor noise: an honest "unknown".
	UnaccountedTokens int `json:"unaccounted_tokens,omitempty"`
	// FaultTokens is the cost of spans the turn NEEDED but the plan had ELIDED
	// (Outcome.Faults) — the UNDER-resident axis. It is NOT part of the ratio; a high
	// fault cost means the window was trimmed too lean (the opposite of noise). Reported so
	// "raise S/N" cannot be gamed by dropping needed spans (that just moves cost here).
	FaultTokens int `json:"fault_tokens"`
	// ResidentTokens is SignalTokens + NoiseTokens + UnaccountedTokens — the total cost of
	// the resident view (== Plan.CostUsed for a plan whose Selections all carry Cost).
	ResidentTokens int `json:"resident_tokens"`
}

// Ratio is the headline signal-to-noise number in [0,1]: SignalTokens / ResidentTokens
// — the fraction of the resident window that pulled weight this turn. 1.0 means every
// resident token was referenced (|resident| == |desired|, the goal); a low ratio means
// most of the window idled (paying to keep — and cache — noise). Unaccounted tokens sit
// in the denominator (they are resident cost), so the ratio never over-claims signal.
// An empty resident view returns 1.0 by convention (no window, no noise — nothing to
// curate), the same fail-to-best posture the planner's empty plan takes.
func (s SignalNoise) Ratio() float64 {
	if s.ResidentTokens <= 0 {
		return 1.0
	}
	return float64(s.SignalTokens) / float64(s.ResidentTokens)
}

// FaultRatio is fault cost over (resident + fault) cost — the UNDER-resident pressure.
// It rises when the plan elided spans the turn then needed (a too-lean window). Read it
// ALONGSIDE Ratio: high Ratio + low FaultRatio is the target (lean and sufficient); high
// Ratio + high FaultRatio is over-trimmed (lean but starving); low Ratio is bloated.
// Returns 0 when there were no faults and no resident cost.
func (s SignalNoise) FaultRatio() float64 {
	denom := s.ResidentTokens + s.FaultTokens
	if denom <= 0 {
		return 0
	}
	return float64(s.FaultTokens) / float64(denom)
}

// ComputeSignalNoise folds a Plan and its witnessed Outcome into the token-weighted S/N
// breakdown. It reads each resident Selection's planned Cost and classifies it by the
// Outcome's per-span labels:
//   - pinned OR in Hits  -> signal
//   - in Wasted (and not pinned) -> noise
//   - otherwise          -> unaccounted (resident, unlabeled)
//
// Fault cost is summed from the ELIDED spans the Outcome marks faulted (a span the turn
// needed but the plan had paged out). Labels that name no resident/elided span are
// ignored (fail-closed: a stale or fabricated id cannot move the metric). The function is
// pure and deterministic — same (plan, outcome) yields the same breakdown.
func ComputeSignalNoise(p Plan, o Outcome) SignalNoise {
	hit := make(map[string]bool, len(o.Hits))
	for _, id := range o.Hits {
		hit[id] = true
	}
	wasted := make(map[string]bool, len(o.Wasted))
	for _, id := range o.Wasted {
		wasted[id] = true
	}
	var sn SignalNoise
	for _, sel := range p.Selected {
		c := sel.Cost
		if c < 0 {
			c = 0
		}
		sn.ResidentTokens += c
		switch {
		case sel.Pinned || hit[sel.ID]:
			sn.SignalTokens += c
		case wasted[sel.ID]:
			sn.NoiseTokens += c
		default:
			sn.UnaccountedTokens += c
		}
	}
	// Fault cost: an Outcome.Faults id names an ELIDED span (paged out, then needed).
	if len(o.Faults) > 0 {
		faulted := make(map[string]bool, len(o.Faults))
		for _, id := range o.Faults {
			faulted[id] = true
		}
		for _, el := range p.Elided {
			if faulted[el.ID] {
				c := el.Cost
				if c < 0 {
					c = 0
				}
				sn.FaultTokens += c
			}
		}
	}
	return sn
}

// Grade renders a one-word health verdict for an operator surface, combining the two
// axes so a single label captures the window's state. The thresholds are intentionally
// coarse (a gauge, not a controller): the point is to make "cache-hit 0.99 but S/N 0.3"
// legible at a glance, not to tune a budget.
//
//	bloated   — Ratio < 0.5: most of the window idled (the cache-hit trap in the open)
//	starving  — FaultRatio > 0.25: the window is trimmed so lean the turn keeps faulting
//	lean      — Ratio >= 0.8 and FaultRatio <= 0.1: resident ≈ desired (the goal)
//	ok        — everything between (acceptable, not yet ideal)
func (s SignalNoise) Grade() string {
	r, fr := s.Ratio(), s.FaultRatio()
	if fr > 0.25 {
		return "starving"
	}
	if r < 0.5 {
		return "bloated"
	}
	if r >= 0.8 && fr <= 0.1 {
		return "lean"
	}
	return "ok"
}
