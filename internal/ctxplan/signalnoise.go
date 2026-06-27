package ctxplan

import "math"

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
	hit := stringSet(o.Hits)
	wasted := stringSet(o.Wasted)
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
	sn.FaultTokens = faultTokens(p, o)
	return sn
}

// stringSet folds a slice of ids into a presence set — the slice-to-set idiom shared by the
// S/N and refcount witnesses (Hits / Wasted / Faults lookups). Pure: a nil/empty slice
// yields an empty (non-nil) map.
func stringSet(ids []string) map[string]bool {
	m := make(map[string]bool, len(ids))
	for _, id := range ids {
		m[id] = true
	}
	return m
}

// faultTokens sums the planned cost of every ELIDED span the Outcome marks faulted (a span
// the turn needed but the plan had paged out) — the under-resident axis. Negative costs
// clamp to zero. Shared by the boolean and attention-witnessed S/N paths, which compute the
// fault axis identically (attention covers resident spans only). Returns 0 when no faults.
func faultTokens(p Plan, o Outcome) int {
	if len(o.Faults) == 0 {
		return 0
	}
	faulted := stringSet(o.Faults)
	total := 0
	for _, el := range p.Elided {
		if faulted[el.ID] {
			c := el.Cost
			if c < 0 {
				c = 0
			}
			total += c
		}
	}
	return total
}

// SpanMass is the WITNESSED attention mass a turn placed on one resident span — the
// rung-1→2 upgrade of the inferred boolean hit. Mass is normalized to [0,1]: it is the
// fraction of the span's resident cost that pulled weight this turn, the continuous a_s
// the #851 formula calls for. ID names a span by Cell.ID, the same key Plan.Selected and
// Outcome.Hits carry. A mass outside [0,1] is clamped on read (fail-closed: a malformed
// witness can never push a span past pure-signal or below pure-noise).
type SpanMass struct {
	ID   string  `json:"id"`
	Mass float64 `json:"mass"`
}

// Attribution is the per-span attention-mass witness for one turn — the self-contained
// shape ComputeSignalNoise's witnessed twin consumes. It is deliberately local to ctxplan
// and depends on nothing else: the kvmmu span-attribution accumulator (#853) adapts INTO
// this map rather than this type reaching across the lane. An empty/nil Attribution means
// "no attention witness available" — the caller falls back to the rung-0 boolean path
// (ComputeSignalNoise), which is how offline / API-only sessions still get an S/N number.
type Attribution map[string]float64

// clampMass folds an attention mass into [0,1] — a malformed witness (negative, or >1 from
// a normalization slip) can never make a span more than pure signal or less than pure noise.
func clampMass(m float64) float64 {
	if m < 0 {
		return 0
	}
	if m > 1 {
		return 1
	}
	return m
}

// SignalNoiseFromAttention folds a Plan and a WITNESSED per-span attention Attribution into
// the same token-weighted S/N breakdown ComputeSignalNoise produces — but with the binary
// a_s ∈ {0,1} hit replaced by the continuous attention mass a_s ∈ [0,1] the #851 formula
// calls for. A resident span's cost is SPLIT by its mass: round(mass·cost) tokens count as
// signal (the share that pulled weight) and the remainder as noise (the share that idled),
// so a half-attended span is half signal, half noise — granularity the boolean hit cannot
// express.
//
// Pins stay pure signal regardless of mass: a pin is signal by construction (the turn
// cannot proceed without it), the same rule ComputeSignalNoise applies, so a pinned span's
// full cost is signal even at mass 0. A resident span the Attribution does not name is
// UNACCOUNTED (the witness saw no attention on it AND did not label it noise) — it sits in
// the denominator as honest unknown, exactly like an unlabeled span in the boolean path, so
// the witnessed ratio never over-claims. Fault cost is read from the same Outcome the
// boolean path uses (attention attribution covers RESIDENT spans only; an elided-then-
// faulted span has no resident mass to witness), so the under-resident axis is identical.
//
// The function is pure and deterministic: same (plan, attribution, outcome) yields the same
// breakdown. Same SignalNoise struct out, so Ratio() / FaultRatio() / Grade() are unchanged.
func SignalNoiseFromAttention(p Plan, attribution Attribution, o Outcome) SignalNoise {
	var sn SignalNoise
	for _, sel := range p.Selected {
		c := sel.Cost
		if c < 0 {
			c = 0
		}
		sn.ResidentTokens += c
		if sel.Pinned {
			// A pin is signal by construction — never split, never idle (matches the boolean path).
			sn.SignalTokens += c
			continue
		}
		mass, witnessed := attribution[sel.ID]
		if !witnessed {
			// No attention witnessed this resident span: honest unknown, like an unlabeled span.
			sn.UnaccountedTokens += c
			continue
		}
		// Split the span's cost by witnessed mass: the attended share is signal, the rest noise.
		signal := int(math.Round(clampMass(mass) * float64(c)))
		if signal > c {
			signal = c
		}
		sn.SignalTokens += signal
		sn.NoiseTokens += c - signal
	}
	// Fault cost is identical to the boolean path: attention covers resident spans only, so an
	// elided-then-faulted span (no resident mass) is read from the Outcome's Faults, as before.
	sn.FaultTokens = faultTokens(p, o)
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
