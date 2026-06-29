package ctxplan

import "math"

// adaptive.go sizes the working set W (Budget.Tokens) to TASK DIFFICULTY instead of leaving
// it a fixed constant. The trade-off the planner navigates is well-defined: a bigger W keeps
// more spans resident (fewer forecast MISSES / demand-page faults, more answer context) at a
// higher resident-token cost; a smaller W compacts harder (more faults, fewer resident
// tokens). A single fixed Budget.Tokens is ONE point on that spectrum. RecommendBudget walks
// it: harder turns get a larger W (down the fault risk), easy turns get a smaller W (down the
// resident cost), bounded at both ends so the recommendation is never 0 and never unbounded.
//
// It is a PURE ADVISORY: it sizes the budget, it never changes plan CORRECTNESS. Optimize is
// still lossless and recoverable at whatever W it is handed — a smaller W elides more spans
// but every elision keeps its recovery handle (faithful.go), so a degenerate (tight) budget
// collapses to "page more back in", never to a lost fact. The frontier stays MONOTONE: a
// larger W never elides a span a smaller W kept, so more budget never hurts recall.
//
// Everything here is DETERMINISTIC (no randomness, no wall clock): the same difficulty
// signals yield byte-identical bounds.

// Difficulty is the deterministic, signal-derived estimate of how hard the upcoming turn is —
// the input that sizes W. It is read off the planner's own types (Forecast, Outcome), not a
// new parallel model: more distinct intents to cover, more pinned spans that MUST be resident,
// a longer forecast horizon, and a higher OBSERVED fault rate on the last turn all push the
// task toward "harder" (and therefore toward a larger W). The fields are the raw signals;
// Score folds them into one [0,1] number.
type Difficulty struct {
	// Intents is the number of distinct reference predicates the forecast must cover this
	// turn (len(Forecast.Intents)). More predicates => a broader resident view is needed to
	// keep the relevant spans hot.
	Intents int
	// Pins is the number of cells that MUST stay resident regardless of score
	// (len(Forecast.Pins)) — the irreducible floor the budget has to pay before anything
	// else fits. High pin PRESSURE means a small W spends the whole budget on pins and faults
	// everything else, so pin count drives W up.
	Pins int
	// Horizon is how many turns ahead the forecast covers (Forecast.Horizon, >=1). A longer
	// horizon predicts a wider set of future references, a mild push toward a larger W.
	Horizon int
	// FaultRate is the OBSERVED forecast-miss fraction from the last turn's Outcome —
	// faults / (hits + faults), in [0,1]. It is the closed-loop signal: a turn that faulted a
	// lot was under-budgeted, so the next turn's W is sized up. 0 when no turn has run yet (no
	// hits and no faults) — the cold-start neutral.
	FaultRate float64
}

// DifficultyFromForecast derives the static difficulty signals from a Forecast alone (no
// outcome yet): intent count, pin count, and horizon. FaultRate is 0 (cold start — nothing
// has faulted), so the recommendation rests purely on the task's declared shape. This is the
// first-turn / no-feedback path.
func DifficultyFromForecast(f Forecast) Difficulty {
	h := f.Horizon
	if h < 1 {
		h = 1 // a forecast with no horizon covers at least the current turn
	}
	return Difficulty{
		Intents: len(f.Intents),
		Pins:    len(f.Pins),
		Horizon: h,
	}
}

// DifficultyFromOutcome derives difficulty from the Forecast AND the WITNESSED Outcome of the
// last turn: the static signals from the forecast plus the observed FaultRate from the
// outcome. This is the closed-loop path — a turn that demand-paged a lot of elided spans back
// in (high faults relative to hits) was under-budgeted, and the next plan's W is sized up to
// catch them. With no hits and no faults the FaultRate is 0 (the turn referenced nothing
// resident or elided — no signal), so it degrades to DifficultyFromForecast.
func DifficultyFromOutcome(f Forecast, o Outcome) Difficulty {
	d := DifficultyFromForecast(f)
	hits := len(o.Hits)
	faults := len(o.Faults)
	if total := hits + faults; total > 0 {
		d.FaultRate = float64(faults) / float64(total)
	}
	return d
}

// Difficulty-signal saturation points. Each raw count is normalized against its saturation:
// at or above it the signal contributes its full weight, so a pathological count (a thousand
// intents) cannot blow W past the ceiling. Chosen as sane "this is already a hard turn"
// points, not tuned — they bound the signal, they are not the model.
const (
	intentSaturation  = 8.0 // ~8 distinct predicates is already a broad turn
	pinSaturation     = 6.0 // ~6 forced-resident spans is heavy pin pressure
	horizonSaturation = 4.0 // forecasting ~4 turns ahead is a wide lookahead
)

// Difficulty-signal weights. They sum to 1.0 so Score lands in [0,1]. Relevance to W:
// the OBSERVED fault rate is the strongest signal (it is direct evidence the last W was too
// small), intent breadth and pin pressure are strong static signals, horizon is a mild prior
// — the same shape DefaultWeights uses for the per-cell benefit score.
const (
	wFault   = 0.45
	wIntent  = 0.25
	wPin     = 0.20
	wHorizon = 0.10
)

// Score folds the raw difficulty signals into one deterministic number in [0,1]: 0 = the
// easiest task the signals can describe (no intents, no pins, single-turn horizon, no observed
// faults), 1 = saturated-hard on every axis. Each count is normalized against its saturation
// (clamped to [0,1]) and combined by the fixed weights, so the result is monotone in every
// signal — more intents, more pins, a longer horizon, or a higher fault rate never LOWERS the
// score. A non-finite or out-of-range input fails closed to the bounded value (a negative
// count reads as 0, a NaN fault rate as 0), so a garbage signal can never push W out of range.
func (d Difficulty) Score() float64 {
	intent := saturate(float64(d.Intents), intentSaturation)
	pin := saturate(float64(d.Pins), pinSaturation)
	horizon := saturate(float64(d.Horizon)-1, horizonSaturation-1) // horizon 1 is the floor (0 lookahead)
	fault := d.FaultRate
	if math.IsNaN(fault) || fault < 0 {
		fault = 0
	}
	if fault > 1 {
		fault = 1
	}
	s := wFault*fault + wIntent*intent + wPin*pin + wHorizon*horizon
	// Defensive clamp: the weights sum to 1 and every term is in [0,1], so s is already in
	// [0,1]; this guards against a future weight edit silently breaking the invariant.
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}

// saturate normalizes a raw count to [0,1] against a saturation point: 0 at count<=0, 1 at
// count>=sat, linear between. A non-positive saturation degrades to "always saturated" (1 for
// any positive count) rather than dividing by zero.
func saturate(count, sat float64) float64 {
	if count <= 0 {
		return 0
	}
	if sat <= 0 || count >= sat {
		return 1
	}
	return count / sat
}

// BudgetBounds is the spectrum RecommendBudget walks: the WORKING-SET floor (the smallest W
// the planner will ever recommend — the "compact aggressively" end) and ceiling (the largest —
// the "keep more resident" end). The recommended W slides between them with difficulty. Both
// bounds are explicit so the caller owns the trade-off envelope (e.g. the model's real context
// window as the ceiling); RecommendBudget never returns a W outside [Floor, Ceil].
type BudgetBounds struct {
	Floor int // smallest W — the aggressive-compaction end (more faults, fewest resident tokens)
	Ceil  int // largest W — the keep-resident end (fewest faults, most resident tokens)
}

// DefaultBudgetBounds is a sane spectrum for a turn whose caller has not pinned the model's
// real window: floor 512 tokens (enough for pins + a few hot spans), ceiling 8192 tokens (a
// generous resident view). It is the seed envelope, the obvious lever a caller overrides with
// the model's actual context window — not a tuned optimum.
func DefaultBudgetBounds() BudgetBounds {
	return BudgetBounds{Floor: 512, Ceil: 8192}
}

// normalized returns the bounds with Floor/Ceil forced sane: a non-positive floor clamps to 1
// (W is never 0 — a degenerate budget still pages everything back in, it never loses a fact,
// but a 0 budget would elide even the pins), and a ceiling below the floor collapses to the
// floor (an inverted or zero ceiling degrades to "no headroom", a fixed W at the floor, never
// an unbounded or negative W).
func (b BudgetBounds) normalized() BudgetBounds {
	floor := b.Floor
	if floor < 1 {
		floor = 1
	}
	ceil := b.Ceil
	if ceil < floor {
		ceil = floor
	}
	return BudgetBounds{Floor: floor, Ceil: ceil}
}

// RecommendBudget sizes the working set W to task difficulty within an explicit envelope. It
// linearly interpolates between the floor (easiest, score 0) and the ceiling (hardest, score
// 1): W = Floor + score*(Ceil-Floor), rounded. The result is a PURE RECOMMENDATION — it is the
// Budget to HAND Optimize, it does not change what Optimize does with it (still lossless, still
// recoverable at any W).
//
// Guarantees, all deterministic:
//   - MONOTONE: a higher difficulty score never returns a smaller W (the frontier never folds
//     back — more budget for a harder task, never less).
//   - BOUNDED: the result is always in [max(1,Floor), Ceil]. Never 0, never unbounded. A
//     degenerate (e.g. inverted or zero) bounds set collapses to a single fixed W at the
//     floor — the documented worst case, a constant budget, not a lost fact.
//   - PURE: same (difficulty, bounds) => byte-identical Budget.
func RecommendBudget(d Difficulty, bounds BudgetBounds) Budget {
	b := bounds.normalized()
	score := d.Score() // already clamped to [0,1]
	span := float64(b.Ceil - b.Floor)
	w := float64(b.Floor) + score*span
	tokens := int(math.Round(w))
	// Round can land a hair outside the envelope only on float error; clamp defensively so the
	// BOUNDED guarantee is exact.
	if tokens < b.Floor {
		tokens = b.Floor
	}
	if tokens > b.Ceil {
		tokens = b.Ceil
	}
	return Budget{Tokens: tokens}
}

// RecommendBudgetForForecast is the convenience first-turn entry: size W from a Forecast alone
// (static difficulty, no outcome feedback) within the bounds. Equivalent to
// RecommendBudget(DifficultyFromForecast(f), bounds).
func RecommendBudgetForForecast(f Forecast, bounds BudgetBounds) Budget {
	return RecommendBudget(DifficultyFromForecast(f), bounds)
}

// RecommendBudgetFromOutcome is the convenience closed-loop entry: size the NEXT turn's W from
// the Forecast plus the WITNESSED Outcome of the last turn (so a turn that faulted a lot gets a
// larger W next). Equivalent to RecommendBudget(DifficultyFromOutcome(f, o), bounds).
func RecommendBudgetFromOutcome(f Forecast, o Outcome, bounds BudgetBounds) Budget {
	return RecommendBudget(DifficultyFromOutcome(f, o), bounds)
}
