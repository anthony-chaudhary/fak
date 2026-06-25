package ctxplan

import (
	"math"
	"strconv"
	"strings"
	"unicode"
)

// Forecast is the "imagine what I'll need next" prediction that drives the planner —
// the generalization of preemptively planning. Instead of waiting to fault a span in
// when a turn references it, the model (or a heuristic) authors a forecast of the
// reference string the next Horizon turns are expected to touch, and the planner
// pre-materializes the O(1) view that best covers it under budget. A forecast is the
// planner's "query": the Intents are its predicate, the Weights its cost constants.
//
// A forecast is cheap and revisable: it is recomputed each turn (or each few turns) and
// is never load-bearing for correctness — a MISS costs one demand-page of the missing
// span (a page fault), never a lost fact, because the store is lossless (see scaling.go
// and faithful.go). So a bad forecast degrades efficiency, not correctness.
type Forecast struct {
	// Intents are the predicted reference strings for the next horizon — what the
	// upcoming turns are expected to ask about (e.g. "refund fee", "auth token
	// rotation", "the failing build step"). They are matched, content-word against
	// content-word, against each cell's role+descriptor, exactly as recall/cdb/memq
	// rank relevance. An empty Intents list means "no prediction"; selection then falls
	// to the utility/durability/recency priors and the pins.
	Intents []string `json:"intents,omitempty"`
	// Horizon is how many turns ahead this forecast is meant to cover (>=1). It is
	// advisory metadata today (carried into the plan + EXPLAIN); a multi-horizon
	// planner that trades a larger resident view for a longer horizon is a named
	// follow-on, not this rung.
	Horizon int `json:"horizon,omitempty"`
	// Pins are cell IDs that MUST be resident regardless of score — the spans a turn
	// cannot proceed without (the system prompt, the active goal, the last user turn).
	// A pin is charged against the budget FIRST. A pin that names a sealed/tombstoned
	// cell is REFUSED, not forced in: a pin can never launder poison into context.
	Pins []string `json:"pins,omitempty"`
	// Weights combine the per-cell benefit signals into one score. Zero value => use
	// DefaultWeights via (Weights).orDefault, so a forecast literal that sets none still
	// plans sensibly.
	Weights Weights `json:"weights,omitempty"`
}

// Weights are the planner's cost constants — the knobs that decide how much each
// benefit signal contributes to a cell's score, the analogue of a query optimizer's
// cost constants. All are non-negative; a zero weight drops that signal entirely.
type Weights struct {
	Relevance  float64 `json:"relevance,omitempty"`  // forecast-intent overlap (the predicted-need signal)
	Utility    float64 `json:"utility,omitempty"`    // learned outcome-utility (recall.Page.Utility, carried in Cell.Attrs["utility"])
	Durability float64 `json:"durability,omitempty"` // durable > bounded > session > turn (a durable fact is worth keeping resident)
	Recency    float64 `json:"recency,omitempty"`    // newer spans slightly favored (a mild prior, not a dominant one)
	// Primacy is the OPPOSITE-END twin of Recency: OLDER spans slightly favored. Recency
	// alone is monotone in step, so the four-term benefit cannot express "the oldest spans
	// (system framing, original constraints) are ALSO load-bearing" - the structural gap a
	// U-shaped 'remove-the-middle' prior fills. DEFAULT 0 (OFF): an EXPERIMENT, not shipped
	// behavior; turn it on with a non-zero weight (0.2 is symmetric with Recency). Gate any
	// claim on a multi-turn fault measure (fak horizon-recovery), never on faith. See
	// docs/explainers/compounding-benefits-of-a-saved-call.md (the r term).
	Primacy float64 `json:"primacy,omitempty"`
}

// DefaultWeights is the seed cost model: relevance dominates (the forecast is the point),
// utility is a strong secondary, durability and recency are mild priors. Tuned to be
// sensible, not optimal — the weights are the obvious lever a later rung learns.
func DefaultWeights() Weights {
	return Weights{Relevance: 1.0, Utility: 0.5, Durability: 0.4, Recency: 0.2}
}

// orDefault returns w, or DefaultWeights if w is the zero value (all four weights 0,
// which would score every cell 0 and make the plan arbitrary).
func (w Weights) orDefault() Weights {
	if w == (Weights{}) {
		return DefaultWeights()
	}
	return w
}

// UtilityMax is the clamp ceiling for the learned outcome-utility signal — the same
// scale recall uses for Page.Utility (a witness-gated, [0,UtilityMax]-clamped Q-value).
// ctxplan reads it off Cell.Attrs["utility"] (a benign, provenance-clean number a
// recall-backed cell carries) and normalizes by this ceiling so utility contributes in
// [0,1] like the other signals. A missing/garbage value normalizes to 0 (the
// fail-closed neutral default — an un-credited span gets no utility boost).
const UtilityMax = 4.0

// signal is the four normalized benefit features for one span — the per-cell analogue of
// pg_statistic. Forecast.Benefit combines them with Weights; Weights.Learn uses the SAME
// vector as its feature row so the scorer and the online learner can never drift apart
// (the weight a span was scored with is the weight its outcome tunes).
type signal struct {
	Relevance  float64
	Utility    float64
	Durability float64
	Recency    float64
	Primacy    float64
}

// signals returns the four benefit features for c, normalizing recency against maxStep.
// Relevance is 0 with no forecast intents; the others are content- or step-driven. It is
// the shared scoring vocabulary Benefit and Weights.Learn both speak.
func (f Forecast) signals(c Span, maxStep int) signal {
	return signal{
		Relevance:  f.relevance(c),
		Utility:    spanUtility(c),
		Durability: durabilityPrior(c.Durability),
		Recency:    recency(c.Step, maxStep),
		// c.Step is the LAST-REFERENCE step (the same signal recency reads), so primacy only
		// down-weights spans untouched-since-old, never a recently re-referenced old span.
		Primacy: primacy(c.Step, maxStep),
	}
}

// Benefit scores one cell for the planner: the weighted sum of its relevance to the
// forecast intents, its learned utility, its durability prior, and its recency, each
// normalized to ~[0,1]. A SEALED or TOMBSTONED cell scores exactly 0 — it is never a
// candidate for the resident view, mirroring the recall/cdb/memq invariant that poison
// and suppressed spans never enter context. maxStep is the largest step in the candidate
// set, used to normalize recency; pass 0 to disable the recency term.
func (f Forecast) Benefit(c Span, maxStep int) float64 {
	if c.Sealed || c.Tombstoned {
		return 0
	}
	w := f.Weights.orDefault()
	s := f.signals(c, maxStep)
	b := w.Relevance*s.Relevance + w.Utility*s.Utility + w.Durability*s.Durability + w.Recency*s.Recency + w.Primacy*s.Primacy
	// Single fail-closed choke point: a non-finite score (a poisoned signal, an Inf
	// weight) collapses to 0 rather than poisoning the planner's sort or DP downstream.
	if math.IsNaN(b) || math.IsInf(b, 0) {
		return 0
	}
	return b
}

// relevance is the fraction of distinct forecast content-tokens that appear in the
// cell's role+descriptor, in [0,1]. With no intents it is 0 (no prediction to match).
// This is the same stopword-aware extractive overlap recall/cdb/memq use as their
// relevance ranker — selection, never the trust boundary.
func (f Forecast) relevance(c Span) float64 {
	q := tokenSet(strings.Join(f.Intents, " "))
	if len(q) == 0 {
		return 0
	}
	doc := tokenSet(c.Role + " " + c.Descriptor)
	hit := 0
	for t := range q {
		if doc[t] {
			hit++
		}
	}
	return float64(hit) / float64(len(q))
}

// spanUtility reads the learned outcome-utility a recall-backed span carries in
// Attrs["utility"] and normalizes it to [0,1] by UtilityMax. A missing or non-numeric
// value is 0 (the neutral default), so a never-credited or non-recall span simply gets
// no utility boost — the order is then driven by the other signals.
func spanUtility(c Span) float64 {
	v := c.Attrs["utility"]
	if v == "" {
		return 0
	}
	u, err := strconv.ParseFloat(v, 64)
	// Reject non-finite explicitly: ParseFloat("NaN") returns NaN with a NIL error, and
	// NaN passes both `u <= 0` and `u > UtilityMax` as false, so without this guard a
	// poisoned Attrs["utility"] would leak NaN into Benefit -> density -> the sort and the
	// DP. Fail closed to the neutral 0.
	if err != nil || math.IsNaN(u) || math.IsInf(u, 0) || u <= 0 {
		return 0
	}
	if u > UtilityMax {
		u = UtilityMax
	}
	return u / UtilityMax
}

// durabilityPrior maps a span's durability class to a keep-resident prior in [0,1]: a
// durable fact (a stated preference, an identity) is more worth keeping resident across
// turns than a turn-scoped deictic. The order matches durabilityRank
// (turn<session<bounded<durable); an unknown class normalizes to turn via NormDurability —
// the fail-closed shortest-lived default.
func durabilityPrior(class string) float64 {
	switch NormDurability(class) {
	case DurabilityDurable:
		return 1.0
	case DurabilityBounded:
		return 0.6
	case DurabilitySession:
		return 0.5
	default: // turn
		return 0.2
	}
}

// recency returns a mild [0,1] preference for newer spans (higher step). With maxStep
// <= 0 it is 0 (recency disabled / a single-step set), so it never divides by zero and
// never dominates — it is a tie-breaker prior, not a ranker.
func recency(step, maxStep int) float64 {
	if maxStep <= 0 || step <= 0 {
		return 0
	}
	if step >= maxStep {
		return 1
	}
	return float64(step) / float64(maxStep)
}

// primacy is the OPPOSITE-END mirror of recency: a mild [0,1] preference for OLDER spans
// (lower last-reference step). It is exactly 1 - recency over the open interval, but pinned
// to 0 at BOTH degenerate ends so it never lifts the newest span and never divides by zero.
// Paired with recency under separate weights it favors both ends over the middle - the
// 'remove-the-middle' prior. EXPERIMENT: contributes nothing unless Weights.Primacy is
// non-zero (DEFAULT 0). step is the LAST-REFERENCE step, so a recently re-touched old span
// has high recency and LOW primacy - primacy only down-weights spans untouched since old.
func primacy(step, maxStep int) float64 {
	if maxStep <= 0 || step <= 0 {
		return 0
	}
	if step >= maxStep {
		return 0
	}
	return float64(maxStep-step) / float64(maxStep)
}

// tokenSet is the lowercased, length>2, content-token set of a string — the same
// extractive tokenization recall/cdb/memq use, kept local so ctxplan adds no coupling.
func tokenSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, t := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(t) > 2 {
			out[t] = true
		}
	}
	return out
}
