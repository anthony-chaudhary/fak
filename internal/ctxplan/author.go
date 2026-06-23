package ctxplan

import "sort"

// author.go — the forecast AUTHOR: the rung that predicts, from the trajectory, the
// reference string the next turns are expected to touch, so the planner can pre-materialize
// the O(1) view that covers it BEFORE the fault. It is the "general preemptive planner" —
// the piece the design note (O1-TURN-CONTEXT-PLANNER-2026-06-23.md §6) names as the
// follow-on to "the forecast is authored, not learned": intents/pins were SUPPLIED; this
// rung AUTHORS the intents from the trajectory itself.
//
// # Why this exists (preemptive, not reactive)
//
// Before this rung the only forecast authoring was a single-message heuristic in the agent
// seam (internal/agent/ctxplan_seam.go: heuristically the LAST user message's content
// words). That is a degenerate predictor: it sees one turn, not the trajectory, it lives in
// the wrong tier (the agent adapter, not the leaf where Forecast is defined), and it cannot
// be reused by the gateway or a demo. Forecast.Learn (learn.go) is the REACTIVE half — it
// revises intents from where the forecast was WRONG (a witnessed fault). This file is the
// PROACTIVE half: it predicts intents from where the session has BEEN (the recent span
// history). The two compose — Propose seeds the forecast from the trajectory; Learn revises
// it from the outcome — closing the "authored, not learned" gap into "authored FROM the
// trajectory, then refined by the outcome".
//
// # The proposer posture (a lossy prediction, never a decision)
//
// A Proposer is a LOSSY PROPOSER in the wirescreen sense (internal/wirescreen/doc.go): it
// emits Intents — a prediction of need, never a decision the system trusts. The planner
// treats a prediction as a hint: a MISS (a span the turn needs that the forecast did not
// predict) costs one demand-page fault, never a lost fact, because the store is lossless
// (forecast.go, faithful.go). So a wrong author degrades EFFICIENCY, never CORRECTNESS —
// the same one-sided posture every proposer rung obeys. The trust boundary does not move:
// the author reasons over SAFE span metadata (role+descriptor) only, exactly as the planner
// does; a sealed/tombstoned span is never predicted into context (it is skipped below), so
// a proposer can never launder poison.
//
// # The deterministic seed and the model-backed follow-on
//
// TrajectoryAuthor (below) is the shipped SEED: a deterministic, model-free predictor that
// scores each content token by RECENCY-WEIGHTED RECURRENCE across the recent trajectory — a
// token that has appeared recently and repeatedly is the strongest signal of what the next
// turns will touch. It is the heuristicScreener analogue (wirescreen RUNG 1 shipped its
// deterministic reference impl first; the model-backed Screener is NEXT). The MODEL-BACKED
// proposer — a small local model emitting predicted reference strings through this same
// Proposer seam — is the higher-tier follow-on (wirescreen RUNG 4: "Seam:
// ctxplan.Forecast.Intents ... This is ctxplan #556"), gated on the outbound transform seam
// that does not yet exist on the flagship passthrough. The interface is defined NOW so the
// model arm slots in later without changing the planner's contract, exactly as RUNG 1
// defined abi.SemanticScreen before any model Screener existed.

// Proposer authors a Forecast by predicting, from the trajectory, what the next Horizon
// turns are expected to reference — the "general preemptive planner" rung (#556). It is the
// one-method seam a model-backed predictor satisfies: emit Intents (the predicted reference
// string), carry the caller's Pins/Weights/Horizon through, and the planner optimizes the
// O(1) view that covers them. A MISS is a demand-page fault, never a lost fact (the store
// is lossless), so a proposer is a lossy predictor the planner never trusts for correctness.
//
// The contract reasons over SAFE span metadata (role+descriptor) — the same vocabulary the
// planner scores against — so a Proposer adds no coupling to sealed bytes and builds as a
// pure foundation-tier value. A model proposer that must read full message bodies reads them
// through the store's trust gate at a higher tier; the seed here needs no bytes.
type Proposer interface {
	Propose(spans []Span) Forecast
}

// DefaultAuthorWindow is the recency tail the author predicts from: the most recent
// DefaultAuthorWindow spans. The recent trajectory is the strongest predictor of the
// immediate next turns, so the author scans a bounded tail rather than the whole history —
// the same posture the index's RecencyWindow takes (a few dozen recent spans), sized a
// little larger so a recurring session topic is caught across a couple of turns of noise.
// It is a conservative seed, not a tuned constant.
const DefaultAuthorWindow = 64

// DefaultAuthorIntents caps a trajectory-authored intent list so the prediction stays O(1)
// — it never grows into an unbounded memory of every token ever seen. It is the proactive
// analogue of learn.go's maxLearnedIntents (which caps the reactive fault-memory): the
// focused top-K predicted tokens, fewer than the accumulated fault set, because a focused
// forecast weights each intent more (Forecast.relevance is a fraction of distinct query
// tokens, so a broad intent list dilutes per-token relevance).
const DefaultAuthorIntents = 16

// TrajectoryAuthor is the deterministic, model-free seed Proposer: it authors a Forecast
// whose Intents are the content tokens with the highest RECENCY-WEIGHTED RECURRENCE across
// the recent trajectory. A token is scored, for every recent span whose role+descriptor
// contains it, by (1 + recency(span, maxStep)): the 1 weights RECURRENCE (a topic that has
// come up repeatedly is likely to come up again — momentum), and the recency term in [0,1]
// weights the MOST RECENT occurrences higher (the immediate-next-turn prior). Recurrence
// therefore dominates and recency breaks near-ties — the two sub-signals a trajectory
// carries about what is coming next.
//
// The zero value is valid (defaults applied): Window=DefaultAuthorWindow,
// MaxIntents=DefaultAuthorIntents, Horizon=1, no Pins, default Weights. It is deterministic
// (map iteration is neutralized by a total (score desc, token asc) tie-break), fail-closed
// (sealed/tombstoned spans are skipped — their content is never predicted into context; an
// empty trajectory yields an empty-intent forecast that falls back to the priors + pins),
// and bounded (the intent list is capped at MaxIntents). Pins, Weights, and Horizon are
// carried through verbatim — the author predicts the Intents; the caller owns the structural
// pins and the cost constants.
type TrajectoryAuthor struct {
	// Window is how many of the most-recent spans the author scans (the recency tail it
	// predicts from). <= 0 => DefaultAuthorWindow.
	Window int
	// MaxIntents caps the authored intent list. <= 0 => DefaultAuthorIntents.
	MaxIntents int
	// Horizon is carried into Forecast.Horizon (advisory; <= 0 => 1).
	Horizon int
	// Pins are carried into Forecast.Pins verbatim — the structural spans a turn cannot
	// proceed without (system prompt, active goal). The author predicts Intents; the
	// caller owns the pins.
	Pins []string
	// Weights are carried into Forecast.Weights verbatim (the zero value => the planner
	// applies DefaultWeights via Weights.orDefault).
	Weights Weights
}

// Propose authors a Forecast from the trajectory: it scans the most-recent Window spans,
// scores each distinct content token by recency-weighted recurrence, and emits the top
// MaxIntents as the forecast's Intents — the predicted reference string the planner
// pre-materializes the O(1) view against. Sealed/tombstoned spans are skipped (their
// content is never predicted into context). The result is deterministic and composes with
// the planner exactly like a hand-supplied Forecast: a span whose content matches the
// trajectory-derived intents gets a high relevance Benefit and is kept resident.
func (a TrajectoryAuthor) Propose(spans []Span) Forecast {
	window := a.Window
	if window <= 0 {
		window = DefaultAuthorWindow
	}
	maxIntents := a.MaxIntents
	if maxIntents <= 0 {
		maxIntents = DefaultAuthorIntents
	}

	// maxStep over ALL spans is the recency-normalization vocabulary (the same recency()
	// uses when scoring), so the author and the planner share one recency scale. A
	// single-step set (maxStep == 0) makes recency() return 0 for every span — handled
	// below by the +1 recurrence base, so a one-span trajectory still predicts its tokens.
	maxStep := 0
	for _, s := range spans {
		if s.Step > maxStep {
			maxStep = s.Step
		}
	}

	// Score each content token by RECURRENCY (base 1 per spanning occurrence) PLUS a
	// recency bonus in [0,1]. Scanning only the recent Window bounds the work; the tail is
	// spans[lo:], append order == step order, so it is the most-recent tail.
	lo := len(spans) - window
	if lo < 0 {
		lo = 0
	}
	scores := map[string]float64{}
	for _, s := range spans[lo:] {
		// Fail-closed: a sealed/tombstoned span can never be resident, so its content must
		// never be predicted into context — predicting it would steer the planner toward a
		// span the trust gate refuses. Skip it entirely (its tokens teach nothing).
		if s.Sealed || s.Tombstoned {
			continue
		}
		w := 1 + recency(s.Step, maxStep) // recurrence (1) + recency bonus ([0,1])
		for t := range tokenSet(s.Role + " " + s.Descriptor) {
			scores[t] += w
		}
	}

	// Rank by (score desc, token asc) — a total order that neutralizes map iteration, so the
	// same trajectory yields a byte-identical intent list. Take the top MaxIntents.
	type scored struct {
		tok   string
		score float64
	}
	ranked := make([]scored, 0, len(scores))
	for t, sc := range scores {
		ranked = append(ranked, scored{tok: t, score: sc})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		return ranked[i].tok < ranked[j].tok
	})
	if len(ranked) > maxIntents {
		ranked = ranked[:maxIntents]
	}
	intents := make([]string, len(ranked))
	for i, c := range ranked {
		intents[i] = c.tok
	}

	horizon := a.Horizon
	if horizon <= 0 {
		horizon = 1
	}
	return Forecast{
		Intents: intents,
		Horizon: horizon,
		Pins:    a.Pins,
		Weights: a.Weights,
	}
}
