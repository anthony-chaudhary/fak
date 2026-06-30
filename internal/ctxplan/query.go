package ctxplan

// query.go is the AGENT-CALLABLE facade over the planner: the typed surface a tool wrapper
// invokes so the MODEL — not just the host — can author its own context view. The goal's core
// verb is "the model can imagine and predict what it needs, query it, set it up, and use that
// as the fresh session's history." Today PlanCells is driven by the host (it hands the planner
// a Forecast + Budget it chose). This file flips the client: a model states its intents (and,
// optionally, its budget and horizon) as a typed PlanQuery, the SAME planner runs, and the
// model gets back a typed PlanView it can inspect (the selected/elided spans, the cost used,
// the EXPLAIN) and adopt as its working set.
//
// It is a THIN, PURE facade — it adds NO planning logic and NO divergence:
//
//   - It lowers a PlanQuery into the existing Forecast{Intents,Horizon,Pins,Weights}.
//   - It resolves the budget: the query's explicit Budget if set, else the adaptive
//     RecommendBudgetForForecast (adaptive.go) sizes a sane default from the query's declared
//     shape (intent count, pin count, horizon) — so an UNDER-SPECIFIED query that names only
//     intents still plans against a sensible working set instead of a magic constant.
//   - It runs the unchanged PlanCells and wraps the resulting Plan in a PlanView.
//
// A PlanView built from a query is therefore IDENTICAL to a direct PlanCells call with the
// same Forecast + resolved Budget — the facade is a typed front door, not a second planner.
// And it inherits every invariant the planner already enforces: a model-authored pin can never
// launder a sealed/tombstoned span into context (Optimize elides it up front), and a
// model-authored query can never exceed the budget (Optimize charges pins first and bounds the
// rest). So letting the MODEL drive the planner changes WHO states the prediction, never the
// honesty of the view it gets back.
//
// Everything here is DETERMINISTIC: the same (query, spans, cost) yields a byte-identical
// PlanView — replanning the same turn twice cannot drift.

// PlanQuery is the model's typed "imagine what I'll need" request — the agent-facing predicate
// the planner runs. It is the model's stated intents (what the upcoming turns will reference)
// plus the two optional knobs a model may set to shape its own working set: a Budget (the
// token cap on the resident view) and a Horizon (how many turns ahead the prediction covers).
// Pins and Weights are carried through verbatim so a model that wants to force a span resident
// or retune the cost constants can, under the same hard constraints the host path obeys.
//
// The ONLY required field is Intents — and even that may be empty (selection then falls to the
// utility/durability/recency priors and the pins, exactly as a Forecast with no intents does).
// An under-specified query is not an error; it plans against sane defaults (see Plan).
type PlanQuery struct {
	// Intents are the predicted reference strings for the next horizon — what the model
	// expects the upcoming turns to ask about (e.g. "refund fee", "auth token rotation").
	// They are the Forecast's predicate, matched content-word against each span's
	// role+descriptor. Empty means "no prediction" (priors + pins decide).
	Intents []string `json:"intents,omitempty"`

	// Budget is the OPTIONAL token cap the model sets on its own resident view. A nil Budget
	// (the under-specified case) means "size it for me": the facade falls back to the adaptive
	// RecommendBudgetForForecast, which sizes a sane working set from the query's declared
	// shape. A non-nil Budget is honored verbatim (after Optimize's own negative-clamp).
	Budget *Budget `json:"budget,omitempty"`

	// Horizon is how many turns ahead this query is meant to cover (>=1). It is advisory
	// metadata carried into the plan + EXPLAIN, and it feeds the adaptive budget default (a
	// longer horizon mildly enlarges the auto-sized working set). 0 means "this turn".
	Horizon int `json:"horizon,omitempty"`

	// Pins are span IDs the model declares MUST be resident regardless of score — charged
	// against the budget first. A pin that names a sealed/tombstoned span is REFUSED by the
	// planner (a model-authored pin cannot launder poison into context), never forced in.
	Pins []string `json:"pins,omitempty"`

	// Weights are the OPTIONAL cost constants the model may retune. The zero value uses
	// DefaultWeights (relevance dominates), so a query that sets none still scores sensibly.
	Weights Weights `json:"weights,omitempty"`

	// Assumptions are fact-like inputs the plan may rely on before acting. They are
	// scored into use/query/refresh classes and carried into PlanView so a low-confidence
	// or stale item cannot silently enter effect context as if it were fresh.
	Assumptions []Assumption `json:"assumptions,omitempty"`
}

// PlanView is the typed result the model inspects and adopts as its fresh history — the
// agent-facing peer of Plan. It carries exactly what a model needs to decide whether to use
// the view: the resident Selected spans (the working set it would adopt), the cold-but-
// recoverable Elided spans (what it gave up, each still one demand-page away), the CostUsed
// (resident tokens, <= Budget by construction), the resolved Budget and Horizon, the
// Faithful verdict (the honesty gate — true iff no span was destroyed), and the Explain
// string (the operator-readable EXPLAIN). The full underlying Plan is embedded for a caller
// that wants the complete accounting (objective, per-span density, the pinned-token split).
//
// PlanView REUSES the planner's own Selection/Elision types — it does not re-shape them — so a
// view's selected/elided sets are bit-for-bit the plan's, and faithful.go's audit applies
// unchanged.
type PlanView struct {
	// Selected is the resident working set in RENDER order (step ascending) — the history the
	// model would set up and continue from. Identical to Plan.Selected.
	Selected []Selection `json:"selected"`
	// Elided is the cold-but-recoverable set — what the view leaves out, each carrying its
	// page-back-in handle. Identical to Plan.Elided. A forecast MISS pages one of these back
	// in (a cheap fault, never a lost fact).
	Elided []Elision `json:"elided"`
	// CostUsed is the resident token cost of Selected — the realized size of the working set,
	// <= Budget by construction (pins may overrun, in which case Plan.OverBudget is set).
	CostUsed int `json:"cost_used"`
	// Budget is the token cap the plan was run under — the model's explicit Budget, or the
	// adaptive default the facade sized when the query left it unset.
	Budget int `json:"budget"`
	// Horizon is the turn lookahead carried from the query (advisory).
	Horizon int `json:"horizon,omitempty"`
	// Faithful is the honesty verdict: true iff the resident+elided sets partition every
	// candidate AND every elided span is recoverable (faithful.go). A model can refuse a view
	// that is not faithful before adopting it.
	Faithful bool `json:"faithful"`
	// Explain is the operator-readable EXPLAIN of the plan (the per-span cost/benefit/density
	// table plus the faithfulness footer) — the "step through this before you trust it"
	// surface the model can read back.
	Explain string `json:"explain"`
	// Plan is the full underlying plan for a caller that wants the complete accounting
	// (objective, pinned-token split, over-budget flag). PlanView is a typed projection of it,
	// never a divergent re-computation.
	Plan Plan `json:"plan"`
	// Assumptions is present when the query supplied assumptions. EffectSafe is false when
	// any item needs a user query or source refresh before an effectful action.
	Assumptions *AssumptionReport `json:"assumptions,omitempty"`
}

// Plan runs the agent's query through the SAME planner the host uses and returns the typed
// view — the pure, deterministic entrypoint a tool wrapper invokes. It performs NO I/O (it
// reads only SAFE span metadata) and couples to NO host: it takes the candidate spans and an
// optional CostModel (nil => TokenCost) directly, so the model-driven path and the host-driven
// path run the identical PlanCells over the identical inputs.
//
// Budget resolution is the only place the query and the host path differ in INPUT (never in
// planning):
//   - q.Budget set      => that budget verbatim (the model chose its own cap).
//   - q.Budget nil       => the adaptive RecommendBudgetForForecast sizes a default from the
//     query's declared shape within DefaultBudgetBounds, so an under-specified query (intents
//     only) still plans against a sensible working set instead of a magic constant.
//
// The returned PlanView is byte-identical to wrapping a direct
// PlanCells(spans, q.forecast(), resolvedBudget, cost) — the facade adds no divergence. A
// tool wrapper (in cmd/fak or internal/agent) registers this as the model-callable tool; that
// registration is a higher-tier follow-on and is intentionally NOT done here (this leaf owns
// only the typed query->plan->view surface).
func (q PlanQuery) Plan(spans []Span, cost CostModel) PlanView {
	f := q.forecast()
	budget := q.resolveBudget(f)
	p := PlanCells(spans, f, budget, cost)
	view := viewOf(p)
	if len(q.Assumptions) > 0 {
		report := AssessAssumptions(q.Assumptions, DefaultAssumptionPolicy())
		view.Assumptions = &report
	}
	return view
}

// forecast lowers the typed query into the planner's existing Forecast. It is a pure field
// copy — the query is a thin, model-facing renaming of the forecast's predicate, not a new
// model. Carrying Pins and Weights through verbatim is what lets a model force a span resident
// or retune the cost constants under the SAME hard constraints the host path obeys.
func (q PlanQuery) forecast() Forecast {
	return Forecast{
		Intents: q.Intents,
		Horizon: q.Horizon,
		Pins:    q.Pins,
		Weights: q.Weights,
	}
}

// resolveBudget returns the query's explicit Budget if set, else the adaptive default sized
// from the forecast's declared shape (intent count, pin count, horizon) within
// DefaultBudgetBounds. This is the ONLY place an under-specified query gains a default: a
// model that names only intents gets a sensibly sized working set, while a model that states
// its own cap is honored verbatim. The adaptive sizing is itself pure and deterministic
// (adaptive.go), so the resolved budget never drifts for the same query.
func (q PlanQuery) resolveBudget(f Forecast) Budget {
	if q.Budget != nil {
		return *q.Budget
	}
	return RecommendBudgetForForecast(f, DefaultBudgetBounds())
}

// viewOf projects a Plan into the typed PlanView — the resident/elided sets verbatim, the cost
// and budget accounting, and the precomputed Faithful verdict + Explain string. It is a pure
// projection (no re-planning), so a view's selected/elided/cost are the plan's exactly.
func viewOf(p Plan) PlanView {
	return PlanView{
		Selected: p.Selected,
		Elided:   p.Elided,
		CostUsed: p.CostUsed,
		Budget:   p.Budget,
		Horizon:  p.Horizon,
		Faithful: Audit(p).Faithful,
		Explain:  p.Explain(),
		Plan:     p,
	}
}
