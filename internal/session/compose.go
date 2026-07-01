package session

// compose.go — the per-session budget COMPOSITION (issue #628, epic #620 track 5).
//
// THE GAP IT CLOSES. The drive state names a Pace (per-turn throttle), but two other
// budgets that govern how hard a session works existed BEFORE the drive state and never
// composed with it:
//
//   - agent.SessionPlanner.Budget — the per-turn ctxplan resident-context window. It is
//     genuinely per-session but set ONCE at construction; throttling a session never
//     touched it, so a "slowed" session still planned its full-width context window.
//   - the matmul-cores FAK_BUDGET (internal/model/budget.go) — the fraction of the
//     machine's cores a forward pass uses. It is resolved ONCE at init and is process-
//     WIDE, so "slow this session" had no per-session core knob at all.
//
// So "slow this one session" meant reaching for a set-once per-session window AND a
// process-global core budget — two knobs, neither of them the Pace the operator just
// set. This file makes Pace.MaxTokensPerTurn the SINGLE knob: it derives both budgets
// from the same throttle, so dialing a session's per-turn output down also drives its
// resident-context window and its core fraction down, proportionally.
//
// WHAT THE THROTTLE IS RELATIVE TO. MaxTokensPerTurn is an absolute per-turn OUTPUT cap;
// "throttle" only has meaning relative to the session's UNTHROTTLED per-turn output
// target (the baseline the consumer knows — e.g. the planner's default max-new-tokens).
// The ratio is that quotient, clamped to (0,1]: a cap at or above the baseline is "no
// throttle" (ratio 1.0), a non-positive cap or a non-positive baseline is "no opinion"
// (ratio 1.0, the byte-for-byte pre-compose path). The composition is PURE — it reads no
// process state — so the derived budgets are reproducible and table-testable, the same
// posture model/budget.go's resolveBudgetWorkers takes for the env path.
//
// HONEST FENCE. Composing a per-session ratio into the per-session planner window is
// sound (the window is the session's own value). Composing it into the process-global
// matmul pool is the "deliberate, measured" change model/budget.go's doc flagged: a live
// model.SetWorkerBudget mutates a process-WIDE worker count, so applying ComposeWorker-
// Fraction per turn is sound only in a SINGLE-session process (the harness / a one-shot
// serve), not under a multi-session gateway sharing one pool. This file computes the
// fraction from the knob; WHERE it is safe to apply it is the consumer's call.
//
// THROUGHPUT (#1585, epic #1570 "managed context"). MaxTokensPerTurn is a CONFIGURED
// knob — an operator or admission control sets it ahead of time. But a session can also
// fall behind its expected pace for reasons nobody configured: GPU contention, a slow
// upstream model, backpressure. Throughput carries that OBSERVED rate (an
// ObservedTokensPerSec measured against an ExpectedTokensPerSec) as its own value —
// deliberately NOT new fields on Pace, since Pace is declared in session.go and this
// package's session.go is concurrently owned by other in-flight work right now; keeping
// the new axis a standalone type declared here means #1585 never has to touch that file.
// ThroughputRatio/ComposePlannerBudgetForThroughput apply the exact same ratio-and-floor
// shape ComposePlannerBudget already uses for the configured cap, so "the session is
// measurably slower right now" shrinks the resident-context window exactly as "the
// operator configured a slower cap" does — without requiring anyone to have set
// MaxTokensPerTurn at all. ComposePace takes a Pace AND a Throughput and folds both
// signals into one Budget by taking whichever is more constraining, so the two knobs
// compose rather than one silently overriding the other.

import "math"

// Throughput is a session's MEASURED recent pace (#1585): how many tokens per
// wall-clock second it actually produced over some recent window the caller owns (e.g. a
// rolling turn-completion tracker), judged against the rate it was expected to sustain.
// It is the runtime-observed twin of Pace.MaxTokensPerTurn, kept as its own type rather
// than fields on Pace itself — see the file header. The zero value means "no observation
// yet" and composes to "no opinion" everywhere it is used.
type Throughput struct {
	// ObservedTokensPerSec is the measured recent rate. 0 means no observation yet.
	ObservedTokensPerSec float64 `json:"observed_tokens_per_sec,omitempty"`
	// ExpectedTokensPerSec is the reference rate ObservedTokensPerSec is judged against
	// (analogous to baselineOutput for MaxTokensPerTurn). 0 means no expectation is
	// configured (no opinion).
	ExpectedTokensPerSec float64 `json:"expected_tokens_per_sec,omitempty"`
}

// MinPlannerBudgetDivisor floors a composed planner budget at base/MinPlannerBudgetDivisor:
// no matter how hard a session is throttled, its resident-context window keeps at least
// this fraction of its baseline, so the structural pins (system / first+last user / the
// active goal) and a minimal recency tail still fit. A proportional floor (rather than a
// machine-wide magic constant) scales with the configured window: a 4096-token base floors
// at 512, a 1024 base at 128. The throttle drives the window DOWN, never to a unusable size.
const MinPlannerBudgetDivisor = 8

// ComposedBudgets is the two derived per-session budgets a single throttle produces from
// one knob (Pace.MaxTokensPerTurn). PlannerBudget is the resident-context window to set on
// agent.SessionPlanner.Budget; WorkerFraction is the matmul FAK_BUDGET fraction in (0,1] to
// feed model.SetWorkerBudget (single-session-sound only — see the file header fence). Ratio
// is the throttle that produced them (1.0 == unthrottled), carried so a consumer can log or
// gate on "by how much" without recomputing.
type ComposedBudgets struct {
	PlannerBudget  int     `json:"planner_budget"`
	WorkerFraction float64 `json:"worker_fraction"`
	Ratio          float64 `json:"ratio"`
}

// ThrottleRatio is the fraction in (0,1] by which this Pace throttles a session's per-turn
// work, relative to baselineOutput — the session's unthrottled per-turn output target. It
// is 1.0 (no throttle) when this Pace voices no opinion (MaxTokensPerTurn <= 0), when there
// is no baseline to scale against (baselineOutput <= 0), or when the cap sits at or above
// the baseline (a cap that does not actually lower the turn is not a throttle). Otherwise it
// is the quotient MaxTokensPerTurn/baselineOutput, a value in (0,1). It is the shared
// primitive both ComposePlannerBudget and ComposeWorkerFraction round identically against.
func (p Pace) ThrottleRatio(baselineOutput int) float64 {
	if p.MaxTokensPerTurn <= 0 || baselineOutput <= 0 {
		return 1.0
	}
	if p.MaxTokensPerTurn >= baselineOutput {
		return 1.0
	}
	return float64(p.MaxTokensPerTurn) / float64(baselineOutput)
}

// ComposePlannerBudget scales a base resident-context window down by this Pace's throttle
// ratio, floored at base/MinPlannerBudgetDivisor so a hard throttle never starves the window
// below a usable size. A non-positive base is returned unchanged (nothing to scale); an
// unthrottled Pace (ratio 1.0) returns the base verbatim, so an un-paced session's planner
// budget is byte-for-byte what it was before this composition existed. The result is the
// value to assign to agent.SessionPlanner.Budget.
func (p Pace) ComposePlannerBudget(basePlannerBudget, baselineOutput int) int {
	if basePlannerBudget <= 0 {
		return basePlannerBudget
	}
	r := p.ThrottleRatio(baselineOutput)
	if r >= 1.0 {
		return basePlannerBudget
	}
	floor := basePlannerBudget / MinPlannerBudgetDivisor
	if floor < 1 {
		floor = 1
	}
	composed := int(math.Round(float64(basePlannerBudget) * r))
	if composed < floor {
		composed = floor
	}
	return composed
}

// ComposeWorkerFraction is the matmul FAK_BUDGET fraction in (0,1] this Pace asks the cores
// to run at — the throttle ratio directly: a session paced to half its baseline output runs
// its forward pass on (about) half the machine. The value is shaped to feed model.SetWorker-
// Budget, which floors any positive fraction to at least one worker, so a deep throttle slows
// a session without ever zeroing its compute. SOUND ONLY in a single-session process (the
// file header fence): the matmul pool is process-global.
func (p Pace) ComposeWorkerFraction(baselineOutput int) float64 {
	return p.ThrottleRatio(baselineOutput)
}

// Compose folds both derived budgets (and the ratio that produced them) into one record —
// the single call a consumer makes to turn the Pace knob into the two real budgets. It is a
// pure projection of ComposePlannerBudget + ComposeWorkerFraction over the same baseline.
func (p Pace) Compose(basePlannerBudget, baselineOutput int) ComposedBudgets {
	return ComposedBudgets{
		PlannerBudget:  p.ComposePlannerBudget(basePlannerBudget, baselineOutput),
		WorkerFraction: p.ComposeWorkerFraction(baselineOutput),
		Ratio:          p.ThrottleRatio(baselineOutput),
	}
}

// ThroughputRatio (#1585, epic #1570 "managed context") is the fraction in (0,1] of its
// expected rate this session is ACTUALLY achieving, judged from a runtime observation rather
// than a configured cap — the measured twin of ThrottleRatio. It is 1.0 ("no constraint")
// when there is no observation yet (ObservedTokensPerSec <= 0), no expectation configured
// (ExpectedTokensPerSec <= 0), or the session is keeping pace or running faster than expected
// (Observed >= Expected — running ahead is never a reason to shrink the window here, exactly
// as ThrottleRatio never widens on a cap above baseline). Otherwise it is the quotient
// Observed/Expected, a value in (0,1): a session running at half its expected throughput
// yields 0.5. NaN/Inf inputs (a corrupt observation) fail closed to 1.0, never past infinity
// or negative.
func (t Throughput) ThroughputRatio() float64 {
	if t.ObservedTokensPerSec <= 0 || t.ExpectedTokensPerSec <= 0 {
		return 1.0
	}
	if math.IsNaN(t.ObservedTokensPerSec) || math.IsInf(t.ObservedTokensPerSec, 0) {
		return 1.0
	}
	if t.ObservedTokensPerSec >= t.ExpectedTokensPerSec {
		return 1.0
	}
	return t.ObservedTokensPerSec / t.ExpectedTokensPerSec
}

// ComposePlannerBudgetForThroughput scales a base resident-context window down by this
// Throughput's observed ratio, floored at base/MinPlannerBudgetDivisor — the same floor
// discipline ComposePlannerBudget applies to the configured MaxTokensPerTurn cap, now driven
// by a measured runtime rate instead of a configured one. A session that is falling behind
// its expected throughput sees its resident-context window shrink proportionally (fewer
// spans to hold hot while it is producing tokens slowly), and a session keeping pace or
// running ahead is untouched — byte-for-byte the base, exactly as an un-observed Throughput
// was before this composition existed. A non-positive base is returned unchanged.
func (t Throughput) ComposePlannerBudgetForThroughput(basePlannerBudget int) int {
	if basePlannerBudget <= 0 {
		return basePlannerBudget
	}
	r := t.ThroughputRatio()
	if r >= 1.0 {
		return basePlannerBudget
	}
	floor := basePlannerBudget / MinPlannerBudgetDivisor
	if floor < 1 {
		floor = 1
	}
	composed := int(math.Round(float64(basePlannerBudget) * r))
	if composed < floor {
		composed = floor
	}
	return composed
}

// ComposePace folds BOTH pace signals — the configured MaxTokensPerTurn cap (p) and the
// observed runtime Throughput (t) — into a single resident-context window: whichever signal
// is more constraining (the smaller of the two composed budgets) wins, so a session that is
// both throttled by configuration AND running behind its expected throughput gets the harder
// of the two shrinks, never the milder one silently overriding the other. This is the
// one-call entry point a caller composing BOTH #628's configured pace and #1585's observed
// throughput into one planner Budget should use, in place of calling ComposePlannerBudget and
// ComposePlannerBudgetForThroughput separately and having to reconcile them by hand.
func (p Pace) ComposePace(t Throughput, basePlannerBudget, baselineOutput int) int {
	configured := p.ComposePlannerBudget(basePlannerBudget, baselineOutput)
	observed := t.ComposePlannerBudgetForThroughput(basePlannerBudget)
	if observed < configured {
		return observed
	}
	return configured
}
