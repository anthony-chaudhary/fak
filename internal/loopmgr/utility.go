package loopmgr

// Loop UTILITY: did the loop actually do anything, and is it failing? (#6497)
//
// The loop fold has always counted how often a loop RAN — Fires, Started, Ended —
// and whether an independent referee witnessed it (Witnessed / WitnessRefused /
// WitnessUnavailable). It never counted the one thing an operator asks first when a
// scheduled loop looks green: did the run SUCCEED, and did it produce anything?
//
// That gap is what the maintenance audit hit. `scout-loop/task-scheduler` and
// `logvault-capture` each recorded two runs and two `end status=failed
// reason=EXIT_NONZERO` rows — four runs, four failures, zero successes — yet every
// projection over the ledger reported them as loops with Runs=2, because Ended
// counts an end regardless of its outcome. The OS scheduler said one thing
// (`LastTaskResult`) and the ledger said another, and nothing in fak's own status
// surface named the disagreement.
//
// So this file adds the utility partition of an ended run. It is a PURE
// classification of an end event that the ledger already writes — no new event kind,
// no new ledger byte, no producer change required for the failure half to work
// today. Every ended run lands in exactly one bucket:
//
//	Ended == Failed + Effects + NoFuel + Unattributed
//
//   - FAILURE      the run ended failed/canceled. Counted, and counted CONSECUTIVELY
//     so a loop that is broken (not merely having a bad day) is loud.
//   - EFFECT       the run ended clean AND declared a useful effect (an issue filed,
//     an artifact captured) via the `useful_effects` metric.
//   - NO_FUEL      the run ended clean and declared, in the typed vocabulary, that
//     there was nothing to do. A no-fuel tick is a SUCCESS, not an effect, and must
//     never be laundered into either.
//   - UNATTRIBUTED the run ended clean but declared neither. This is the honest
//     bucket, and it is deliberately not called "success": all it proves is that a
//     child exited 0. The issue's demand — "acting mode produces a witnessed effect,
//     or a typed no-fuel result, not just child exit" — is measurable exactly because
//     this bucket exists and is separate.
//
// Cost is summed alongside, with the number of runs that actually reported one, so a
// zero cost on an un-instrumented loop reads as "never measured" rather than "free".

// The typed end-reason and metric keys a loop run declares its own utility with.
// They are part of the ledger's public vocabulary: a producer writes them on the
// `end` event, and this fold is the only reader that interprets them.
const (
	// ReasonNoFuel is the typed end reason for a run that completed successfully and
	// found nothing to do. It exists so "the scout found no lead" cannot be reported
	// as either a failure or an effect.
	ReasonNoFuel = "NO_FUEL"

	// MetricUsefulEffects is the count of useful effects a completed run produced
	// (issues filed, artifacts captured, studies written). > 0 makes the run an
	// EFFECT run; absent or 0 leaves it UNATTRIBUTED.
	MetricUsefulEffects = "useful_effects"

	// MetricCostMilliUSD is the run's cost in thousandths of a US dollar. Integer
	// milli-USD keeps the ledger's int64-metric contract exact (no float drift in a
	// hash-chained record).
	MetricCostMilliUSD = "cost_milli_usd"
)

// FailureAlertThreshold is the consecutive-failure count at which a loop is ALERTING
// rather than merely unlucky. Two is "the first repeated failure": one failure is a
// data point, two in a row is a broken loop, and both quarantined loops in #6497 hit
// it on their second scheduled day.
const FailureAlertThreshold uint64 = 2

// EndOutcome is the closed utility classification of one ended run. Every end event
// maps to exactly one value, so the four counters partition Ended exactly.
type EndOutcome string

const (
	// OutcomeFailure: the run ended failed or canceled — it did not complete.
	// Canceled joins failed because a canceled run produced no result either, which
	// is the same convention internal/operatortouches already reads ends with.
	OutcomeFailure EndOutcome = "failure"
	// OutcomeEffect: the run completed and declared >= 1 useful effect.
	OutcomeEffect EndOutcome = "effect"
	// OutcomeNoFuel: the run completed and declared, in the typed vocabulary, that
	// there was no work available.
	OutcomeNoFuel EndOutcome = "no_fuel"
	// OutcomeUnattributed: the run completed but declared neither an effect nor
	// no-fuel — a bare child exit 0, which proves the process ran and nothing more.
	OutcomeUnattributed EndOutcome = "unattributed"
)

// ClassifyEnd maps one `end` event onto its utility outcome. It is pure and total:
// any event (including one that is not an end) yields a value, and an end with no
// status is read through the same claimed-done fallback the fold uses, so the
// classification can never disagree with the state the snapshot records.
//
// Order matters and is fixed: a failing status wins over any declared effect (a run
// that filed an issue and then crashed did not complete), and an explicit no-fuel
// reason wins over an absent effect metric.
func ClassifyEnd(ev Event) EndOutcome {
	switch fallbackStatus(ev.Status, StatusClaimedDone) {
	case StatusFailed, StatusCanceled:
		return OutcomeFailure
	}
	if ev.Reason == ReasonNoFuel {
		return OutcomeNoFuel
	}
	if ev.Metrics[MetricUsefulEffects] > 0 {
		return OutcomeEffect
	}
	return OutcomeUnattributed
}

// applyEndUtility folds one end event into the snapshot's utility counters. Called
// from the single EventEnd arm of the fold, so the incremental SummarizeFrom path
// and the from-empty Summarize path share one definition by construction.
func (s *LoopSnapshot) applyEndUtility(ev Event) {
	switch ClassifyEnd(ev) {
	case OutcomeFailure:
		s.Failed++
		s.ConsecutiveFailures++
	case OutcomeEffect:
		s.Effects++
		s.ConsecutiveFailures = 0
	case OutcomeNoFuel:
		s.NoFuel++
		s.ConsecutiveFailures = 0
	default:
		s.Unattributed++
		s.ConsecutiveFailures = 0
	}
	// Cost is orthogonal to the outcome: a failed run still burned tokens, and an
	// operator triaging a broken loop wants to know how much it is burning to fail.
	if v, ok := ev.Metrics[MetricCostMilliUSD]; ok {
		s.CostMilliUSD += v
		s.CostedRuns++
	}
}

// FailureAlert reports whether the loop has failed FailureAlertThreshold times in a
// row — the first repeated failure. It is the one-field gate an operator pane or a
// `--check` exit code keys on, so the alert line is defined once here rather than
// re-derived at each surface.
func (s LoopSnapshot) FailureAlert() bool {
	return s.ConsecutiveFailures >= FailureAlertThreshold
}

// NeverSucceeded reports the shape #6497 was filed about: the loop has ended at
// least one run and EVERY one of them failed. It is stronger than FailureAlert — a
// loop can alert after a bad streak yet still have a history of good runs — and it
// is what makes "zero successful recorded runs" a fact the status surface states
// instead of one an operator has to reconstruct from the raw ledger.
func (s LoopSnapshot) NeverSucceeded() bool {
	return s.Ended > 0 && s.Failed == s.Ended
}
