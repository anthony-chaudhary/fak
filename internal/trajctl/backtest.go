package trajctl

// backtest.go — issue #2573, epic #2533: the REPLAY BACKTEST — qualify a candidate scorer
// against recorded history BEFORE it steers anything live.
//
// The calibration meter (#2566, calibration.go) grades scorers on the rows they ALREADY
// wrote into the live ledger, so it can only tell you a method was mis-calibrated after it
// has been steering sessions wrong for a while. This leaf moves that verdict earlier. A
// candidate scorer is REPLAYED over a recorded corpus — a previously exported trajctl
// ledger whose objectives already carry their witnessed W3 outcome — and the numbers it
// freshly computes are correlated against those known outcomes, optionally alongside the
// incumbent method it would replace. Nothing is appended, nothing is steered, and no model
// is called: the whole pass is a pure fold over recorded history, which is exactly what
// makes replay-as-fitness the near-free oracle for a scoring-method change.
//
// # The qualification gate: a scorer ships with its backtest
//
// Before a new or changed scoring method is registered for live use it must earn a
// QUALIFIED verdict on a recorded corpus. Two conditions, both cheap:
//
//  1. Its readings track the witnessed outcome at least as well as the calibration meter's
//     WELL_CALIBRATED band (r >= calibrationWellThreshold) — the same bar the live meter
//     uses, so a scorer cannot pass offline on a threshold the live pane would fail it on.
//  2. It does not MATERIALLY regress the incumbent it replaces (within
//     backtestRegressionTolerance) — "new" is not "better", and a rewrite that reads
//     history worse than the method it replaces has no business steering.
//
// # Why the verdict is typed, and why INCONCLUSIVE is not a pass
//
// "The backtest could not run" and "the backtest ran and the scorer lost" are different
// facts, and collapsing them is precisely how a bad scorer ships anyway — a tool error that
// degrades to a silent skip reads, to everything downstream, exactly like a clean bill of
// health. So the vocabulary is closed and total: QUALIFIED, NOT_QUALIFIED (it ran, the
// scorer lost), INCONCLUSIVE (it ran, the corpus could not decide), and BACKTEST_ERROR (it
// could not run at all). Every outcome also carries a closed BacktestReason code, so a
// caller never parses prose, and INCONCLUSIVE is deliberately NOT QUALIFIED: a corpus too
// thin to refute a scorer has not endorsed it either.
//
// # What the replay hands the scorer — and the limit that follows
//
// The corpus is the only input, so the evidence window is rebuilt from it alone: the
// objective's recorded score rows become PriorScores, their recorded commit evidence
// becomes PhaseCommits (keyed by the phase id the commit scorer stamps into
// EvidenceRef.Detail), and the resolver verifies exactly those pointers the corpus
// recorded — a pointer the corpus never saw reads unknown, so an un-grounded claim scores
// 0 rather than being credited. That keeps the replay offline and deterministic: no git, no
// clock, no network.
//
// The honest limit, stated so it can be checked: the window is reconstructed AS RECORDED,
// with no look-ahead fence. A corpus row stamped after the outcome it is being correlated
// against is still visible to the candidate, so a scorer that simply echoes the recorded W3
// rows will qualify trivially. This backtest measures CALIBRATION, not resistance to
// reading the answer; anti-gaming is the audit fold's job (audit.go), and a time-fenced
// replay window is the obvious follow-on if a scorer is ever suspected of the echo.
//
// A corpus also records no sessions, so a session-fed scorer (activity-progress-divergence)
// replays against an empty session set and emits nothing — which surfaces as INCONCLUSIVE
// with a THIN_CORPUS reason rather than as a pass. That is the intended shape: this corpus
// cannot qualify that scorer, and it says so.

import "fmt"

// BacktestSchema is the pinned schema id for a replay-backtest report.
const BacktestSchema = "fak-trajctl-backtest/1"

// BacktestVerdict is the closed backtest vocabulary. Exactly one of the four holds for
// every backtest, and only BacktestQualified permits a scorer to go live.
type BacktestVerdict string

const (
	// BacktestQualified: the backtest ran and the candidate earned its way to the live
	// spin — it tracks the witnessed outcome and does not regress the incumbent.
	BacktestQualified BacktestVerdict = "QUALIFIED"
	// BacktestNotQualified: the backtest ran and the candidate LOST. It read recorded
	// history too weakly, backwards, or worse than the method it would replace.
	BacktestNotQualified BacktestVerdict = "NOT_QUALIFIED"
	// BacktestInconclusive: the backtest ran and the corpus could not decide — too few
	// paired samples, or no variance to correlate. Not a pass: an undecided scorer has
	// not been qualified, it has only failed to be refuted.
	BacktestInconclusive BacktestVerdict = "INCONCLUSIVE"
	// BacktestErrored: the backtest could NOT RUN — a missing candidate, an unreadable
	// corpus, or a corpus carrying no witnessed outcome. A tool fault, never a verdict on
	// the scorer, and never silently a skip.
	BacktestErrored BacktestVerdict = "BACKTEST_ERROR"
)

// BacktestReason is the closed reason code behind a verdict, so a caller can branch on the
// cause without reading Detail's prose.
type BacktestReason string

const (
	// BacktestTracksOutcome accompanies QUALIFIED: the readings move with the outcome.
	BacktestTracksOutcome BacktestReason = "TRACKS_OUTCOME"
	// BacktestWeakCorrelation: positive but under the well-calibrated bar — better than
	// noise, not trustworthy enough to steer on.
	BacktestWeakCorrelation BacktestReason = "WEAK_CORRELATION"
	// BacktestAntiCorrelated: the candidate's high readings precede low outcomes. The
	// named enemy — it would steer the wrong way.
	BacktestAntiCorrelated BacktestReason = "ANTI_CORRELATED"
	// BacktestRegressesIncumbent: the candidate reads history materially worse than the
	// incumbent method it would replace.
	BacktestRegressesIncumbent BacktestReason = "REGRESSES_INCUMBENT"
	// BacktestThinCorpus: fewer than two paired samples, or no variance in the readings or
	// the outcomes. Widen the corpus; do not ship on it.
	BacktestThinCorpus BacktestReason = "THIN_CORPUS"
	// BacktestNoCandidate: no candidate scorer was supplied to replay.
	BacktestNoCandidate BacktestReason = "NO_CANDIDATE_SCORER"
	// BacktestUnknownScorer: the named method resolves to no registered scorer.
	BacktestUnknownScorer BacktestReason = "UNKNOWN_SCORER"
	// BacktestNotReplayable: the named scorer cannot run offline (it needs a gateway, a
	// clock, or a network), so recorded history cannot qualify it here.
	BacktestNotReplayable BacktestReason = "SCORER_NOT_REPLAYABLE"
	// BacktestNoCorpus: the corpus path is missing or unreadable.
	BacktestNoCorpus BacktestReason = "CORPUS_UNREADABLE"
	// BacktestNoRecordedOutcome: the corpus parsed but carries no witnessed W3 outcome to
	// calibrate against, so there is no ground truth to be right or wrong about.
	BacktestNoRecordedOutcome BacktestReason = "NO_RECORDED_OUTCOME"
)

// backtestRegressionTolerance is how far a candidate's coefficient may trail the
// incumbent's and still qualify. A finite corpus carries sampling noise, so demanding a
// strictly-not-worse reading would refuse candidates that merely tied; this is the noise
// room, deliberately small — anything past it is a real regression, not jitter.
const backtestRegressionTolerance = 0.05

// BacktestCase is one replayed objective: the outcome recorded for it, and what the
// candidate said when replayed against its rebuilt evidence window. Scored is false when
// the candidate emitted no row for the objective — recorded as a visible case with zero
// rows rather than dropped, so a scorer that silently declines half the corpus is legible
// in the report instead of quietly shrinking the denominator.
type BacktestCase struct {
	ObjectiveID string  `json:"objective_id"`
	Outcome     float64 `json:"outcome"`
	Rows        int     `json:"candidate_rows"`
	Candidate   float64 `json:"candidate_value"`
	Scored      bool    `json:"scored"`
}

// BacktestReport is the schema-pinned calibration report a backtest emits. Candidate is the
// replayed candidate's calibration against the recorded outcomes; Incumbent is the same
// measure for the method it would replace (nil when none was named); Delta is
// candidate-minus-incumbent, positive when the candidate reads history better.
type BacktestReport struct {
	Schema      string             `json:"schema"`
	Method      string             `json:"method"`
	Version     string             `json:"version"`
	GroundTruth string             `json:"ground_truth_method"`
	Objectives  int                `json:"corpus_objectives"`
	Outcomes    int                `json:"corpus_outcomes"`
	Candidate   ScorerCalibration  `json:"candidate"`
	Incumbent   *ScorerCalibration `json:"incumbent,omitempty"`
	Delta       float64            `json:"delta"`
	Cases       []BacktestCase     `json:"cases,omitempty"`
	Verdict     BacktestVerdict    `json:"verdict"`
	Reason      BacktestReason     `json:"reason"`
	Detail      string             `json:"detail"`
}

// Qualified reports whether the candidate earned the live spin. It is true for exactly one
// verdict, so a caller can never accidentally read a tool error or an undecided corpus as a
// pass.
func (r BacktestReport) Qualified() bool { return r.Verdict == BacktestQualified }

// BacktestRefusal builds a BACKTEST_ERROR report for a backtest that could not run at all —
// an unresolvable scorer name, an unreadable corpus. It exists so the shell's refusals carry
// the same schema and the same closed vocabulary as a completed run: a caller reading JSON
// always gets a report, never an empty body plus a nonzero exit.
func BacktestRefusal(method string, reason BacktestReason, detail string) BacktestReport {
	return BacktestReport{
		Schema:      BacktestSchema,
		Method:      method,
		GroundTruth: GroundTruthMethod,
		Verdict:     BacktestErrored,
		Reason:      reason,
		Detail:      detail,
	}
}

// Backtest replays candidate over the recorded corpus s and returns its qualification
// report. incumbent may be nil; when supplied it is replayed over the SAME rebuilt windows,
// so the comparison is apples-to-apples rather than candidate-now against incumbent-then.
// Pure: it appends nothing, reads no clock, and touches no network.
func (s State) Backtest(candidate, incumbent Scorer) BacktestReport {
	rep := BacktestReport{
		Schema:      BacktestSchema,
		GroundTruth: GroundTruthMethod,
		Objectives:  len(s.Objectives),
	}
	if candidate == nil {
		return backtestErrored(rep, BacktestNoCandidate, "no candidate scorer to replay")
	}
	rep.Method, rep.Version = candidate.Method(), candidate.Version()

	outcome, have := s.finalGroundTruth()
	ids := make([]string, 0, len(have))
	for _, id := range s.ObjectiveIDs() {
		if have[id] {
			ids = append(ids, id)
		}
	}
	rep.Outcomes = len(ids)
	if rep.Outcomes == 0 {
		return backtestErrored(rep, BacktestNoRecordedOutcome, fmt.Sprintf(
			"corpus carries %d objective(s) but no %s W3 outcome to calibrate against — there is no ground truth to be right or wrong about",
			rep.Objectives, GroundTruthMethod))
	}

	xs, ys, cases := s.replayScorer(candidate, outcome, ids)
	rep.Cases = cases
	rep.Candidate = backtestCalibration(candidate, xs, ys)
	if incumbent != nil {
		ix, iy, _ := s.replayScorer(incumbent, outcome, ids)
		inc := backtestCalibration(incumbent, ix, iy)
		rep.Incumbent = &inc
		if rep.Candidate.Measured && inc.Measured {
			rep.Delta = rep.Candidate.Coefficient - inc.Coefficient
		}
	}
	return backtestDecide(rep)
}

// replayScorer runs sc over every objective with a recorded outcome and pools one (reading,
// outcome) pair per emitted row — the same pooling the live calibration meter uses, so a
// multi-row scorer is measured identically offline and online.
func (s State) replayScorer(sc Scorer, outcome map[string]float64, ids []string) (xs, ys []float64, cases []BacktestCase) {
	cases = make([]BacktestCase, 0, len(ids))
	for _, id := range ids {
		rows := sc.Score(s.Objectives[id], s.replayWindow(id))
		c := BacktestCase{ObjectiveID: id, Outcome: outcome[id], Rows: len(rows)}
		var sum float64
		for _, row := range rows {
			xs = append(xs, row.Value)
			ys = append(ys, outcome[id])
			sum += row.Value
		}
		if len(rows) > 0 {
			c.Scored = true
			c.Candidate = sum / float64(len(rows))
		}
		cases = append(cases, c)
	}
	return xs, ys, cases
}

// replayWindow rebuilds, from the corpus alone, the evidence window a scorer would have
// folded for objectiveID: the objective's recorded rows as the prior curve, their recorded
// commit evidence as the phase→commit bindings (EvidenceRef.Detail carries the phase id the
// commit scorer stamps), and a resolver that verifies exactly the pointers the corpus
// recorded. A pointer absent from the corpus reads unknown, never verified, so replay can
// only credit evidence the recording actually saw.
func (s State) replayWindow(objectiveID string) EvidenceWindow {
	prior := s.ScoresFor(objectiveID)
	commits := map[string][]string{}
	bound := map[string]bool{}
	recorded := map[string]bool{}
	var stamp int64
	for _, row := range prior {
		if row.UnixMillis > stamp {
			stamp = row.UnixMillis
		}
		for _, ev := range row.Evidence {
			recorded[backtestRefKey(ev.Kind, ev.Ref)] = true
			if ev.Kind != "commit" || ev.Detail == "" {
				continue
			}
			key := ev.Detail + "\x00" + ev.Ref
			if bound[key] {
				continue // the same commit recorded twice for a phase binds it once
			}
			bound[key] = true
			commits[ev.Detail] = append(commits[ev.Detail], ev.Ref)
		}
	}
	return EvidenceWindow{
		PhaseCommits: commits,
		PriorScores:  prior,
		Resolve: func(ref EvidenceRef) EvidenceStatus {
			if recorded[backtestRefKey(ref.Kind, ref.Ref)] {
				return EvidenceVerified
			}
			return EvidenceUnknown
		},
		UnixMillis: stamp,
	}
}

// backtestRefKey keys an evidence pointer by kind and ref, ignoring Detail: the same commit
// recorded under two different phase details is still the same witnessed object. The NUL
// separator cannot occur in either field, so distinct pairs cannot collide.
func backtestRefKey(kind, ref string) string { return kind + "\x00" + ref }

// backtestCalibration measures one replayed scorer's readings against the paired outcomes,
// reusing the live meter's Pearson fold and banding so an offline verdict and the live
// scorer pane can never drift apart.
func backtestCalibration(sc Scorer, xs, ys []float64) ScorerCalibration {
	c := ScorerCalibration{Method: sc.Method(), Version: sc.Version(), Samples: len(xs)}
	if r, ok := pearson(xs, ys); ok {
		c.Measured = true
		c.Coefficient = r
		c.Verdict = calibrationVerdict(r)
		c.Detail = fmt.Sprintf("r=%+.2f over %d replayed sample(s) vs the %s outcome", r, c.Samples, GroundTruthMethod)
		return c
	}
	c.Verdict = CalibrationInsufficient
	c.Detail = fmt.Sprintf("insufficient signal: %d replayed sample(s) with no variance to correlate — needs >=2 objectives with differing outcomes AND differing readings", c.Samples)
	return c
}

// backtestDecide applies the qualification gate to a completed replay. The order matters:
// an unmeasurable corpus is undecided (never a failure of the scorer), a backwards reading
// is called out ahead of a merely weak one, and the incumbent comparison is the last gate a
// well-calibrated candidate must still pass.
func backtestDecide(rep BacktestReport) BacktestReport {
	switch {
	case !rep.Candidate.Measured:
		rep.Verdict, rep.Reason = BacktestInconclusive, BacktestThinCorpus
		rep.Detail = fmt.Sprintf("corpus cannot decide: %s — INCONCLUSIVE is not a pass; widen the corpus before shipping %s",
			rep.Candidate.Detail, rep.Method)
	case rep.Candidate.Coefficient < 0:
		rep.Verdict, rep.Reason = BacktestNotQualified, BacktestAntiCorrelated
		rep.Detail = fmt.Sprintf("%s ANTI-correlates with the witnessed outcome (%s): its high readings precede low outcomes, so it would steer the wrong way",
			rep.Method, rep.Candidate.Detail)
	case rep.Candidate.Coefficient < calibrationWellThreshold:
		rep.Verdict, rep.Reason = BacktestNotQualified, BacktestWeakCorrelation
		rep.Detail = fmt.Sprintf("%s reads recorded history too weakly to steer on (%s); qualification needs r >= %+.2f",
			rep.Method, rep.Candidate.Detail, calibrationWellThreshold)
	case rep.Incumbent != nil && rep.Incumbent.Measured && rep.Candidate.Coefficient < rep.Incumbent.Coefficient-backtestRegressionTolerance:
		rep.Verdict, rep.Reason = BacktestNotQualified, BacktestRegressesIncumbent
		rep.Detail = fmt.Sprintf("%s r=%+.2f trails incumbent %s r=%+.2f (delta %+.2f, tolerance %.2f): new is not better",
			rep.Method, rep.Candidate.Coefficient, rep.Incumbent.Method, rep.Incumbent.Coefficient, rep.Delta, backtestRegressionTolerance)
	default:
		rep.Verdict, rep.Reason = BacktestQualified, BacktestTracksOutcome
		rep.Detail = fmt.Sprintf("%s tracks the witnessed outcome across %d replayed objective(s) (%s)%s",
			rep.Method, rep.Outcomes, rep.Candidate.Detail, backtestIncumbentNote(rep))
	}
	return rep
}

// backtestIncumbentNote appends the incumbent comparison to a qualifying detail line, or
// says plainly that none was named — a QUALIFIED verdict should never leave a reader
// guessing whether anything was compared against.
func backtestIncumbentNote(rep BacktestReport) string {
	if rep.Incumbent == nil {
		return "; no incumbent named, so this qualifies calibration only"
	}
	if !rep.Incumbent.Measured {
		return fmt.Sprintf("; incumbent %s was unmeasurable on this corpus, so no regression check was possible", rep.Incumbent.Method)
	}
	return fmt.Sprintf("; vs incumbent %s r=%+.2f (delta %+.2f)", rep.Incumbent.Method, rep.Incumbent.Coefficient, rep.Delta)
}

// backtestErrored stamps a could-not-run outcome onto a partially built report, keeping the
// corpus counts already gathered so the reader can see how much was read before the fault.
func backtestErrored(rep BacktestReport, reason BacktestReason, detail string) BacktestReport {
	rep.Verdict, rep.Reason, rep.Detail = BacktestErrored, reason, detail
	return rep
}
