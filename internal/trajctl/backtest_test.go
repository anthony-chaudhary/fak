package trajctl

import (
	"fmt"
	"testing"
)

// backtest_test.go — issue #2573. The witness set for the replay backtest, built so the
// qualification gate cannot pass vacuously: every corpus below is exercised by BOTH a
// tracking scorer and a deliberately KNOWN-BAD one, so a green run proves the gate
// discriminates rather than proving it says QUALIFIED to anything handed to it.

// trackingFixtureScorer reads the recorded judge rows in the replayed window's prior curve
// and reports their latest value. On a corpus whose judge rows were honest it tracks the
// witnessed outcome, so it is the positive control.
type trackingFixtureScorer struct{}

func (trackingFixtureScorer) Method() string  { return "fixture-tracking" }
func (trackingFixtureScorer) Version() string { return "1" }

func (trackingFixtureScorer) Score(obj Objective, win EvidenceWindow) []ScoreRow {
	value, ok := latestFixtureJudge(obj.ID, win.PriorScores)
	if !ok {
		return nil
	}
	return []ScoreRow{{ObjectiveID: obj.ID, Value: value, Method: "fixture-tracking", Version: "1", Witness: W1}}
}

// invertedFixtureScorer is the KNOWN-BAD scorer the done condition names: it reports the
// exact complement of the honest reading, so it scores high precisely where the objective
// was abandoned. It is well-formed, emits a row for every objective, and stays inside
// [0,1] — it is wrong in the one way that matters, which is why NOT_QUALIFIED here is a
// real refusal and not a validation error in disguise.
type invertedFixtureScorer struct{}

func (invertedFixtureScorer) Method() string  { return "fixture-inverted" }
func (invertedFixtureScorer) Version() string { return "1" }

func (invertedFixtureScorer) Score(obj Objective, win EvidenceWindow) []ScoreRow {
	value, ok := latestFixtureJudge(obj.ID, win.PriorScores)
	if !ok {
		return nil
	}
	return []ScoreRow{{ObjectiveID: obj.ID, Value: 1 - value, Method: "fixture-inverted", Version: "1", Witness: W1}}
}

// flatFixtureScorer always reports the same number. It is neither right nor wrong, which is
// exactly the shape a corpus cannot decide — the INCONCLUSIVE control.
type flatFixtureScorer struct{}

func (flatFixtureScorer) Method() string  { return "fixture-flat" }
func (flatFixtureScorer) Version() string { return "1" }

func (flatFixtureScorer) Score(obj Objective, _ EvidenceWindow) []ScoreRow {
	return []ScoreRow{{ObjectiveID: obj.ID, Value: 0.5, Method: "fixture-flat", Version: "1", Witness: W1}}
}

// partialFixtureScorer is the PARTIALLY mis-calibrated shape: it reads history correctly for
// most objectives and exactly backwards for a deterministic minority. That is the realistic
// middle of the band — not a scorer that is wrong about everything, but one whose readings
// are only loosely coupled to reality. Modulus tunes how large the wrong-way minority is, so
// one fixture yields both a barely-positive candidate (a small modulus inverts more
// objectives) and a decent-but-beatable one, without either being hand-fitted per assertion.
type partialFixtureScorer struct {
	method  string
	modulus int
}

func (p partialFixtureScorer) Method() string { return p.method }
func (partialFixtureScorer) Version() string  { return "1" }

func (p partialFixtureScorer) Score(obj Objective, win EvidenceWindow) []ScoreRow {
	value, ok := latestFixtureJudge(obj.ID, win.PriorScores)
	if !ok {
		return nil
	}
	if fixtureObjectiveIndex(obj.ID)%p.modulus == 1 {
		value = 1 - value // this objective is read backwards
	}
	return []ScoreRow{{ObjectiveID: obj.ID, Value: clamp01(value), Method: p.method, Version: "1", Witness: W1}}
}

// fixtureObjectiveIndex reads the trailing digit of a spread-corpus objective id ("obj0" ..
// "obj7"). It is the fixture's stand-in for "which objective is this", so the wrong-way
// minority is deterministic and the resulting coefficient is reproducible run to run.
func fixtureObjectiveIndex(id string) int {
	if id == "" {
		return 0
	}
	c := id[len(id)-1]
	if c < '0' || c > '9' {
		return 0
	}
	return int(c - '0')
}

// latestFixtureJudge returns the latest recorded "seed-judge" reading for an objective from
// a replayed prior curve. The fixtures read the corpus through the window exactly as a real
// scorer would, so the replay plumbing itself is under test.
func latestFixtureJudge(objectiveID string, prior []ScoreRow) (float64, bool) {
	var out float64
	found := false
	for _, row := range prior {
		if row.ObjectiveID == objectiveID && row.Method == "seed-judge" {
			out, found = row.Value, true
		}
	}
	return out, found
}

// backtestCorpus builds a recorded corpus: three objectives with DIFFERING witnessed W3
// outcomes, each carrying an honest recorded judge reading a replayed scorer can fold.
func backtestCorpus(t *testing.T) State {
	t.Helper()
	rows := []Row{
		ObjectiveRecord(Objective{ID: "shipped", Statement: "ship the widget", Status: StatusMet}),
		ObjectiveRecord(Objective{ID: "partial", Statement: "ship half the widget", Status: StatusActive}),
		ObjectiveRecord(Objective{ID: "dropped", Statement: "ship the abandoned widget", Status: StatusAbandoned}),
		ScoreRecord(ScoreRow{ObjectiveID: "shipped", Value: 1.0, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 10}),
		ScoreRecord(ScoreRow{ObjectiveID: "partial", Value: 0.5, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 10}),
		ScoreRecord(ScoreRow{ObjectiveID: "dropped", Value: 0.0, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 10}),
		ScoreRecord(ScoreRow{ObjectiveID: "shipped", Value: 0.9, Method: "seed-judge", Version: "1", Witness: W1, UnixMillis: 20}),
		ScoreRecord(ScoreRow{ObjectiveID: "partial", Value: 0.5, Method: "seed-judge", Version: "1", Witness: W1, UnixMillis: 20}),
		ScoreRecord(ScoreRow{ObjectiveID: "dropped", Value: 0.1, Method: "seed-judge", Version: "1", Witness: W1, UnixMillis: 20}),
	}
	for _, r := range rows {
		if err := Validate(r); err != nil {
			t.Fatalf("corpus row invalid: %v", err)
		}
	}
	return Fold(rows)
}

// TestBacktestQualifiesATrackingScorerAndRefusesTheKnownBadOne is the done-condition
// witness, run as ONE test on ONE corpus so neither half can be explained away: the same
// recorded history that qualifies an honest scorer must refuse the inverted one. If the gate
// were vacuous — always QUALIFIED, or always refusing — one of these two assertions fails.
func TestBacktestQualifiesATrackingScorerAndRefusesTheKnownBadOne(t *testing.T) {
	corpus := backtestCorpus(t)

	good := corpus.Backtest(trackingFixtureScorer{}, nil)
	if good.Verdict != BacktestQualified {
		t.Fatalf("tracking scorer verdict = %s (%s), want %s: %s", good.Verdict, good.Reason, BacktestQualified, good.Detail)
	}
	if good.Reason != BacktestTracksOutcome {
		t.Errorf("tracking scorer reason = %s, want %s", good.Reason, BacktestTracksOutcome)
	}
	if !good.Qualified() {
		t.Error("Qualified() must be true for a QUALIFIED verdict")
	}
	if good.Schema != BacktestSchema {
		t.Errorf("schema = %q, want %q", good.Schema, BacktestSchema)
	}
	if good.Outcomes != 3 || good.Objectives != 3 {
		t.Errorf("corpus counts = %d objectives / %d outcomes, want 3/3", good.Objectives, good.Outcomes)
	}
	if good.Candidate.Coefficient <= 0 || !good.Candidate.Measured {
		t.Errorf("tracking candidate should measure a positive coefficient, got %+v", good.Candidate)
	}

	bad := corpus.Backtest(invertedFixtureScorer{}, nil)
	if bad.Verdict != BacktestNotQualified {
		t.Fatalf("known-bad scorer verdict = %s (%s), want %s: %s", bad.Verdict, bad.Reason, BacktestNotQualified, bad.Detail)
	}
	if bad.Reason != BacktestAntiCorrelated {
		t.Errorf("known-bad scorer reason = %s, want %s", bad.Reason, BacktestAntiCorrelated)
	}
	if bad.Qualified() {
		t.Error("a known-bad scorer must never report Qualified()")
	}
	if bad.Candidate.Coefficient >= 0 {
		t.Errorf("inverted candidate should anti-correlate, got r=%+.2f", bad.Candidate.Coefficient)
	}
	// The report is a CALIBRATION report, not a bare verdict: it must carry the per-objective
	// replay so an operator can see WHERE the scorer disagreed with history.
	if len(bad.Cases) != 3 {
		t.Fatalf("want one case per replayed objective, got %d", len(bad.Cases))
	}
	for _, c := range bad.Cases {
		if !c.Scored || c.Rows != 1 {
			t.Errorf("case %s should carry the one replayed row, got %+v", c.ObjectiveID, c)
		}
	}
}

// backtestSpreadCorpus builds a wider recorded corpus — eight objectives whose witnessed
// outcomes are spread evenly across the unit interval — so a replayed scorer's coefficient
// can land anywhere in the band rather than being pinned to +-1 by three coarse points.
func backtestSpreadCorpus(t *testing.T) State {
	t.Helper()
	rows := make([]Row, 0, 24)
	for i := 0; i < 8; i++ {
		id := fmt.Sprintf("obj%d", i)
		value := float64(i) / 7
		rows = append(rows,
			ObjectiveRecord(Objective{ID: id, Statement: "objective " + id, Status: StatusMet}),
			ScoreRecord(ScoreRow{ObjectiveID: id, Value: value, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 10}),
			ScoreRecord(ScoreRow{ObjectiveID: id, Value: value, Method: "seed-judge", Version: "1", Witness: W1, UnixMillis: 20}),
		)
	}
	for _, r := range rows {
		if err := Validate(r); err != nil {
			t.Fatalf("corpus row invalid: %v", err)
		}
	}
	return Fold(rows)
}

// TestBacktestWeakCorrelationIsRefusedApartFromAntiCorrelation pins the middle band: a
// scorer that leans the right way but not hard enough is refused with its OWN reason, so
// "not good enough yet" is never reported as "steers backwards".
func TestBacktestWeakCorrelationIsRefusedApartFromAntiCorrelation(t *testing.T) {
	rep := backtestSpreadCorpus(t).Backtest(partialFixtureScorer{method: "fixture-weak", modulus: 3}, nil)
	if rep.Verdict != BacktestNotQualified || rep.Reason != BacktestWeakCorrelation {
		t.Fatalf("weak scorer = %s/%s, want %s/%s (r=%+.2f): %s",
			rep.Verdict, rep.Reason, BacktestNotQualified, BacktestWeakCorrelation, rep.Candidate.Coefficient, rep.Detail)
	}
	// The band, not the exact number: it must lean the right way yet miss the bar, which is
	// what separates this refusal from the anti-correlated one above it.
	if rep.Candidate.Coefficient < 0 || rep.Candidate.Coefficient >= calibrationWellThreshold {
		t.Errorf("the weak control must sit in [0, %.2f), got r=%+.2f", calibrationWellThreshold, rep.Candidate.Coefficient)
	}
}

// TestBacktestThinCorpusIsInconclusiveNotAPass is the anti-false-green guard: a corpus that
// cannot decide must never read as qualification. Both shapes are covered — a scorer with no
// variance in its readings, and a corpus with only one objective.
func TestBacktestThinCorpusIsInconclusiveNotAPass(t *testing.T) {
	flat := backtestCorpus(t).Backtest(flatFixtureScorer{}, nil)
	if flat.Verdict != BacktestInconclusive || flat.Reason != BacktestThinCorpus {
		t.Fatalf("flat scorer = %s/%s, want %s/%s: %s", flat.Verdict, flat.Reason, BacktestInconclusive, BacktestThinCorpus, flat.Detail)
	}
	if flat.Qualified() {
		t.Error("INCONCLUSIVE must never report Qualified() — an undecided corpus has not endorsed the scorer")
	}

	single := Fold([]Row{
		ObjectiveRecord(Objective{ID: "only", Statement: "the only objective", Status: StatusMet}),
		ScoreRecord(ScoreRow{ObjectiveID: "only", Value: 1.0, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 1}),
		ScoreRecord(ScoreRow{ObjectiveID: "only", Value: 0.9, Method: "seed-judge", Version: "1", Witness: W1, UnixMillis: 2}),
	})
	one := single.Backtest(trackingFixtureScorer{}, nil)
	if one.Verdict != BacktestInconclusive || one.Reason != BacktestThinCorpus {
		t.Fatalf("single-objective corpus = %s/%s, want %s/%s: %s", one.Verdict, one.Reason, BacktestInconclusive, BacktestThinCorpus, one.Detail)
	}
}

// TestBacktestToolErrorsAreTypedApartFromAFailedQualification is the confusion risk the
// issue names, pinned: a backtest that could not RUN reports BACKTEST_ERROR with a cause,
// never NOT_QUALIFIED and never a silent skip that downstream reads as a pass.
func TestBacktestToolErrorsAreTypedApartFromAFailedQualification(t *testing.T) {
	missing := backtestCorpus(t).Backtest(nil, nil)
	if missing.Verdict != BacktestErrored || missing.Reason != BacktestNoCandidate {
		t.Fatalf("nil candidate = %s/%s, want %s/%s", missing.Verdict, missing.Reason, BacktestErrored, BacktestNoCandidate)
	}
	if missing.Qualified() {
		t.Error("a tool error must never report Qualified()")
	}
	if missing.Schema != BacktestSchema || missing.Detail == "" {
		t.Errorf("a refusal must still carry the pinned schema and a cause, got %+v", missing)
	}

	// A corpus with objectives but no witnessed W3 outcome cannot grade anything. That is a
	// property of the CORPUS, so it is a tool error, not a verdict on the scorer.
	noTruth := Fold([]Row{
		ObjectiveRecord(Objective{ID: "a", Statement: "a", Status: StatusActive}),
		ObjectiveRecord(Objective{ID: "b", Statement: "b", Status: StatusActive}),
		ScoreRecord(ScoreRow{ObjectiveID: "a", Value: 0.9, Method: "seed-judge", Version: "1", Witness: W1, UnixMillis: 1}),
		ScoreRecord(ScoreRow{ObjectiveID: "b", Value: 0.1, Method: "seed-judge", Version: "1", Witness: W1, UnixMillis: 1}),
	}).Backtest(trackingFixtureScorer{}, nil)
	if noTruth.Verdict != BacktestErrored || noTruth.Reason != BacktestNoRecordedOutcome {
		t.Fatalf("outcome-free corpus = %s/%s, want %s/%s: %s", noTruth.Verdict, noTruth.Reason, BacktestErrored, BacktestNoRecordedOutcome, noTruth.Detail)
	}
	if noTruth.Objectives != 2 {
		t.Errorf("a refusal should still report what it read, got %d objectives", noTruth.Objectives)
	}

	// An empty corpus is likewise a tool fault and not a qualification.
	empty := Fold(nil).Backtest(trackingFixtureScorer{}, nil)
	if empty.Verdict != BacktestErrored || empty.Reason != BacktestNoRecordedOutcome {
		t.Fatalf("empty corpus = %s/%s, want %s/%s", empty.Verdict, empty.Reason, BacktestErrored, BacktestNoRecordedOutcome)
	}
}

// TestBacktestRefusesARegressionAgainstTheIncumbent pins the second half of the gate: a
// candidate can track the outcome well enough to pass on its own and still be refused for
// reading history worse than the method it would replace.
func TestBacktestRefusesARegressionAgainstTheIncumbent(t *testing.T) {
	corpus := backtestSpreadCorpus(t)

	// A candidate that clears the well-calibrated bar on its own, so the refusal below can
	// only come from the incumbent comparison and not from a weak reading.
	candidate := partialFixtureScorer{method: "fixture-good-enough", modulus: 4}
	solo := corpus.Backtest(candidate, nil)
	if solo.Verdict != BacktestQualified {
		t.Fatalf("the regression control must qualify on its own (r=%+.2f), got %s/%s: %s",
			solo.Candidate.Coefficient, solo.Verdict, solo.Reason, solo.Detail)
	}

	rep := corpus.Backtest(candidate, trackingFixtureScorer{})
	if rep.Incumbent == nil {
		t.Fatal("an incumbent was supplied, so the report must carry its calibration")
	}
	if rep.Verdict != BacktestNotQualified || rep.Reason != BacktestRegressesIncumbent {
		t.Fatalf("candidate = %s/%s, want %s/%s: %s", rep.Verdict, rep.Reason, BacktestNotQualified, BacktestRegressesIncumbent, rep.Detail)
	}
	if rep.Delta >= 0 {
		t.Errorf("delta should be negative when the candidate trails the incumbent, got %+.2f", rep.Delta)
	}

	// And the mirror: the same scorer as both candidate and incumbent ties, so the
	// regression check must not refuse a candidate that merely equals the incumbent.
	tie := corpus.Backtest(trackingFixtureScorer{}, trackingFixtureScorer{})
	if tie.Verdict != BacktestQualified {
		t.Fatalf("a tie against the incumbent must qualify, got %s/%s: %s", tie.Verdict, tie.Reason, tie.Detail)
	}
	if tie.Delta != 0 {
		t.Errorf("a tie should read delta 0, got %+.2f", tie.Delta)
	}
}

// TestBacktestReplaysRecordedEvidenceThroughTheRealScorer proves the replay window is
// genuinely rebuilt from the corpus rather than fabricated: the SHIPPED commit scorer, run
// offline over recorded commit evidence, reproduces the per-phase fractions the recording
// implies — and credits nothing for a phase whose commit the corpus never witnessed.
func TestBacktestReplaysRecordedEvidenceThroughTheRealScorer(t *testing.T) {
	plan := []PlanPhase{{ID: "p1", Title: "one"}, {ID: "p2", Title: "two"}}
	corpus := Fold([]Row{
		ObjectiveRecord(Objective{ID: "both", Statement: "both phases witnessed", Plan: plan, Status: StatusMet}),
		ObjectiveRecord(Objective{ID: "half", Statement: "one phase witnessed", Plan: plan, Status: StatusActive}),
		ObjectiveRecord(Objective{ID: "none", Statement: "no phase witnessed", Plan: plan, Status: StatusAbandoned}),
		ScoreRecord(ScoreRow{ObjectiveID: "both", Value: 1.0, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 5,
			Evidence: []EvidenceRef{{Kind: "commit", Ref: "aaa1", Detail: "p1"}, {Kind: "commit", Ref: "aaa2", Detail: "p2"}}}),
		ScoreRecord(ScoreRow{ObjectiveID: "half", Value: 0.5, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 5,
			Evidence: []EvidenceRef{{Kind: "commit", Ref: "bbb1", Detail: "p1"}}}),
		ScoreRecord(ScoreRow{ObjectiveID: "none", Value: 0.0, Method: CommitScorerMethod, Version: "1", Witness: W3, UnixMillis: 5}),
	})

	rep := corpus.Backtest(CommitProgressScorer{}, nil)
	if rep.Verdict != BacktestQualified {
		t.Fatalf("replaying the commit scorer over its own recorded evidence should qualify, got %s/%s: %s", rep.Verdict, rep.Reason, rep.Detail)
	}
	want := map[string]float64{"both": 1.0, "half": 0.5, "none": 0.0}
	for _, c := range rep.Cases {
		if got := want[c.ObjectiveID]; c.Candidate != got {
			t.Errorf("replayed %s = %.2f, want %.2f (the recorded evidence implies it)", c.ObjectiveID, c.Candidate, got)
		}
		if !c.Scored {
			t.Errorf("case %s should be scored", c.ObjectiveID)
		}
	}

	// The resolver is fail-closed: a phase whose commit the corpus never recorded must not
	// be credited, even when the objective declares the phase.
	win := corpus.replayWindow("none")
	if len(win.PhaseCommits) != 0 {
		t.Errorf("an objective with no recorded commit evidence must replay with no phase bindings, got %+v", win.PhaseCommits)
	}
	if st := win.Resolve(EvidenceRef{Kind: "commit", Ref: "never-recorded"}); st != EvidenceUnknown {
		t.Errorf("an unrecorded pointer must resolve unknown, got %s", st)
	}
	if st := corpus.replayWindow("half").Resolve(EvidenceRef{Kind: "commit", Ref: "bbb1"}); st != EvidenceVerified {
		t.Errorf("a recorded pointer must resolve verified, got %s", st)
	}
}

// TestBacktestScorerWithNoRowsIsVisibleNotDropped guards the silent-skip failure mode at the
// case level: a scorer that declines an objective leaves a legible unscored case behind
// instead of quietly shrinking the sample the verdict rests on.
func TestBacktestScorerWithNoRowsIsVisibleNotDropped(t *testing.T) {
	// The activity scorer needs sessions, which a ledger corpus never records, so it emits
	// nothing — the shape the doc comment predicts.
	rep := backtestCorpus(t).Backtest(ActivityDivergenceScorer{}, nil)
	if rep.Verdict != BacktestInconclusive || rep.Reason != BacktestThinCorpus {
		t.Fatalf("a session-fed scorer on a ledger corpus = %s/%s, want %s/%s: %s",
			rep.Verdict, rep.Reason, BacktestInconclusive, BacktestThinCorpus, rep.Detail)
	}
	if len(rep.Cases) != 3 {
		t.Fatalf("every replayed objective must leave a case, got %d", len(rep.Cases))
	}
	for _, c := range rep.Cases {
		if c.Scored || c.Rows != 0 {
			t.Errorf("case %s should record that the scorer declined it, got %+v", c.ObjectiveID, c)
		}
	}
	if rep.Candidate.Samples != 0 {
		t.Errorf("no rows means no samples, got %d", rep.Candidate.Samples)
	}
}

// TestBacktestRefusalCarriesTheClosedVocabulary covers the shell's refusal constructor: a
// backtest that never reached the corpus still emits a schema-pinned report a caller can
// branch on.
func TestBacktestRefusalCarriesTheClosedVocabulary(t *testing.T) {
	rep := BacktestRefusal("no-such-method", BacktestUnknownScorer, "no scorer registered under that name")
	if rep.Schema != BacktestSchema || rep.Verdict != BacktestErrored || rep.Reason != BacktestUnknownScorer {
		t.Fatalf("refusal = %+v", rep)
	}
	if rep.Qualified() {
		t.Error("a refusal must never report Qualified()")
	}
	if rep.GroundTruth != GroundTruthMethod {
		t.Errorf("ground truth = %q, want %q", rep.GroundTruth, GroundTruthMethod)
	}
}
