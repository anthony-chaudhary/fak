package trajctl

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// cannedJudge is a JudgeClient with a scripted verdict/usage/error, recording
// the request it was handed so a test can assert the budget cap traveled into
// the call.
type cannedJudge struct {
	verdict JudgeVerdict
	usage   JudgeUsage
	err     error
	seen    JudgeRequest
	calls   int
}

func (c *cannedJudge) Judge(req JudgeRequest) (JudgeVerdict, JudgeUsage, error) {
	c.calls++
	c.seen = req
	return c.verdict, c.usage, c.err
}

func docsObjective() Objective {
	return Objective{
		ID:        "docs-migrate",
		Statement: "Migrate the observability docs to the new layout and cross-link them.",
		Status:    StatusActive,
	}
}

// TestJudgeScorerEmitsW1Row is the done-condition witness at the fold level: a
// docs-shaped objective plus a canned verdict yields exactly one W1 row whose
// value is the clamped progress and whose evidence carries the verdict blob.
func TestJudgeScorerEmitsW1Row(t *testing.T) {
	client := &cannedJudge{
		verdict: JudgeVerdict{Progress: 0.4, Met: false, Rationale: "two of five doc pages moved"},
		usage:   JudgeUsage{Tokens: 120},
	}
	s := NewJudgeScorer(client, 512)
	rows := s.Score(docsObjective(), EvidenceWindow{UnixMillis: 7})
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	row := rows[0]
	if row.Witness != W1 {
		t.Errorf("witness = %q, want W1 (must never outrank W2/W3)", row.Witness)
	}
	if row.Value != 0.4 {
		t.Errorf("value = %v, want 0.4", row.Value)
	}
	if row.Method != JudgeScorerMethod || row.Version != JudgeScorerVersion {
		t.Errorf("method/version = %q/%q", row.Method, row.Version)
	}
	if row.UnixMillis != 7 {
		t.Errorf("unix_millis = %d, want 7 (window stamp propagated)", row.UnixMillis)
	}
	if len(row.Evidence) != 1 || row.Evidence[0].Kind != "judge-verdict" {
		t.Fatalf("evidence = %+v, want one judge-verdict ref", row.Evidence)
	}
	var back JudgeVerdict
	if err := json.Unmarshal([]byte(row.Evidence[0].Detail), &back); err != nil {
		t.Fatalf("evidence detail is not the verdict blob: %v", err)
	}
	if back.Rationale != "two of five doc pages moved" {
		t.Errorf("verdict blob lost the rationale: %+v", back)
	}
	// The whole row must survive ledger validation so it can be appended.
	if err := validateScore(row); err != nil {
		t.Errorf("emitted row fails validation: %v", err)
	}
}

// TestJudgeScorerBudgetCapTravelsIntoRequest proves the per-call cap is the
// request's output ceiling — the source-side half of the budget enforcement.
func TestJudgeScorerBudgetCapTravelsIntoRequest(t *testing.T) {
	client := &cannedJudge{verdict: JudgeVerdict{Progress: 0.2}, usage: JudgeUsage{Tokens: 50}}
	s := NewJudgeScorer(client, 256)
	_ = s.Score(docsObjective(), EvidenceWindow{})
	if client.seen.MaxTokens != 256 {
		t.Fatalf("request MaxTokens = %d, want the 256 cap forwarded", client.seen.MaxTokens)
	}
}

// TestJudgeScorerRejectsOverBudgetReturn is the budget-cap witness the issue
// names: a call whose reported spend exceeds the cap earns NO row, so a runaway
// judge call cannot credit progress or hide its cost.
func TestJudgeScorerRejectsOverBudgetReturn(t *testing.T) {
	client := &cannedJudge{
		verdict: JudgeVerdict{Progress: 0.9, Met: true},
		usage:   JudgeUsage{Tokens: 4096}, // blew past the cap
	}
	s := JudgeScorer{Client: client, MaxCallTokens: 512}
	if rows := s.Score(docsObjective(), EvidenceWindow{}); rows != nil {
		t.Fatalf("over-budget call must yield no row, got %+v", rows)
	}
}

// TestJudgeScorerFailsClosed covers the no-spend boundaries: a nil client, a
// non-positive cap, a client error, and a closed objective all yield no row —
// and the nil-client / closed-objective paths must not even call the client.
func TestJudgeScorerFailsClosed(t *testing.T) {
	if rows := (JudgeScorer{Client: nil, MaxCallTokens: 512}).Score(docsObjective(), EvidenceWindow{}); rows != nil {
		t.Errorf("nil client must yield no row, got %+v", rows)
	}
	client := &cannedJudge{verdict: JudgeVerdict{Progress: 1}, usage: JudgeUsage{Tokens: 10}}
	if rows := (JudgeScorer{Client: client, MaxCallTokens: 0}).Score(docsObjective(), EvidenceWindow{}); rows != nil {
		t.Errorf("non-positive cap must yield no row, got %+v", rows)
	}
	if client.calls != 0 {
		t.Errorf("a no-budget scorer must not call the client, got %d calls", client.calls)
	}
	errClient := &cannedJudge{err: errors.New("upstream 500")}
	if rows := (JudgeScorer{Client: errClient, MaxCallTokens: 512}).Score(docsObjective(), EvidenceWindow{}); rows != nil {
		t.Errorf("client error must yield no row, got %+v", rows)
	}
	closed := docsObjective()
	closed.Status = StatusMet
	metClient := &cannedJudge{verdict: JudgeVerdict{Progress: 1}, usage: JudgeUsage{Tokens: 10}}
	if rows := (JudgeScorer{Client: metClient, MaxCallTokens: 512}).Score(closed, EvidenceWindow{}); rows != nil {
		t.Errorf("closed objective must yield no row, got %+v", rows)
	}
	if metClient.calls != 0 {
		t.Errorf("a closed objective must not call the client, got %d calls", metClient.calls)
	}
}

// TestJudgeScorerClampsProgress guards the [0,1] invariant the ledger requires:
// a model that returns an out-of-range progress is clamped, not rejected.
func TestJudgeScorerClampsProgress(t *testing.T) {
	client := &cannedJudge{verdict: JudgeVerdict{Progress: 1.7}, usage: JudgeUsage{Tokens: 10}}
	rows := NewJudgeScorer(client, 512).Score(docsObjective(), EvidenceWindow{})
	if len(rows) != 1 || rows[0].Value != 1 {
		t.Fatalf("progress 1.7 must clamp to value 1, got %+v", rows)
	}
}

// conjunctiveObjective is docsObjective plus a two-criterion rubric marked
// conjunctive, the fixture the #3926 all-pass-AND tests score against.
func conjunctiveObjective() Objective {
	obj := docsObjective()
	obj.Rubric = &Rubric{
		Source:      "test-model",
		Conjunctive: true,
		Criteria: []RubricCriterion{
			{ID: "c1", Text: "The migration landed (load-bearing)."},
			{ID: "c2", Text: "Cross-links resolve."},
		},
	}
	return obj
}

// rubricConjunctiveRef returns the single rubric-conjunctive summary ref the
// fold emits in conjunctive mode, or false if none is present.
func rubricConjunctiveRef(rows []ScoreRow) (EvidenceRef, bool) {
	if len(rows) != 1 {
		return EvidenceRef{}, false
	}
	for _, ev := range rows[0].Evidence {
		if ev.Kind == "rubric-conjunctive" {
			return ev, true
		}
	}
	return EvidenceRef{}, false
}

// TestJudgeScorerRejectsScalarAboveCriterionSupport captures #5951's failure
// class: the judge cannot claim complete progress while its own itemization says
// that the rubric is only half complete. The fold fails closed with no W1 row.
func TestJudgeScorerRejectsScalarAboveCriterionSupport(t *testing.T) {
	obj := conjunctiveObjective()
	obj.Rubric.Conjunctive = false
	client := &cannedJudge{
		verdict: JudgeVerdict{
			Progress:  1,
			Met:       true,
			Rationale: "complete",
			Criteria: []RubricFinding{
				{ID: "c1", Progress: 1},
				{ID: "c2", Progress: 0},
			},
		},
		usage: JudgeUsage{Tokens: 80},
	}

	if rows := NewJudgeScorer(client, 512).Score(obj, EvidenceWindow{}); len(rows) != 0 {
		t.Fatalf("contradictory scalar produced rows: %+v", rows)
	}
}

// TestJudgeScorerAcceptsScalarAtCriterionSupport pins the non-regression side:
// an itemized soft verdict whose scalar does not exceed its findings still
// yields the ordinary W1 row.
func TestJudgeScorerAcceptsScalarAtCriterionSupport(t *testing.T) {
	obj := conjunctiveObjective()
	obj.Rubric.Conjunctive = false
	client := &cannedJudge{
		verdict: JudgeVerdict{
			Progress:  0.5,
			Rationale: "one of two complete",
			Criteria: []RubricFinding{
				{ID: "c1", Progress: 1},
				{ID: "c2", Progress: 0},
			},
		},
		usage: JudgeUsage{Tokens: 80},
	}

	rows := NewJudgeScorer(client, 512).Score(obj, EvidenceWindow{})
	if len(rows) != 1 || rows[0].Value != 0.5 {
		t.Fatalf("consistent itemized verdict = %+v, want one row at 0.5", rows)
	}
}

// TestJudgeScorerConjunctiveAllPass is the #3926 done-condition's happy half:
// every criterion at the pass threshold makes the W1 value 1.0 — and it is the
// hard AND, not the soft progress, that decides it (the verdict's soft progress
// is low but the value is a full 1.0 because all criteria passed).
func TestJudgeScorerConjunctiveAllPass(t *testing.T) {
	client := &cannedJudge{
		verdict: JudgeVerdict{
			Progress:  0.3, // soft fold would credit only 0.3
			Rationale: "both criteria met",
			Criteria: []RubricFinding{
				{ID: "c1", Progress: 1, Note: "migration landed"},
				{ID: "c2", Progress: 1, Note: "links resolve"},
			},
		},
		usage: JudgeUsage{Tokens: 90},
	}
	rows := NewJudgeScorer(client, 512).Score(conjunctiveObjective(), EvidenceWindow{UnixMillis: 5})
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	if rows[0].Value != 1 {
		t.Errorf("all criteria pass ⇒ value 1.0, got %v (soft progress must not leak in)", rows[0].Value)
	}
	if rows[0].Witness != W1 {
		t.Errorf("conjunctive fold must stay W1, got %q", rows[0].Witness)
	}
	ref, ok := rubricConjunctiveRef(rows)
	if !ok || ref.Ref != "all-pass" {
		t.Errorf("all-pass must be cited by a rubric-conjunctive ref, got %+v", rows[0].Evidence)
	}
	// Per-criterion attribution is preserved alongside the AND summary.
	n := 0
	for _, ev := range rows[0].Evidence {
		if ev.Kind == "rubric-criterion" {
			n++
		}
	}
	if n != 2 {
		t.Errorf("per-criterion refs must be preserved, got %d", n)
	}
	if err := validateScore(rows[0]); err != nil {
		t.Errorf("conjunctive row fails ledger validation: %v", err)
	}
}

// TestJudgeScorerConjunctiveOneFails is the issue's core motivation: an
// objective the soft fold would score "80% done" scores 0.0 under the AND when
// a single load-bearing criterion is unmet, and the failing criterion is named
// in evidence so the failure is attributable.
func TestJudgeScorerConjunctiveOneFails(t *testing.T) {
	client := &cannedJudge{
		verdict: JudgeVerdict{
			Progress:  0.8, // soft fold would read a misleading 80%
			Rationale: "migration done but links broken",
			Criteria: []RubricFinding{
				{ID: "c1", Progress: 1, Note: "migration landed"},
				{ID: "c2", Progress: 0.5, Note: "half the links still dangle"},
			},
		},
		usage: JudgeUsage{Tokens: 110},
	}
	rows := NewJudgeScorer(client, 512).Score(conjunctiveObjective(), EvidenceWindow{})
	if len(rows) != 1 || rows[0].Value != 0 {
		t.Fatalf("one unmet criterion ⇒ value 0.0, got %+v", rows)
	}
	ref, ok := rubricConjunctiveRef(rows)
	if !ok || ref.Ref != "fail" || !strings.Contains(ref.Detail, "c2") {
		t.Errorf("failing criterion c2 must be cited, got %+v", ref)
	}
	if err := validateScore(rows[0]); err != nil {
		t.Errorf("failed-AND row fails ledger validation: %v", err)
	}
}

// TestJudgeScorerConjunctiveMissingFinding proves the AND fails closed: a
// criterion the verdict simply omitted cannot be assumed passing — the objective
// scores 0.0 and the missing criterion is named.
func TestJudgeScorerConjunctiveMissingFinding(t *testing.T) {
	client := &cannedJudge{
		verdict: JudgeVerdict{
			Progress: 1,                                        // model claims fully done…
			Criteria: []RubricFinding{{ID: "c1", Progress: 1}}, // …but only reported c1
		},
		usage: JudgeUsage{Tokens: 40},
	}
	rows := NewJudgeScorer(client, 512).Score(conjunctiveObjective(), EvidenceWindow{})
	if len(rows) != 1 || rows[0].Value != 0 {
		t.Fatalf("a missing criterion finding ⇒ value 0.0 (fail-closed), got %+v", rows)
	}
	ref, ok := rubricConjunctiveRef(rows)
	if !ok || ref.Ref != "fail" || !strings.Contains(ref.Detail, "c2") {
		t.Errorf("missing criterion c2 must be named, got %+v", ref)
	}
}

// TestJudgeScorerConjunctiveOptIn guards the out-of-scope boundary: a rubric
// left in the default (non-conjunctive) mode keeps the soft-progress fold and
// emits NO rubric-conjunctive ref, so the opt-in never disturbs existing rows.
func TestJudgeScorerConjunctiveOptIn(t *testing.T) {
	obj := conjunctiveObjective()
	obj.Rubric.Conjunctive = false
	client := &cannedJudge{
		verdict: JudgeVerdict{
			Progress: 0.5,
			Criteria: []RubricFinding{{ID: "c1", Progress: 0.5}, {ID: "c2", Progress: 0.5}},
		},
		usage: JudgeUsage{Tokens: 60},
	}
	rows := NewJudgeScorer(client, 512).Score(obj, EvidenceWindow{})
	if len(rows) != 1 || rows[0].Value != 0.5 {
		t.Fatalf("soft mode must keep clamp01(progress)=0.5, got %+v", rows)
	}
	if _, ok := rubricConjunctiveRef(rows); ok {
		t.Errorf("soft mode must emit no rubric-conjunctive ref, got %+v", rows[0].Evidence)
	}
}

// TestConjunctiveValueFailsClosed pins the helper's nil/empty boundary: with no
// rubric to AND over there is nothing to prove, so it credits 0.0.
func TestConjunctiveValueFailsClosed(t *testing.T) {
	if v, _ := conjunctiveValue(nil, []RubricFinding{{ID: "c1", Progress: 1}}); v != 0 {
		t.Errorf("nil rubric must fail closed to 0.0, got %v", v)
	}
	if v, _ := conjunctiveValue(&Rubric{}, []RubricFinding{{ID: "c1", Progress: 1}}); v != 0 {
		t.Errorf("empty rubric must fail closed to 0.0, got %v", v)
	}
}

// TestJudgeScorerRegistersHonestlyBelowW3 pins the ranking guarantee: the judge
// scorer registers alongside the deterministic scorers and its rung is W1, so a
// consumer that also has the W3 commit scorer's row always has the stronger one.
func TestJudgeScorerRegistersHonestlyBelowW3(t *testing.T) {
	reg := NewRegistry()
	if err := reg.Register(CommitProgressScorer{}); err != nil {
		t.Fatalf("register commit scorer: %v", err)
	}
	if err := reg.Register(JudgeScorer{}); err != nil {
		t.Fatalf("register judge scorer: %v", err)
	}
	js, ok := reg.Get(JudgeScorerMethod)
	if !ok {
		t.Fatalf("judge scorer did not register under %q", JudgeScorerMethod)
	}
	if got := js.(JudgeScorer).Method(); got != JudgeScorerMethod {
		t.Fatalf("registered method = %q", got)
	}
	// Confirm the emitted rung is strictly W1 (not W2/W3) via a live fold.
	client := &cannedJudge{verdict: JudgeVerdict{Progress: 0.5}, usage: JudgeUsage{Tokens: 10}}
	rows := (JudgeScorer{Client: client, MaxCallTokens: 512}).Score(docsObjective(), EvidenceWindow{})
	if len(rows) != 1 || rows[0].Witness != W1 {
		t.Fatalf("judge rung must be W1, got %+v", rows)
	}
}
