package trajctl

import (
	"encoding/json"
	"errors"
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
