package workerenvelope

import (
	"strings"
	"testing"
)

// TestParseValidFixture proves a well-formed shipped envelope round-trips
// through Parse (decode + validate) and preserves every field.
func TestParseValidFixture(t *testing.T) {
	const fixture = `{
		"status": "shipped",
		"issue": 1795,
		"commit_sha": "c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90",
		"tests_run": ["go test ./internal/workerenvelope/"],
		"witness": "commit c99f5c02"
	}`

	r, err := Parse([]byte(fixture))
	if err != nil {
		t.Fatalf("Parse of valid fixture failed: %v", err)
	}
	if r.Status != StatusShipped {
		t.Errorf("status = %q, want %q", r.Status, StatusShipped)
	}
	if r.Issue != 1795 {
		t.Errorf("issue = %d, want 1795", r.Issue)
	}
	if r.CommitSHA != "c99f5c02a1b2c3d4e5f60718293a4b5c6d7e8f90" {
		t.Errorf("commit_sha = %q, unexpected", r.CommitSHA)
	}
	if len(r.TestsRun) != 1 || r.TestsRun[0] != "go test ./internal/workerenvelope/" {
		t.Errorf("tests_run = %v, unexpected", r.TestsRun)
	}
	if r.Witness != "commit c99f5c02" {
		t.Errorf("witness = %q, unexpected", r.Witness)
	}
	if r.Blocker != "" {
		t.Errorf("blocker = %q, want empty", r.Blocker)
	}
}

// TestParseShortSHA proves a 7-char short SHA is accepted on a shipped result.
func TestParseShortSHA(t *testing.T) {
	const fixture = `{"status":"shipped","issue":42,"commit_sha":"c99f5c0","witness":"log: run.log"}`
	if _, err := Parse([]byte(fixture)); err != nil {
		t.Fatalf("short-SHA shipped fixture should validate, got: %v", err)
	}
}

// TestValidateBlockedAndNotYet proves the two non-shipped statuses validate
// when they name a blocker.
func TestValidateBlockedAndNotYet(t *testing.T) {
	for _, st := range []Status{StatusBlocked, StatusNotYet} {
		r := Result{Status: st, Issue: 7, Blocker: "peer WIP breaks internal/model build"}
		if err := r.Validate(); err != nil {
			t.Errorf("%s result with a blocker should validate, got: %v", st, err)
		}
	}
}

// TestValidateMalformed drives the malformed fixtures: each must fail, and the
// error must name the field that broke the contract.
func TestValidateMalformed(t *testing.T) {
	cases := []struct {
		name    string
		r       Result
		wantSub string // substring the error must contain
	}{
		{
			name:    "shipped missing witness",
			r:       Result{Status: StatusShipped, Issue: 1, CommitSHA: "c99f5c02"},
			wantSub: "witness",
		},
		{
			name:    "shipped missing commit_sha",
			r:       Result{Status: StatusShipped, Issue: 1, Witness: "log: run.log"},
			wantSub: "commit_sha",
		},
		{
			name:    "shipped carries a blocker",
			r:       Result{Status: StatusShipped, Issue: 1, CommitSHA: "c99f5c02", Witness: "commit c99f5c02", Blocker: "flaky"},
			wantSub: "must not carry a blocker",
		},
		{
			name:    "blocked missing blocker",
			r:       Result{Status: StatusBlocked, Issue: 1},
			wantSub: "requires a blocker",
		},
		{
			name:    "not_yet missing blocker",
			r:       Result{Status: StatusNotYet, Issue: 1},
			wantSub: "requires a blocker",
		},
		{
			name:    "issue <= 0",
			r:       Result{Status: StatusBlocked, Issue: 0, Blocker: "x"},
			wantSub: "issue must be > 0",
		},
		{
			name:    "negative issue",
			r:       Result{Status: StatusBlocked, Issue: -5, Blocker: "x"},
			wantSub: "issue must be > 0",
		},
		{
			name:    "bad sha shape (too short)",
			r:       Result{Status: StatusShipped, Issue: 1, CommitSHA: "abc12", Witness: "w"},
			wantSub: "hex sha",
		},
		{
			name:    "bad sha shape (non-hex)",
			r:       Result{Status: StatusBlocked, Issue: 1, CommitSHA: "zzzzzzz", Blocker: "x"},
			wantSub: "hex sha",
		},
		{
			name:    "unknown status",
			r:       Result{Status: Status("done"), Issue: 1},
			wantSub: "invalid status",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.r.Validate()
			if err == nil {
				t.Fatalf("expected validation error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q does not mention %q", err.Error(), tc.wantSub)
			}
		})
	}
}

// TestParseRejectsMalformedJSON proves Parse surfaces a decode error (distinct
// from a validation error) for syntactically broken JSON.
func TestParseRejectsMalformedJSON(t *testing.T) {
	_, err := Parse([]byte(`{"status": "shipped", "issue":`))
	if err == nil {
		t.Fatal("expected decode error for truncated JSON, got nil")
	}
	if !strings.Contains(err.Error(), "decode") {
		t.Errorf("error %q should mention decode", err.Error())
	}
}

// TestParseMalformedFixtureFails is the ticket's explicit witness: a malformed
// (well-formed JSON, contract-violating) fixture parses through JSON but fails
// Validate.
func TestParseMalformedFixtureFails(t *testing.T) {
	const fixture = `{"status":"shipped","issue":1795,"commit_sha":"c99f5c02"}` // no witness
	if _, err := Parse([]byte(fixture)); err == nil {
		t.Fatal("malformed fixture (shipped without witness) should fail Parse")
	}
}
