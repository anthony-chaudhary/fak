package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// TestGuardAccuracyComplaintsIntakeIsAdvisory drives the `--complaints` seam
// end-to-end (#2821): a JSON file of agent-authored over-block appeals folds into
// the scorecard as a machine-readable field-FP intake, yet -- being a cheap-to-file
// self-report -- never adds debt or reds the gate. This pins the wiring and the
// anti-gaming contract from the CLI, deterministically and without any network.
func TestGuardAccuracyComplaintsIntakeIsAdvisory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "complaints.json")
	body := `{"complaints":[
		{"kind":"false-positive","summary":"grep of git push in docs escalated","occurrences":4},
		{"kind":"over-broad","summary":"commit message mentioning rm escalated","occurrences":2},
		{"kind":"latency","summary":"gate felt slow","occurrences":9}
	]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runGuardAccuracy(&stdout, &stderr, []string{"--workspace", "/repo", "--complaints", path, "--json"})
	if code != 0 {
		t.Fatalf("runGuardAccuracy exit=%d stderr=%s", code, stderr.String())
	}

	var payload scorecard.Payload
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("parse payload: %v\n%s", err, stdout.String())
	}
	if !payload.OK {
		t.Fatalf("complaints must not red the gate; verdict=%q reason=%q", payload.Verdict, payload.Reason)
	}
	// Only the two over-block appeals count; latency is excluded. Occurrences sum 4+2=6.
	if got := payload.Corpus["field_fp_appeals"]; got != float64(2) {
		t.Fatalf("field_fp_appeals = %v, want 2", got)
	}
	if got := payload.Corpus["field_fp_occurrences"]; got != float64(6) {
		t.Fatalf("field_fp_occurrences = %v, want 6", got)
	}
	// The subjective intake must not contaminate the ground-truth corpus FP number.
	if got := payload.Corpus["fp_count"]; got != float64(0) {
		t.Fatalf("corpus fp_count = %v, want 0 (complaints must not inflate the labeled FP rate)", got)
	}
}

// TestGuardAccuracyWithoutComplaintsIsUnchanged pins the additive contract: with no
// --complaints flag the emitted payload carries no intake keys, so the default
// `fak score guard-accuracy` surface is exactly what it was before #2821.
func TestGuardAccuracyWithoutComplaintsIsUnchanged(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runGuardAccuracy(&stdout, &stderr, []string{"--workspace", "/repo", "--json"})
	if code != 0 {
		t.Fatalf("runGuardAccuracy exit=%d stderr=%s", code, stderr.String())
	}
	var payload scorecard.Payload
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("parse payload: %v\n%s", err, stdout.String())
	}
	if _, ok := payload.Corpus["field_fp_appeals"]; ok {
		t.Fatal("field_fp_appeals present with no --complaints flag")
	}
}
