package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionobs"
)

// seedLearnCorpus is a corpus with a clear, non-circular discriminator: waste sessions
// carry heavy guard friction, value sessions almost none. Both cohorts clear the
// contrast's minimum so the report is separable.
func seedLearnCorpus() []sessionobs.Record {
	var c []sessionobs.Record
	for i := 0; i < 3; i++ {
		c = append(c, sessionobs.Record{
			SessionID: "v", AssistantTurns: 10, ToolCalls: 20, ReadOnlyCalls: 12, OutputTokens: 3000,
			Outcome: sessionobs.OutcomeShipped, Signals: sessionobs.Signals{Commits: 1},
		})
	}
	for i := 0; i < 3; i++ {
		c = append(c, sessionobs.Record{
			SessionID: "w", AssistantTurns: 9, ToolCalls: 18, ReadOnlyCalls: 6, OutputTokens: 1500,
			Outcome: sessionobs.OutcomeStopped, Signals: sessionobs.Signals{StopEvents: 1, GuardRefusals: 6},
		})
	}
	return c
}

func writeLearnCorpusFile(t *testing.T, recs []sessionobs.Record) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "corpus.jsonl")
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	for _, r := range recs {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSessionsLearnReadsCorpus(t *testing.T) {
	path := writeLearnCorpusFile(t, seedLearnCorpus())
	var out, errb bytes.Buffer
	if rc := runSessions(&out, &errb, []string{"learn", "--corpus", path}); rc != 0 {
		t.Fatalf("learn exit=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "guard_refusals") {
		t.Errorf("expected the value-vs-waste contrast, got:\n%s", out.String())
	}
}

func TestSessionsLearnJSONEnvelope(t *testing.T) {
	path := writeLearnCorpusFile(t, seedLearnCorpus())
	var out, errb bytes.Buffer
	if rc := runSessions(&out, &errb, []string{"learn", "--corpus", path, "--json"}); rc != 0 {
		t.Fatalf("learn --json exit=%d stderr=%s", rc, errb.String())
	}
	var env learnEnvelope
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("envelope is not JSON: %v\n%s", err, out.String())
	}
	if env.Schema != learnSchema || !env.OK {
		t.Errorf("envelope schema/ok wrong: %+v", env)
	}
	if env.Finding != "sessionobs_learn" {
		t.Errorf("a separable corpus with a discriminator should find sessionobs_learn, got %q", env.Finding)
	}
	if env.Contrast.TopFeature != "guard_refusals" {
		t.Errorf("guard_refusals is the planted discriminator, got top=%q", env.Contrast.TopFeature)
	}
}

func TestSessionsLearnMissingCorpusIsAdvisory(t *testing.T) {
	// A missing committed corpus must NOT error a garden member (which would red the
	// gate): under --json it is a well-formed advisory envelope with exit 0.
	var out, errb bytes.Buffer
	missing := filepath.Join(t.TempDir(), "nope.jsonl")
	rc := runSessions(&out, &errb, []string{"learn", "--corpus", missing, "--json"})
	if rc != 0 {
		t.Fatalf("missing corpus --json should exit 0 (advisory), got %d (stderr %s)", rc, errb.String())
	}
	var env learnEnvelope
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("envelope is not JSON: %v\n%s", err, out.String())
	}
	if env.OK || env.Verdict != "ACTION" || env.Finding != "sessionobs_corpus_missing" {
		t.Errorf("missing corpus should be advisory ACTION, got %+v", env)
	}
}

func TestSessionsLearnRegisteredInGarden(t *testing.T) {
	// The honest witness for the sessionobs loop_consumes rung: the learn loop is a
	// registered garden member. The score path reads this same predicate.
	if !sessionsLearnRegistered() {
		t.Fatal("sessions_learn must be registered in gardenbundle.Members (the loop_consumes rung)")
	}
}
