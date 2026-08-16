package harnesscreationreceipt

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func validReceipt() Receipt {
	start := time.Date(2026, 8, 16, 1, 2, 3, 0, time.UTC)
	return Receipt{Schema: Schema, RunID: "run-a8f29c", ParticipantID: "person-b7e14d", ParticipantClass: "unfamiliar-builder", PriorFamiliarity: "none", Track: "ten-minute", Independent: true, Artifact: "fak@v0.44.0", ArtifactDigest: "sha256:abc", OS: "linux-amd64", CPU: "x86_64", Toolchain: "go1.26", NetworkState: "online install; offline selfcheck", CacheState: "empty", StartedAt: start, StoppedAt: start.Add(5 * time.Minute), ElapsedSeconds: 300, Commands: []Command{{Command: "fak harness init", Exit: 0}}, FilesChanged: []string{"product/config.go"}, Rebuilds: 1, RebuildSeconds: 20, Outcome: "success", HelpRequests: 0, Transcript: "receipts/run/transcript.txt", Receipt: "receipts/run/README.md"}
}
func TestParseAndEvaluateIndependentReceipt(t *testing.T) {
	r := validReceipt()
	raw, _ := json.Marshal(r)
	got, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	out := Evaluate(got)
	if !out.Valid || out.Row.ParticipantID != r.ParticipantID || out.Row.Outcome != "success" {
		t.Fatalf("result=%+v", out)
	}
}
func TestFailureRemainsEligibleRow(t *testing.T) {
	r := validReceipt()
	r.Outcome = "failure"
	raw, _ := json.Marshal(r)
	got, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	if Evaluate(got).Row.Outcome != "failure" {
		t.Fatal("failure was dropped")
	}
}
func TestParseFailsClosedAndWeekendRequiresWitness(t *testing.T) {
	for name, mutate := range map[string]func(*Receipt){"clock": func(r *Receipt) { r.StoppedAt = r.StartedAt }, "commands": func(r *Receipt) { r.Commands = nil }, "slug": func(r *Receipt) { r.ParticipantID = "Jane Doe" }, "weekend": func(r *Receipt) { r.Track = "weekend" }} {
		t.Run(name, func(t *testing.T) {
			r := validReceipt()
			mutate(&r)
			raw, _ := json.Marshal(r)
			_, err := Parse(raw)
			if err == nil {
				t.Fatal("expected refusal")
			}
			if name == "weekend" && !strings.Contains(err.Error(), "independent_authorship") {
				t.Fatal(err)
			}
		})
	}
}
func TestWeekendWitnessAccepted(t *testing.T) {
	r := validReceipt()
	r.Track = "weekend"
	r.IndependentAuthorship = "signed artifact statement"
	r.Conformance = "receipt/conformance.json"
	raw, _ := json.Marshal(r)
	if _, err := Parse(raw); err != nil {
		t.Fatal(err)
	}
}

func TestCheckUniqueRejectsRunAndParticipantDuplicates(t *testing.T) {
	r := validReceipt()
	row := Evaluate(r).Row
	for _, study := range []string{
		`{"runs":[{"id":"run-a8f29c","participant_id":"person-other"}]}`,
		`{"runs":[{"id":"run-other","participant_id":"person-b7e14d"}]}`,
	} {
		if err := CheckUnique([]byte(study), row); err == nil {
			t.Fatal("expected duplicate refusal")
		}
	}
	if err := CheckUnique([]byte(`{"runs":[]}`), row); err != nil {
		t.Fatal(err)
	}
}
