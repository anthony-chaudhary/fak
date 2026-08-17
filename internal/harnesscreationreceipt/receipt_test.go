package harnesscreationreceipt

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func validReceipt() Receipt {
	start := time.Date(2026, 8, 16, 1, 2, 3, 0, time.UTC)
	return Receipt{Schema: Schema, RunID: "run-a8f29c", ParticipantID: "person-b7e14d", ParticipantClass: "unfamiliar-builder", PriorFamiliarity: "none", Track: "ten-minute", Arm: "fak", PairID: "pair-a8f29c", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a8f29c", PairOrder: "fak-first", ArmPosition: 1, Independent: true, Artifact: "fak@v0.44.0", ArtifactDigest: "sha256:abc", OS: "linux-amd64", CPU: "x86_64", Toolchain: "go1.26", NetworkState: "online install; offline selfcheck", CacheState: "empty", StartedAt: start, StoppedAt: start.Add(5 * time.Minute), ElapsedSeconds: 300, Commands: []Command{{Command: "fak harness init", Exit: 0}}, FilesChanged: []string{"product/config.go"}, Rebuilds: 1, RebuildSeconds: 20, Outcome: "success", HelpRequests: 0, Transcript: "receipts/run/transcript.txt", Receipt: "receipts/run/README.md"}
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

func TestParseAdmitsEarlyFailureWithoutProductMutation(t *testing.T) {
	r := validReceipt()
	r.Outcome = "failure"
	r.FilesChanged = nil
	r.Rebuilds = 0
	r.RebuildSeconds = 0
	r.Commands = []Command{{Command: "retrieve artifact", Exit: 1}}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Parse(raw)
	if err != nil {
		t.Fatalf("early failure refused: %v", err)
	}
	if got.Outcome != "failure" || len(got.FilesChanged) != 0 || got.Rebuilds != 0 {
		t.Fatalf("early failure changed: %+v", got)
	}
}

func TestParseKeepsSuccessEvidenceFloorAndRejectsContradictions(t *testing.T) {
	tests := map[string]Receipt{}
	missing := validReceipt()
	missing.FilesChanged = nil
	missing.Rebuilds = 0
	missing.RebuildSeconds = 0
	tests["success missing product evidence"] = missing

	negative := validReceipt()
	negative.Outcome = "failure"
	negative.Rebuilds = -1
	tests["negative rebuild"] = negative

	partial := validReceipt()
	partial.Outcome = "failure"
	partial.FilesChanged = nil
	partial.Rebuilds = 0
	partial.RebuildSeconds = 3
	tests["duration without rebuild"] = partial

	for name, receipt := range tests {
		t.Run(name, func(t *testing.T) {
			raw, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := Parse(raw); err == nil {
				t.Fatal("contradictory receipt accepted")
			}
		})
	}
}

func TestParseRejectsElapsedClockMismatch(t *testing.T) {
	r := validReceipt()
	r.ElapsedSeconds--
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Parse(raw)
	if err == nil || !strings.Contains(err.Error(), "elapsed_seconds 299 does not match started_at/stopped_at interval 300") {
		t.Fatalf("Parse() error = %v, want elapsed clock mismatch", err)
	}
}

func TestParseAcceptsFractionalClockInterval(t *testing.T) {
	r := validReceipt()
	r.StoppedAt = r.StoppedAt.Add(500 * time.Millisecond)
	r.ElapsedSeconds = 300.5
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}

	got, err := Parse(raw)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if got.ElapsedSeconds != 300.5 {
		t.Fatalf("elapsed_seconds = %v, want 300.5", got.ElapsedSeconds)
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
	r.Arm = "fak"
	r.IndependentAuthorship = "signed artifact statement"
	r.Conformance = "receipt/conformance.json"
	raw, _ := json.Marshal(r)
	if _, err := Parse(raw); err != nil {
		t.Fatal(err)
	}
}

func TestCheckUniqueRejectsTaskDigestOutsideProtocol(t *testing.T) {
	row := Evaluate(validReceipt()).Row
	study := []byte(`{"protocol":{"task_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"runs":[]}`)
	if err := CheckUnique(study, row); err == nil || !strings.Contains(err.Error(), "protocol task_digest") {
		t.Fatalf("protocol digest drift accepted: %v", err)
	}
}

func TestCheckUniqueAdmitsSecondPairedArmAndRejectsDuplicates(t *testing.T) {
	r := validReceipt()
	row := Evaluate(r).Row
	study := `{"protocol":{"task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"runs":[{"id":"run-other","participant_id":"person-b7e14d","track":"ten-minute","arm":"baseline","pair_id":"pair-a8f29c","task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","machine_id":"machine-a8f29c","os":"linux-amd64","cpu":"x86_64","network_state":"online install; offline selfcheck","cache_state":"empty","pair_order":"fak-first","arm_position":2}]}`
	if err := CheckUnique([]byte(study), row); err != nil {
		t.Fatalf("second paired arm refused: %v", err)
	}
	for name, existing := range map[string]string{
		"run":              `{"id":"run-a8f29c","participant_id":"person-other","track":"ten-minute","arm":"baseline","pair_id":"pair-other","task_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","machine_id":"machine-other","os":"linux-amd64","cpu":"x86_64","network_state":"online install; offline selfcheck","cache_state":"empty","pair_order":"baseline-first","arm_position":1}`,
		"pair participant": `{"id":"run-other","participant_id":"person-other","track":"ten-minute","arm":"baseline","pair_id":"pair-a8f29c","task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","machine_id":"machine-a8f29c","os":"linux-amd64","cpu":"x86_64","network_state":"online install; offline selfcheck","cache_state":"empty","pair_order":"fak-first","arm_position":2}`,
		"attempt":          `{"id":"run-other","participant_id":"person-b7e14d","track":"ten-minute","arm":"fak","pair_id":"pair-other","task_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","machine_id":"machine-other","os":"linux-amd64","cpu":"x86_64","network_state":"online install; offline selfcheck","cache_state":"empty","pair_order":"baseline-first","arm_position":1}`,
		"pair arm":         `{"id":"run-other","participant_id":"person-other","track":"ten-minute","arm":"fak","pair_id":"pair-a8f29c","task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","machine_id":"machine-a8f29c","os":"linux-amd64","cpu":"x86_64","network_state":"online install; offline selfcheck","cache_state":"empty","pair_order":"fak-first","arm_position":2}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := CheckUnique([]byte(`{"protocol":{"task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"runs":[`+existing+`]}`), row); err == nil {
				t.Fatal("expected duplicate refusal")
			}
		})
	}
}

func TestParseRejectsArmOrderMismatch(t *testing.T) {
	r := validReceipt()
	r.ArmPosition = 2
	raw, _ := json.Marshal(r)
	if _, err := Parse(raw); err == nil || !strings.Contains(err.Error(), "arm_position") {
		t.Fatalf("wrong arm position accepted: %v", err)
	}
	r = validReceipt()
	r.PairOrder = ""
	raw, _ = json.Marshal(r)
	if _, err := Parse(raw); err == nil || !strings.Contains(err.Error(), "pair_order") {
		t.Fatalf("missing pair order accepted: %v", err)
	}
}

func TestCheckUniqueRejectsPairedEnvelopeDrift(t *testing.T) {
	row := Evaluate(validReceipt()).Row
	fields := map[string]func(*StudyRow){
		"task": func(r *StudyRow) {
			r.TaskDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
		},
		"machine": func(r *StudyRow) { r.MachineID = "machine-other" },
		"os":      func(r *StudyRow) { r.OS = "windows-amd64" },
		"cpu":     func(r *StudyRow) { r.CPU = "arm64" },
		"network": func(r *StudyRow) { r.NetworkState = "offline" },
		"cache":   func(r *StudyRow) { r.CacheState = "warm" },
	}
	for name, mutate := range fields {
		t.Run(name, func(t *testing.T) {
			existing := row
			existing.ID = "run-other"
			existing.Arm = "baseline"
			existing.ArmPosition = 2
			mutate(&existing)
			raw, _ := json.Marshal(struct {
				Protocol struct {
					TaskDigest string `json:"task_digest"`
				} `json:"protocol"`
				Runs []StudyRow `json:"runs"`
			}{Protocol: struct {
				TaskDigest string `json:"task_digest"`
			}{TaskDigest: row.TaskDigest}, Runs: []StudyRow{existing}})
			if err := CheckUnique(raw, row); err == nil || !strings.Contains(err.Error(), "envelope") {
				t.Fatalf("%s drift accepted: %v", name, err)
			}
		})
	}
}
