package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnesscreationreceipt"
	"github.com/anthony-chaudhary/fak/internal/harnesscreationstudy"
)

func TestHarnessStudyCrossoverCLI(t *testing.T) {
	p := filepath.Join(t.TempDir(), "study.json")
	raw := `{"schema":"fak.harness-crossover-study/v1alpha1","id":"x","tasks":[{"id":"c","domain":"coding"},{"id":"l","domain":"legal"},{"id":"i","domain":"integrated"}],"weights":{"switch_action_seconds":10},"alternatives":[{"id":"native","kind":"native-profile","documentation":[{"url":"https://example.test","retrieved":"2026-08-15"}],"setup":{"value":1,"provenance":"modeled"},"maintenance":{"value":1,"provenance":"modeled"},"runs":[{"task_id":"c","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"l","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"i","explanation":{"provenance":"modeled"},"provenance":"modeled"}]},{"id":"fak","kind":"contextual-harness","documentation":[{"url":"https://example.test","retrieved":"2026-08-15"}],"setup":{"value":2,"provenance":"modeled"},"maintenance":{"value":2,"provenance":"modeled"},"runs":[{"task_id":"c","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"l","explanation":{"provenance":"modeled"},"provenance":"modeled"},{"task_id":"i","explanation":{"provenance":"modeled"},"provenance":"modeled"}]}]}`
	if err := os.WriteFile(p, []byte(raw), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"study", "crossover", "--input", p})
	if code != 0 || !strings.Contains(out.String(), `"winner": "native"`) {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errb.String())
	}
}

func TestHarnessStudyCreationCLIKeepsFailedBuildersInDenominator(t *testing.T) {
	p := filepath.Join(t.TempDir(), "study.json")
	raw := `{"schema":"fak.harness-creation-study/v1alpha1","id":"creation-1","protocol":{"frozen":true,"ten_minute_limit_seconds":600,"assistance_policy":"task-card-and-help-only","failures_in_denominator":true,"task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","parity":{"frozen":true,"minimum_pairs":2,"max_median_elapsed_ratio":1.25,"counterbalanced_order":true}},"baseline":{"id":"tuned-alt","runnable":true,"tuned":true,"frozen":true,"evidence":"receipts/baseline.json"},"runs":[{"id":"maintainer","participant_id":"maintainer","track":"ten-minute","participant_class":"maintainer-calibration","independent":false,"outcome":"success","elapsed_seconds":10,"receipt":"receipts/m.json"},{"id":"builder-a","participant_id":"builder-a","track":"ten-minute","participant_class":"unfamiliar-builder","independent":true,"outcome":"timeout","elapsed_seconds":600,"receipt":"receipts/a.json"}]}`
	if err := os.WriteFile(p, []byte(raw), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"study", "creation", "--input", p})
	if code != 0 || !strings.Contains(out.String(), `"calibration_runs": 1`) || !strings.Contains(out.String(), `"failures": 1`) || !strings.Contains(out.String(), `"claim_status": "not_yet"`) || !strings.Contains(out.String(), `"complete_pairs": 0`) {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errb.String())
	}
}

func TestHarnessStudyReceiptKeepsEarlyFailureInPairedDenominator(t *testing.T) {
	dir := t.TempDir()
	studyPath := filepath.Join(dir, "study.json")
	study := harnesscreationstudy.Study{
		Schema: harnesscreationstudy.Schema,
		ID:     "early-failure-e2e",
		Protocol: harnesscreationstudy.Protocol{
			TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Frozen: true, TenMinuteLimitSeconds: 600, AssistancePolicy: "task-card-and-help-only", FailuresInDenominator: true,
			Parity: harnesscreationstudy.MatchedStudySpec{Frozen: true, MinimumPairs: 2, MaxMedianElapsedRatio: 1.25, CounterbalancedOrder: true},
		},
		Baseline: harnesscreationstudy.Baseline{ID: "tuned-alt", Runnable: true, Tuned: true, Frozen: true, Evidence: "baseline.json"},
	}
	studyRaw, err := json.Marshal(study)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(studyPath, studyRaw, 0600); err != nil {
		t.Fatal(err)
	}
	start := time.Date(2026, 8, 16, 1, 2, 3, 0, time.UTC)
	receipt := harnesscreationreceipt.Receipt{
		Schema: harnesscreationreceipt.Schema, RunID: "run-early-failure", ParticipantID: "person-early", ParticipantClass: "unfamiliar-builder", PriorFamiliarity: "none",
		Track: "ten-minute", Arm: "fak", PairID: "pair-early", TaskDigest: study.Protocol.TaskDigest, MachineID: "machine-early", PairOrder: "fak-first", ArmPosition: 1, Independent: true,
		Artifact: "fak@v1", ArtifactDigest: "sha256:abc", OS: "linux", CPU: "x86_64", Toolchain: "go1.26", NetworkState: "online", CacheState: "empty",
		StartedAt: start, StoppedAt: start.Add(9 * time.Second), ElapsedSeconds: 9, Commands: []harnesscreationreceipt.Command{{Command: "retrieve fak", Exit: 1}},
		Outcome: "failure", Transcript: "early.txt", Receipt: "early.json",
	}
	receiptRaw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(dir, "receipt.json")
	if err := os.WriteFile(receiptPath, receiptRaw, 0600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runHarness(&out, &errOut, []string{"study", "receipt", "--input", receiptPath, "--study", studyPath}); code != 0 {
		t.Fatalf("early failure code=%d err=%s", code, errOut.String())
	}
	var result harnesscreationreceipt.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	study.Runs = append(study.Runs, harnesscreationstudy.Run{
		ID: result.Row.ID, ParticipantID: result.Row.ParticipantID, Track: result.Row.Track, Arm: result.Row.Arm, PairID: result.Row.PairID, TaskDigest: result.Row.TaskDigest, MachineID: result.Row.MachineID,
		PairOrder: result.Row.PairOrder, ArmPosition: result.Row.ArmPosition, ParticipantClass: result.Row.ParticipantClass, Independent: result.Row.Independent, OS: result.Row.OS, CPU: result.Row.CPU,
		NetworkState: result.Row.NetworkState, CacheState: result.Row.CacheState, Outcome: result.Row.Outcome, ElapsedSeconds: result.Row.ElapsedSeconds, Receipt: result.Row.Receipt,
	})
	report := harnesscreationstudy.Evaluate(study)
	if report.Parity.IncompletePairs != 1 || report.Parity.FakSuccesses != 0 || report.Parity.ClaimStatus != "not_yet" {
		t.Fatalf("early failure missing from denominator: %+v", report.Parity)
	}
}

func TestHarnessStudyReceiptRowsFormCompletePair(t *testing.T) {
	dir := t.TempDir()
	studyPath := filepath.Join(dir, "study.json")
	study := harnesscreationstudy.Study{
		Schema: harnesscreationstudy.Schema,
		ID:     "paired-e2e",
		Protocol: harnesscreationstudy.Protocol{
			TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Frozen: true, TenMinuteLimitSeconds: 600, AssistancePolicy: "task-card-and-help-only", FailuresInDenominator: true,
			Parity: harnesscreationstudy.MatchedStudySpec{Frozen: true, MinimumPairs: 2, MaxMedianElapsedRatio: 1.25, CounterbalancedOrder: true},
		},
		Baseline: harnesscreationstudy.Baseline{ID: "tuned-alt", Runnable: true, Tuned: true, Frozen: true, Evidence: "baseline.json"},
	}
	writeStudy := func() {
		t.Helper()
		raw, err := json.Marshal(study)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(studyPath, raw, 0600); err != nil {
			t.Fatal(err)
		}
	}
	writeStudy()
	start := time.Date(2026, 8, 17, 1, 2, 3, 0, time.UTC)
	for _, pair := range []struct{ id, participant, order string }{{"pair-a", "person-a", "fak-first"}, {"pair-b", "person-b", "baseline-first"}} {
		for _, arm := range []string{"fak", "baseline"} {
			position := 2
			if (pair.order == "fak-first" && arm == "fak") || (pair.order == "baseline-first" && arm == "baseline") {
				position = 1
			}
			receipt := harnesscreationreceipt.Receipt{
				Schema: harnesscreationreceipt.Schema, RunID: "run-" + pair.id + "-" + arm, ParticipantID: pair.participant, ParticipantClass: "unfamiliar-builder", PriorFamiliarity: "none",
				Track: "ten-minute", Arm: arm, PairID: pair.id, TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-" + pair.id, PairOrder: pair.order, ArmPosition: position, Independent: true, Artifact: arm + "@v1", ArtifactDigest: "sha256:abc", OS: "linux", CPU: "amd64", Toolchain: "go1.26", NetworkState: "online install; offline selfcheck", CacheState: "empty",
				StartedAt: start, StoppedAt: start.Add(100 * time.Second), ElapsedSeconds: 100, Commands: []harnesscreationreceipt.Command{{Command: "build", Exit: 0}}, FilesChanged: []string{"product/config.go"}, Rebuilds: 1, RebuildSeconds: 10, Outcome: "success", Transcript: arm + ".txt", Receipt: arm + ".json",
			}
			receiptPath := filepath.Join(dir, pair.id+"-"+arm+".json")
			raw, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(receiptPath, raw, 0600); err != nil {
				t.Fatal(err)
			}
			var out, errb bytes.Buffer
			if code := runHarness(&out, &errb, []string{"study", "receipt", "--input", receiptPath, "--study", studyPath}); code != 0 {
				t.Fatalf("arm=%s code=%d err=%s", arm, code, errb.String())
			}
			var result harnesscreationreceipt.Result
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			study.Runs = append(study.Runs, harnesscreationstudy.Run{
				ID: result.Row.ID, ParticipantID: result.Row.ParticipantID, Track: result.Row.Track, Arm: result.Row.Arm, PairID: result.Row.PairID, TaskDigest: result.Row.TaskDigest, MachineID: result.Row.MachineID, OS: result.Row.OS, CPU: result.Row.CPU, NetworkState: result.Row.NetworkState, CacheState: result.Row.CacheState, PairOrder: result.Row.PairOrder, ArmPosition: result.Row.ArmPosition,
				ParticipantClass: result.Row.ParticipantClass, Independent: result.Row.Independent, Outcome: result.Row.Outcome, ElapsedSeconds: result.Row.ElapsedSeconds, Receipt: result.Row.Receipt,
			})
			writeStudy()
		}
	}
	drift := harnesscreationreceipt.Receipt{
		Schema: harnesscreationreceipt.Schema, RunID: "run-pair-c-fak", ParticipantID: "person-c", ParticipantClass: "unfamiliar-builder", PriorFamiliarity: "none",
		Track: "ten-minute", Arm: "fak", PairID: "pair-c", TaskDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", MachineID: "machine-c", PairOrder: "fak-first", ArmPosition: 1, Independent: true,
		Artifact: "fak@v1", ArtifactDigest: "sha256:abc", OS: "linux", CPU: "x86_64", Toolchain: "go1.26", NetworkState: "online", CacheState: "empty",
		StartedAt: time.Date(2026, 8, 16, 1, 2, 3, 0, time.UTC), StoppedAt: time.Date(2026, 8, 16, 1, 4, 3, 0, time.UTC), ElapsedSeconds: 120,
		Commands: []harnesscreationreceipt.Command{{Command: "build", Exit: 0}}, FilesChanged: []string{"product/config.go"}, Rebuilds: 1, RebuildSeconds: 10, Outcome: "success", Transcript: "drift.txt", Receipt: "drift.json",
	}
	driftRaw, err := json.Marshal(drift)
	if err != nil {
		t.Fatal(err)
	}
	driftPath := filepath.Join(dir, "drift.json")
	if err := os.WriteFile(driftPath, driftRaw, 0600); err != nil {
		t.Fatal(err)
	}
	var driftOut, driftErr bytes.Buffer
	if code := runHarness(&driftOut, &driftErr, []string{"study", "receipt", "--input", driftPath, "--study", studyPath}); code == 0 || !strings.Contains(driftErr.String(), "protocol task_digest") {
		t.Fatalf("protocol digest drift code=%d err=%s", code, driftErr.String())
	}

	report := harnesscreationstudy.Evaluate(study)
	if report.Parity.CompletePairs != 2 || report.Parity.ClaimStatus != "supported" {
		t.Fatalf("receipt rows did not form supported pair: %+v", report.Parity)
	}
}
