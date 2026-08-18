package harnesscontrolpacket

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnesscontrolstudy"
)

func TestReceiptLifecycleDerivesClockIdentityAndArtifact(t *testing.T) {
	dir := createReceiptTestPacket(t, "default-control")
	start := time.Date(2026, 8, 18, 3, 0, 0, 0, time.UTC)
	r, err := StartReceipt(ReceiptStartOptions{Dir: dir, ParticipantID: "person-a", PairID: "pair-a", PairOrder: "default-first", Now: func() time.Time { return start }})
	if err != nil {
		t.Fatal(err)
	}
	if r.ArmPosition != 1 || r.StartedAt != start.Format(time.RFC3339Nano) || !strings.HasPrefix(r.TaskDigest, "sha256:") || r.BaseLockID == "" {
		t.Fatalf("started receipt = %+v", r)
	}
	commands := filepath.Join(dir, "commands.txt")
	artifact := filepath.Join(dir, "result.json")
	os.WriteFile(commands, []byte("./fak harness inspect\n./fak harness verify-run\n"), 0o600)
	os.WriteFile(artifact, []byte(`{"done":true}`), 0o600)
	r, err = FinalizeReceipt(ReceiptFinalizeOptions{Dir: dir, ArtifactPath: artifact, CommandsPath: commands, Succeeded: true, Verified: true, Confidence: 4, InspectCaptured: true, PreviewCaptured: true, RuntimeVerifyCaptured: true, Now: func() time.Time { return start.Add(90 * time.Second) }})
	if err != nil {
		t.Fatal(err)
	}
	if r.ElapsedSeconds != 90 || len(r.Commands) != 2 || !strings.HasPrefix(r.ArtifactDigest, "sha256:") || r.StoppedAt == "" {
		t.Fatalf("final receipt = %+v", r)
	}
}

func TestReceiptLifecycleRequiresSecondArmPreference(t *testing.T) {
	dir := createReceiptTestPacket(t, "scratch")
	start := time.Date(2026, 8, 18, 3, 0, 0, 0, time.UTC)
	if _, err := StartReceipt(ReceiptStartOptions{Dir: dir, ParticipantID: "person-a", PairID: "pair-a", PairOrder: "default-first", Now: func() time.Time { return start }}); err != nil {
		t.Fatal(err)
	}
	commands, artifact := filepath.Join(dir, "commands.txt"), filepath.Join(dir, "result.json")
	os.WriteFile(commands, []byte("./fak harness resolve\n"), 0o600)
	os.WriteFile(artifact, []byte("ok"), 0o600)
	_, err := FinalizeReceipt(ReceiptFinalizeOptions{Dir: dir, ArtifactPath: artifact, CommandsPath: commands, Confidence: 3, Now: func() time.Time { return start.Add(time.Second) }})
	if err == nil || !strings.Contains(err.Error(), "preference") {
		t.Fatalf("err=%v", err)
	}
}

func createReceiptTestPacket(t *testing.T, arm string) string {
	t.Helper()
	root := t.TempDir()
	materials := filepath.Join(root, "materials")
	os.MkdirAll(materials, 0o755)
	os.WriteFile(filepath.Join(materials, "arm-card.md"), []byte(arm), 0o600)
	os.WriteFile(filepath.Join(materials, "task-card.md"), []byte("task"), 0o600)
	if arm == "default-control" {
		os.WriteFile(filepath.Join(materials, "product.json"), []byte("{}"), 0o600)
		os.WriteFile(filepath.Join(materials, "selection.json"), []byte("{}"), 0o600)
		os.WriteFile(filepath.Join(materials, "kernel-component.txt"), []byte("kernel"), 0o600)
		os.WriteFile(filepath.Join(materials, "product.lock.json"), []byte(`{"id":"sha256:`+strings.Repeat("b", 64)+`"}`), 0o600)
	}
	receipt := harnesscontrolstudy.Receipt{Schema: harnesscontrolstudy.ReceiptSchema, StudyID: "default-control-vs-scratch-2026-08", ParticipantID: "person-random", ArtifactDigest: "sha256:REPLACE"}
	raw, _ := json.Marshal(receipt)
	binary, receiptPath := filepath.Join(root, "fak"), filepath.Join(root, "receipt.json")
	os.WriteFile(binary, []byte("binary"), 0o700)
	os.WriteFile(receiptPath, raw, 0o600)
	dir := filepath.Join(root, "packet")
	if _, err := Create(CreateOptions{Arm: arm, MaterialsDir: materials, BinaryPath: binary, ReceiptPath: receiptPath, OutputDir: dir, SourceCommit: strings.Repeat("a", 40), BinaryVersion: "study"}); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestReceiptLifecycleProducesEvaluatorAcceptedPairs(t *testing.T) {
	root := t.TempDir()
	start := time.Date(2026, 8, 18, 4, 0, 0, 0, time.UTC)
	var links []harnesscontrolstudy.ReceiptLink
	var taskDigest string
	for pairIndex, order := range []string{"default-first", "scratch-first"} {
		arms := []string{"default-control", "scratch"}
		if order == "scratch-first" {
			arms = []string{"scratch", "default-control"}
		}
		for armIndex, arm := range arms {
			dir := createReceiptTestPacket(t, arm)
			clock := start.Add(time.Duration(pairIndex*10+armIndex) * time.Minute)
			if _, err := StartReceipt(ReceiptStartOptions{Dir: dir, ParticipantID: fmt.Sprintf("person-%d", pairIndex), PairID: fmt.Sprintf("pair-%d", pairIndex), PairOrder: order, Now: func() time.Time { return clock }}); err != nil {
				t.Fatal(err)
			}
			commands, artifact := filepath.Join(dir, "commands.txt"), filepath.Join(dir, "result.json")
			os.WriteFile(commands, []byte("./fak harness command\n"), 0o600)
			os.WriteFile(artifact, []byte("ok"), 0o600)
			opts := ReceiptFinalizeOptions{Dir: dir, ArtifactPath: artifact, CommandsPath: commands, Succeeded: true, Verified: true, Confidence: 4, InspectCaptured: arm == "default-control", PreviewCaptured: arm == "default-control", RuntimeVerifyCaptured: arm == "default-control", Now: func() time.Time { return clock.Add(30 * time.Second) }}
			if armIndex == 1 {
				opts.Preference, opts.PreferenceReason = "default-control", "visible controls"
			}
			if _, err := FinalizeReceipt(opts); err != nil {
				t.Fatal(err)
			}
			raw, _ := os.ReadFile(filepath.Join(dir, "receipt.json"))
			sum := sha256.Sum256(raw)
			path := filepath.Join(root, fmt.Sprintf("pair-%d-%s.json", pairIndex, arm))
			os.WriteFile(path, raw, 0o600)
			links = append(links, harnesscontrolstudy.ReceiptLink{Source: filepath.Base(path), Digest: "sha256:" + hex.EncodeToString(sum[:])})
			var receipt harnesscontrolstudy.Receipt
			json.Unmarshal(raw, &receipt)
			taskDigest = receipt.TaskDigest
		}
	}
	study := harnesscontrolstudy.Study{Schema: harnesscontrolstudy.Schema, StudyID: "default-control-vs-scratch-2026-08", TaskDigest: taskDigest, MinPairs: 2, Rows: links}
	studyRaw, _ := json.Marshal(study)
	studyPath := filepath.Join(root, "study.json")
	os.WriteFile(studyPath, studyRaw, 0o600)
	report, err := harnesscontrolstudy.Evaluate(studyPath)
	if err != nil || report.AdmissiblePairs != 2 || report.Verdict != "measured" {
		t.Fatalf("report=%+v err=%v", report, err)
	}
}
