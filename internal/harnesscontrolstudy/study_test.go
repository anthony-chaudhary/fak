package harnesscontrolstudy

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEvaluatePairedStudy(t *testing.T) {
	dir := t.TempDir()
	taskDigest := "sha256:" + strings.Repeat("a", 64)
	rows := []Receipt{
		receipt("person-a", "pair-a", "default-first", "default-control", 1, 42, 5, ""),
		receipt("person-a", "pair-a", "default-first", "scratch", 2, 95, 3, "default-control"),
		receipt("person-b", "pair-b", "scratch-first", "scratch", 1, 82, 3, ""),
		receipt("person-b", "pair-b", "scratch-first", "default-control", 2, 38, 5, "default-control"),
	}
	study := Study{Schema: Schema, StudyID: "study", TaskDigest: taskDigest, MinPairs: 2}
	for i := range rows {
		rows[i].StudyID = study.StudyID
		rows[i].TaskDigest = taskDigest
		raw, _ := json.Marshal(rows[i])
		name := filepath.Join("receipts", string(rune('a'+i))+".json")
		path := filepath.Join(dir, name)
		os.MkdirAll(filepath.Dir(path), 0o755)
		os.WriteFile(path, raw, 0o600)
		study.Rows = append(study.Rows, ReceiptLink{Source: filepath.ToSlash(name), Digest: digest(raw)})
	}
	raw, _ := json.Marshal(study)
	path := filepath.Join(dir, "study.json")
	os.WriteFile(path, raw, 0o600)
	report, err := Evaluate(path)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "measured" || report.AdmissiblePairs != 2 || report.PreferDefaultControl != 2 || report.DefaultControl.MedianSeconds != 40 || report.Scratch.MedianSeconds != 88.5 {
		t.Fatalf("report=%+v", report)
	}
}

func TestEvaluateRefusesMissingRawReceipt(t *testing.T) {
	dir := t.TempDir()
	study := Study{Schema: Schema, StudyID: "study", TaskDigest: "sha256:" + strings.Repeat("a", 64), MinPairs: 2, Rows: []ReceiptLink{{Source: "missing.json", Digest: "sha256:" + strings.Repeat("b", 64)}}}
	raw, _ := json.Marshal(study)
	path := filepath.Join(dir, "study.json")
	os.WriteFile(path, raw, 0o600)
	if _, err := Evaluate(path); err == nil {
		t.Fatal("expected missing receipt refusal")
	}
}

func receipt(person, pair, order, arm string, pos int, seconds float64, confidence int, preference string) Receipt {
	base := ""
	inspect, preview := false, false
	if arm == "default-control" {
		base = "sha256:" + strings.Repeat("c", 64)
		inspect = true
		preview = true
	}
	return Receipt{Schema: ReceiptSchema, ParticipantID: person, PairID: pair, PairOrder: order, Arm: arm, ArmPosition: pos, StartedAt: "2026-08-17T12:00:00Z", StoppedAt: timeAt(seconds), ElapsedSeconds: seconds, Succeeded: true, Verified: true, Confidence: confidence, Commands: []string{"run"}, ArtifactDigest: "sha256:" + strings.Repeat("d", 64), BinaryVersion: "v1", BinaryCommit: "abc", BaseLockID: base, InspectCaptured: inspect, PreviewCaptured: preview, Preference: preference, PreferenceReason: func() string {
		if pos == 2 {
			return "clearer and faster"
		}
		return ""
	}()}
}
func timeAt(seconds float64) string {
	return "2026-08-17T12:" + func() string { m := int(seconds) / 60; s := int(seconds) % 60; return fmt.Sprintf("%02d:%02dZ", m, s) }()
}
