package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessControlStudyCLIReportsNotYetWithoutPairs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "study.json")
	raw := `{"schema":"fak-harness-control-study/1","study_id":"pilot","task_digest":"sha256:` + strings.Repeat("a", 64) + `","min_pairs":2,"rows":[]}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"study", "control", "--input", path})
	var report struct {
		Verdict string   `json:"verdict"`
		Reasons []string `json:"reasons"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if code != 3 || errb.Len() != 0 || report.Verdict != "not_yet" || len(report.Reasons) == 0 {
		t.Fatalf("code=%d stderr=%q report=%+v", code, errb.String(), report)
	}
}

func TestHarnessControlPacketCLIRejectsCrossArmLeakAndVerifies(t *testing.T) {
	root := t.TempDir()
	materials := filepath.Join(root, "scratch")
	if err := os.MkdirAll(materials, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{"arm-card.md": "scratch", "task-card.md": "task"} {
		if err := os.WriteFile(filepath.Join(materials, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	binary := filepath.Join(root, "fak")
	receipt := filepath.Join(root, "receipt.json")
	os.WriteFile(binary, []byte("binary"), 0o755)
	os.WriteFile(receipt, []byte("{}"), 0o644)
	packet := filepath.Join(root, "packet")
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"study", "control", "packet", "create", "--arm", "scratch", "--materials", materials, "--binary", binary, "--receipt", receipt, "--output", packet, "--source-commit", "abc123", "--binary-version", "study abc123"})
	if code != 0 || !strings.Contains(out.String(), "HARNESS CONTROL PACKET | CREATED") {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	out.Reset()
	if code := runHarness(&out, &errb, []string{"study", "control", "packet", "verify", "--dir", packet}); code != 0 || !strings.Contains(out.String(), "HARNESS CONTROL PACKET | VERIFIED") {
		t.Fatalf("verify code=%d stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if _, err := os.Stat(filepath.Join(packet, "product.json")); !os.IsNotExist(err) {
		t.Fatalf("scratch packet leaked product: %v", err)
	}
}
