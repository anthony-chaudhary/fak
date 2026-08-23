package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunLifecyclePreviewAndApply(t *testing.T) {
	p := filepath.Join(t.TempDir(), "tx.json")
	b, _ := json.Marshal(lifecycle.Transaction{ID: "tx-1", Stage: "partial_pause", Members: []lifecycle.Member{{ID: "a", State: "paused"}}})
	if err := os.WriteFile(p, b, 0600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runLifecycle(&out, &errOut, []string{"cancel", "--tx", p}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"operation": "resume"`) {
		t.Fatalf("preview=%s", out.String())
	}
	var tx lifecycle.Transaction
	raw, _ := os.ReadFile(p)
	json.Unmarshal(raw, &tx)
	if tx.Stage != "partial_pause" {
		t.Fatal("preview mutated transaction")
	}
	out.Reset()
	if code := runLifecycle(&out, &errOut, []string{"cancel", "--tx", p, "--apply"}); code != 0 {
		t.Fatalf("apply code=%d err=%s", code, errOut.String())
	}
	raw, _ = os.ReadFile(p)
	json.Unmarshal(raw, &tx)
	if tx.Outcome != "cancelled" {
		t.Fatalf("applied=%s", raw)
	}
}
