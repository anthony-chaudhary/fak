package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFormatMicroParentReceiptRowsReadOnlyAndHonest(t *testing.T) {
	rows, err := formatMicroParentReceiptRows([]byte(`{"schema":"fak-micro-selfcheck/2","parent_task_id":"parent","verdict":"PASS","children":[{"work_unit_id":"alpha","lease_id":"lease-a","session_id":"session-a","state":"stopped","effect_digest":"sha256:abc","witnessed":true},{"work_unit_id":"beta"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(rows, "\n")
	for _, want := range []string{"receipt  parent · PASS · children 2", "lease lease-a · session session-a · stopped · effect sha256:abc", "beta · lease not yet · session not yet · not yet · effect not yet"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q:\n%s", want, got)
		}
	}
}
func TestFormatMicroParentReceiptRowsRejectsSemanticFork(t *testing.T) {
	if _, err := formatMicroParentReceiptRows([]byte(`{"schema":"other/1"}`)); err == nil {
		t.Fatal("accepted unknown schema")
	}
}

func TestRunInfoReceiptCapturesReadOnlyRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, []byte(`{"schema":"fak-micro-selfcheck/2","parent_task_id":"parent","verdict":"PASS","children":[{"work_unit_id":"alpha"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runInfo(&out, &errOut, []string{"--receipt", path}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if got := out.String(); !strings.Contains(got, "receipt  parent · PASS") || !strings.Contains(got, "effect not yet") {
		t.Fatalf("dishonest render:\n%s", got)
	}
}
