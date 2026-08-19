package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDoctorDefaultsSelfcheckInNonFakRepository(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("benchmark repository\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if rc := runDoctor(nil, &out, &errOut, []string{"defaults-selfcheck", "--workspace", root, "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s output=%s", rc, errOut.String(), out.String())
	}
	var report defaultsSelfcheckReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.OK || !report.NonFAK || len(report.Rows) < 12 || len(report.Postures) != 5 {
		t.Fatalf("report=%+v", report)
	}
	for _, row := range report.Rows {
		if row.State == "fail" {
			t.Fatalf("failed row: %+v", row)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "created.txt")); err != nil {
		t.Fatalf("tool mutation witness missing: %v", err)
	}
}
