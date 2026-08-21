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
	if !report.OK || !report.NonFAK {
		t.Fatalf("report=%+v", report)
	}
	wantRows := map[string]bool{
		"bounded-repository-tools": true, "caveman-and-ponytail": true,
		"compact-history": true, "stale-read-elision": true, "cold-tool-deferral": true,
		"vcache-anchor": true, "minimum-prefix-steering": true,
		"calibrated-read-pricing": true, "calibrated-write-pricing": true,
		"ttl-tier-steering": true, "cross-backend-vcache-signals": true,
		"openai-cold-tool-deferral": true,
	}
	for _, row := range report.Rows {
		if row.State == "fail" {
			t.Fatalf("failed row: %+v", row)
		}
		delete(wantRows, row.Name)
	}
	if len(wantRows) != 0 {
		t.Fatalf("defaults selfcheck missing required rows: %v", wantRows)
	}
	wantPostures := map[string]bool{
		"agent//" + shrinkWireForeignProxy:               true,
		"guard/claude/" + shrinkWireAnthropicPassthrough: true,
		"guard/codex/" + shrinkWireForeignProxy:          true,
		"serve//" + shrinkWireInKernel:                   true,
		"serve//" + shrinkWireForeignProxy:               true,
	}
	for _, posture := range report.Postures {
		delete(wantPostures, posture.Entrypoint+"/"+posture.Harness+"/"+posture.Wire)
	}
	if len(wantPostures) != 0 {
		t.Fatalf("defaults selfcheck missing launch postures: %v", wantPostures)
	}
	if _, err := os.Stat(filepath.Join(root, "created.txt")); err != nil {
		t.Fatalf("tool mutation witness missing: %v", err)
	}
}
