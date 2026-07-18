package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDoctorMoversRendersSection is the end-to-end doctor render witness (#2472):
// it runs `fak doctor movers` against a temp ledger + issue feed and asserts the
// section renders both the fastest-moving modules and the dormant-with-open-
// issues candidates. Dormant last_dates are anchored in 2020 so the row is
// dormant for any real test clock (the command reads the wall clock for `now`).
func TestDoctorMoversRendersSection(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "module-versions.jsonl")
	ledgerBody := `{"schema":"fak-module-versions/1","ts":"2020-01-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":5,"last_commit":"g1","last_date":"2020-01-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2020-03-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":20,"last_commit":"g2","last_date":"2020-03-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2020-01-01T00:00:00Z","module":"internal/idle","kind":"internal","rev":2,"last_commit":"i1","last_date":"2020-01-01T00:00:00Z"}
`
	if err := os.WriteFile(ledger, []byte(ledgerBody), 0o644); err != nil {
		t.Fatal(err)
	}
	issues := filepath.Join(dir, "issues.json")
	if err := os.WriteFile(issues, []byte(`[{"number":42,"title":"idle rot","paths":["internal/idle/x.go"]}]`), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runDoctorMovers(&out, &errb, []string{"--ledger", ledger, "--issues", issues})
	if rc != 0 {
		t.Fatalf("runDoctorMovers rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"== fak doctor: module movers ==",
		"Δr+15  r5→r20  internal/gateway",
		"dormant modules with open issues",
		"internal/idle",
		"issues: #42",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("`fak doctor movers` output missing %q\n---\n%s", want, got)
		}
	}
}

// TestDoctorMoversMissingLedger witnesses the read-error exit: a missing ledger
// path is a clean exit 1 with an actionable hint, not a panic.
func TestDoctorMoversMissingLedger(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runDoctorMovers(&out, &errb, []string{"--ledger", filepath.Join(t.TempDir(), "nope.jsonl")})
	if rc != 1 {
		t.Fatalf("missing ledger rc = %d, want 1", rc)
	}
	if !strings.Contains(errb.String(), "no ledger at") {
		t.Errorf("missing-ledger stderr = %q, want an actionable hint", errb.String())
	}
}
