package toon

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestKnownSkipReasonsRegisteredInDosToml is the structural half of #3066's
// Witness DoD: every member of the closed skip-reason vocabulary must be declared
// in the workspace dos.toml [reasons] table, so `dos_check_reason` resolves each
// token (known, refusable) and `dos_refuse_reasons` lists it. KnownSkipReasons is
// the producer of record; this test fails loudly if a reason is added to the Go
// constant set without a matching [reasons.<TOKEN>] registration (the drift that
// otherwise silently surfaces as an UNCLASSIFIED reason at runtime).
func TestKnownSkipReasonsRegisteredInDosToml(t *testing.T) {
	content := readRepoDosToml(t)
	for _, r := range KnownSkipReasons() {
		header := "[reasons." + string(r) + "]"
		if !strings.Contains(content, header) {
			t.Errorf("SkipReason %q has no %s table in dos.toml — dos_check_reason would return known=false (UNCLASSIFIED drift)", r, header)
		}
	}
}

// readRepoDosToml reads the repo-root dos.toml located relative to this test's
// own source path (internal/toon), so the lookup is independent of the test's
// working directory.
func readRepoDosToml(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}
