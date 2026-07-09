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
//
// It binds BOTH halves of "known, refusable": the [reasons.<TOKEN>] table must
// exist AND declare refusal = true, and cite issue #3066 in see_also so the
// registration stays traceable to this gate. A header alone (e.g. left with
// refusal = false, or copied without the #3066 provenance) would make
// dos_check_reason resolve the token as non-refusable — a silent regression the
// header-presence check could not catch.
func TestKnownSkipReasonsRegisteredInDosToml(t *testing.T) {
	content := readRepoDosToml(t)
	for _, r := range KnownSkipReasons() {
		header := "[reasons." + string(r) + "]"
		if !strings.Contains(content, header) {
			t.Errorf("SkipReason %q has no %s table in dos.toml — dos_check_reason would return known=false (UNCLASSIFIED drift)", r, header)
			continue
		}
		block := dosReasonBlock(content, header)
		if !reasonFieldTrue(block, "refusal") {
			t.Errorf("SkipReason %q is registered but not marked refusal = true — dos_check_reason would resolve it as non-refusable", r)
		}
		if !strings.Contains(block, "issue #3066") {
			t.Errorf("SkipReason %q registration does not cite issue #3066 in its table — the gate provenance is unbound", r)
		}
	}
}

// dosReasonBlock returns the text of the [reasons.<TOKEN>] table named by header:
// from the header line up to (but excluding) the next top-level [section] or EOF.
// This lets the binding assertions scope to a single reason's fields rather than
// matching a sibling table's refusal/summary by accident.
func dosReasonBlock(content, header string) string {
	i := strings.Index(content, header)
	if i < 0 {
		return ""
	}
	rest := content[i+len(header):]
	if j := strings.Index(rest, "\n["); j >= 0 {
		return content[i : i+len(header)+j]
	}
	return content[i:]
}

// reasonFieldTrue reports whether block contains a `field = true` line, tolerant
// of the aligned whitespace the dos.toml [reasons] tables use (e.g. "refusal  = true").
func reasonFieldTrue(block, field string) bool {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, field))
		if strings.HasPrefix(rest, "=") && strings.TrimSpace(rest[1:]) == "true" {
			return true
		}
	}
	return false
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
