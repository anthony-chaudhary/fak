package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// TestKnownFrictionTouchedTrees: only file-MUTATING tool events contribute a
// tree, absolute paths are made repo-relative, Grep/Bash/Read operands are
// dropped, and the result is deduped. The host OS builds the absolute paths so
// filepath.IsAbs/Rel behave natively on Windows and POSIX alike.
func TestKnownFrictionTouchedTrees(t *testing.T) {
	root := t.TempDir()
	abs := func(rel string) string { return filepath.Join(root, filepath.FromSlash(rel)) }
	events := []trajctl.ToolEvent{
		{Tool: "Edit", Target: abs("internal/terminalbench/packet_pins_test.go")},
		{Tool: "Edit", Target: abs("internal/terminalbench/packet_pins_test.go")}, // dup -> collapses
		{Tool: "Write", Target: "cmd/fak/guard_stophook.go"},                      // already repo-relative
		{Tool: "Grep", Target: "func Match"},                                      // pattern, not a tree
		{Tool: "Bash", Target: "go test ./internal/terminalbench/"},               // command, not a tree
		{Tool: "Read", Target: abs("internal/knownbad/knownbad.go")},              // browse, not a modification
		{Tool: "Edit", Target: ""},                                                // empty operand
	}
	got := knownFrictionTouchedTrees(events, root)
	want := []string{"internal/terminalbench/packet_pins_test.go", "cmd/fak/guard_stophook.go"}
	if len(got) != len(want) {
		t.Fatalf("touched = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("touched[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

// TestKnownFrictionAdvisoryLineFires: touched trees that intersect a LIVE
// signature produce a heads-up naming that signature; the line is advisory and
// carries the reason so the agent can act on it.
func TestKnownFrictionAdvisoryLineFires(t *testing.T) {
	rec := knownbad.NewRecord("test", []string{"internal/terminalbench/**"}, "peer changed packet doc w/o refreshing hash pin", "agentX", "", 1000, 3600)
	touched := []string{"internal/terminalbench/packet_pins_test.go"}
	line := knownFrictionAdvisoryLine([]knownbad.Record{rec}, touched, 1500) // within ttl -> live
	if line == "" {
		t.Fatal("expected an advisory line for a touched tree intersecting a live signature, got empty")
	}
	for _, want := range []string{rec.Signature, "reason=test", "known-bad", "Advisory only", "don't re-fix"} {
		if !strings.Contains(line, want) {
			t.Errorf("advisory line missing %q\nline: %s", want, line)
		}
	}
}

// TestKnownFrictionAdvisoryLineSilent: the three fail-silent gates -- no touched
// trees, a touched tree that intersects nothing live, and an EXPIRED signature
// (its bounded TTL lapsed) -- each yield "".
func TestKnownFrictionAdvisoryLineSilent(t *testing.T) {
	live := knownbad.NewRecord("test", []string{"internal/terminalbench/**"}, "n", "a", "", 1000, 3600)
	expired := knownbad.NewRecord("test", []string{"internal/terminalbench/**"}, "n", "a", "", 1000, 100) // dead by 1100

	if got := knownFrictionAdvisoryLine([]knownbad.Record{live}, nil, 1500); got != "" {
		t.Errorf("no touched trees should be silent, got: %s", got)
	}
	if got := knownFrictionAdvisoryLine([]knownbad.Record{live}, []string{"cmd/fak/orient.go"}, 1500); got != "" {
		t.Errorf("non-intersecting touched tree should be silent, got: %s", got)
	}
	if got := knownFrictionAdvisoryLine([]knownbad.Record{expired}, []string{"internal/terminalbench/x.go"}, 2000); got != "" {
		t.Errorf("expired signature should be silent, got: %s", got)
	}
}

// TestKnownFrictionAdvisoryLineCaps: more live matches than the cap names the
// first knownFrictionAdvisoryCap and summarises the rest as a trailing count, so
// the heads-up stays one bounded line.
func TestKnownFrictionAdvisoryLineCaps(t *testing.T) {
	trees := []string{"a", "b", "c", "d"}
	records := make([]knownbad.Record, 0, len(trees))
	touched := make([]string, 0, len(trees))
	for _, tr := range trees {
		records = append(records, knownbad.NewRecord("test", []string{tr + "/**"}, "n", "a", "", 1000, 3600))
		touched = append(touched, tr+"/x.go")
	}
	line := knownFrictionAdvisoryLine(records, touched, 1500)
	if !strings.Contains(line, "4 LIVE known-bad") {
		t.Errorf("expected a total count of 4, line: %s", line)
	}
	if !strings.Contains(line, "+1 more") {
		t.Errorf("expected %d named + trailing '+1 more', line: %s", knownFrictionAdvisoryCap, line)
	}
}
