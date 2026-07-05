package main

import (
	"strings"
	"testing"
)

// #2429 done-condition: `fak memory-read` is the session-start injection path —
// it must route every fact body through the same read-time trust gate `fak
// memory recall` applies, so a stale-claim note never enters the digest as fact.
// fixtureMemoryStore (memory_recall_test.go) exercises the REAL
// DefaultArtifactVerifier: fresh.md names a path that exists, stale.md names a
// deleted package.
func TestMemoryRead_withholdsStaleRendersFresh(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	code := runMemoryRead(&out, &errb, []string{"--store", dir})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "## Gate helper (fresh.md)") {
		t.Errorf("still-true note must render; got:\n%s", s)
	}
	if !strings.Contains(s, "The memory algebra executor lives in internal/memq/exec.go.") {
		t.Errorf("fresh note body must render verbatim; got:\n%s", s)
	}
	if !strings.Contains(s, "internal/gonepkg/gone.go") {
		t.Errorf("withheld footer must name the failing claim as evidence; got:\n%s", s)
	}
	if strings.Contains(s, "## Moved fix (stale.md)") {
		t.Errorf("stale note body must never render into the digest; got:\n%s", s)
	}
	if !strings.Contains(s, "withheld (never injected as fact):") || !strings.Contains(s, "stale.md") {
		t.Errorf("stale note must be named in a withheld footer, not silently dropped; got:\n%s", s)
	}
}

// --index-only must stay a raw MEMORY.md echo (untouched by the gate — the
// index text carries no fact bodies to withhold).
func TestMemoryRead_indexOnlyUnaffectedByGate(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	code := runMemoryRead(&out, &errb, []string{"--store", dir, "--index-only"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if strings.Contains(s, "internal/memq/exec.go") {
		t.Errorf("--index-only must not emit fact bodies; got:\n%s", s)
	}
	if !strings.Contains(s, "Gate helper") {
		t.Errorf("--index-only must still emit the MEMORY.md index text; got:\n%s", s)
	}
}
