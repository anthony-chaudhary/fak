package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/assumecheck"
)

// TestAssumeWiringMatchesDeclaredStatus binds the shell's witness-dispatch table to
// the registry's declared per-row WitnessStatus: a row is marked wired if and only
// if this shell actually has a gatherer for it, and every gatherer key is a
// registered id. This is what keeps the declared marker from drifting into a lie
// when C3 (#3821) adds or moves drivers.
func TestAssumeWiringMatchesDeclaredStatus(t *testing.T) {
	for _, a := range assumecheck.Registry() {
		_, wired := assumeWitnessGatherers[a.ID]
		if wired != (a.WitnessStatus == assumecheck.WitnessWired) {
			t.Fatalf("assumption %q declares witness status %q but shell wiring is %v", a.ID, a.WitnessStatus, wired)
		}
	}
	for id := range assumeWitnessGatherers {
		if _, ok := assumecheck.Lookup(id); !ok {
			t.Fatalf("witness gatherer wired for unregistered assumption %q", id)
		}
	}
}

// TestAssumeCheckDeclaredOnlyIsUnverifiable proves the proof-by-default posture end
// to end: checking a registered-but-unwired row yields UNVERIFIABLE (exit 4) with
// the wiring gap named as the explanation — never a fabricated HOLDS. Runs entirely
// off the declared registry; a declared-only row touches no disk or roster.
func TestAssumeCheckDeclaredOnlyIsUnverifiable(t *testing.T) {
	for _, a := range assumecheck.Registry() {
		if a.WitnessStatus != assumecheck.WitnessDeclaredOnly {
			continue
		}
		var out, errBuf bytes.Buffer
		code := runAssume(&out, &errBuf, []string{"check", a.ID})
		if code != 4 {
			t.Fatalf("check %q: exit=%d (stderr=%q), want 4 (UNVERIFIABLE, fail-closed)", a.ID, code, errBuf.String())
		}
		s := out.String()
		if !strings.Contains(s, string(assumecheck.OutcomeUnverifiable)) {
			t.Fatalf("check %q output carries no UNVERIFIABLE outcome:\n%s", a.ID, s)
		}
		if strings.Contains(s, "outcome    : "+string(assumecheck.OutcomeHolds)) {
			t.Fatalf("check %q fabricated a HOLDS for a declared-only row:\n%s", a.ID, s)
		}
		if !strings.Contains(s, "declared-only") {
			t.Fatalf("check %q does not explain the declared-only wiring gap:\n%s", a.ID, s)
		}
	}
}

// TestAssumeListEnumeratesRegistry proves `fak assume list` shows every registered
// row and distinguishes wired from declared-only wiring.
func TestAssumeListEnumeratesRegistry(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runAssume(&out, &errBuf, []string{"list"}); code != 0 {
		t.Fatalf("list: exit=%d (stderr=%q)", code, errBuf.String())
	}
	s := out.String()
	for _, a := range assumecheck.Registry() {
		if !strings.Contains(s, a.ID) {
			t.Fatalf("list omits registered assumption %q:\n%s", a.ID, s)
		}
	}
	if !strings.Contains(s, string(assumecheck.WitnessWired)) || !strings.Contains(s, string(assumecheck.WitnessDeclaredOnly)) {
		t.Fatalf("list does not distinguish wired from declared-only wiring:\n%s", s)
	}
}

// TestAssumeCheckUnknownIDNamesTheMenu proves an unknown id is a usage error (exit
// 2) that names the known ids from the registry, not a guessed check.
func TestAssumeCheckUnknownIDNamesTheMenu(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runAssume(&out, &errBuf, []string{"check", "no-such-assumption"}); code != 2 {
		t.Fatalf("unknown id: exit=%d, want 2", code)
	}
	for _, a := range assumecheck.Registry() {
		if !strings.Contains(errBuf.String(), a.ID) {
			t.Fatalf("unknown-id usage error omits known id %q:\n%s", a.ID, errBuf.String())
		}
	}
}
