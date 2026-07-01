package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/claimcheck"
)

// TestClaimCheckWitnessFor drives the #2153 scaffold verb end to end: a claim of each
// headline class yields exit 0 with a plan naming the class, the cheapest command, and
// (for the test-proof classes) a skeleton; --json round-trips the WitnessPlan.
func TestClaimCheckWitnessFor(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runClaimCheck(&out, &errb, strings.NewReader(""), []string{"--witness-for", "shipped the resolver fix and pushed to main"}); code != 0 {
		t.Fatalf("witness-for exit = %d, want 0 (stderr=%q)", code, errb.String())
	}
	if s := out.String(); !strings.Contains(s, "class:     shipped") || !strings.Contains(s, "dos verify") {
		t.Fatalf("shipped plan missing from output:\n%s", s)
	}

	out.Reset()
	errb.Reset()
	if code := runClaimCheck(&out, &errb, strings.NewReader(""), []string{"--witness-for", "the pane renders ANSI garbage on resize", "--json"}); code != 0 {
		t.Fatalf("witness-for --json exit = %d, want 0 (stderr=%q)", code, errb.String())
	}
	var plan claimcheck.WitnessPlan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("plan JSON: %v\n%s", err, out.String())
	}
	if plan.Class != claimcheck.WitnessVisual || plan.Skeleton == "" || plan.Command == "" {
		t.Fatalf("visual plan = %+v, want visual class with skeleton + command", plan)
	}
}

// TestClaimCheckWitnessSelfTest pins the corpus witness: the built-in
// one-claim-per-class fixture grades clean, exit 0.
func TestClaimCheckWitnessSelfTest(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runClaimCheck(&out, &errb, strings.NewReader(""), []string{"--witness-self-test"}); code != 0 {
		t.Fatalf("witness-self-test exit = %d, want 0\n%s%s", code, out.String(), errb.String())
	}
	if s := out.String(); !strings.Contains(s, "classified as expected") {
		t.Fatalf("self-test summary missing:\n%s", s)
	}
}
