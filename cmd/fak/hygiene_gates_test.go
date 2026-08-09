package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// cmd/fak/hygiene_gates_test.go — the #5604 contract: `fak hygiene --gates <name>` either names
// gates that exist or REFUSES, and a run that skipped a gate says so.
//
// The defect these pin: gateFilter used to upper-case the CSV into a want set with no membership
// check, so `--gates NOSUCHGATE` built a non-empty selection, matched no registered gate, ran
// ZERO checks, and exited 0 with an empty findings payload — output byte-identical to a genuine
// clean sweep over the whole tree. A lint whose failure mode is a green tick is worse than no
// lint, because a pass is the one outcome nobody investigates.

// TestGateFilter_UnknownNameRefused is the failure-first case: every shape of wrong name — a
// typo, a hyphenation, the singular of a plural — must be an error, not an empty selection.
func TestGateFilter_UnknownNameRefused(t *testing.T) {
	for _, csv := range []string{
		"NOSUCHGATE",
		"doc-placement",       // hyphenated, not the registered DOC_PLACEMENT
		"BROKEN_LINKS",        // plural of BROKEN_LINK
		"DOC_PLACEMENT,BOGUS", // one good, one bad — the bad one still refuses
		"INDEX_SYNC,,NOPE",    // empty entries skipped, the unknown still refuses
	} {
		want, err := gateFilter(csv)
		if err == nil {
			t.Errorf("gateFilter(%q) = %v, nil — an unknown gate name must refuse, not select %d gate(s)",
				csv, want, len(want))
			continue
		}
		if want != nil {
			t.Errorf("gateFilter(%q) returned a non-nil selection alongside its error: %v", csv, want)
		}
		// The refusal has to be actionable: it names the bad input AND the valid set.
		if !strings.Contains(err.Error(), "valid:") {
			t.Errorf("gateFilter(%q) error %q does not list the valid gate set", csv, err)
		}
	}
}

// TestGateFilter_EmptySelectionRefused pins the other half of "an empty gate set is never a
// silent pass": a value that parses to no names at all is a refusal, not a full sweep and not a
// zero-check pass.
func TestGateFilter_EmptySelectionRefused(t *testing.T) {
	for _, csv := range []string{",", " , ", ",,,"} {
		if want, err := gateFilter(csv); err == nil {
			t.Errorf("gateFilter(%q) = %v, nil — a selection that names no gate must refuse", csv, want)
		}
	}
}

// TestGateFilter_KnownNamesAccepted keeps the working path working: every registered gate name
// resolves, case-insensitively, including a DefaultOff ratchet (which is selectable by name even
// though it is skipped in the default sweep).
func TestGateFilter_KnownNamesAccepted(t *testing.T) {
	gates := hooks.HygieneGates()
	if len(gates) == 0 {
		t.Fatal("HygieneGates() is empty — this test would be vacuous")
	}
	for _, g := range gates {
		for _, form := range []string{g.Name, strings.ToLower(g.Name), " " + g.Name + " "} {
			want, err := gateFilter(form)
			if err != nil {
				t.Errorf("gateFilter(%q) refused a registered gate: %v", form, err)
				continue
			}
			if !want[g.Name] {
				t.Errorf("gateFilter(%q) did not select %s (got %v)", form, g.Name, want)
			}
		}
	}
}

// TestGateFilter_EmptyMeansAll keeps the default: no --gates is the whole sweep (nil want), not
// an empty selection.
func TestGateFilter_EmptyMeansAll(t *testing.T) {
	want, err := gateFilter("")
	if err != nil {
		t.Fatalf("gateFilter(\"\") errored: %v", err)
	}
	if want != nil {
		t.Errorf("gateFilter(\"\") = %v, want nil (nil means 'run every gate')", want)
	}
}

// TestRunHygiene_UnknownGateExitsNonZero drives the real command surface end to end. This is the
// case the issue reports: the whole point is that the exit code is NOT 0.
//
// It also pins the exit code AWAY from 2. Exit 2 means could-not-run, which is the code that
// sends `make hygiene` to its Python fallback — that sweep would run, pass, and bury the typo,
// reinstating the silent green. A usage refusal must hard-fail instead.
func TestRunHygiene_UnknownGateExitsNonZero(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runHygiene(&out, &errb, []string{"--gates", "NOSUCHGATE"})
	if rc == 0 {
		t.Fatalf("runHygiene(--gates NOSUCHGATE) exited 0 having run nothing\nstdout: %s\nstderr: %s",
			out.String(), errb.String())
	}
	if rc == 2 {
		t.Errorf("runHygiene(--gates NOSUCHGATE) exited 2 (could-not-run) — that routes make hygiene " +
			"to the Python fallback, which passes and hides the typo; want a hard refusal")
	}
	if !strings.Contains(errb.String(), "NOSUCHGATE") {
		t.Errorf("refusal does not name the offending gate; stderr: %s", errb.String())
	}
	if strings.Contains(out.String(), "hygiene OK") {
		t.Errorf("refused run still reported success on stdout: %s", out.String())
	}
}

// TestEmitHygieneJSON_ReportsSkips pins the second silent-green: a gate whose Check errored is
// fail-open (correct — one broken checker must not wedge the tree) but must be COUNTED and
// NAMED, under the same keys `fak hooks pre-commit --json` uses, so a consumer can tell "clean"
// from "clean because SECRET_SHAPE never ran".
func TestEmitHygieneJSON_ReportsSkips(t *testing.T) {
	var out, errb bytes.Buffer
	emitHygieneJSON(&out, &errb, nil, nil, []string{"SECRET_SHAPE", "BROKEN_LINK"}, runScope{Population: scopePopulationTree})

	var got struct {
		Count        int      `json:"count"`
		SkippedGates []string `json:"skipped_gates"`
		SkippedCount int      `json:"skipped_count"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("emitHygieneJSON did not emit valid JSON: %v\n%s", err, out.String())
	}
	if got.SkippedCount != 2 {
		t.Errorf("skipped_count = %d, want 2", got.SkippedCount)
	}
	if len(got.SkippedGates) != 2 || got.SkippedGates[0] != "SECRET_SHAPE" {
		t.Errorf("skipped_gates = %v, want [SECRET_SHAPE BROKEN_LINK]", got.SkippedGates)
	}
	// The degraded payload must remain distinguishable from a clean one even though BOTH carry
	// zero findings — that indistinguishability is the whole defect.
	if got.Count != 0 {
		t.Errorf("count = %d, want 0 for this fixture", got.Count)
	}
}

// TestEmitHygieneJSON_CleanRunReportsNoSkips is the other side of the pair: a clean run emits the
// keys with empty/zero values (never absent, never null), so a consumer can parse one shape.
func TestEmitHygieneJSON_CleanRunReportsNoSkips(t *testing.T) {
	var out, errb bytes.Buffer
	emitHygieneJSON(&out, &errb, nil, nil, nil, runScope{Population: scopePopulationTree})

	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("emitHygieneJSON did not emit valid JSON: %v\n%s", err, out.String())
	}
	for _, k := range []string{"skipped_gates", "skipped_count"} {
		if _, ok := got[k]; !ok {
			t.Errorf("clean run omitted %q — the key must always be present so one shape parses", k)
		}
	}
	if got["skipped_gates"] == nil {
		t.Error("skipped_gates is null on a clean run; want an empty array")
	}
}
