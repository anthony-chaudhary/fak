package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// hooks_precommit_skipped_test.go — #5299: a pre-commit gate whose Check returns an error is
// SKIPPED (fail-open, and that stays: a broken checker must never wedge every commit on a shared
// trunk), but the skip must not be SILENT. Before this, "all gates ran clean" and "PUBLIC_LEAK
// errored and the rest ran clean" produced byte-identical output and the same exit 0, so a
// persistently-broken security gate could go unnoticed indefinitely. These tests pin the three
// halves of the done condition together: the gate is NAMED, the skip is COUNTED (on stderr and
// in --json), and the exit code still says 0.

// gateEvidenceMarker stands in for whatever a scanning gate had in hand when it failed — the
// PUBLIC_LEAK / SECRET_SHAPE gates read the staged diff hunting for secret material, so their
// error strings are exactly the thing that must never be echoed into a commit log. The report
// prints the gate NAME and an error CLASS only; these tests assert the marker never surfaces.
const gateEvidenceMarker = "do-not-print-this-gate-evidence"

// withPreCommitGates swaps the hook's gate set for the duration of one test. Injection is the
// only way to witness a could-not-run gate: no real gate errors on a healthy fixture repo.
func withPreCommitGates(t *testing.T, gs ...hooks.Gate) {
	t.Helper()
	prev := preCommitGates
	preCommitGates = func() []hooks.Gate { return gs }
	t.Cleanup(func() { preCommitGates = prev })
}

// cannotRunGate is a gate that cannot reach its evidence, wrapping the real ErrCouldNotRun
// sentinel and carrying evidence-shaped text in its message.
func cannotRunGate(name string) hooks.Gate {
	return hooks.Gate{Name: name, Check: func(*hooks.StagedDiff) ([]hooks.Finding, error) {
		return nil, fmt.Errorf("reading staged evidence %s: %w", gateEvidenceMarker, hooks.ErrCouldNotRun)
	}}
}

// cleanGate is a gate that runs and finds nothing.
func cleanGate(name string) hooks.Gate {
	return hooks.Gate{Name: name, Check: func(*hooks.StagedDiff) ([]hooks.Finding, error) { return nil, nil }}
}

func TestPreCommitCouldNotRunGateIsNamedCountedAndStillExitsZero(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	withPreCommitGates(t, cannotRunGate("CANARY_SCAN"), cleanGate("CANARY_CLEAN"))

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo})
	report := errb.String()

	t.Run("exit_code_stays_0", func(t *testing.T) {
		// The half most likely to regress: reporting the skip must not become a refusal.
		if code != 0 {
			t.Fatalf("a gate that could not run must not block the commit: exit %d; stderr=%s", code, report)
		}
	})
	t.Run("stderr_names_the_skipped_gate", func(t *testing.T) {
		if !strings.Contains(report, "CANARY_SCAN") {
			t.Fatalf("stderr must name the gate that was skipped; got %q", report)
		}
		if !strings.Contains(report, "could not run") {
			t.Fatalf("stderr must say the gate could not run; got %q", report)
		}
	})
	t.Run("stderr_carries_a_nonzero_skipped_count", func(t *testing.T) {
		if !strings.Contains(report, "1 of 2 gate(s) skipped") {
			t.Fatalf("stderr must carry the skipped count so a degraded run is legible; got %q", report)
		}
	})
	t.Run("stderr_never_carries_the_gate_error_text", func(t *testing.T) {
		// A failing PUBLIC_LEAK / SECRET_SHAPE gate can hold matched secret material in its
		// error; the report must stay a gate name plus a fixed class string.
		if strings.Contains(report, gateEvidenceMarker) {
			t.Fatalf("the skip report leaked the gate's error text into stderr; got %q", report)
		}
	})
	t.Run("the_other_gates_still_run", func(t *testing.T) {
		// Fail-open means SKIP THE ONE, not abandon the rest.
		if strings.Contains(report, "CANARY_CLEAN") {
			t.Fatalf("the healthy gate must not be reported as skipped; got %q", report)
		}
	})
}

func TestPreCommitJSONCarriesTheSkippedGateLedger(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	withPreCommitGates(t, cannotRunGate("CANARY_SCAN"), cleanGate("CANARY_CLEAN"))

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo, "--json"})
	var payload struct {
		Count        int      `json:"count"`
		SkippedGates []string `json:"skipped_gates"`
		SkippedCount int      `json:"skipped_count"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("--json output did not parse: %v; stdout=%s", err, out.String())
	}

	t.Run("exit_code_stays_0", func(t *testing.T) {
		if code != 0 {
			t.Fatalf("a gate that could not run must not block the commit: exit %d; stderr=%s", code, errb.String())
		}
	})
	t.Run("skipped_count_is_nonzero", func(t *testing.T) {
		if payload.SkippedCount != 1 {
			t.Fatalf("skipped_count = %d, want 1; stdout=%s", payload.SkippedCount, out.String())
		}
	})
	t.Run("skipped_gates_names_the_gate", func(t *testing.T) {
		if len(payload.SkippedGates) != 1 || payload.SkippedGates[0] != "CANARY_SCAN" {
			t.Fatalf("skipped_gates = %v, want [CANARY_SCAN]", payload.SkippedGates)
		}
	})
	t.Run("stdout_never_carries_the_gate_error_text", func(t *testing.T) {
		if strings.Contains(out.String(), gateEvidenceMarker) {
			t.Fatalf("the JSON ledger leaked the gate's error text; got %s", out.String())
		}
	})
}

// TestEnabledGateNamesExcludesOperatorDisabledGates pins the one judgment the skip ledger has to
// make: a gate an operator turned OFF (or escaped for this commit) is intent, not degradation,
// and must never inflate the skipped count — otherwise every commit with a softened gate would
// read as a degraded run and the signal would be worthless. This is the helper the hook uses to
// name the tail of gates a spent wall-clock budget drops.
func TestEnabledGateNamesExcludesOperatorDisabledGates(t *testing.T) {
	t.Setenv("FAK_TEST_CANARY_OFF_GUARD", "off")
	t.Setenv("FAK_TEST_CANARY_ESCAPE", "1")
	gs := []hooks.Gate{
		{Name: "CANARY_ON"},
		{Name: "CANARY_OFF", ModeEnv: "FAK_TEST_CANARY_OFF_GUARD"},
		{Name: "CANARY_ESCAPED", EscapeEnv: "FAK_TEST_CANARY_ESCAPE"},
		{Name: "CANARY_ADVISORY", DefaultMode: "warn"}, // warn still RUNS, so it still counts
	}
	got := enabledGateNames(gs)
	want := []string{"CANARY_ON", "CANARY_ADVISORY"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("enabledGateNames = %v, want %v", got, want)
	}
}

func TestPreCommitFullGateRunReportsNoSkips(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	withPreCommitGates(t, cleanGate("CANARY_ONE"), cleanGate("CANARY_TWO"))

	t.Run("human_report_stays_quiet", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo}); code != 0 {
			t.Fatalf("a clean gate set must pass: exit %d; stderr=%s", code, errb.String())
		}
		if strings.Contains(errb.String(), "skipped") {
			t.Fatalf("a run where every gate delivered a verdict must report no skips; got %q", errb.String())
		}
	})
	t.Run("json_ledger_is_present_and_zero", func(t *testing.T) {
		// Always-present keys: a consumer must never have to read "absent" as "none".
		var out, errb bytes.Buffer
		if code := runHooks(&out, &errb, []string{"pre-commit", "--root", repo, "--json"}); code != 0 {
			t.Fatalf("a clean gate set must pass: exit %d; stderr=%s", code, errb.String())
		}
		var payload map[string]any
		if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
			t.Fatalf("--json output did not parse: %v; stdout=%s", err, out.String())
		}
		got, ok := payload["skipped_count"]
		if !ok {
			t.Fatalf("skipped_count must be present on a full run; stdout=%s", out.String())
		}
		if n, isNum := got.(float64); !isNum || n != 0 {
			t.Fatalf("skipped_count = %v, want 0; stdout=%s", got, out.String())
		}
		if _, ok := payload["skipped_gates"]; !ok {
			t.Fatalf("skipped_gates must be present on a full run; stdout=%s", out.String())
		}
	})
}
