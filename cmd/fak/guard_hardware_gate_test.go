package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hwgatelint"
)

// guard_hardware_gate_test.go — the dedicated "local machine is the compute boundary" hook:
// runGuardHardwareGateGate reads the final assistant turn itself, folds it through
// internal/hwgatelint, and applies the off|shadow|warn|enforce ladder. These tests pin the
// self-contained transcript read, the ladder decisions and the fixed sanctioned-route redirect,
// the env inheritance from the operator-directed posture, and the precedence over the
// operator-directed gate.

// TestNormalizeGuardHardwareGateMode covers the closed vocabulary: empty defaults to the warn
// soak, every named rung round-trips, and an unknown value is an error callers treat as fail-open.
func TestNormalizeGuardHardwareGateMode(t *testing.T) {
	for _, tc := range []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", guardHardwareGateModeWarn, false},
		{"  ", guardHardwareGateModeWarn, false},
		{"WARN", guardHardwareGateModeWarn, false},
		{"off", guardPreCompactModeOff, false},
		{"Shadow", guardPreCompactModeShadow, false},
		{"enforce", guardPreCompactModeEnforce, false},
		{"loud", "", true},
	} {
		got, err := normalizeGuardHardwareGateMode(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("normalize(%q): want error, got %q", tc.in, got)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("normalize(%q) = %q, %v; want %q, nil", tc.in, got, err, tc.want)
		}
	}
	// The env-boundary total form never errors and pins a bad string to warn (advisory), never enforce.
	if got := guardHardwareGateNormalizedOrWarn("loud"); got != guardHardwareGateModeWarn {
		t.Errorf("guardHardwareGateNormalizedOrWarn(bad) = %q, want warn", got)
	}
	if got := guardHardwareGateNormalizedOrWarn("enforce"); got != guardPreCompactModeEnforce {
		t.Errorf("guardHardwareGateNormalizedOrWarn(enforce) = %q, want enforce", got)
	}
}

// sampleHardwareGateFinding is the finding a local-GPU blocker turn produces — the case enforce
// should BLOCK and redirect to a sanctioned compute node.
func sampleHardwareGateFinding() hwgatelint.Finding {
	return hwgatelint.Finding{
		Class: "NO_LOCAL_GPU",
		Node:  "GCP `fak-realmodel` (L4 sm_89) or the DGX via `dgxbridge`",
	}
}

// TestGuardHardwareGateDecide pins the pure ladder off a fired finding. Unlike the
// operator-directed gate there is no HUMAN_RESIDUAL carve-out: every hardware-gated stop is
// redirectable (the fleet always offers a node or an operator handoff), so enforce always blocks.
func TestGuardHardwareGateDecide(t *testing.T) {
	f := sampleHardwareGateFinding()
	for _, tc := range []struct {
		name     string
		mode     string
		wantExit int
		wantDisp guardStopDisposition
		wantSub  string // stderr substring
	}{
		{"shadow allows", guardPreCompactModeShadow, 0, stopDispHardwareGateShadow, "shadow"},
		{"warn allows + soaks", guardHardwareGateModeWarn, 0, stopDispHardwareGateWarn, "soak mode"},
		{"enforce blocks + redirects", guardPreCompactModeEnforce, 2, stopDispHardwareGateContinue, "control point"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr strings.Builder
			exit, disp, fired := guardHardwareGateDecide(&stderr, tc.mode, f)
			if exit != tc.wantExit || disp != tc.wantDisp || !fired {
				t.Fatalf("decide(%q) = exit %d disp %q fired %v; want %d/%q/true",
					tc.mode, exit, disp, fired, tc.wantExit, tc.wantDisp)
			}
			if !strings.Contains(stderr.String(), tc.wantSub) {
				t.Errorf("stderr missing %q:\n%s", tc.wantSub, stderr.String())
			}
		})
	}
}

// TestRunGuardHardwareGateGateInert pins the fail-open non-firing cases: off, a bad mode string,
// and an empty transcript path all leave the gate inert (exit 0, no disposition, not fired).
func TestRunGuardHardwareGateGateInert(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode string
		path string
	}{
		{"off is inert", guardPreCompactModeOff, "does-not-exist.jsonl"},
		{"bad mode fails open", "loud", "does-not-exist.jsonl"},
		{"empty path is inert", guardPreCompactModeEnforce, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr strings.Builder
			exit, disp, fired := runGuardHardwareGateGate(&stderr, tc.mode, tc.path)
			if exit != 0 || disp != "" || fired {
				t.Fatalf("inert case fired: exit %d disp %q fired %v", exit, disp, fired)
			}
		})
	}
}

// TestRunGuardHardwareGateGateReadsTranscript proves the gate is self-contained: it reads the final
// prose turn from the fixture itself and scans it — a hardware-gated turn enforces (exit 2), a clean
// turn stays inert, and a final turn that STILL called a tool is not a stopping-to-declare-a-blocker.
func TestRunGuardHardwareGateGateReadsTranscript(t *testing.T) {
	dir := t.TempDir()

	// Gated: the final prose turn declares a local-GPU blocker → enforce BLOCKS and redirects.
	gated := filepath.Join(dir, "gated.jsonl")
	writeStopTranscriptFixture(t, gated,
		`{"type":"assistant","message":{"role":"assistant","content":"Wired the loader. There is no GPU on this host, so I can't run the CUDA device witness."}}`,
	)
	var stderr strings.Builder
	exit, disp, fired := runGuardHardwareGateGate(&stderr, guardPreCompactModeEnforce, filepath.ToSlash(gated))
	if exit != 2 || disp != stopDispHardwareGateContinue || !fired {
		t.Fatalf("gated enforce = exit %d disp %q fired %v; want 2/continue/true; stderr=%s", exit, disp, fired, stderr.String())
	}
	if !strings.Contains(stderr.String(), "control point") {
		t.Errorf("redirect missing from gated enforce:\n%s", stderr.String())
	}

	// Clean: a final turn with no hardware blocker → inert (allow the stop).
	clean := filepath.Join(dir, "clean.jsonl")
	writeStopTranscriptFixture(t, clean,
		`{"type":"assistant","message":{"role":"assistant","content":"All tests pass. Shipped the refactor and committed."}}`,
	)
	var stderr2 strings.Builder
	if exit2, _, fired2 := runGuardHardwareGateGate(&stderr2, guardPreCompactModeEnforce, filepath.ToSlash(clean)); exit2 != 0 || fired2 {
		t.Fatalf("clean enforce = exit %d fired %v; want 0/false; stderr=%s", exit2, fired2, stderr2.String())
	}
}

// TestGuardHardwareGateContinueInjectsNode pins that a BLOCKED hardware-gated stop injects the
// finding's natural node and the fixed redirect back to the model, and that the defensive fallbacks
// yield a usable message when a finding leaves Class/Node blank (no empty "()" or dangling clause).
func TestGuardHardwareGateContinueInjectsNode(t *testing.T) {
	f := sampleHardwareGateFinding()
	var stderr strings.Builder
	exit, disp, fired := guardHardwareGateDecide(&stderr, guardPreCompactModeEnforce, f)
	if exit != 2 || disp != stopDispHardwareGateContinue || !fired {
		t.Fatalf("enforce hardware-gated = exit %d disp %q fired %v; want 2/continue/true", exit, disp, fired)
	}
	for _, want := range []string{f.Node, string(f.Class), "Route to", "no allowed path"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("continue message missing %q:\n%s", want, stderr.String())
		}
	}

	// Defensive fallbacks: an unclassified finding must still produce a usable message — the generic
	// class label and the generic fleet-menu node — never an empty parenthetical.
	var stderr2 strings.Builder
	guardHardwareGateDecide(&stderr2, guardPreCompactModeEnforce, hwgatelint.Finding{})
	msg := stderr2.String()
	if !strings.Contains(msg, "(local-hardware)") {
		t.Errorf("blank Class did not fall back to the generic label:\n%s", msg)
	}
	if !strings.Contains(msg, "sanctioned compute node for the task") {
		t.Errorf("blank Node did not fall back to the generic fleet menu:\n%s", msg)
	}
	if strings.Contains(msg, "()") {
		t.Errorf("blank fields produced an empty parenthetical:\n%s", msg)
	}
}

// TestRunGuardStopHookHardwareGateEnforceBlocks is the end-to-end enforce path: a clean stop whose
// final turn declares "no GPU on this host" is BLOCKED (exit 2) so the agent gets the
// sanctioned-compute-node redirect and dispatches the work instead of stopping at the local boundary.
func TestRunGuardStopHookHardwareGateEnforceBlocks(t *testing.T) {
	srv := stopHookQuietGauge(t)
	dir := t.TempDir()
	fixture := filepath.Join(dir, "nogpu.jsonl")
	writeStopTranscriptFixture(t, fixture,
		`{"type":"user","message":{"role":"user","content":"run the device witness"}}`,
		`{"type":"assistant","message":{"role":"assistant","content":"Wired the loader. There is no GPU on this host, so I can't run the CUDA device witness."}}`,
	)
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"transcript_path":"`+filepath.ToSlash(fixture)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--hardware-gate", guardPreCompactModeEnforce,
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (headless hardware-gated stop is blocked); stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"control point", "Do not stop for lack of local hardware", "no allowed path"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("redirect guidance missing %q:\n%s", want, stderr.String())
		}
	}
}

// TestRunGuardStopHookHardwareGateWarnSoaks proves the shipped default rung: warn prints the
// would-enforce redirect for the operator but still ALLOWS the stop (exit 0), so the pathology can
// soak before promotion to enforce.
func TestRunGuardStopHookHardwareGateWarnSoaks(t *testing.T) {
	srv := stopHookQuietGauge(t)
	dir := t.TempDir()
	fixture := filepath.Join(dir, "nogpu.jsonl")
	writeStopTranscriptFixture(t, fixture,
		`{"type":"assistant","message":{"role":"assistant","content":"Done. Can't run the CUDA suite — no GPU on this laptop."}}`,
	)
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"transcript_path":"`+filepath.ToSlash(fixture)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--hardware-gate", guardHardwareGateModeWarn,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (warn soak allows the stop); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "[warn]") || !strings.Contains(stderr.String(), "soak mode") {
		t.Errorf("warn soak line missing:\n%s", stderr.String())
	}
}

// TestRunGuardStopHookHardwareGatePrecedesOperatorDirected proves the precedence wiring: a final
// turn that is BOTH hardware-gated ("no GPU here") AND operator-directed ("do you want me to…") is
// caught by the hardware gate FIRST — the more specific "wrong machine" misroute with a fixed remedy
// wins over the generic "act on your own question" guidance. Both gates run in enforce; the redirect
// (not the choicetriage remediation) is what reaches the model.
func TestRunGuardStopHookHardwareGatePrecedesOperatorDirected(t *testing.T) {
	srv := stopHookQuietGauge(t)
	dir := t.TempDir()
	fixture := filepath.Join(dir, "both.jsonl")
	writeStopTranscriptFixture(t, fixture,
		`{"type":"assistant","message":{"role":"assistant","content":"There is no GPU on this host. Do you want me to try running it somewhere else?"}}`,
	)
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"transcript_path":"`+filepath.ToSlash(fixture)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--hardware-gate", guardPreCompactModeEnforce,
		"--operator-directed", guardPreCompactModeEnforce,
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (blocked); stderr=%s", code, stderr.String())
	}
	// The hardware-gate redirect won: its signature line is present, the operator-directed one is not.
	if !strings.Contains(stderr.String(), "control point") {
		t.Errorf("expected the hardware-gate redirect to win:\n%s", stderr.String())
	}
	if strings.Contains(stderr.String(), "no operator to answer") {
		t.Errorf("operator-directed message should not fire when the hardware gate already blocked:\n%s", stderr.String())
	}
}

// TestInstallGuardStopHookInjectsHardwareGateEnv pins that the hardware-gate mode is threaded into
// the Stop-hook child, inheriting the RESOLVED operator-directed posture (both are operator-absent-
// capped headless gates), and that a bad string is pinned to warn (advisory), never enforce.
func TestInstallGuardStopHookInjectsHardwareGateEnv(t *testing.T) {
	dir := t.TempDir()
	_, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, guardPreCompactModeEnforce)
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	got := ""
	for _, kv := range env {
		if kv[0] == guardStopHookHardwareGateEnvMode {
			got = kv[1]
		}
	}
	if got != guardPreCompactModeEnforce {
		t.Fatalf("hardware-gate env = %q, want enforce (inherited from operator-directed posture)", got)
	}
	// A bad operator-directed mode string caps BOTH gates to warn.
	_, env2, _, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, "loud")
	if err != nil {
		t.Fatalf("install(bad mode): %v", err)
	}
	got2 := ""
	for _, kv := range env2 {
		if kv[0] == guardStopHookHardwareGateEnvMode {
			got2 = kv[1]
		}
	}
	if got2 != guardHardwareGateModeWarn {
		t.Fatalf("bad-mode hardware-gate env = %q, want warn", got2)
	}
}
