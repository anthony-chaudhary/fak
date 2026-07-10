package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// guard_stops_headlesslint_test.go — the guard-side headlesslint fold: a Stop hook
// that reads the final assistant turn now records whether that turn ended by asking
// a human instead of acting. These tests pin the sensor (applyHeadlessLintSignal),
// the transcript-reader wiring (including the tool-use gate), and the operator-facing
// tally, all observe-only and fail-open.

// TestApplyHeadlessLintSignal covers the sensor directly: a clean final turn is a
// no-op, a permission ask populates the top finding with its choicetriage
// remediation, and an authority-bearing ask is routed to HUMAN_RESIDUAL (a genuine
// escalation, never suppressed as an obvious action).
func TestApplyHeadlessLintSignal(t *testing.T) {
	// Clean turn: nothing operator-directed -> every field stays zero.
	clean := &guardStopTranscript{Read: true}
	applyHeadlessLintSignal(clean, "All tests pass. Committed and pushed. Done.")
	if clean.OperatorDirected || clean.OperatorDirectedCount != 0 || clean.OperatorDirectedClass != "" {
		t.Errorf("clean turn should be a no-op, got %+v", clean)
	}

	// Permission ask: the obvious next step, phrased as a question. Class is the
	// linguistic shape; the choicetriage disposition is TAKE_OBVIOUS (just do it).
	perm := &guardStopTranscript{Read: true}
	applyHeadlessLintSignal(perm, "The change is ready. Do you want me to push it?")
	if !perm.OperatorDirected {
		t.Fatalf("permission ask not detected: %+v", perm)
	}
	if perm.OperatorDirectedClass != "PERMISSION_ASK" {
		t.Errorf("class = %q, want PERMISSION_ASK", perm.OperatorDirectedClass)
	}
	if perm.OperatorDirectedDisposition != "TAKE_OBVIOUS" {
		t.Errorf("disposition = %q, want TAKE_OBVIOUS", perm.OperatorDirectedDisposition)
	}
	if perm.OperatorDirectedCount < 1 || strings.TrimSpace(perm.OperatorDirectedResolve) == "" {
		t.Errorf("count/resolve unset: count=%d resolve=%q", perm.OperatorDirectedCount, perm.OperatorDirectedResolve)
	}

	// Authority ask: naming a release/approval gate earns HUMAN_RESIDUAL — the guard
	// records a real escalation to route, not an inline question to a person.
	auth := &guardStopTranscript{Read: true}
	applyHeadlessLintSignal(auth, "Waiting for your approval to release the build.")
	if !auth.OperatorDirected || auth.OperatorDirectedDisposition != "HUMAN_RESIDUAL" {
		t.Errorf("authority ask should fold to HUMAN_RESIDUAL, got %+v", auth)
	}
}

// TestReadGuardStopTranscriptOperatorDirected proves the reader wiring end-to-end over
// a written transcript: a prose-only final turn that asks a human is flagged, while a
// final turn that still made a tool call is NOT — a turn with a pending tool is not
// stopping-to-ask, so the fold is correctly gated on last_had_tool_use.
func TestReadGuardStopTranscriptOperatorDirected(t *testing.T) {
	dir := t.TempDir()

	// (a) prose-only end_turn that asks a human.
	proseP := filepath.Join(dir, "prose.jsonl")
	writeStopTranscriptFixture(t, proseP,
		`{"type":"user","message":{"role":"user","content":"handle the timeout"}}`,
		`{"type":"assistant","message":{"role":"assistant","content":"Fixed the handler. Do you want me to push the changes?"}}`,
	)
	sig := readGuardStopTranscript(proseP)
	if sig == nil || !sig.Read {
		t.Fatalf("prose transcript not read: %+v", sig)
	}
	if !sig.OperatorDirected || sig.OperatorDirectedClass != "PERMISSION_ASK" {
		t.Errorf("prose final turn should be operator-directed PERMISSION_ASK, got %+v", sig)
	}

	// (b) the SAME question, but the final turn also fires a tool call. The turn is
	// not ending the session on an unanswered question, so it is not flagged.
	toolP := filepath.Join(dir, "tool.jsonl")
	writeStopTranscriptFixture(t, toolP,
		`{"type":"assistant","message":{"role":"assistant","content":"Do you want me to push?"}}`,
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Do you want me to push?"},{"type":"tool_use","name":"Bash"}]}}`,
	)
	sigTool := readGuardStopTranscript(toolP)
	if sigTool == nil || !sigTool.Read {
		t.Fatalf("tool transcript not read: %+v", sigTool)
	}
	if !sigTool.LastHadToolUse {
		t.Fatalf("fixture invalid: final turn should have a tool use")
	}
	if sigTool.OperatorDirected {
		t.Errorf("final turn with a tool call must not be flagged operator-directed, got %+v", sigTool)
	}
}

// TestSummarizeGuardStopsOperatorDirected folds a ledger holding one operator-directed
// stop and confirms the tally counts it and the human render names the headless-directed
// headline plus the per-row remediation on a guard-ended row.
func TestSummarizeGuardStopsOperatorDirected(t *testing.T) {
	rows := []guardStopRecord{
		{
			Schema:      guardStopRecordSchema,
			Ts:          "2026-07-01T00:00:00Z",
			Disposition: string(stopDispCleanCompletion),
			Kind:        string(stopKindClean),
			Transcript:  &guardStopTranscript{Read: true, OperatorDirected: true, OperatorDirectedClass: "PERMISSION_ASK", OperatorDirectedDisposition: "TAKE_OBVIOUS"},
		},
		{
			Schema:      guardStopRecordSchema,
			Ts:          "2026-07-01T00:01:00Z",
			Disposition: string(stopDispBlindGiveUp),
			Kind:        string(stopKindStandDown),
			Transcript:  &guardStopTranscript{Read: true, OperatorDirected: true, OperatorDirectedClass: "CONFIRMATION_WAIT", OperatorDirectedDisposition: "TAKE_OBVIOUS"},
		},
	}
	var b strings.Builder
	for _, r := range rows {
		js, _ := json.Marshal(r)
		b.Write(js)
		b.WriteByte('\n')
	}
	sum := summarizeGuardStops(b.String(), 10)
	if sum.OperatorDirected != 2 {
		t.Errorf("OperatorDirected = %d, want 2", sum.OperatorDirected)
	}
	human := renderGuardStopsSummary(sum)
	if !strings.Contains(human, "asking a human instead of acting") {
		t.Errorf("render missing operator-directed headline:\n%s", human)
	}
	// The stand-down row is in Recent, so its per-row remediation note must appear.
	if !strings.Contains(human, "asked a human: CONFIRMATION_WAIT → TAKE_OBVIOUS") {
		t.Errorf("render missing per-row operator-directed note:\n%s", human)
	}
}

// writeStopTranscriptFixture writes the given JSONL lines to path for transcript-reader fixtures.
func writeStopTranscriptFixture(t *testing.T, path string, lines ...string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write fixture %s: %v", path, err)
	}
}
