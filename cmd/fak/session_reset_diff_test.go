package main

// session_reset_diff_test.go — exercises `fak session reset-diff` (issue #1575): the
// offline CLI front end over internal/sessionreset.DiffReset. Proves the CLI wires a
// JSON transcript request through the SAME BuildSeed/BuildResetTransaction path the
// live gateway reset hook uses, and renders a concrete before/after delta — not just
// that a struct got populated.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runResetDiff drives runSessionResetDiff with a stdin string and captured streams,
// returning (exit, stdout, stderr).
func runResetDiff(t *testing.T, stdin string, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := runSessionResetDiff(strings.NewReader(stdin), &out, &errb, args)
	return code, out.String(), errb.String()
}

// sampleResetDiffRequest is a transcript rich enough to exercise all four buckets:
// a durable preference (summarized), a verbatim tail (survived), an ephemeral line
// (expired), and a system preamble (must-requery via the warm prefix).
const sampleResetDiffRequest = `{
  "old_trace": "trace-parent",
  "new_trace": "trace-child",
  "fresh_budget_tok": 75,
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant for the fak repo."},
    {"role": "user", "content": "Help me add a budget-triggered session reset to fak."},
    {"role": "assistant", "content": "Sure. Let me look at the session package."},
    {"role": "user", "content": "I prefer concise answers."},
    {"role": "user", "content": "it's 3pm and the build is currently running"},
    {"role": "assistant", "content": "Got it."},
    {"role": "user", "content": "Now wire the gateway hook."}
  ]
}`

// TestSessionResetDiffCLIRendersHumanReadableExplain proves the default text render
// names every bucket the issue asks for and reflects the lineage from the request.
func TestSessionResetDiffCLIRendersHumanReadableExplain(t *testing.T) {
	code, out, errb := runResetDiff(t, sampleResetDiffRequest)
	if code != 0 {
		t.Fatalf("reset-diff exit=%d, want 0\nstderr=%s", code, errb)
	}
	if !strings.Contains(out, "trace-parent -> trace-child") {
		t.Fatalf("Explain output missing lineage: %s", out)
	}
	for _, want := range []string{"SURVIVED", "SUMMARIZED", "MUST-REQUERY", "EXPIRED"} {
		if !strings.Contains(out, want) {
			t.Fatalf("Explain output missing bucket %q:\n%s", want, out)
		}
	}
}

// TestSessionResetDiffCLIJSONShowsConcreteDelta proves --json emits a ResetDiff whose
// buckets carry an actual before/after delta: the durable fact folded into
// Summarized, the ephemeral line's digest in Expired, and a warm-prefix row in
// MustRequery — not merely a populated-but-empty struct.
func TestSessionResetDiffCLIJSONShowsConcreteDelta(t *testing.T) {
	code, out, errb := runResetDiff(t, sampleResetDiffRequest, "--json")
	if code != 0 {
		t.Fatalf("reset-diff --json exit=%d, want 0\nstderr=%s", code, errb)
	}
	var diff struct {
		OldTrace    string `json:"old_trace"`
		NewTrace    string `json:"new_trace"`
		SeedDigest  string `json:"seed_digest"`
		BeforeSpans int    `json:"before_spans"`
		AfterChars  int    `json:"after_chars"`
		Survived    []struct {
			Name string `json:"name"`
			Text string `json:"text"`
		} `json:"survived"`
		Summarized []struct {
			Name string `json:"name"`
			Text string `json:"text"`
		} `json:"summarized"`
		MustRequery []struct {
			Reason string `json:"reason"`
			Digest string `json:"digest"`
		} `json:"must_requery"`
		Expired []struct {
			Digest string `json:"digest"`
			Reason string `json:"reason"`
		} `json:"expired"`
	}
	if err := json.Unmarshal([]byte(out), &diff); err != nil {
		t.Fatalf("reset-diff --json output is not valid JSON: %v\n%s", err, out)
	}
	if diff.OldTrace != "trace-parent" || diff.NewTrace != "trace-child" {
		t.Fatalf("lineage = %s -> %s, want trace-parent -> trace-child", diff.OldTrace, diff.NewTrace)
	}
	if diff.SeedDigest == "" {
		t.Fatal("seed_digest missing from JSON output")
	}
	if diff.BeforeSpans != 7 {
		t.Fatalf("before_spans = %d, want 7", diff.BeforeSpans)
	}
	if diff.AfterChars == 0 {
		t.Fatal("after_chars = 0, want a non-empty carried-over seed")
	}

	foundDurable := false
	for _, p := range diff.Summarized {
		if strings.Contains(p.Text, "I prefer concise answers") {
			foundDurable = true
		}
	}
	if !foundDurable {
		t.Fatalf("durable preference missing from Summarized bucket: %+v", diff.Summarized)
	}

	for _, buckets := range [][]struct {
		Name string `json:"name"`
		Text string `json:"text"`
	}{diff.Survived, diff.Summarized} {
		for _, p := range buckets {
			if strings.Contains(p.Text, "3pm") {
				t.Fatalf("ephemeral line leaked into the after-state part %q: %q", p.Name, p.Text)
			}
		}
	}
	if len(diff.Expired) == 0 {
		t.Fatal("expected at least one expired span (the ephemeral line)")
	}
	for _, s := range diff.Expired {
		if s.Digest == "" {
			t.Fatalf("expired span missing digest: %+v", s)
		}
	}

	foundWarmPrefix := false
	for _, s := range diff.MustRequery {
		if s.Reason == "warm_prefix_replay" && s.Digest != "" {
			foundWarmPrefix = true
		}
	}
	if !foundWarmPrefix {
		t.Fatalf("expected a warm_prefix_replay row with a digest in MustRequery: %+v", diff.MustRequery)
	}
}

// TestSessionResetDiffCLIMarkdown proves --md renders the shareable report with one
// section per bucket.
func TestSessionResetDiffCLIMarkdown(t *testing.T) {
	code, out, errb := runResetDiff(t, sampleResetDiffRequest, "--md")
	if code != 0 {
		t.Fatalf("reset-diff --md exit=%d, want 0\nstderr=%s", code, errb)
	}
	for _, want := range []string{
		"# Session reset diff",
		"## Survived (carried forward near-verbatim)",
		"## Summarized (folded/distilled into the seed)",
		"## Must re-query (cold, recoverable via an explicit follow-up)",
		"## Expired (dropped, no recovery handle beyond the digest)",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("Markdown output missing %q:\n%s", want, out)
		}
	}
}

// TestSessionResetDiffCLIRequiresNewTrace proves a missing new_trace is a loud usage
// error (exit 2), not a silently empty diff.
func TestSessionResetDiffCLIRequiresNewTrace(t *testing.T) {
	code, _, errb := runResetDiff(t, `{"old_trace":"trace-parent"}`)
	if code != 2 {
		t.Fatalf("exit=%d, want 2 for a missing new_trace", code)
	}
	if !strings.Contains(errb, "new_trace") {
		t.Fatalf("stderr should mention new_trace: %s", errb)
	}
}

// TestSessionResetDiffCLIRejectsMalformedInput proves invalid/unknown-field JSON
// fails loudly (exit 2) rather than silently zeroing a mistyped field.
func TestSessionResetDiffCLIRejectsMalformedInput(t *testing.T) {
	code, _, errb := runResetDiff(t, `{"new_trce":"typo"}`)
	if code != 2 {
		t.Fatalf("exit=%d, want 2 for malformed/unknown-field JSON\nstderr=%s", code, errb)
	}
}

// TestSessionResetDiffCLIReadsFromInFlag proves --in FILE works as an alternative to
// stdin, matching the callavoid/dispatch --in convention.
func TestSessionResetDiffCLIReadsFromInFlag(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "req.json")
	if err := os.WriteFile(path, []byte(sampleResetDiffRequest), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	code, out, errb := runResetDiff(t, "", "--in", path, "--json")
	if code != 0 {
		t.Fatalf("reset-diff --in exit=%d, want 0\nstderr=%s", code, errb)
	}
	if !strings.Contains(out, "trace-child") {
		t.Fatalf("--in output missing expected trace: %s", out)
	}
}
