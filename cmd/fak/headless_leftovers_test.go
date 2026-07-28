package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunHeadlessLintLeftovers exercises the run-level end-of-run fold behind
// `fak headless-lint --leftovers` end to end (#3670): a final summary that narrates
// deferred work with zero issues filed exits 1 (refused); the same summary once the
// follow-ups are filed (--issues-filed) or an operator escape (--override) is given
// exits 0 (clean).
func TestRunHeadlessLintLeftovers(t *testing.T) {
	const summary = "Shipped the fix, tests pass, committed abc1234.\n" +
		"There are two more things worth doing: backoff and a docs pass."

	run := func(argv ...string) (int, string) {
		var out, errb bytes.Buffer
		code := runHeadlessLint(&out, &errb, strings.NewReader(""), argv)
		return code, out.String() + errb.String()
	}

	// Arm 1 — narrated leftovers, zero issues filed -> refused (exit 1).
	if code, s := run("--leftovers", summary); code != 1 {
		t.Fatalf("arm1: want exit 1 (refused), got %d\noutput: %s", code, s)
	}

	// Arm 2 — same summary, follow-ups filed as gh issues -> clean (exit 0).
	if code, s := run("--leftovers", "--issues-filed", "2", summary); code != 0 {
		t.Fatalf("arm2: want exit 0 (clean) with 2 issues filed, got %d\noutput: %s", code, s)
	}

	// Operator escape — "genuinely nothing left" -> clean (exit 0).
	if code, s := run("--leftovers", "--override", summary); code != 0 {
		t.Fatalf("override: want exit 0 (clean), got %d\noutput: %s", code, s)
	}

	// A completed-work summary carries no leftover narration -> clean.
	if code, _ := run("--leftovers", "Implemented the parser, tests pass, pushed."); code != 0 {
		t.Fatalf("clean summary: want exit 0, got %d", code)
	}

	// JSON mode still refuses arm 1 and emits the schema tag.
	code, s := run("--leftovers", "--json", summary)
	if code != 1 {
		t.Fatalf("json arm1: want exit 1, got %d", code)
	}
	if !strings.Contains(s, "fak-leftovers-fold/1") || !strings.Contains(s, "leftovers_unfiled") {
		t.Errorf("json arm1: expected schema + verdict in output, got: %s", s)
	}
}

// TestRunHeadlessLintLeftoversCountsEvidenceNotTheClaim is #5425's headline: with a
// transcript in hand, `fak headless-lint --leftovers` counts the issues a run filed from
// that run's own tool-use record, and a --issues-filed number the run asserts about
// itself no longer buys anything it cannot show.
//
// Three arms, in the order that matters:
//  1. the run CLAIMS 5 filed and its transcript evidences none -> still refused (exit 1);
//  2. the run claims 5 and its transcript evidences 2 -> clean, but the report says 2,
//     with the superseded claim of 5 kept visible beside it;
//  3. no usable transcript -> UNKNOWN (exit 3), and the JSON omits issues_filed entirely
//     rather than reporting a zero nobody witnessed.
func TestRunHeadlessLintLeftoversCountsEvidenceNotTheClaim(t *testing.T) {
	dir := t.TempDir()
	const summary = "Shipped the fix, tests pass, committed abc1234. I filed 5 follow-up issues.\n" +
		"There are two more things worth doing: backoff and a docs pass."

	run := func(argv ...string) (int, string) {
		var out, errb bytes.Buffer
		code := runHeadlessLint(&out, &errb, strings.NewReader(""), argv)
		return code, out.String() + errb.String()
	}

	// Arm 1 — the claim is loud, the record is empty. A transcript whose only tool calls
	// are ordinary work (and whose PROSE says an issue was filed) evidences no filing.
	empty := filepath.Join(dir, "no-filings.jsonl")
	writeGuardLeftoversTranscript(t, empty, []toolUseFixture{
		{Name: "functions.shell_command", Input: map[string]any{"command": "go test ./internal/headlesslint/"}},
		{Name: "functions.shell_command", Input: map[string]any{"description": "I filed the two follow-ups as gh issue create", "command": "git status --porcelain"}},
	})
	code, s := run("--leftovers", "--transcript", empty, "--issues-filed", "5", "--json", summary)
	if code != 1 {
		t.Fatalf("arm1: a claim of 5 with zero evidenced filings must still refuse; got exit %d\noutput: %s", code, s)
	}
	rep := decodeLeftoversJSON(t, s)
	if rep.Verdict != "leftovers_unfiled" {
		t.Errorf("arm1: verdict = %q, want leftovers_unfiled", rep.Verdict)
	}
	if rep.IssuesFiled == nil || *rep.IssuesFiled != 0 {
		t.Errorf("arm1: issues_filed = %v, want a witnessed 0", rep.IssuesFiled)
	}
	if rep.IssuesFiledClaimed == nil || *rep.IssuesFiledClaimed != 5 {
		t.Errorf("arm1: the superseded claim of 5 must stay visible, got %v", rep.IssuesFiledClaimed)
	}
	if rep.IssuesFiledSource != "transcript" {
		t.Errorf("arm1: source = %q, want transcript", rep.IssuesFiledSource)
	}

	// Arm 2 — the record shows two real filings; the claim of 5 does not raise the count.
	filed := filepath.Join(dir, "two-filings.jsonl")
	writeGuardLeftoversTranscript(t, filed, []toolUseFixture{
		{Name: "functions.shell_command", Input: map[string]any{"command": `gh issue create --title "backoff" --body "done condition"`}},
		{Name: "functions.shell_command", Input: map[string]any{"command": `fak issue create --title "docs pass" --body "done condition"`}},
	})
	code, s = run("--leftovers", "--transcript", filed, "--issues-filed", "5", "--json", summary)
	if code != 0 {
		t.Fatalf("arm2: two evidenced filings clear the doctrine; got exit %d\noutput: %s", code, s)
	}
	rep = decodeLeftoversJSON(t, s)
	if rep.IssuesFiled == nil || *rep.IssuesFiled != 2 {
		t.Fatalf("arm2: issues_filed = %v, want the evidenced 2 — not the claimed 5", rep.IssuesFiled)
	}
	if rep.IssuesFiledClaimed == nil || *rep.IssuesFiledClaimed != 5 {
		t.Errorf("arm2: claim of 5 should be recorded beside the evidence, got %v", rep.IssuesFiledClaimed)
	}

	// Arm 3 — a transcript was requested and there is nothing to read. Unknown, not zero:
	// a distinct exit code, a distinct verdict, and NO issues_filed field at all.
	missing := filepath.Join(dir, "absent.jsonl")
	code, s = run("--leftovers", "--transcript", missing, "--json", summary)
	if code != 3 {
		t.Fatalf("arm3: an absent transcript must be unknown (exit 3), not a refusal or a pass; got %d\noutput: %s", code, s)
	}
	rep = decodeLeftoversJSON(t, s)
	if rep.Verdict != "leftovers_filing_unknown" {
		t.Errorf("arm3: verdict = %q, want leftovers_filing_unknown", rep.Verdict)
	}
	if rep.IssuesFiled != nil {
		t.Errorf("arm3: issues_filed must be absent when unknown, got %d", *rep.IssuesFiled)
	}
	if rep.IssuesFiledKnown {
		t.Errorf("arm3: issues_filed_known must be false")
	}
	if strings.Contains(s, `"issues_filed":`) {
		t.Errorf("arm3: the JSON must not print a count nobody witnessed:\n%s", s)
	}

	// The human render of arm 3 has to say "unknown" out loud, not print a zero.
	_, human := run("--leftovers", "--transcript", missing, summary)
	if !strings.Contains(human, "UNKNOWN") || !strings.Contains(human, "unknown is not zero") {
		t.Errorf("arm3 render should state the count is unknown, got:\n%s", human)
	}
}

// TestHeadlessLintUsageDocumentsTranscriptEvidence keeps the new flag discoverable: a
// flag that only exists in the source is a flag no operator will ever reach for.
func TestHeadlessLintUsageDocumentsTranscriptEvidence(t *testing.T) {
	for _, want := range []string{"--transcript", "issues_filed_source", "filing unknown (3)", "DEPRECATED"} {
		if !strings.Contains(headlessLintUsage, want) {
			t.Errorf("headless-lint usage must document %q", want)
		}
	}
	// The flag defaults must also be reachable through the FlagSet's own help.
	var out, errb bytes.Buffer
	if code := runHeadlessLint(&out, &errb, strings.NewReader(""), []string{"--nonexistent-flag"}); code != 2 {
		t.Fatalf("bad flag: want usage exit 2, got %d", code)
	}
	if !strings.Contains(errb.String(), "--transcript session.jsonl") {
		t.Errorf("usage output missing the --transcript synopsis:\n%s", errb.String())
	}
}

// leftoversJSON mirrors the emitted LeftoversReport with POINTER counts, so a test can
// tell an absent count from a zero one exactly as a downstream reader would.
type leftoversJSON struct {
	Verdict            string `json:"verdict"`
	Narrated           int    `json:"narrated"`
	IssuesFiled        *int   `json:"issues_filed"`
	IssuesFiledKnown   bool   `json:"issues_filed_known"`
	IssuesFiledSource  string `json:"issues_filed_source"`
	IssuesFiledClaimed *int   `json:"issues_filed_claimed"`
}

func decodeLeftoversJSON(t *testing.T, s string) leftoversJSON {
	t.Helper()
	var rep leftoversJSON
	if err := json.Unmarshal([]byte(s), &rep); err != nil {
		t.Fatalf("decode report: %v\noutput: %s", err, s)
	}
	return rep
}
