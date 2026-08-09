package guardcomplaint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// codexRolloutJournal reproduces the #5213 evidence shape: ONE Codex session whose
// rollout showed three interruption points — two POLICY_BLOCK/Bash refusals and a
// later SELF_MODIFY/Bash refusal — and nothing that identified WHICH call each one
// refused. Every row shares the session, so the session alone cannot select a denial.
func codexRolloutJournal(t *testing.T) []string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	lines := []string{
		`{"seq":53,"call_seq":41,"ts_unix_nano":530,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","by":"monitor","trace_id":"codex-019f71eb","args_digest":"sha256:installer"}`,
		`{"seq":90,"call_seq":63,"ts_unix_nano":900,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","by":"monitor","trace_id":"codex-019f71eb","args_digest":"sha256:hostmaint"}`,
		`{"seq":104,"call_seq":77,"ts_unix_nano":1040,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"SELF_MODIFY","by":"monitor","trace_id":"codex-019f71eb","args_digest":"sha256:agentsmd"}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return []string{path}
}

// TestSelectDenialBindsSessionAndCallID is the #5213 regression witness: #5197 and
// #5194 both filed rationale-only because their journal matching was ambiguous. The
// selectors that existed then — reason/tool, and the session — all still match several
// rows of one busy session, so an appeal could not name the call it was appealing.
// Session + call id now selects exactly one denial without scanning ambiguous rows.
func TestSelectDenialBindsSessionAndCallID(t *testing.T) {
	paths := codexRolloutJournal(t)

	// The pre-#5213 selectors: both still ambiguous, and both correctly refuse to
	// attach a witness rather than binding the appeal to the newest plausible row.
	ambiguous := []struct {
		name    string
		sel     DenialSelector
		matches int
	}{
		{"reason and tool", DenialSelector{Reason: "POLICY_BLOCK", Tool: "Bash"}, 2},
		{"session alone", DenialSelector{TraceID: "codex-019f71eb"}, 3},
	}
	for _, tc := range ambiguous {
		t.Run(tc.name, func(t *testing.T) {
			got := SelectDenial(paths, tc.sel)
			if !got.Ambiguous || got.Matches != tc.matches || got.Evidence != nil {
				t.Fatalf("selection = %+v, want %d-match ambiguity with no attached evidence", got, tc.matches)
			}
		})
	}

	// Session + call id: one row each, including the two that share reason AND tool.
	for _, tc := range []struct {
		name    string
		callSeq uint64
		wantSeq uint64
		digest  string
	}{
		{"first policy block", 41, 53, "sha256:installer"},
		{"second policy block", 63, 90, "sha256:hostmaint"},
		{"self modify", 77, 104, "sha256:agentsmd"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := SelectDenial(paths, DenialSelector{TraceID: "codex-019f71eb", CallSeq: tc.callSeq})
			if got.Ambiguous || got.Matches != 1 || got.Evidence == nil {
				t.Fatalf("selection = %+v, want exactly one match", got)
			}
			if got.Evidence.Seq != tc.wantSeq || got.Evidence.CallSeq != tc.callSeq {
				t.Fatalf("evidence seq/call_seq = %d/%d, want %d/%d",
					got.Evidence.Seq, got.Evidence.CallSeq, tc.wantSeq, tc.callSeq)
			}
			// The selected row must be the refused CALL, not merely a row of the
			// right class: its args digest pins which command was refused.
			if got.Evidence.ArgsDigest != tc.digest {
				t.Fatalf("evidence args digest = %q, want %q", got.Evidence.ArgsDigest, tc.digest)
			}
		})
	}
}

// TestSelectDenialCallIDFailsClosed pins the fail-closed half: a call id that no row
// carries selects nothing, and a row with no call id at all is never substituted for
// one that was asked for by id. An appeal that cannot find its call files
// witness-less and says so, rather than attaching an unrelated denial.
func TestSelectDenialCallIDFailsClosed(t *testing.T) {
	paths := codexRolloutJournal(t)
	if got := SelectDenial(paths, DenialSelector{TraceID: "codex-019f71eb", CallSeq: 999}); got.Evidence != nil || got.Matches != 0 || got.Ambiguous {
		t.Fatalf("unknown call id = %+v, want honest no-match", got)
	}
	// A call id paired with the WRONG session is contradictory, not a near-miss.
	if got := SelectDenial(paths, DenialSelector{TraceID: "some-other-session", CallSeq: 41}); got.Evidence != nil || got.Matches != 0 {
		t.Fatalf("cross-session call id = %+v, want honest no-match", got)
	}

	// A journal predating call_seq carries no call id: it must not answer a
	// selection made by call id, however well its reason/tool/session match.
	legacy := filepath.Join(t.TempDir(), "legacy.jsonl")
	row := `{"seq":7,"ts_unix_nano":70,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","by":"monitor","trace_id":"codex-019f71eb"}`
	if err := os.WriteFile(legacy, []byte(row+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	legacyPaths := []string{legacy}
	if got := SelectDenial(legacyPaths, DenialSelector{TraceID: "codex-019f71eb", CallSeq: 41}); got.Evidence != nil || got.Matches != 0 {
		t.Fatalf("call-id selection against a call_seq-less journal = %+v, want honest no-match", got)
	}
	// Without a call-id selector that same legacy row still selects, so adding the
	// field never narrows what an existing caller could already reach.
	if got := SelectDenial(legacyPaths, DenialSelector{TraceID: "codex-019f71eb"}); got.Evidence == nil || got.Evidence.Seq != 7 {
		t.Fatalf("legacy selection without a call id = %+v, want the row to still bind", got)
	}
}

// TestEvidenceBlockNamesTheRefusedCall proves the appeal ARTIFACT carries the call
// identity, not just the reason class — the #5213 complaint that an operator could
// not tell what was refused. The rendered body also names the exact selector that
// reproduces the selection, so a reader can re-derive the witness themselves.
func TestEvidenceBlockNamesTheRefusedCall(t *testing.T) {
	got := SelectDenial(codexRolloutJournal(t), DenialSelector{TraceID: "codex-019f71eb", CallSeq: 63})
	if got.Evidence == nil {
		t.Fatalf("selection = %+v, want evidence", got)
	}
	c := Complaint{
		Kind:      "over-broad",
		Reason:    "POLICY_BLOCK",
		Tool:      "Bash",
		Summary:   "policy block refused an installer edit",
		Rationale: "the call edited an installer script, not a guarded path",
		Evidence:  got.Evidence,
	}
	body := c.evidenceBlock()
	for _, want := range []string{
		"- call id: `63`",
		"- journal seq: `90`",
		"- trace id: `codex-019f71eb`",
		"- args digest: `sha256:hostmaint`",
		"--from-journal --trace-id codex-019f71eb --call-seq 63",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("evidence block missing %q:\n%s", want, body)
		}
	}
}

// TestEvidenceCarriesMatchedRule pins the second half of #5213: an appeal must name
// the RUNG that refused, not only the refusal's class. ("monitor","POLICY_BLOCK")
// alone covers the recursive-delete, out-of-tree-write and deny-regex rungs at once,
// so a class-only witness cannot say which rule the appeal is arguing against.
func TestEvidenceCarriesMatchedRule(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	lines := []string{
		`{"seq":1,"call_seq":11,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","by":"monitor","trace_id":"s","deny_rule":"rm_rf"}`,
		`{"seq":2,"call_seq":12,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","by":"monitor","trace_id":"s","deny_rule":"out_of_tree_write"}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	paths := []string{path}

	// Same reason, same tool, same session, same decider — different rungs.
	for _, tc := range []struct {
		callSeq uint64
		rule    string
	}{
		{11, "rm_rf"},
		{12, "out_of_tree_write"},
	} {
		got := SelectDenial(paths, DenialSelector{TraceID: "s", CallSeq: tc.callSeq})
		if got.Evidence == nil || got.Evidence.DenyRule != tc.rule {
			t.Fatalf("call %d evidence = %+v, want deny rule %q", tc.callSeq, got.Evidence, tc.rule)
		}
		if body := (Complaint{Evidence: got.Evidence}).evidenceBlock(); !strings.Contains(body, "- matched rule: `"+tc.rule+"`") {
			t.Fatalf("evidence block missing matched rule %q:\n%s", tc.rule, body)
		}
	}
}

// TestEvidenceDropsUnknownDenyRuleWhole is the disclosure witness. deny_rule is
// rendered verbatim into a GitHub issue, so its value space must stay exactly the
// compile-time closed set. A value outside that set is dropped WHOLE — never
// trimmed, filtered, or truncated into the set, which is the partial-credit failure
// that leaked an env-assignment value into args_label before #5863.
func TestEvidenceDropsUnknownDenyRuleWhole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tampered.jsonl")
	lines := []string{
		`{"seq":1,"call_seq":11,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","trace_id":"s","deny_rule":"AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI"}`,
		`{"seq":2,"call_seq":12,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","trace_id":"s","deny_rule":"rm_rf /home/user/.ssh/id_rsa"}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	paths := []string{path}

	// A value with no member spelling at all: dropped entirely.
	got := SelectDenial(paths, DenialSelector{TraceID: "s", CallSeq: 11})
	if got.Evidence == nil {
		t.Fatalf("selection = %+v, want the row to still bind", got)
	}
	if got.Evidence.DenyRule != "" {
		t.Fatalf("deny rule = %q, want it dropped whole", got.Evidence.DenyRule)
	}

	// A member id followed by attacker-supplied text: canonicalization keeps ONLY
	// the authored leading atom, so the trailing path can never reach the body.
	got = SelectDenial(paths, DenialSelector{TraceID: "s", CallSeq: 12})
	if got.Evidence == nil || got.Evidence.DenyRule != "rm_rf" {
		t.Fatalf("evidence = %+v, want the bare member id rm_rf", got.Evidence)
	}
	body := (Complaint{Evidence: got.Evidence}).evidenceBlock()
	for _, leak := range []string{"id_rsa", "/home/user", "wJalrXUtnFEMI", "AWS_SECRET"} {
		if strings.Contains(body, leak) {
			t.Fatalf("evidence block leaked %q:\n%s", leak, body)
		}
	}
	// The audit structure that matters is still there.
	if !strings.Contains(body, "- matched rule: `rm_rf`") {
		t.Fatalf("evidence block lost the matched rule:\n%s", body)
	}
}
