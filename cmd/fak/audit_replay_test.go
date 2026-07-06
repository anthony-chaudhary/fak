package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// mintRow / soundChain / writeRowsFile are shared fixture builders defined in
// audit_diagnose_test.go (same package): they stamp a sound hash chain so the replay fold runs
// over rows the real journal.VerifyRows accepts.

func TestReplayRows_DeterministicSession(t *testing.T) {
	// Two distinct calls, each recorded twice with the SAME verdict — a reproducible session.
	rows := []journal.Row{
		{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:a"},
		{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "SELF_MODIFY", ArgsDigest: "sha256:b"},
		{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:a"},
		{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "SELF_MODIFY", ArgsDigest: "sha256:b"},
	}
	rep := replayRows("x", mintChain(rows))
	if !rep.Deterministic {
		t.Fatalf("identical calls with identical verdicts must be deterministic: %+v", rep)
	}
	if len(rep.Findings) != 0 {
		t.Fatalf("no findings expected, got %+v", rep.Findings)
	}
	if rep.DecisionRows != 4 || rep.DistinctCalls != 2 {
		t.Fatalf("want 4 decision rows over 2 distinct calls, got rows=%d calls=%d", rep.DecisionRows, rep.DistinctCalls)
	}
}

func TestReplayRows_NonDeterministicVerdictDivergence(t *testing.T) {
	// The SAME call identity (Bash + sha256:cmd) recorded first ALLOW then DENY — the floor
	// answered the same question two ways: a non-determinism finding.
	rows := []journal.Row{
		{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:cmd", ArgsLabel: "command=ls"},
		{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", ArgsDigest: "sha256:cmd", ArgsLabel: "command=ls"},
		{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:cmd", ArgsLabel: "command=ls"},
	}
	rep := replayRows("x", mintChain(rows))
	if rep.Deterministic {
		t.Fatalf("verdict divergence for one call identity must be non-deterministic: %+v", rep)
	}
	if len(rep.Findings) != 1 {
		t.Fatalf("want exactly one finding, got %+v", rep.Findings)
	}
	f := rep.Findings[0]
	if f.Tool != "Bash" || f.ArgsDigest != "sha256:cmd" || f.ArgsLabel != "command=ls" {
		t.Fatalf("finding identity wrong: %+v", f)
	}
	if f.Occurrences != 3 || len(f.Outcomes) != 2 {
		t.Fatalf("want 3 occurrences over 2 outcomes, got %+v", f)
	}
	// Most-frequent first: ALLOW occurred twice, DENY once.
	if f.Outcomes[0].Verdict != "ALLOW" || f.Outcomes[0].Count != 2 {
		t.Fatalf("outcomes[0] should be ALLOW x2, got %+v", f.Outcomes[0])
	}
	if f.Outcomes[1].Verdict != "DENY" || f.Outcomes[1].Reason != "POLICY_BLOCK" || f.Outcomes[1].Count != 1 {
		t.Fatalf("outcomes[1] should be DENY[POLICY_BLOCK] x1, got %+v", f.Outcomes[1])
	}
}

func TestReplayRows_SameDigestDifferentToolsAreDistinctCalls(t *testing.T) {
	// A shared digest under two different tools is two DIFFERENT call identities — neither
	// diverges on its own, so the session stays deterministic.
	rows := []journal.Row{
		{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:shared"},
		{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", ArgsDigest: "sha256:shared"},
	}
	rep := replayRows("x", mintChain(rows))
	if !rep.Deterministic || rep.DistinctCalls != 2 {
		t.Fatalf("same digest under two tools = 2 distinct deterministic calls, got %+v", rep)
	}
}

func TestReplayRows_IgnoresNonAdjudicationRows(t *testing.T) {
	// QUARANTINE (result-side) and VDSO_HIT rows are not call adjudications: even sharing a
	// digest with a DECIDE they must not manufacture a divergence.
	rows := []journal.Row{
		{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:x"},
		{Kind: "QUARANTINE", Tool: "Read", Verdict: "QUARANTINE", Reason: "SECRET_EXFIL", ArgsDigest: "sha256:x", ResultDigest: "sha256:r"},
		{Kind: "VDSO_HIT", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:x"},
	}
	rep := replayRows("x", mintChain(rows))
	if !rep.Deterministic {
		t.Fatalf("non-adjudication rows must not create findings: %+v", rep)
	}
	if rep.DecisionRows != 1 || rep.DistinctCalls != 1 {
		t.Fatalf("only the DECIDE row is a call adjudication, got rows=%d calls=%d", rep.DecisionRows, rep.DistinctCalls)
	}
}

func TestReplayRows_UnkeyableRowsCountedNotAsserted(t *testing.T) {
	// An adjudicated call whose args were blob-backed with no inline digest cannot be keyed;
	// it is counted as un-keyable, never a finding.
	rows := []journal.Row{
		{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE"},
		{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK"},
	}
	rep := replayRows("x", mintChain(rows))
	if !rep.Deterministic {
		t.Fatalf("un-keyable rows must not be asserted: %+v", rep)
	}
	if rep.DecisionRows != 2 || rep.UnkeyableRows != 2 || rep.DistinctCalls != 0 {
		t.Fatalf("want 2 decision/2 unkeyable/0 distinct, got %+v", rep)
	}
}

func TestReplayRows_EmptyJournalIsDeterministic(t *testing.T) {
	rep := replayRows("x", nil)
	if !rep.Deterministic || !rep.ChainLinearOK || rep.Rows != 0 {
		t.Fatalf("empty journal should be a trivially deterministic, chain-ok replay: %+v", rep)
	}
}

func TestReplayRows_InterleavedChainStillFolds(t *testing.T) {
	// A concurrent-writer interleave breaks the linear chain but must NOT stop the determinism
	// fold; the chain note points at `fak audit diagnose` and determinism is still asserted.
	prefix := soundChain(t, 2)
	head := prefix[len(prefix)-1].Hash
	a3, _ := mintRow(3, head, journal.Row{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:a"})
	b3, _ := mintRow(3, head, journal.Row{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", ArgsDigest: "sha256:b"})
	rows := []journal.Row{prefix[0], prefix[1], a3, b3}

	rep := replayRows("x", rows)
	if rep.ChainLinearOK {
		t.Fatalf("interleaved rows should not verify as a linear chain: %+v", rep)
	}
	if !strings.Contains(rep.ChainNote, "audit diagnose") {
		t.Fatalf("chain note should point at diagnose: %q", rep.ChainNote)
	}
	if !rep.Deterministic {
		t.Fatalf("interleave with no verdict divergence is still deterministic: %+v", rep)
	}
}

func TestRunAuditReplay_DeterministicExitsZero(t *testing.T) {
	rows := mintChain([]journal.Row{
		{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:a"},
		{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:a"},
	})
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	code := runAuditReplay(&stdout, &stderr, path, false)
	if code != 0 {
		t.Fatalf("deterministic replay should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "DETERMINISTIC") {
		t.Fatalf("render should name DETERMINISTIC:\n%s", stdout.String())
	}
}

func TestRunAuditReplay_NonDeterministicExitsOneAndRenders(t *testing.T) {
	rows := mintChain([]journal.Row{
		{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:cmd", ArgsLabel: "command=ls"},
		{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", ArgsDigest: "sha256:cmd", ArgsLabel: "command=ls"},
	})
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	code := runAuditReplay(&stdout, &stderr, path, false)
	if code != 1 {
		t.Fatalf("non-deterministic replay should exit 1, got %d (stderr=%s)", code, stderr.String())
	}
	for _, want := range []string{
		"NON-DETERMINISTIC",
		"diverged: Bash",
		"args_digest=sha256:cmd",
		"call_label=command=ls",
		"ALLOW",
		"POLICY_BLOCK",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestRunAuditReplay_JSONShape(t *testing.T) {
	rows := mintChain([]journal.Row{
		{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE", ArgsDigest: "sha256:cmd"},
		{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", ArgsDigest: "sha256:cmd"},
	})
	path := filepath.Join(t.TempDir(), "g.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	if code := runAuditReplay(&stdout, &stderr, path, true); code != 1 {
		t.Fatalf("non-deterministic --json should still exit 1, got %d", code)
	}
	var rep replayReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("--json output not parseable: %v\n%s", err, stdout.String())
	}
	if rep.Deterministic || len(rep.Findings) != 1 {
		t.Fatalf("json payload should carry one finding: %+v", rep)
	}
	if rep.Findings[0].Tool != "Bash" || len(rep.Findings[0].Outcomes) != 2 {
		t.Fatalf("json finding wrong: %+v", rep.Findings[0])
	}
}

func TestRunAuditReplay_MissingFileErrors(t *testing.T) {
	// A read of a genuinely broken path (a directory) is a setup error → exit 2. (A merely
	// absent file reads as the empty journal — journal.ReadRows treats not-exist as no rows.)
	var stdout, stderr bytes.Buffer
	code := runAuditReplay(&stdout, &stderr, t.TempDir(), false)
	if code != 2 {
		t.Fatalf("unreadable path should exit 2, got %d (stderr=%s)", code, stderr.String())
	}
}

// mintChain stamps a sound, linearly-verifiable hash chain over the given content rows so the
// replay fold and the chain-status check both run against rows journal.VerifyRows accepts.
func mintChain(content []journal.Row) []journal.Row {
	out := make([]journal.Row, 0, len(content))
	prev := ""
	for i, r := range content {
		minted, h := mintRow(uint64(i+1), prev, r)
		out = append(out, minted)
		prev = h
	}
	return out
}
