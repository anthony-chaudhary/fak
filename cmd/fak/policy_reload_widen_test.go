package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestDiffPolicyWideningReportsCapabilityDeltaDeterministically(t *testing.T) {
	old := adjudicator.Policy{
		Posture:         adjudicator.PostureFailClosed,
		Allow:           map[string]bool{"kept": true},
		AllowPrefix:     []string{"read_"},
		Deny:            map[string]abi.ReasonCode{"kept-deny": abi.ReasonPolicyBlock, "removed-deny": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{"kept/**", "removed/**"},
	}
	next := adjudicator.Policy{
		Posture:         adjudicator.PostureAdmitAndLog,
		Allow:           map[string]bool{"kept": true, "z-new": true, "a-new": true},
		AllowPrefix:     []string{"read_", "write_"},
		Deny:            map[string]abi.ReasonCode{"kept-deny": abi.ReasonDefaultDeny},
		SelfModifyGlobs: []string{"kept/**"},
	}
	got := diffPolicyWidening(old, next).String()
	want := "added_allow=a-new,z-new; added_allow_prefix=write_; removed_deny=removed-deny; removed_self_modify_globs=removed/**; posture=fail_closed->admit_and_log"
	if got != want {
		t.Fatalf("delta=%q want %q", got, want)
	}
}

func TestDiffPolicyWideningAllowsNarrowingAndRelabel(t *testing.T) {
	old := adjudicator.Policy{
		Posture:         adjudicator.PostureAdmitAndLog,
		Allow:           map[string]bool{"removed": true},
		AllowPrefix:     []string{"removed_"},
		Deny:            map[string]abi.ReasonCode{"relabeled": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{"kept/**"},
	}
	next := adjudicator.Policy{
		Posture:         adjudicator.PostureFailClosed,
		Deny:            map[string]abi.ReasonCode{"relabeled": abi.ReasonDefaultDeny, "new-deny": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{"kept/**", "new/**"},
	}
	if delta := diffPolicyWidening(old, next); !delta.Empty() {
		t.Fatalf("narrowing reported as widening: %s", delta.String())
	}
}

func TestPolicyWideningConfirmationEnv(t *testing.T) {
	for _, value := range []string{"1", "true", "YES"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv(policyReloadWidenConfirmEnv, value)
			if !policyReloadWidenConfirmed() {
				t.Fatalf("%q did not confirm", value)
			}
		})
	}
	t.Setenv(policyReloadWidenConfirmEnv, "0")
	if policyReloadWidenConfirmed() {
		t.Fatal("0 confirmed widening")
	}
	err := policyWideningError(policyWideningDelta{AddedAllow: []string{"danger"}})
	if !strings.Contains(err.Error(), policyReloadWidenConfirmEnv) || !strings.Contains(err.Error(), "added_allow=danger") {
		t.Fatalf("error lacks operator action or delta: %v", err)
	}
}

func TestReloadPolicyRejectsWideningKeepsLastGoodAndJournalsDelta(t *testing.T) {
	journal.ResetActiveForTest()
	auditDir := t.TempDir()
	j, err := journal.Enable(filepath.Join(auditDir, "audit.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(journal.ResetActiveForTest)
	t.Cleanup(func() { adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy()) })
	t.Setenv(policyReloadWidenConfirmEnv, "")
	t.Setenv("FAK_GUARD_ALLOW_OVERLAY", filepath.Join(t.TempDir(), "missing-allow.json"))
	t.Setenv("FAK_GUARD_DENY_OVERLAY", filepath.Join(t.TempDir(), "missing-deny.json"))

	dir := t.TempDir()
	strictPath := filepath.Join(dir, "strict.json")
	widePath := filepath.Join(dir, "wide.json")
	writePolicyFile(t, strictPath, `{"posture":"fail_closed","allow":["read"],"deny":{"danger":"POLICY_BLOCK"},"self_modify_globs":["protected/**"]}`)
	writePolicyFile(t, widePath, `{"posture":"admit_and_log","allow":["read","danger"],"deny":{},"self_modify_globs":[]}`)
	applyPolicy(strictPath) // launch-time install is deliberately not reload-gated

	_, _, err = reloadPolicy(widePath)
	if err == nil || !strings.Contains(err.Error(), "added_allow=danger") || !strings.Contains(err.Error(), "removed_deny=danger") || !strings.Contains(err.Error(), "protected/**") || !strings.Contains(err.Error(), "posture=fail_closed->admit_and_log") {
		t.Fatalf("reload error lacks widening delta: %v", err)
	}
	if verdict := adjudicateToolForReloadTest("danger"); verdict.Kind != abi.VerdictDeny {
		t.Fatalf("rejected reload replaced last-good floor: %+v", verdict)
	}
	rows := j.Recent(10)
	last := rows[len(rows)-1]
	if last.ConfigSwap == nil || last.ConfigSwap.Outcome != journal.ConfigSwapRejected || !strings.Contains(last.ConfigSwap.Reason, "added_allow=danger") {
		t.Fatalf("journal lacks rejected widening delta: %+v", last)
	}

	t.Setenv(policyReloadWidenConfirmEnv, "1")
	_, summary, err := reloadPolicy(widePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(summary, "confirmed_widening: added_allow=danger") {
		t.Fatalf("success response lacks confirmed delta: %q", summary)
	}
	if verdict := adjudicateToolForReloadTest("danger"); verdict.Kind != abi.VerdictAllow {
		t.Fatalf("confirmed reload did not apply: %+v", verdict)
	}
	last = j.Recent(10)[len(j.Recent(10))-1]
	if last.ConfigSwap == nil || last.ConfigSwap.Outcome != journal.ConfigSwapOK || !strings.Contains(last.ConfigSwap.Reason, "confirmed_widening: added_allow=danger") {
		t.Fatalf("journal lacks confirmed widening delta: %+v", last)
	}
}

func TestApplyPolicyLaunchDoesNotRequireWideningConfirmation(t *testing.T) {
	t.Cleanup(func() { adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy()) })
	t.Setenv(policyReloadWidenConfirmEnv, "")
	t.Setenv("FAK_GUARD_ALLOW_OVERLAY", filepath.Join(t.TempDir(), "missing-allow.json"))
	t.Setenv("FAK_GUARD_DENY_OVERLAY", filepath.Join(t.TempDir(), "missing-deny.json"))
	path := filepath.Join(t.TempDir(), "launch.json")
	writePolicyFile(t, path, `{"posture":"admit_and_log","allow":["launch-wide"]}`)
	applyPolicy(path)
	if verdict := adjudicateToolForReloadTest("launch-wide"); verdict.Kind != abi.VerdictAllow {
		t.Fatalf("launch-time apply was gated: %+v", verdict)
	}
}

func writePolicyFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func adjudicateToolForReloadTest(tool string) abi.Verdict {
	return adjudicator.Default.Adjudicate(context.Background(), &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}})
}
