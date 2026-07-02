package dispatchtick

import (
	"strings"
	"testing"
)

func TestClassifyNoCommitReason(t *testing.T) {
	banner := "> build · glm-4.5-air\n"
	cases := []struct {
		name string
		tail string
		size int64
		want string
	}{
		{"self-modify guard refusal", "guard summary\nrefused: reason=SELF_MODIFY edit to cmd/fak\n", 90000, NoCommitSelfModify},
		{"policy block guard refusal", "final turn\nPOLICY_BLOCK: push refused\n", 4096, NoCommitPolicyBlock},
		{"self-modify outranks policy block", "SELF_MODIFY then POLICY_BLOCK\n", 500, NoCommitSelfModify},
		{"claude cap banner", "You've hit your usage limit · resets 8pm\n", 700, NoCommitAuthWall},
		{"codex cap banner", "You've hit your usage limit. Visit https://chatgpt.com/codex\n", 700, NoCommitAuthWall},
		{"glm quota wall", "Limit Exhausted: your limit will reset at 2026-07-03\n", 120000, NoCommitAuthWall},
		{"glm usage wall", "usage limit reached for zai-coding-plan\n", 300, NoCommitAuthWall},
		{"off-trunk guard refusal", "commit refused: OFF_TRUNK\n", 2000, NoCommitOffTrunk},
		{"banner-only no-op under stub floor", banner, int64(len(banner)), NoCommitBannerNoop},
		{"banner over stub floor is not a no-op", banner + strings.Repeat("x", StubLogMaxBytes), StubLogMaxBytes + 30, NoCommitUnknown},
		{"banner with unknown size is not a no-op", banner, -1, NoCommitUnknown},
		{"no recognized signature", "worker ran and exited\n", 900, NoCommitUnknown},
		{"empty tail", "", 0, NoCommitUnknown},
	}
	for _, tc := range cases {
		if got := ClassifyNoCommitReason(tc.tail, tc.size); got != tc.want {
			t.Errorf("%s: ClassifyNoCommitReason = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// TestHeldNoCommitIssuesHoldsOnlyReblockableGuardRefusals pins the pick-held-invariant
// rung (#1396): only the two structural guard refusals (self_modify / policy_block)
// hold their issue -- an auth wall re-probes after the time cooldown, a banner no-op
// is owned by the backend-health gate, and witnessed/unwitnessed slots hold nothing.
func TestHeldNoCommitIssuesHoldsOnlyReblockableGuardRefusals(t *testing.T) {
	records := []WitnessRecord{
		{Issue: 1338, Claim: ClaimNoCommit, Reason: NoCommitSelfModify},
		{Issue: 1339, Claim: ClaimNoCommit, Reason: NoCommitPolicyBlock},
		{Issue: 1340, Claim: ClaimNoCommit, Reason: NoCommitAuthWall},
		{Issue: 1341, Claim: ClaimNoCommit, Reason: NoCommitBannerNoop},
		{Issue: 1342, Claim: ClaimNoCommit, Reason: NoCommitOffTrunk},
		{Issue: 1343, Claim: ClaimNoCommit, Reason: NoCommitUnknown},
		{Issue: 1344, Claim: ClaimWitnessed, SHA: "abc123"},
		{Issue: 1345, Claim: ClaimUnwitnessed, SHA: "def456"},
	}
	held := HeldNoCommitIssues(records)
	if len(held) != 2 || !held[1338] || !held[1339] {
		t.Fatalf("held = %v, want exactly {1338, 1339}", held)
	}
	if got := HeldNoCommitIssues(nil); len(got) != 0 {
		t.Fatalf("held over no records = %v, want empty", got)
	}
}

func TestSubjectCitesIssue(t *testing.T) {
	cases := []struct {
		subject string
		issue   int
		want    bool
	}{
		{"fix(dispatchtick): hold guard-blocked issues #1324 (fak dispatchtick)", 1324, true},
		{"fix: treat same-tick ready as positive (#1324)", 1324, true},
		{"#1324 leading cite", 1324, true},
		{"fix: prefix match must not bind #13245", 1324, false},
		{"fix: glued token binds nothing x#1324", 1324, false},
		{"fix: hyphen-glued token binds nothing -#1324", 1324, false},
		{"fix: another issue #99", 1324, false},
		{"", 1324, false},
	}
	for _, tc := range cases {
		if got := SubjectCitesIssue(tc.subject, tc.issue); got != tc.want {
			t.Errorf("SubjectCitesIssue(%q, %d) = %v, want %v", tc.subject, tc.issue, got, tc.want)
		}
	}
}

// TestFirstResolvingSHA parses the `git log --pretty=format:%H<US>%s` stream the way
// the Python witness sweep does: newest-first, first subject citing #issue wins.
func TestFirstResolvingSHA(t *testing.T) {
	lines := strings.Join([]string{
		"aaa111\x1ffeat(other): unrelated work (#999)",
		"bbb222\x1ffix(dispatchtick): resolve the pick hold #1396 (fak dispatchtick)",
		"ccc333\x1ffix(dispatchtick): older cite #1396",
	}, "\n")
	if got := FirstResolvingSHA(lines, 1396); got != "bbb222" {
		t.Fatalf("FirstResolvingSHA = %q, want bbb222 (newest citing commit)", got)
	}
	if got := FirstResolvingSHA(lines, 1234); got != "" {
		t.Fatalf("FirstResolvingSHA for uncited issue = %q, want empty", got)
	}
	if got := FirstResolvingSHA("", 1396); got != "" {
		t.Fatalf("FirstResolvingSHA over empty log = %q, want empty", got)
	}
}

func TestCommitWitnessed(t *testing.T) {
	cases := []struct {
		verdict string
		witness string
		want    bool
	}{
		{"OK", WitnessOK, true},
		{"ok", WitnessOK, true},
		{"OK", "subject-only", false},
		{"CLAIM_UNWITNESSED", WitnessOK, false},
		{"", "", false},
	}
	for _, tc := range cases {
		if got := CommitWitnessed(tc.verdict, tc.witness); got != tc.want {
			t.Errorf("CommitWitnessed(%q, %q) = %v, want %v", tc.verdict, tc.witness, got, tc.want)
		}
	}
}

// TestWitnessRecordMapMirrorsSidecarShape pins the .witness sidecar payload shape the
// Python dispatcher writes (tools/issue_resolve_dispatch.py witness_exited_workers):
// a no-commit record carries a reason and explicit nulls; a graded record carries no
// reason key. Downstream readers (the operator card, the Python picker) parse both.
func TestWitnessRecordMapMirrorsSidecarShape(t *testing.T) {
	noCommit := WitnessRecord{Issue: 1338, Log: "resolve-1338-20260702-000000.log", Claim: ClaimNoCommit, Reason: NoCommitSelfModify}
	m := noCommit.Map()
	if m["issue"] != 1338 || m["claim"] != ClaimNoCommit || m["reason"] != NoCommitSelfModify {
		t.Fatalf("no-commit map = %#v", m)
	}
	for _, key := range []string{"sha", "verdict", "witness"} {
		if v, ok := m[key]; !ok || v != nil {
			t.Fatalf("no-commit map[%q] = %v (present %v), want explicit null", key, v, ok)
		}
	}
	graded := WitnessRecord{Issue: 1344, Log: "resolve-1344-20260702-000000.log", SHA: "abc123", Claim: ClaimWitnessed, Verdict: "OK", Witness: WitnessOK}
	gm := graded.Map()
	if gm["sha"] != "abc123" || gm["verdict"] != "OK" || gm["witness"] != WitnessOK {
		t.Fatalf("graded map = %#v", gm)
	}
	if _, ok := gm["reason"]; ok {
		t.Fatalf("graded record must not carry a reason key: %#v", gm)
	}
}
