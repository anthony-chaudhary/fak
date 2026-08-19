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
		{"claude cap banner", "You've hit your usage limit · resets 8pm\n", 700, NoCommitUsageCap},
		{"codex cap banner", "You've hit your usage limit. Visit https://chatgpt.com/codex\n", 700, NoCommitUsageCap},
		{"glm quota wall", "Limit Exhausted: your limit will reset at 2026-07-03\n", 120000, NoCommitUsageCap},
		{"glm usage wall", "usage limit reached for zai-coding-plan\n", 300, NoCommitUsageCap},
		{"login wall is genuine auth (401 /login)", "API Error: 401 Please run /login\n", 300, NoCommitAuthWall},
		{"not-logged-in is genuine auth", "Not logged in. Run /login to continue\n", 300, NoCommitAuthWall},
		{"credit wall is genuine auth", "Your credit balance is too low to run this request\n", 300, NoCommitAuthWall},
		{"unknown/unentitled model is switchable", "error: model \"fable\" is not available for this account\n", 300, NoCommitModelUnknown},
		{"transient overload is rate limit", "API Error: 529 overloaded_error\n", 300, NoCommitRateLimit},
		// Codex/OpenAI-wire throttles name the 429 family WITHOUT a literal "429" in the tail;
		// each must grade rate_limit (not unknown) so the concurrency-backoff term counts it.
		{"codex per-minute throttle (no 429 code)", "Rate limit reached for gpt-5-codex on tokens per min (TPM). Please try again in 2s.\n", 300, NoCommitRateLimit},
		{"codex too-many-requests (no 429 code)", "stream error: Too Many Requests; retrying 1/5\n", 300, NoCommitRateLimit},
		{"codex rate_limit_exceeded code", "{\"error\":{\"code\":\"rate_limit_exceeded\"}}\n", 300, NoCommitRateLimit},
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

// TestGradeTestRun pins the #3838 test-run grader: the ONLY path to GREEN is a test
// that actually ran AND passed. A runner that never fired (ran=false) grades UNRUN
// regardless of the passed bit, so a disabled/faulted/no-test-package run can never
// masquerade as a pass; a run that fired and failed grades RED.
func TestGradeTestRun(t *testing.T) {
	cases := []struct {
		ran    bool
		passed bool
		want   string
	}{
		{true, true, ClaimTestGreen},
		{true, false, ClaimTestRed},
		{false, false, ClaimTestUnrun},
		{false, true, ClaimTestUnrun}, // never-ran outranks a stale passed bit
	}
	for _, tc := range cases {
		if got := GradeTestRun(tc.ran, tc.passed); got != tc.want {
			t.Errorf("GradeTestRun(ran=%v, passed=%v) = %q, want %q", tc.ran, tc.passed, got, tc.want)
		}
	}
}

// TestWitnessRecordMapEmitsTestClaim pins the #3838 sidecar rung: a graded record with
// a TestClaim emits a test_claim key ALONGSIDE the diff-shape verdict/witness, while a
// record with no test claim (a no-commit slot) omits the key entirely, so its sidecar
// stays byte-identical to before this rung.
func TestWitnessRecordMapEmitsTestClaim(t *testing.T) {
	green := WitnessRecord{Issue: 3838, SHA: "abc123", Claim: ClaimWitnessed, Verdict: "OK", Witness: WitnessOK, TestClaim: ClaimTestGreen}
	if gm := green.Map(); gm["test_claim"] != ClaimTestGreen || gm["witness"] != WitnessOK {
		t.Fatalf("green map = %#v, want test_claim GREEN alongside diff-witness", gm)
	}
	red := WitnessRecord{Issue: 3839, SHA: "def456", Claim: ClaimWitnessed, Verdict: "OK", Witness: WitnessOK, TestClaim: ClaimTestRed}
	if rm := red.Map(); rm["test_claim"] != ClaimTestRed {
		t.Fatalf("red map = %#v, want test_claim RED even on a diff-witnessed commit", rm)
	}
	bare := WitnessRecord{Issue: 3840, Claim: ClaimNoCommit, Reason: NoCommitSelfModify}
	if _, ok := bare.Map()["test_claim"]; ok {
		t.Fatalf("no-commit record must omit test_claim: %#v", bare.Map())
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
		// The model-switchable trio is handled by Layer-2 downgrade re-dispatch, NOT by
		// holding — so a usage_cap / model_unknown / rate_limit slot is never held out.
		{Issue: 1346, Claim: ClaimNoCommit, Reason: NoCommitUsageCap},
		{Issue: 1347, Claim: ClaimNoCommit, Reason: NoCommitModelUnknown},
		{Issue: 1348, Claim: ClaimNoCommit, Reason: NoCommitRateLimit},
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

// TestModelSwitchableReason pins which no-commit classes Layer-2 re-dispatch acts on:
// the switchable trio (a different model can clear them) vs the guard refusals, the
// genuine auth wall, off-trunk, banner no-op, and unknown (a model switch cannot).
func TestModelSwitchableReason(t *testing.T) {
	for _, r := range []string{NoCommitUsageCap, NoCommitModelUnknown, NoCommitRateLimit} {
		if !ModelSwitchableReason(r) {
			t.Errorf("ModelSwitchableReason(%q) = false, want true", r)
		}
	}
	for _, r := range []string{
		NoCommitSelfModify, NoCommitPolicyBlock, NoCommitAuthWall,
		NoCommitOffTrunk, NoCommitBannerNoop, NoCommitUnknown, "",
	} {
		if ModelSwitchableReason(r) {
			t.Errorf("ModelSwitchableReason(%q) = true, want false", r)
		}
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

// TestNextDowngradeModel walks the downgrade ladder: a seat-default slot advances to the
// chain head, a pinned slot to the next rung, the last rung exhausts, and an off-ladder
// model falls to the head.
func TestNextDowngradeModel(t *testing.T) {
	chain := []string{"claude-opus-4-8", "claude-sonnet-5"}
	if m, ok := NextDowngradeModel("", chain); !ok || m != "claude-opus-4-8" {
		t.Fatalf("seat-default -> %q,%v want claude-opus-4-8,true", m, ok)
	}
	if m, ok := NextDowngradeModel("claude-opus-4-8", chain); !ok || m != "claude-sonnet-5" {
		t.Fatalf("opus -> %q,%v want claude-sonnet-5,true", m, ok)
	}
	if _, ok := NextDowngradeModel("claude-sonnet-5", chain); ok {
		t.Fatalf("last rung must exhaust the ladder")
	}
	if _, ok := NextDowngradeModel("anything", nil); ok {
		t.Fatalf("empty chain must exhaust")
	}
	if m, ok := NextDowngradeModel("some-off-ladder-model", chain); !ok || m != "claude-opus-4-8" {
		t.Fatalf("off-ladder -> %q,%v want head,true", m, ok)
	}
}

// TestModelDowngradeReDispatch maps only the model-switchable no-commit slots to their next
// rung; guard refusals, auth walls, witnessed commits, and exhausted ladders are omitted.
func TestModelDowngradeReDispatch(t *testing.T) {
	chain := []string{"claude-opus-4-8", "claude-sonnet-5"}
	records := []WitnessRecord{
		{Issue: 10, Claim: ClaimNoCommit, Reason: NoCommitUsageCap},                            // seat-default -> opus
		{Issue: 11, Claim: ClaimNoCommit, Reason: NoCommitRateLimit, Model: "claude-opus-4-8"}, // -> sonnet
		{Issue: 12, Claim: ClaimNoCommit, Reason: NoCommitSelfModify},                          // guard refusal: omitted
		{Issue: 13, Claim: ClaimNoCommit, Reason: NoCommitAuthWall},                            // auth wall: omitted
		{Issue: 14, Claim: ClaimWitnessed},                                                     // shipped: omitted
		{Issue: 15, Claim: ClaimNoCommit, Reason: NoCommitRateLimit, Model: "claude-sonnet-5"}, // exhausted: omitted
	}
	got := ModelDowngradeReDispatch(records, chain)
	want := map[int]string{10: "claude-opus-4-8", 11: "claude-sonnet-5"}
	if len(got) != len(want) {
		t.Fatalf("downgrade map = %v, want %v", got, want)
	}
	for issue, model := range want {
		if got[issue] != model {
			t.Fatalf("issue %d -> %q, want %q", issue, got[issue], model)
		}
	}
}

func TestWitnessRecordMapCarriesExactSessionIdentity(t *testing.T) {
	m := (WitnessRecord{SessionID: "sess-3330", RegistrationID: "reg-3330", Claim: ClaimWitnessed}).Map()
	if m["session_id"] != "sess-3330" || m["registration_id"] != "reg-3330" {
		t.Fatalf("identity fields missing from sidecar: %#v", m)
	}
}
