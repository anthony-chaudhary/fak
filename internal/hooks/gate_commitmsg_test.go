package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// TestSuggestGradeableSubject_correctsDeterministicFailures asserts that the two DETERMINISTIC
// gradeability failures — a near-miss conventional type and an inflected leading verb — earn a
// concrete, self-verified rewrite, while any case that would require a guess earns "".
func TestSuggestGradeableSubject_correctsDeterministicFailures(t *testing.T) {
	cases := []struct {
		name    string
		subject string
		want    string
	}{
		// Near-miss type: the description verb is already fine, only the type is wrong.
		{"feature->feat", "feature(gateway): add the reclaim path", "feat(gateway): add the reclaim path"},
		{"fixes-type->fix", "fixes(policy): correct the rule", "fix(policy): correct the rule"},
		{"documentation->docs", "documentation: clarify the runbook", "docs: clarify the runbook"},
		{"tests->test", "tests(gateway): cover the slot path", "test(gateway): cover the slot path"},
		{"bang preserved", "feature(api)!: add the breaking flag", "feat(api)!: add the breaking flag"},

		// Inflected leading verb: type is valid, the verb just needs its imperative base.
		{"past-ed", "feat(gateway): added a retry", "feat(gateway): add a retry"},
		{"past-wired", "feat(gateway): wired the seam", "feat(gateway): wire the seam"},
		{"gerund-caching", "perf(cache): caching the results", "perf(cache): cache the results"},
		{"gerund-wiring", "feat(x): wiring the panel", "feat(x): wire the panel"},
		{"third-person-fixes", "fix(x): fixes the leak", "fix(x): fix the leak"},
		{"doubled-pinning", "refactor(x): pinning the default", "refactor(x): pin the default"},

		// Both wrong at once: near-miss type AND inflected verb.
		{"type+verb", "feature(gateway): added a retry", "feat(gateway): add a retry"},

		// No safe suggestion — must stay "".
		{"empty", "", ""},
		{"no-conventional-prefix", "fixed the parser crash", ""},
		{"unknown-type-no-correction", "improvement(x): add a thing", ""},
		{"genuinely-noun-led", "feat(gateway): posture improvements", ""},
		{"decorated-lead", "feat(x): `added` a retry", ""},
		{"already-gradeable", "feat(gateway): add a retry", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := suggestGradeableSubject(tc.subject)
			if got != tc.want {
				t.Fatalf("suggestGradeableSubject(%q) = %q, want %q", tc.subject, got, tc.want)
			}
			// Every non-empty suggestion must actually be gradeable — the self-verify contract.
			if got != "" {
				if ok, why := CommitMsgVerdict(got); !ok {
					t.Fatalf("suggested subject %q is not gradeable: %s", got, why)
				}
			}
		})
	}
}

// TestImperativeBase_membershipDecides spot-checks the over-generative base resolver: a form that
// derives from a recognized verb resolves, an unrelated noun does not.
func TestImperativeBase_membershipDecides(t *testing.T) {
	cases := map[string]string{
		"added":   "add",
		"caching": "cache",
		"wiring":  "wire",
		"fixes":   "fix",
		"pinning": "pin",
		"built":   "build",
		"posture": "", // noun, no verb base
		"the":     "", // stopword
		"retry":   "", // noun
		"add":     "add",
	}
	for in, want := range cases {
		if got := imperativeBase(in); got != want {
			t.Errorf("imperativeBase(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestLintCommitMessage_gradeabilitySuggestionComposesTrailer is the end-to-end proof: a subject
// that is BOTH non-gradeable (wrong type) AND unstamped earns a single suggested subject that fixes
// both — the gradeable rewrite with the path-implied trailer appended.
func TestLintCommitMessage_gradeabilitySuggestionComposesTrailer(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("feature(gateway): added the slot reclaim path", []string{"internal/gateway/server.go"}, root)
	if r.OK {
		t.Fatalf("expected NOT ok for a non-gradeable, unstamped subject")
	}
	if r.Gradeable {
		t.Fatalf("subject leads with an unknown type; should not be gradeable")
	}
	want := "feat(gateway): add the slot reclaim path (fak gateway)"
	if r.SuggestedSubject != want {
		t.Fatalf("SuggestedSubject = %q, want %q", r.SuggestedSubject, want)
	}
	// The composed suggestion must itself pass the gate.
	if ok, why := CommitMsgVerdict(r.SuggestedSubject); !ok {
		t.Fatalf("composed suggestion %q is not gradeable: %s", r.SuggestedSubject, why)
	}
}

// TestCommitMsgVerdict_isolateVerbAccepted is the #3912 regression fixture. `fak commit --preview`
// graded `test(codex): isolate direct continuation override (#3023) (fak cmd)` as ungradeable
// (55/F, BLOCKED) because `isolate` was missing from commitVerbs, while the mutating `fak commit`
// — which never routes the subject through this gate — committed the byte-identical subject and
// scored it 100/A. Two commands must grade one subject one way. `isolate` names a real, checkable
// action, so both now accept it: the shared verdict is gradeable, and the full preview lint that
// `--preview` renders returns an unblocked A-grade matching what the mutating commit reported.
func TestCommitMsgVerdict_isolateVerbAccepted(t *testing.T) {
	const reported = "test(codex): isolate direct continuation override (#3023) (fak cmd)"

	// CommitMsgVerdict is the shared gradeability function both `--preview` and the mutating
	// path's deriveCommitMessageStamp route through; it must now accept the exact #3912 subject.
	if ok, why := CommitMsgVerdict(reported); !ok {
		t.Fatalf("CommitMsgVerdict rejected the #3912 subject as ungradeable: %s", why)
	}

	// The end-to-end pre-commit lint (`--preview`'s report) must carry no blocking issue and grade
	// A, so the preview no longer discards a subject the mutating commit would accept.
	root := writeLintRepo(t)
	r := LintCommitMessage(reported, []string{"cmd/fak/sessions_codex_hook_test.go"}, root)
	if !r.OK {
		t.Fatalf("preview lint blocked the #3912 subject; issues=%v", r.Issues)
	}
	if !r.Gradeable {
		t.Fatalf("the #3912 subject must be witness-gradeable")
	}
	if r.Grade != "A" {
		t.Errorf("preview grade = %s (score %d); want A to match the mutating commit's 100/A", r.Grade, r.Score)
	}
}

// TestLintCommitMessage_nounLedNoFalseSuggestion guards the negative: a genuinely noun-led subject
// (no deterministic fix) gets prose advice, not a fabricated rewrite.
func TestLintCommitMessage_nounLedNoFalseSuggestion(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("feat(gateway): posture improvements", []string{"internal/gateway/server.go"}, root)
	if r.Gradeable {
		t.Fatalf("noun-led description should not be gradeable")
	}
	if r.SuggestedSubject != "" {
		t.Fatalf("must not fabricate a rewrite for a noun-led subject, got %q", r.SuggestedSubject)
	}
	if !hasIssueContaining(r, "witness-gradeable") {
		t.Fatalf("want the witness-gradeable prose advice, got %v", r.Issues)
	}
}

// TestCommitMsgVerdict_rejectsSingleParentPseudoMerge proves issue #10882:
// A commit claiming to be a "Merge " must actually have >= 2 topological parents.
// Single-parent pseudo-merges are rejected with MERGE_WITNESS_FAIL.
func TestCommitMsgVerdict_rejectsSingleParentPseudoMerge(t *testing.T) {
	root := t.TempDir()
	runGit := func(args ...string) string {
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		windowgate.ConfigureBackgroundCommand(cmd)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}

	runGit("init", "-q")
	runGit("config", "user.name", "Test")
	runGit("config", "user.email", "test@example.com")
	runGit("config", "commit.gpgsign", "false")

	// 1. Initial single commit (0 parents)
	runGit("commit", "--allow-empty", "-m", "initial commit")

	// Attempting a pseudo-merge on single-parent branch
	ok, why := CommitMsgVerdictWithGit("Merge origin/main into main", root)
	if ok {
		t.Fatalf("CommitMsgVerdictWithGit accepted single-parent pseudo-merge")
	}
	if !strings.Contains(why, "MERGE_WITNESS_FAIL") {
		t.Errorf("why %q does not contain MERGE_WITNESS_FAIL", why)
	}

	// 2. Add another commit so HEAD has 1 parent
	runGit("commit", "--allow-empty", "-m", "feat(core): add feature (fak core)")
	ok, why = CommitMsgVerdictWithGit("Merge branch 'feature'", root)
	if ok {
		t.Fatalf("CommitMsgVerdictWithGit accepted single-parent pseudo-merge on second commit")
	}
	if !strings.Contains(why, "MERGE_WITNESS_FAIL") {
		t.Errorf("why %q does not contain MERGE_WITNESS_FAIL", why)
	}

	// 3. Test LintCommitMessage rejects it
	dosToml := "[lanes]\nconcurrent = [\"gateway\"]\n[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n"
	_ = os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644)
	r := LintCommitMessage("Merge branch 'feature'", []string{"internal/gateway/x.go"}, root)
	if r.OK {
		t.Fatalf("LintCommitMessage accepted single-parent pseudo-merge; report: %+v", r)
	}
	if r.Gradeable {
		t.Errorf("single-parent pseudo-merge should not be gradeable")
	}
	if !hasIssueContaining(r, "MERGE_WITNESS_FAIL") {
		t.Errorf("expected issue containing MERGE_WITNESS_FAIL; got issues=%v", r.Issues)
	}

	// 4. Create a branch and perform a real merge
	mainBranch := runGit("rev-parse", "--abbrev-ref", "HEAD")
	runGit("checkout", "-q", "-b", "side-branch")
	runGit("commit", "--allow-empty", "-m", "feat(side): side change (fak core)")
	runGit("checkout", "-q", mainBranch)
	runGit("commit", "--allow-empty", "-m", "feat(main): main change (fak core)")

	// Start merge with --no-commit: MERGE_HEAD exists
	runGit("merge", "--no-ff", "--no-commit", "side-branch")

	ok, why = CommitMsgVerdictWithGit("Merge branch 'side-branch'", root)
	if !ok {
		t.Fatalf("CommitMsgVerdictWithGit rejected real in-flight merge: %s", why)
	}
	r = LintCommitMessage("Merge branch 'side-branch'", []string{"internal/gateway/x.go"}, root)
	if !r.OK || r.StampKind != "exempt" {
		t.Fatalf("LintCommitMessage rejected real in-flight merge: ok=%v kind=%s issues=%v", r.OK, r.StampKind, r.Issues)
	}

	// Complete merge: HEAD has 2 parents
	runGit("commit", "-m", "Merge branch 'side-branch'")
	ok, why = CommitMsgVerdictWithGit("Merge branch 'side-branch'", root)
	if !ok {
		t.Fatalf("CommitMsgVerdictWithGit rejected committed multi-parent merge: %s", why)
	}
}
