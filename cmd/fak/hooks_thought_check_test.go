package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/issuecheck"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func validManagedThoughtCheckFake(t *testing.T) *thoughtCheckFakeGH {
	return validManagedThoughtCheckFakeForIssue(t, testThoughtCheckIssue().Number)
}

func validManagedThoughtCheckFakeForIssue(t *testing.T, issueNumber int) *thoughtCheckFakeGH {
	t.Helper()
	issue := testThoughtCheckIssue()
	issue.Number = issueNumber
	body, err := issuecheck.FormatReviewComment(issue, testThoughtCheckReview(t, issue))
	if err != nil {
		t.Fatal(err)
	}
	return &thoughtCheckFakeGH{
		issue: issue,
		comments: []thoughtCheckCommentJSON{
			thoughtCheckComment(401, body, "fak-review-bot"),
		},
	}
}

func TestManagedThoughtCheckAdmissionUsesDurableIntentWithoutIssueEnv(t *testing.T) {
	_, wt, base := newSingleReapFixture(t)
	if err := workerworktree.SaveIntent(wt, base, "fix(cmd): reviewed worker (#17) (fak cmd)", []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	t.Setenv(managedIssueEnv, "")
	t.Setenv(managedThoughtCheckModeEnv, "enforce")
	fake := validManagedThoughtCheckFakeForIssue(t, 17)
	withManagedThoughtCheckRunner(t, fake.run, nil)

	admission := managedWorkerThoughtCheckAdmission(context.Background(), wt)
	if !admission.OK || admission.Issue != 17 || !admission.Required {
		t.Fatalf("intent-bound admission = %+v, want valid issue #17", admission)
	}

	before := len(fake.calls)
	t.Setenv(managedIssueEnv, "18")
	admission = managedWorkerThoughtCheckAdmission(context.Background(), wt)
	if !admission.Blocks() || !admission.BindingError || !strings.Contains(admission.Reason, "intent binds #17") {
		t.Fatalf("mismatched env admission = %+v, want hard binding refusal", admission)
	}
	if len(fake.calls) != before {
		t.Fatalf("mismatched env reached GitHub: calls before=%d after=%d", before, len(fake.calls))
	}
}

func TestManagedThoughtCheckAdmissionMissingIntentFailsClosedButHumanCommitIsUnbound(t *testing.T) {
	_, wt, _ := newSingleReapFixture(t)
	t.Setenv(managedIssueEnv, "")
	t.Setenv(managedThoughtCheckModeEnv, "off")
	managed := managedWorkerThoughtCheckAdmission(context.Background(), wt)
	if !managed.Blocks() || !managed.BindingError || !strings.Contains(managed.Reason, "intent") {
		t.Fatalf("managed admission without intent = %+v, want fail closed even in off mode", managed)
	}

	human := managedWorkerThoughtCheckAdmission(context.Background(), t.TempDir())
	if human.Required || !human.OK || human.Blocks() {
		t.Fatalf("unmanaged human admission = %+v, want compatible no-op", human)
	}
}

func withManagedThoughtCheckRunner(t *testing.T, runner thoughtCheckRunner, wantDir *string) {
	t.Helper()
	previous := managedThoughtCheckRunnerFactory
	managedThoughtCheckRunnerFactory = func(_ context.Context, dir string) thoughtCheckRunner {
		if wantDir != nil {
			*wantDir = dir
		}
		return runner
	}
	t.Cleanup(func() { managedThoughtCheckRunnerFactory = previous })
}

func TestManagedThoughtCheckAdmissionValidAndSelectedIssueOnly(t *testing.T) {
	fake := validManagedThoughtCheckFake(t)
	admission := verifyManagedThoughtCheck(context.Background(), fake.issue.Number, "", fake.run)
	if !admission.OK || admission.CommentID != 401 || admission.IssueDigest == "" || admission.CatalogVersion != issuecheck.CatalogVersion {
		t.Fatalf("admission = %+v", admission)
	}
	for _, call := range fake.calls {
		joined := strings.Join(call, " ")
		if strings.Contains(joined, "issue list") || strings.Contains(joined, "/issues?") {
			t.Fatalf("admission scanned backlog instead of selected issue: %v", call)
		}
	}
}

func TestManagedThoughtCheckAdmissionTrustsStableRepoOwnerNotViewer(t *testing.T) {
	issue := testThoughtCheckIssue()
	body, err := issuecheck.FormatReviewComment(issue, testThoughtCheckReview(t, issue))
	if err != nil {
		t.Fatal(err)
	}
	fake := &thoughtCheckFakeGH{
		issue: issue, repoOwner: "stable-owner", viewer: "different-credential",
		comments: []thoughtCheckCommentJSON{thoughtCheckComment(402, body, "stable-owner")},
	}
	admission := verifyManagedThoughtCheck(context.Background(), issue.Number, "enforce", fake.run)
	if !admission.OK || admission.CommentID != 402 {
		t.Fatalf("stable-owner admission = %+v", admission)
	}
	for _, call := range fake.calls {
		if len(call) > 1 && call[1] == "user" {
			t.Fatalf("read-only admission incorrectly trusted credential viewer: %v", call)
		}
	}
}

func TestManagedThoughtCheckAdmissionRefusesMissingStaleDuplicateAndNetwork(t *testing.T) {
	issue := testThoughtCheckIssue()
	valid := validManagedThoughtCheckFake(t)
	staleBody := valid.comments[0].Body
	staleIssue := issue
	staleIssue.Body += "\nMaterial acceptance change."

	tests := []struct {
		name   string
		runner thoughtCheckRunner
		want   string
	}{
		{name: "missing", runner: (&thoughtCheckFakeGH{issue: issue}).run, want: "missing"},
		{name: "stale", runner: (&thoughtCheckFakeGH{issue: staleIssue, comments: []thoughtCheckCommentJSON{thoughtCheckComment(1, staleBody, "fak-review-bot")}}).run, want: "stale"},
		{name: "duplicate", runner: (&thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{
			thoughtCheckComment(1, valid.comments[0].Body, "fak-review-bot"),
			thoughtCheckComment(2, valid.comments[0].Body, "fak-review-bot"),
		}}).run, want: "multiple"},
		{name: "network", runner: func(context.Context, ...string) ([]byte, error) { return nil, errors.New("network down") }, want: "network down"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			admission := verifyManagedThoughtCheck(context.Background(), issue.Number, "enforce", tc.runner)
			if admission.OK || !admission.Blocks() || !strings.Contains(strings.ToLower(admission.Reason), strings.ToLower(tc.want)) {
				t.Fatalf("admission = %+v, want blocking %q", admission, tc.want)
			}
		})
	}
}

func TestManagedThoughtCheckModesDefaultEnforceObserveOffAndInvalid(t *testing.T) {
	missing := &thoughtCheckFakeGH{issue: testThoughtCheckIssue()}
	if got := verifyManagedThoughtCheck(context.Background(), 17, "", missing.run); !got.Blocks() || got.Mode != "enforce" {
		t.Fatalf("default admission = %+v, want enforce refusal", got)
	}
	if got := verifyManagedThoughtCheck(context.Background(), 17, "observe", missing.run); got.Blocks() || got.OK || got.Mode != "observe" {
		t.Fatalf("observe admission = %+v, want visible nonblocking refusal", got)
	}
	calls := 0
	if got := verifyManagedThoughtCheck(context.Background(), 17, "off", func(context.Context, ...string) ([]byte, error) { calls++; return nil, errors.New("called") }); !got.OK || got.Blocks() || calls != 0 {
		t.Fatalf("off admission = %+v calls=%d, want skipped success", got, calls)
	}
	if got := verifyManagedThoughtCheck(context.Background(), 17, "invalid", missing.run); !got.Blocks() || !strings.Contains(got.Reason, managedThoughtCheckModeEnv) {
		t.Fatalf("invalid mode admission = %+v, want fail closed", got)
	}
}

func TestManagedThoughtCheckAdmissionUsesAggregateDeadlineAndRepoRoot(t *testing.T) {
	t.Setenv(managedIssueEnv, "17")
	t.Setenv(managedThoughtCheckModeEnv, "enforce")
	var gotDir string
	previous := managedThoughtCheckRunnerFactory
	managedThoughtCheckRunnerFactory = func(ctx context.Context, dir string) thoughtCheckRunner {
		gotDir = dir
		return func(context.Context, ...string) ([]byte, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}
	}
	t.Cleanup(func() { managedThoughtCheckRunnerFactory = previous })
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()
	admission := managedWorkerThoughtCheckAdmission(ctx, "C:/bound/repo")
	if !admission.Blocks() || !strings.Contains(strings.ToLower(admission.Reason), "deadline") {
		t.Fatalf("deadline admission = %+v", admission)
	}
	if gotDir != "C:/bound/repo" {
		t.Fatalf("runner dir = %q, want resolved repo root", gotDir)
	}
}

func TestManagedPreCommitRefusesBeforeOrdinaryGates(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	t.Setenv(managedIssueEnv, "17")
	t.Setenv(managedThoughtCheckModeEnv, "")
	missing := &thoughtCheckFakeGH{issue: testThoughtCheckIssue()}
	withManagedThoughtCheckRunner(t, missing.run, nil)
	var ordinary atomic.Int32
	withPreCommitGates(t, hooks.Gate{Name: "MUST_NOT_RUN", Check: func(*hooks.StagedDiff) ([]hooks.Finding, error) {
		ordinary.Add(1)
		return nil, nil
	}})
	var out, errb bytes.Buffer
	if code := runHooksPreCommit(&out, &errb, []string{"--root", repo}); code != 1 {
		t.Fatalf("pre-commit exit=%d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	if ordinary.Load() != 0 || !strings.Contains(errb.String(), managedThoughtCheckReason) {
		t.Fatalf("ordinary gates=%d stderr=%q", ordinary.Load(), errb.String())
	}
}

func TestManagedPreCommitObserveAndOffAreExplicitRollbackModes(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	for _, mode := range []string{"observe", "off"} {
		t.Run(mode, func(t *testing.T) {
			repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
			t.Setenv(managedIssueEnv, "17")
			t.Setenv(managedThoughtCheckModeEnv, mode)
			missing := &thoughtCheckFakeGH{issue: testThoughtCheckIssue()}
			calls := 0
			withManagedThoughtCheckRunner(t, func(ctx context.Context, args ...string) ([]byte, error) {
				calls++
				return missing.run(ctx, args...)
			}, nil)
			var ordinary atomic.Int32
			withPreCommitGates(t, hooks.Gate{Name: "ROLLBACK_REACHED", Check: func(*hooks.StagedDiff) ([]hooks.Finding, error) {
				ordinary.Add(1)
				return nil, nil
			}})
			var out, errb bytes.Buffer
			if code := runHooksPreCommit(&out, &errb, []string{"--root", repo}); code != 0 || ordinary.Load() != 1 {
				t.Fatalf("mode=%s exit=%d ordinary=%d stdout=%q stderr=%q", mode, code, ordinary.Load(), out.String(), errb.String())
			}
			if mode == "observe" && (calls == 0 || !strings.Contains(errb.String(), managedThoughtCheckSchema)) {
				t.Fatalf("observe must emit typed nonblocking refusal: calls=%d stderr=%q", calls, errb.String())
			}
			if mode == "off" && calls != 0 {
				t.Fatalf("off mode invoked GitHub %d time(s)", calls)
			}
		})
	}
}

func TestManagedPreCommitShellBackstopFailsClosedOnlyInEnforce(t *testing.T) {
	path := filepath.Join(repoRootFromTest(t), "tools", "githooks", "pre-commit")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, want := range []string{
		`FAK_THOUGHT_CHECK_MODE:-enforce`,
		`enforce|observe|off`,
		`! command -v fak`,
		`managed worker thought-check verifier failed closed`,
		`"$_fak_thought_mode" = "enforce"`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("managed pre-commit shell backstop missing %q", want)
		}
	}
}

func TestManagedThoughtCheckRollbackIsOnPreCommitUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHooksPreCommit(&out, &errb, []string{"--help"}); code != 2 {
		t.Fatalf("help exit=%d, want usage exit 2", code)
	}
	for _, want := range []string{managedThoughtCheckModeEnv, "enforce|observe|off", "default enforce"} {
		if !strings.Contains(errb.String(), want) {
			t.Errorf("pre-commit usage missing %q: %s", want, errb.String())
		}
	}
}

func TestWorktreeLandThoughtCheckBinding(t *testing.T) {
	_, wt, base := newSingleReapFixture(t)
	if err := workerworktree.SaveIntent(wt, base, "fix(cmd): enforce review (#9568) (fak cmd)", []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	msg := filepath.Join(filepath.Dir(wt), ".fak-worker-intents", filepath.Base(wt)+".message")
	if issue, required, reason := worktreeLandThoughtCheckBinding(wt, msg, ""); issue != 9568 || !required || reason != "" {
		t.Fatalf("binding = issue=%d required=%v reason=%q", issue, required, reason)
	}
	ambiguous := filepath.Join(t.TempDir(), "ambiguous.message")
	if err := os.WriteFile(ambiguous, []byte("fix(cmd): ambiguous (#9568 #17) (fak cmd)\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, required, reason := worktreeLandThoughtCheckBinding(t.TempDir(), ambiguous, ""); !required || !strings.Contains(reason, "exactly one") {
		t.Fatalf("ambiguous binding required=%v reason=%q", required, reason)
	}
	operatorMsg := filepath.Join(t.TempDir(), "operator.message")
	if err := os.WriteFile(operatorMsg, []byte("chore: unbound operator land\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, required, reason := worktreeLandThoughtCheckBinding(t.TempDir(), operatorMsg, ""); required || reason != "" {
		t.Fatalf("unbound operator land required=%v reason=%q", required, reason)
	}
}

func TestWorktreeLandThoughtCheckRejectsCoordinatorMessageSubstitution(t *testing.T) {
	_, wt, base := newSingleReapFixture(t)
	message := "fix(cmd): enforce review (#9568) (fak cmd)"
	if err := workerworktree.SaveIntent(wt, base, message, []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	canonical := filepath.Join(filepath.Dir(wt), ".fak-worker-intents", filepath.Base(wt)+".message")
	alternate := filepath.Join(t.TempDir(), "same-issue.message")
	if err := os.WriteFile(alternate, []byte(message+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, required, reason := worktreeLandThoughtCheckBinding(wt, alternate, ""); !required || !strings.Contains(reason, "coordinator-saved commit message path") {
		t.Fatalf("alternate same-issue path required=%v reason=%q", required, reason)
	}

	altered := "fix(cmd): substituted same issue body (#9568) (fak cmd)\n"
	if err := os.WriteFile(canonical, []byte(altered), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, required, reason := worktreeLandThoughtCheckBinding(wt, canonical, ""); !required || !strings.Contains(reason, "message mirror") {
		t.Fatalf("altered same-issue body required=%v reason=%q", required, reason)
	}
}

func TestWorktreeLandThoughtCheckUsesAggregateDeadline(t *testing.T) {
	t.Setenv(managedIssueEnv, "17")
	t.Setenv(managedThoughtCheckModeEnv, "enforce")
	previousFactory := managedThoughtCheckRunnerFactory
	managedThoughtCheckRunnerFactory = func(ctx context.Context, _ string) thoughtCheckRunner {
		return func(context.Context, ...string) ([]byte, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}
	}
	previousTimeout := managedThoughtCheckLandMax
	managedThoughtCheckLandMax = 15 * time.Millisecond
	t.Cleanup(func() {
		managedThoughtCheckRunnerFactory = previousFactory
		managedThoughtCheckLandMax = previousTimeout
	})
	hook := composeWorktreeThoughtCheckVerify(t.TempDir(), "", nil)
	ok, detail := hook(t.TempDir())
	if ok || !strings.Contains(strings.ToLower(detail), "deadline") || !strings.Contains(detail, managedThoughtCheckReason) {
		t.Fatalf("land timeout verdict ok=%v detail=%q", ok, detail)
	}
}

func TestCommitTreeLandCannotBypassMissingThoughtCheck(t *testing.T) {
	repo, wt, base := newSingleReapFixture(t)
	if err := os.WriteFile(filepath.Join(wt, "owned.txt"), []byte("worker edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	message := "fix(cmd): enforce review (#17) (fak cmd)"
	if err := workerworktree.SaveIntent(wt, base, message, []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	rows, err := workerworktree.Inventory(repo, nil)
	if err != nil || len(rows) != 1 {
		t.Fatalf("inventory rows=%+v err=%v", rows, err)
	}
	var msgFile string
	for i, arg := range rows[0].LandArgv {
		if arg == "--msg-file" && i+1 < len(rows[0].LandArgv) {
			msgFile = rows[0].LandArgv[i+1]
		}
	}
	if msgFile == "" {
		t.Fatalf("inventory land argv lacks message: %v", rows[0].LandArgv)
	}
	missing := &thoughtCheckFakeGH{issue: testThoughtCheckIssue()}
	withManagedThoughtCheckRunner(t, missing.run, nil)
	t.Setenv(managedIssueEnv, "")
	t.Setenv(managedThoughtCheckModeEnv, "enforce")
	hook := composeWorktreeThoughtCheckVerify(wt, msgFile, nil)
	res := workerworktree.Land(repo, wt, base, msgFile, []string{"owned.txt"}, hook, nil)
	if res.OK || res.Applied || res.Committed || !strings.Contains(res.Reason, managedThoughtCheckReason) {
		t.Fatalf("land bypassed thought-check: %+v", res)
	}
	cmd := exec.Command("git", "-C", repo, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil || strings.TrimSpace(string(out)) != base {
		t.Fatalf("trunk moved despite refusal: head=%q base=%q err=%v", out, base, err)
	}
}

func TestCommitTreeLandCannotSubstituteAlternateReviewedIssue(t *testing.T) {
	repo, wt, base := newSingleReapFixture(t)
	if err := os.WriteFile(filepath.Join(wt, "owned.txt"), []byte("worker edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := workerworktree.SaveIntent(wt, base, "fix(cmd): assigned issue (#17) (fak cmd)", []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	alternateMessage := filepath.Join(t.TempDir(), "alternate.message")
	if err := os.WriteFile(alternateMessage, []byte("fix(cmd): alternate reviewed issue (#18) (fak cmd)\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	alternate := validManagedThoughtCheckFakeForIssue(t, 18)
	withManagedThoughtCheckRunner(t, alternate.run, nil)
	t.Setenv(managedIssueEnv, "")
	t.Setenv(managedThoughtCheckModeEnv, "enforce")

	hook := composeWorktreeThoughtCheckVerify(wt, alternateMessage, nil)
	res := workerworktree.Land(repo, wt, base, alternateMessage, []string{"owned.txt"}, hook, nil)
	if res.OK || res.Applied || res.Committed || !strings.Contains(res.Reason, "coordinator-saved commit message path") {
		t.Fatalf("alternate reviewed issue bypassed durable intent: %+v", res)
	}
	if len(alternate.calls) != 0 {
		t.Fatalf("alternate issue reached GitHub verifier before binding refusal: %v", alternate.calls)
	}
	cmd := exec.Command("git", "-C", repo, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil || strings.TrimSpace(string(out)) != base {
		t.Fatalf("trunk moved despite alternate-issue refusal: head=%q base=%q err=%v", out, base, err)
	}
}

func TestCommitTreeLandProceedsWithValidThoughtCheck(t *testing.T) {
	repo, wt, base := newSingleReapFixture(t)
	if err := os.WriteFile(filepath.Join(wt, "owned.txt"), []byte("reviewed worker edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	message := "fix(cmd): enforce review (#17) (fak cmd)"
	if err := workerworktree.SaveIntent(wt, base, message, []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	rows, err := workerworktree.Inventory(repo, nil)
	if err != nil || len(rows) != 1 {
		t.Fatalf("inventory rows=%+v err=%v", rows, err)
	}
	var msgFile string
	for i, arg := range rows[0].LandArgv {
		if arg == "--msg-file" && i+1 < len(rows[0].LandArgv) {
			msgFile = rows[0].LandArgv[i+1]
		}
	}
	valid := validManagedThoughtCheckFake(t)
	withManagedThoughtCheckRunner(t, valid.run, nil)
	t.Setenv(managedIssueEnv, "")
	t.Setenv(managedThoughtCheckModeEnv, "enforce")
	hook := composeWorktreeThoughtCheckVerify(wt, msgFile, nil)
	res := workerworktree.Land(repo, wt, base, msgFile, []string{"owned.txt"}, hook, nil)
	if !res.OK || !res.Applied || !res.Committed {
		t.Fatalf("valid reviewed land refused: %+v", res)
	}
	cmd := exec.Command("git", "-C", repo, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil || strings.TrimSpace(string(out)) == base {
		t.Fatalf("trunk did not advance after valid review: head=%q base=%q err=%v", out, base, err)
	}
}
