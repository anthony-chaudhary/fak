package safecommit

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

// fakeGit is a scriptable Runner: it answers each git invocation from a small reply table
// keyed on a join of the argv, records every argv it saw, and lets a test assert exactly
// which git commands were (and were NOT) issued. err is returned only to model
// git-not-executable.
type fakeGit struct {
	// reply maps a key (see keyFor) to a canned (stdout, code). A missing key returns
	// ("", 0) — a benign default that suits the read probes.
	reply map[string]reply
	calls [][]string
	err   error // non-nil => git could not be executed (infra failure on every call)
}

type reply struct {
	out  string
	code int
}

// keyFor reduces an argv to a stable lookup key. The first token always participates; for
// the multi-form subcommands the test cares about, a second token disambiguates.
func keyFor(args []string) string {
	if len(args) == 0 {
		return ""
	}
	switch args[0] {
	case "rev-parse":
		// "rev-parse --git-dir", "rev-parse HEAD", "rev-parse -q --verify MERGE_HEAD",
		// "rev-parse --verify --quiet origin/main"
		return strings.Join(args, " ")
	case "merge-base":
		// "merge-base HEAD origin/main"
		return strings.Join(args, " ")
	case "diff":
		if len(args) > 1 && args[1] == "--no-ext-diff" {
			return "diff-review"
		}
		// The stale-base guard issues two diffs that must be distinguishable:
		//   "diff <mb> origin/main -- P"  (peer-added since the fork)  -> "--" at index 3
		//   "diff origin/main -- P"       (origin vs working tree)     -> "--" at index 2
		// Key on the position of the "--" pathspec separator: index 2 is the single-ref
		// working-tree form; otherwise it is the two-ref peer form.
		if len(args) >= 3 && args[2] == "--" {
			return "diff-wt"
		}
		return "diff-peer"
	case "symbolic-ref", "status", "diff-tree", "commit", "push", "ls-files":
		return args[0]
	default:
		return args[0]
	}
}

func (f *fakeGit) run(_ context.Context, _ string, args ...string) (string, int, error) {
	if f.err != nil {
		return "", -1, f.err
	}
	f.calls = append(f.calls, append([]string(nil), args...))
	if r, ok := f.reply[keyFor(args)]; ok {
		return r.out, r.code, nil
	}
	return "", 0, nil
}

func (f *fakeGit) sawSubcommand(sub string) bool {
	for _, c := range f.calls {
		if len(c) > 0 && c[0] == sub {
			return true
		}
	}
	return false
}

// commitArgv returns the argv of the single `git commit ...` call, or nil if none issued.
func (f *fakeGit) commitArgv() []string {
	for _, c := range f.calls {
		if len(c) > 0 && c[0] == "commit" {
			return c
		}
	}
	return nil
}

func (f *fakeGit) argvFor(sub string) []string {
	for _, c := range f.calls {
		if len(c) > 0 && c[0] == sub {
			return c
		}
	}
	return nil
}

// okLock is a LockFunc that always grants a no-op lock and records release.
func okLock(released *bool) LockFunc {
	return func(LockOptions) (func(), error) {
		return func() {
			if released != nil {
				*released = true
			}
		}, nil
	}
}

// busyLock always reports the lock held.
func busyLock(LockOptions) (func(), error) { return nil, ErrLockBusy }

// onTrunkBase is the reply table for a healthy on-trunk repo with a staged change; tests
// overlay the few keys they vary.
func onTrunkBase() map[string]reply {
	return map[string]reply{
		"rev-parse --git-dir":              {out: ".git", code: 0},
		"symbolic-ref":                     {out: "main\n", code: 0},
		"rev-parse -q --verify MERGE_HEAD": {out: "", code: 0},
		"status":                           {out: " M internal/foo/bar.go\n", code: 0},
		"rev-parse HEAD":                   {out: "abc123\n", code: 0},
		"commit":                           {out: "", code: 0},
		"diff-tree":                        {out: "internal/foo/bar.go\n", code: 0},
		"diff-review":                      {out: "diff --git a/internal/foo/bar.go b/internal/foo/bar.go\n+change\n", code: 0},
		"ls-files":                         {out: "", code: 0},
		"push":                             {out: "", code: 0},
	}
}

func baseOpts() Options {
	return Options{
		Dir:     "/repo",
		Paths:   []string{"internal/foo/bar.go"},
		Message: "fix(foo): correct the bar — keep the cache prefix\n\n(fak safecommit)",
		SignOff: true,
	}
}

func decisionRecorder(t *testing.T) (*witness.Recorder, *[]witness.Decision) {
	t.Helper()
	var captured []witness.Decision
	runner := func(_ context.Context, _ string, args ...string) (string, int, error) {
		for i, a := range args {
			if a != "-F" || i+1 >= len(args) {
				continue
			}
			body, err := os.ReadFile(args[i+1])
			if err != nil {
				return "", 1, err
			}
			var d witness.Decision
			if err := json.Unmarshal([]byte(strings.TrimSpace(string(body))), &d); err != nil {
				return "", 1, err
			}
			captured = append(captured, d)
		}
		return "", 0, nil
	}
	return witness.NewRecorderWithRunner(runner, ""), &captured
}

func TestPathspecRace_isTheHeadlineGuard(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	// A peer raced: diff-tree shows our path PLUS a peer's file that no requested path covers.
	g.reply["diff-tree"] = reply{out: "internal/foo/bar.go\ninternal/peer/swept.go\n", code: 0}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if !res.Committed || res.Verified {
		t.Fatalf("race: want Committed && !Verified, got Committed=%v Verified=%v", res.Committed, res.Verified)
	}
	if res.Reason != ReasonPathspecRace {
		t.Fatalf("race: want reason %q, got %q", ReasonPathspecRace, res.Reason)
	}
	if len(res.RacedExtra) != 1 || res.RacedExtra[0] != "internal/peer/swept.go" {
		t.Fatalf("race: want RacedExtra=[internal/peer/swept.go], got %v", res.RacedExtra)
	}
	// The non-destructive remedy: NEVER reset/revert/push to "fix" a raced commit.
	for _, forbidden := range []string{"reset", "revert", "push"} {
		if g.sawSubcommand(forbidden) {
			t.Fatalf("race: must not issue %q; calls=%v", forbidden, g.calls)
		}
	}
	if res.Pushed {
		t.Fatalf("race: must not report Pushed")
	}
}

func TestDecisionRecorderRecordsPathspecAssertionPass(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	opts := baseOpts()
	rec, captured := decisionRecorder(t)
	opts.Recorder = rec

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if !res.Verified {
		t.Fatalf("happy path should verify, got %+v", res)
	}
	if len(*captured) != 1 {
		t.Fatalf("expected one recorded decision, got %d: %+v", len(*captured), *captured)
	}
	got := (*captured)[0]
	if got.Op != "safecommit" || got.Verdict != witness.VerdictAssertPass {
		t.Fatalf("recorded decision = %+v, want safecommit/assert-pass", got)
	}
	if got.PathspecAssertion != "committed-set==requested-set" {
		t.Fatalf("PathspecAssertion = %q", got.PathspecAssertion)
	}
	if len(got.Tree) != 1 || got.Tree[0] != "internal/foo/bar.go" {
		t.Fatalf("Tree = %+v", got.Tree)
	}
}

func TestDecisionRecorderRecordsPathspecRaceFailure(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["diff-tree"] = reply{out: "internal/foo/bar.go\ninternal/peer/swept.go\n", code: 0}
	opts := baseOpts()
	rec, captured := decisionRecorder(t)
	opts.Recorder = rec

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonPathspecRace {
		t.Fatalf("want PATHSPEC_RACE, got %+v", res)
	}
	if len(*captured) != 1 {
		t.Fatalf("expected one recorded decision, got %d: %+v", len(*captured), *captured)
	}
	got := (*captured)[0]
	if got.Op != "safecommit" || got.Verdict != witness.VerdictAssertFail || got.ReasonClass != ReasonPathspecRace {
		t.Fatalf("recorded decision = %+v, want safecommit/assert-fail/PATHSPEC_RACE", got)
	}
	if got.PathspecAssertion != "committed-set!=requested-set" {
		t.Fatalf("PathspecAssertion = %q", got.PathspecAssertion)
	}
}

func TestOffTrunk_refusesBeforeCommitting(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["symbolic-ref"] = reply{out: "feature/x\n", code: 0}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Reason != ReasonOffTrunk {
		t.Fatalf("want OFF_TRUNK, got %q", res.Reason)
	}
	if g.sawSubcommand("commit") {
		t.Fatalf("off-trunk must not attempt a commit; calls=%v", g.calls)
	}
	if !strings.Contains(res.Detail, "feature/x") {
		t.Fatalf("detail should name the branch, got %q", res.Detail)
	}
	if !strings.Contains(res.Detail, "expected development branch main") {
		t.Fatalf("detail should name the expected development branch, got %q", res.Detail)
	}
}

func TestDetachedHead_isOffTrunk(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	// symbolic-ref exits non-zero on a detached HEAD.
	g.reply["symbolic-ref"] = reply{out: "", code: 128}

	res, _ := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if res.Reason != ReasonOffTrunk {
		t.Fatalf("detached HEAD should be OFF_TRUNK, got %q", res.Reason)
	}
	if !strings.Contains(res.Detail, "detached") {
		t.Fatalf("detail should mention detached, got %q", res.Detail)
	}
}

func TestConfiguredDevelopmentBranchAllowsDev(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[branch_roles]\ndevelopment_branch = \"dev\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["symbolic-ref"] = reply{out: "dev\n", code: 0}
	opts := baseOpts()
	opts.Dir = root

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Verified {
		t.Fatalf("dev branch should be accepted by configured development branch, got %+v", res)
	}
	if !g.sawSubcommand("commit") {
		t.Fatalf("expected commit on configured development branch; calls=%v", g.calls)
	}
}

func TestConfiguredDevelopmentBranchRefusesMainAfterCutover(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[branch_roles]\ndevelopment_branch = \"dev\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["symbolic-ref"] = reply{out: "main\n", code: 0}
	opts := baseOpts()
	opts.Dir = root

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Reason != ReasonOffTrunk {
		t.Fatalf("want OFF_TRUNK for main when configured dev branch is dev, got %+v", res)
	}
	if !strings.Contains(res.Detail, "expected development branch dev") {
		t.Fatalf("detail should name configured dev branch, got %q", res.Detail)
	}
	if g.sawSubcommand("commit") {
		t.Fatalf("off-trunk must not attempt commit; calls=%v", g.calls)
	}
}

func TestExplicitTrunkOverrideWinsOverBranchRole(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[branch_roles]\ndevelopment_branch = \"dev\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["symbolic-ref"] = reply{out: "main\n", code: 0}
	opts := baseOpts()
	opts.Dir = root
	opts.Trunk = "main"

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Verified {
		t.Fatalf("explicit trunk override should accept main, got %+v", res)
	}
}

func TestMergeInProgress_refuses(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["rev-parse -q --verify MERGE_HEAD"] = reply{out: "deadbeef\n", code: 0}

	res, _ := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if res.Reason != ReasonMergeInProgress {
		t.Fatalf("want MERGE_IN_PROGRESS, got %q", res.Reason)
	}
	if g.sawSubcommand("commit") {
		t.Fatalf("merge-in-progress must not commit; calls=%v", g.calls)
	}
}

func TestCommitUsesMessageFileNotDashM(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Verified {
		t.Fatalf("happy path should verify, got %+v", res)
	}
	argv := g.commitArgv()
	if argv == nil {
		t.Fatalf("no commit issued")
	}
	joined := strings.Join(argv, " ")
	// -F <file> must be present; -m must NOT — the em-dash/multiline message would misparse.
	if !strings.Contains(joined, "-F ") {
		t.Fatalf("commit must use -F <file>, got %v", argv)
	}
	for _, a := range argv {
		if a == "-m" {
			t.Fatalf("commit must never use -m, got %v", argv)
		}
	}
	// Sign-off and the pathspec-on-the-commit (-- before paths) must be there.
	if !strings.Contains(joined, " -s ") && argv[1] != "-s" {
		t.Fatalf("commit should sign off (-s), got %v", argv)
	}
	if !strings.Contains(joined, " -- internal/foo/bar.go") {
		t.Fatalf("commit must put the pathspec on the commit after --, got %v", argv)
	}
}

func TestLockBusy_isAValueNotAnError(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	res, err := CommitWith(context.Background(), g.run, busyLock, baseOpts())
	if err != nil {
		t.Fatalf("LOCK_BUSY must be a value, not an error; got err=%v", err)
	}
	if res.Reason != ReasonLockBusy {
		t.Fatalf("want LOCK_BUSY, got %q", res.Reason)
	}
	if g.sawSubcommand("commit") {
		t.Fatalf("lock-busy must not commit; calls=%v", g.calls)
	}
}

func TestHookRefused_surfacesTheMessage(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["commit"] = reply{out: "PUBLIC_LEAK: refusing — token-shaped string in internal/foo/bar.go\n", code: 1}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Reason != ReasonHookRefused {
		t.Fatalf("want HOOK_REFUSED, got %q", res.Reason)
	}
	if res.Committed {
		t.Fatalf("a refused commit must not report Committed")
	}
	if !strings.Contains(res.Detail, "PUBLIC_LEAK") {
		t.Fatalf("detail should carry the hook message, got %q", res.Detail)
	}
}

func TestReviewRefuteBlocksBeforeCommit(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	locked := false
	opts := baseOpts()
	opts.Review = &ReviewOptions{
		Model:     "cheap-scout",
		Objective: "keep the loop honest",
		Reviewer: func(_ context.Context, req modelroute.ReviewRequest) (modelroute.ReviewResult, error) {
			if !strings.Contains(req.Diff, "+change") {
				t.Fatalf("reviewer did not receive diff: %q", req.Diff)
			}
			if req.Objective != "keep the loop honest" {
				t.Fatalf("objective = %q", req.Objective)
			}
			return modelroute.ReviewResult{
				Model:   req.Model,
				Verdict: modelroute.ReviewRefute,
				Reason:  "missing a regression test",
			}, nil
		},
	}
	lock := func(LockOptions) (func(), error) {
		locked = true
		return func() {}, nil
	}

	res, err := CommitWith(context.Background(), g.run, lock, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Reason != ReasonReviewRefuted {
		t.Fatalf("reason = %q, want %q", res.Reason, ReasonReviewRefuted)
	}
	if res.Review == nil || res.Review.Verdict != modelroute.ReviewRefute {
		t.Fatalf("review result not recorded: %+v", res.Review)
	}
	if g.sawSubcommand("commit") || locked {
		t.Fatalf("refuted review must stop before lock/commit; locked=%v calls=%v", locked, g.calls)
	}
}

func TestReviewUnavailableFailsOpenAndRecords(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	opts := baseOpts()
	opts.Review = &ReviewOptions{
		Model: "cheap-scout",
		Reviewer: func(context.Context, modelroute.ReviewRequest) (modelroute.ReviewResult, error) {
			return modelroute.ReviewResult{}, errors.New("review endpoint refused connection")
		},
	}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Verified || res.Reason != "" {
		t.Fatalf("unavailable reviewer must fail open to a verified commit, got %+v", res)
	}
	if res.Review == nil || res.Review.Verdict != modelroute.ReviewUnavailable {
		t.Fatalf("unavailable review was not recorded: %+v", res.Review)
	}
	if !strings.Contains(res.Review.Reason, "refused connection") {
		t.Fatalf("review reason lost connection error: %+v", res.Review)
	}
	if !g.sawSubcommand("commit") {
		t.Fatalf("fail-open review should still commit; calls=%v", g.calls)
	}
}

func TestNothingStaged_failsFastLockFree(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["status"] = reply{out: "", code: 0}
	locked := false
	lock := func(LockOptions) (func(), error) { locked = true; return func() {}, nil }

	res, _ := CommitWith(context.Background(), g.run, lock, baseOpts())
	if res.Reason != ReasonNothingStaged {
		t.Fatalf("want NOTHING_STAGED, got %q", res.Reason)
	}
	if locked {
		t.Fatalf("nothing-staged must fail BEFORE taking the lock")
	}
}

func TestNotARepo(t *testing.T) {
	g := &fakeGit{reply: map[string]reply{
		"rev-parse --git-dir": {out: "fatal: not a git repository", code: 128},
	}}
	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Reason != ReasonNotARepo {
		t.Fatalf("want NOT_A_REPO, got %q", res.Reason)
	}
}

func TestNoPaths(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	opts := baseOpts()
	opts.Paths = nil
	res, _ := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if res.Reason != ReasonNoPath {
		t.Fatalf("want NO_PATHS, got %q", res.Reason)
	}
	if g.sawSubcommand("rev-parse") {
		t.Fatalf("no-paths is a pure check; must not touch git")
	}
}

func TestEmptyMessage(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	opts := baseOpts()
	opts.Message = "   \n  "
	res, _ := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if res.Reason != ReasonEmptyMessage {
		t.Fatalf("want EMPTY_MESSAGE, got %q", res.Reason)
	}
}

func TestGitMissing_isAnInfraError(t *testing.T) {
	g := &fakeGit{err: errors.New("exec: \"git\": executable file not found in $PATH")}
	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err == nil {
		t.Fatalf("git-missing must return a non-nil infra error, got reason=%q", res.Reason)
	}
}

func TestDeletionIsNotARace(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	// status shows the path deleted; diff-tree lists exactly the requested (now deleted) path.
	g.reply["status"] = reply{out: " D internal/foo/bar.go\n", code: 0}
	g.reply["diff-tree"] = reply{out: "internal/foo/bar.go\n", code: 0}

	res, _ := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if !res.Verified || res.Reason != "" {
		t.Fatalf("a deletion of the requested path is exactly-requested, not a race; got %+v", res)
	}
}

func TestDeletionStagingUsesPathspecScopedAll(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["status"] = reply{out: " D internal/foo/bar.go\n", code: 0}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if !res.Verified || res.Reason != "" {
		t.Fatalf("deletion should commit cleanly, got %+v", res)
	}
	want := []string{"add", "--all", "--", "internal/foo/bar.go"}
	if got := g.argvFor("add"); strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("add argv = %q, want %q", got, want)
	}
}

func TestRequestedDirectoryCoversItsFiles(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	// Requesting a directory; committed files all live under it — no false race.
	g.reply["diff-tree"] = reply{out: "internal/foo/a.go\ninternal/foo/sub/b.go\n", code: 0}
	opts := baseOpts()
	opts.Paths = []string{"internal/foo"}

	res, _ := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if !res.Verified {
		t.Fatalf("a requested directory should cover its files; got reason=%q extra=%v", res.Reason, res.RacedExtra)
	}
}

func TestWindowsBackslashPathNormalizes(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["diff-tree"] = reply{out: "internal/foo/bar.go\n", code: 0}
	opts := baseOpts()
	opts.Paths = []string{`internal\foo\bar.go`} // requested with backslashes

	res, _ := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if !res.Verified {
		t.Fatalf("backslash path should normalize and match the forward-slash committed path; got reason=%q extra=%v", res.Reason, res.RacedExtra)
	}
}

func TestHappyPath_verifiesAndPushes(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	released := false
	opts := baseOpts()
	opts.Push = true

	res, err := CommitWith(context.Background(), g.run, okLock(&released), opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Committed || !res.Verified || !res.Pushed {
		t.Fatalf("happy+push: want Committed&&Verified&&Pushed, got %+v", res)
	}
	if res.SHA == "" || res.HeadBefore == "" {
		t.Fatalf("happy path should record SHA and HeadBefore, got %+v", res)
	}
	if !released {
		t.Fatalf("the lock must be released (defer unlock)")
	}
}

func TestPushRejected_leavesCommitIntact(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	g.reply["push"] = reply{out: "! [rejected] main -> main (non-fast-forward)\n", code: 1}
	opts := baseOpts()
	opts.Push = true

	res, _ := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if !res.Verified {
		t.Fatalf("the commit verified; a push rejection must not unset Verified, got %+v", res)
	}
	if res.Pushed {
		t.Fatalf("push was rejected; must not report Pushed")
	}
	if res.Reason != ReasonPushRejected {
		t.Fatalf("want PUSH_REJECTED, got %q", res.Reason)
	}
	if g.sawSubcommand("reset") {
		t.Fatalf("a push rejection must never unwind the commit; calls=%v", g.calls)
	}
}

func TestNoPushWithoutFlag(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	res, _ := CommitWith(context.Background(), g.run, okLock(nil), baseOpts()) // Push=false
	if res.Pushed {
		t.Fatalf("must not push without the flag")
	}
	if g.sawSubcommand("push") {
		t.Fatalf("must not invoke git push without the flag; calls=%v", g.calls)
	}
}

// realDirOpts builds Options pointed at a real on-disk dir (the symlink-escape guard
// resolves landed paths against the filesystem, unlike the string-only race guard).
func realDirOpts(dir string, paths []string) Options {
	return Options{
		Dir:     dir,
		Paths:   paths,
		Message: "fix(x): contained change\n\n(fak safecommit)",
		SignOff: true,
	}
}

// TestSymlinkEscape_isRefused proves the CVE-2025-53109 class is caught: a symlink created
// INSIDE the lease that points OUTSIDE it passes the path-string race guard (the committed
// path still starts with the lease prefix) but its resolved target escapes the lease, so the
// post-commit assertion must refuse rather than report clean.
func TestSymlinkEscape_isRefused(t *testing.T) {
	root := t.TempDir()
	lease := filepath.Join(root, "lease")
	outside := filepath.Join(root, "outside")
	for _, d := range []string{lease, outside} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// the real secret lives outside the lease
	if err := os.WriteFile(filepath.Join(outside, "secret.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// a symlink INSIDE the lease pointing at the outside file
	link := filepath.Join(lease, "evil.go")
	if err := os.Symlink(filepath.Join(outside, "secret.go"), link); err != nil {
		t.Skipf("symlinks unavailable on this host (%v) — guard is still compiled and unit-covered elsewhere", err)
	}

	rep := onTrunkBase()
	rep["status"] = reply{out: " M lease/evil.go\n", code: 0}
	rep["diff-tree"] = reply{out: "lease/evil.go\n", code: 0}
	g := &fakeGit{reply: rep}
	var released bool

	res, err := CommitWith(context.Background(), g.run, okLock(&released), realDirOpts(root, []string{"lease"}))
	if err != nil {
		t.Fatalf("infra error: %v", err)
	}
	if res.Reason != ReasonSymlinkEscape {
		t.Fatalf("reason = %q, want %q (symlink escaping the lease must refuse)", res.Reason, ReasonSymlinkEscape)
	}
	if res.Verified {
		t.Fatalf("a symlink-escaping commit must not be Verified")
	}
	if res.Pushed {
		t.Fatalf("a refused commit must never push")
	}
	if len(res.RacedExtra) == 0 || res.RacedExtra[0] != "lease/evil.go" {
		t.Fatalf("RacedExtra = %v, want [lease/evil.go]", res.RacedExtra)
	}
}

// TestInLeaseSymlink_passes is the paired positive case: a symlink that stays INSIDE the
// lease resolves to a target the lease covers, so the guard must NOT refuse it (otherwise it
// is merely rejecting all symlinks, not escapes).
func TestInLeaseSymlink_passes(t *testing.T) {
	root := t.TempDir()
	lease := filepath.Join(root, "lease")
	if err := os.MkdirAll(lease, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(lease, "real.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(lease, "alias.go")
	if err := os.Symlink(filepath.Join(lease, "real.go"), link); err != nil {
		t.Skipf("symlinks unavailable on this host (%v)", err)
	}

	rep := onTrunkBase()
	rep["status"] = reply{out: " M lease/alias.go\n", code: 0}
	rep["diff-tree"] = reply{out: "lease/alias.go\n", code: 0}
	g := &fakeGit{reply: rep}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), realDirOpts(root, []string{"lease"}))
	if err != nil {
		t.Fatalf("infra error: %v", err)
	}
	if res.Reason != "" {
		t.Fatalf("reason = %q, want clean (an in-lease symlink must pass)", res.Reason)
	}
	if !res.Verified {
		t.Fatalf("an in-lease symlink commit must verify")
	}
}

// TestLandedEscapesLease_directUnit exercises the containment helper without needing OS
// symlink privilege (unavailable to an unprivileged Windows process), so the guard's logic
// is behaviorally verified on every host, not only where symlinks can be created. It drives
// the two decisive branches with plain real files: a landed path whose resolved target the
// requested lease covers (clean) vs one the lease does NOT cover (escaped).
func TestLandedEscapesLease_directUnit(t *testing.T) {
	root := t.TempDir()
	for _, d := range []string{"lease", "outside"} {
		if err := os.MkdirAll(filepath.Join(root, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "lease", "ok.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "outside", "secret.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// in-lease landed path: resolved target is covered -> not escaped.
	if got := landedEscapesLease(root, "lease/ok.go\n", []string{"lease"}); len(got) != 0 {
		t.Fatalf("in-lease path flagged as escape: %v", got)
	}
	// a landed path naming a real file OUTSIDE the lease (the post-resolution containment
	// the symlink case reduces to): resolved target is not covered -> escaped.
	if got := landedEscapesLease(root, "outside/secret.go\n", []string{"lease"}); len(got) != 1 || got[0] != "outside/secret.go" {
		t.Fatalf("outside path not flagged as escape: %v", got)
	}
	// a landed path that does not exist on disk carries no symlink to escape through and is
	// left to the string-level guard -> not flagged here.
	if got := landedEscapesLease(root, "lease/ghost.go\n", []string{"lease"}); len(got) != 0 {
		t.Fatalf("non-existent path flagged as escape: %v", got)
	}
	// dir == "" disables the check.
	if got := landedEscapesLease("", "outside/secret.go\n", []string{"lease"}); len(got) != 0 {
		t.Fatalf("empty dir did not disable the check: %v", got)
	}
}
