package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitintent"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// withCommitFn swaps the commitFn seam for the duration of a test.
func withCommitFn(t *testing.T, fn func(context.Context, safecommit.Options) (safecommit.Result, error)) {
	t.Helper()
	prevCommit, prevBusy, prevWait, prevAssess := commitFn, commitLaneBusyFn, commitLaneWaitFn, syncAssess
	commitFn = fn
	commitLaneBusyFn = func(string) (bool, int) { return false, 0 }
	commitLaneWaitFn = func(string, time.Duration) (bool, safecommit.LockWaitReceipt) {
		return true, safecommit.LockWaitReceipt{}
	}
	syncAssess = func(context.Context, safesync.Options) (safesync.Assessment, error) {
		return safesync.Assessment{OK: true, State: safesync.StateInSync, TargetRef: "origin/main", Branch: "main"}, nil
	}
	t.Cleanup(func() {
		commitFn = prevCommit
		commitLaneBusyFn = prevBusy
		commitLaneWaitFn = prevWait
		syncAssess = prevAssess
	})
}

func TestRenderCommitSyncAdvisoryUsesCurrentRefsWithoutFetch(t *testing.T) {
	tests := []struct {
		name string
		info safesync.Assessment
		want []string
	}{
		{
			name: "up to date",
			info: safesync.Assessment{OK: true, State: safesync.StateInSync, TargetRef: "origin/main", Branch: "main"},
			want: []string{"in-sync with origin/main", "current remote-tracking refs", "no fetch"},
		},
		{
			name: "behind",
			info: safesync.Assessment{State: safesync.StateBehind, TargetRef: "origin/main", Branch: "main"},
			want: []string{"behind origin/main", "fak sync check --fetch --remote origin --branch main", "integrate origin/main in place"},
		},
		{
			name: "diverged",
			info: safesync.Assessment{State: safesync.StateDiverged, TargetRef: "origin/main", Branch: "main"},
			want: []string{"diverged origin/main", "fak sync check --fetch --remote origin --branch main", "integrate origin/main in place"},
		},
		{
			name: "unavailable upstream",
			info: safesync.Assessment{State: safesync.StateNoRemoteRef, TargetRef: "origin/main", Branch: "main", Reason: "remote-tracking ref origin/main not found; fetch first"},
			want: []string{"unavailable origin/main", "no fetch", "fak sync check --fetch --remote origin --branch main"},
		},
	}

	oldAssess := syncAssess
	t.Cleanup(func() { syncAssess = oldAssess })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			syncAssess = func(_ context.Context, opts safesync.Options) (safesync.Assessment, error) {
				calls++
				if opts.Fetch {
					t.Fatal("commit advisory must not fetch")
				}
				if opts.Remote != "origin" {
					t.Fatalf("remote = %q, want origin", opts.Remote)
				}
				if opts.Branch != "main" {
					t.Fatalf("branch = %q, want main", opts.Branch)
				}
				return tt.info, nil
			}
			var out bytes.Buffer
			renderCommitSyncAdvisory(context.Background(), &out, t.TempDir(), "main")
			if calls != 1 {
				t.Fatalf("assessment calls = %d, want 1", calls)
			}
			for _, want := range tt.want {
				if !strings.Contains(out.String(), want) {
					t.Fatalf("advisory = %q, want %q", out.String(), want)
				}
			}
		})
	}
}

func TestRenderCommitSyncAdvisoryAssessmentRunsNoFetchOrMutation(t *testing.T) {
	oldAssess := syncAssess
	t.Cleanup(func() { syncAssess = oldAssess })

	var commands [][]string
	syncAssess = func(ctx context.Context, opts safesync.Options) (safesync.Assessment, error) {
		if opts.Fetch {
			t.Fatal("commit advisory requested a fetch")
		}
		opts.Runner = func(_ context.Context, _ string, args ...string) safesync.RunResult {
			commands = append(commands, append([]string(nil), args...))
			command := strings.Join(args, " ")
			switch command {
			case "rev-parse --verify HEAD":
				return safesync.RunResult{Stdout: []byte("local-head\n")}
			case "rev-parse --verify origin/main":
				return safesync.RunResult{Stdout: []byte("remote-head\n")}
			case "merge-base --is-ancestor remote-head local-head":
				return safesync.RunResult{Code: 1}
			case "merge-base --is-ancestor local-head remote-head":
				return safesync.RunResult{}
			case "diff --name-status -z local-head remote-head":
				return safesync.RunResult{}
			default:
				return safesync.RunResult{Code: 2, Stderr: []byte("unexpected git command")}
			}
		}
		return safesync.Assess(ctx, opts)
	}

	var out bytes.Buffer
	renderCommitSyncAdvisory(context.Background(), &out, "repo", "main")
	if !strings.Contains(out.String(), "behind origin/main") {
		t.Fatalf("advisory = %q, want runner-backed behind assessment", out.String())
	}
	if len(commands) == 0 {
		t.Fatal("assessment issued no read commands")
	}
	mutating := map[string]bool{
		"fetch": true, "pull": true, "merge": true, "stash": true, "reset": true,
		"clean": true, "add": true, "commit": true, "push": true,
	}
	for _, args := range commands {
		if len(args) > 0 && mutating[args[0]] {
			t.Fatalf("commit advisory issued mutating git command: git %s", strings.Join(args, " "))
		}
	}
}

func TestRunCommitPreviewReportsUpstreamWithoutFetch(t *testing.T) {
	oldAssess := syncAssess
	t.Cleanup(func() { syncAssess = oldAssess })
	fetched := false
	syncAssess = func(_ context.Context, opts safesync.Options) (safesync.Assessment, error) {
		fetched = opts.Fetch
		return safesync.Assessment{State: safesync.StateBehind, TargetRef: "origin/main", Branch: "main"}, nil
	}

	var out, errOut bytes.Buffer
	code := runCommit(&out, &errOut, []string{
		"--preview",
		"--dir", t.TempDir(),
		"--trunk", "main",
		"--path", "cmd/fak/commit.go",
		"-m", "feat(commit): surface remote divergence before local commits accumulate (fak cmd)\n\nCloses #8777",
	})
	if code != 0 {
		t.Fatalf("preview code = %d, want 0; stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
	if fetched {
		t.Fatal("preview advisory requested a fetch")
	}
	if !strings.Contains(errOut.String(), "behind origin/main") || !strings.Contains(errOut.String(), "fak sync check --fetch") {
		t.Fatalf("preview stderr = %q, want behind advisory and fetch next step", errOut.String())
	}
}

func TestRunCommitAdvisesImmediatelyBeforeCommitWithoutBlocking(t *testing.T) {
	for _, tt := range []struct {
		name string
		info safesync.Assessment
		want string
	}{
		{name: "behind", info: safesync.Assessment{State: safesync.StateBehind, TargetRef: "origin/main", Branch: "main"}, want: "behind origin/main"},
		{name: "diverged", info: safesync.Assessment{State: safesync.StateDiverged, TargetRef: "origin/main", Branch: "main"}, want: "diverged origin/main"},
		{name: "unavailable", info: safesync.Assessment{State: safesync.StateNoRemoteRef, TargetRef: "origin/main", Branch: "main"}, want: "unavailable origin/main"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var events []string
			withCommitFn(t, func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
				events = append(events, "commit")
				return safecommit.Result{Committed: true, Verified: true, SHA: "abc123", Paths: opts.Paths}, nil
			})
			fetched := false
			syncAssess = func(_ context.Context, opts safesync.Options) (safesync.Assessment, error) {
				events = append(events, "assess")
				fetched = opts.Fetch
				return tt.info, nil
			}

			var out, errOut bytes.Buffer
			code := runCommit(&out, &errOut, []string{
				"--no-build-check",
				"--dir", t.TempDir(),
				"--trunk", "main",
				"--path", "cmd/fak/commit.go",
				"-m", "feat(commit): surface remote divergence before local commits accumulate (fak commit)\n\nCloses #8777",
			})
			if code != 0 {
				t.Fatalf("commit code = %d, want 0; stdout=%q stderr=%q", code, out.String(), errOut.String())
			}
			if fetched {
				t.Fatal("real commit advisory requested a fetch")
			}
			if got := strings.Join(events, ","); got != "assess,commit" {
				t.Fatalf("event order = %q, want assessment immediately before commit", got)
			}
			if !strings.Contains(errOut.String(), tt.want) {
				t.Fatalf("stderr = %q, want %q advisory", errOut.String(), tt.want)
			}
		})
	}
}

func TestRunCommitBusyLaneSkipsBuildCheckAndCommit(t *testing.T) {
	oldBusy, oldWait, oldBuild, oldCommit := commitLaneBusyFn, commitLaneWaitFn, commitBuildCheckGate, commitFn
	t.Cleanup(func() {
		commitLaneBusyFn, commitLaneWaitFn, commitBuildCheckGate, commitFn = oldBusy, oldWait, oldBuild, oldCommit
	})

	commitLaneBusyFn = func(string) (bool, int) { return true, 4242 }
	commitLaneWaitFn = func(string, time.Duration) (bool, safecommit.LockWaitReceipt) {
		return false, safecommit.LockWaitReceipt{
			ElapsedNS:      int64(10 * time.Second),
			DeadlineNS:     int64(10 * time.Second),
			HolderPID:      4242,
			HolderAlive:    true,
			LockAgeSeconds: 17,
		}
	}
	buildCalled := false
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) {
		buildCalled = true
		return safecommit.BuildCheckPassed, ""
	}
	commitCalled := false
	commitFn = func(context.Context, safecommit.Options) (safecommit.Result, error) {
		commitCalled = true
		return safecommit.Result{}, nil
	}

	var out, errOut bytes.Buffer
	code := runCommit(&out, &errOut, []string{"--path", "cmd/fak/main.go", "-m", "fix(cmd): reduce commit stampedes (fak cmd)"})
	if code != safecommit.ExitLockBusy {
		t.Fatalf("code = %d, want %d; stderr=%s", code, safecommit.ExitLockBusy, errOut.String())
	}
	if buildCalled || commitCalled {
		t.Fatalf("busy lane called build=%v commit=%v, want neither", buildCalled, commitCalled)
	}
	if got := errOut.String(); !strings.Contains(got, "skipped build-check") ||
		!strings.Contains(got, "holder pid 4242") ||
		!strings.Contains(got, "elapsed wait:") ||
		!strings.Contains(got, "deadline:") {
		t.Fatalf("stderr = %q, want bounded live-holder refusal evidence", got)
	}

	out.Reset()
	errOut.Reset()
	code = runCommit(&out, &errOut, []string{"--json", "--path", "cmd/fak/main.go", "-m", "fix(cmd): reduce commit stampedes (fak cmd)"})
	if code != safecommit.ExitLockBusy {
		t.Fatalf("JSON code = %d, want %d; stderr=%s", code, safecommit.ExitLockBusy, errOut.String())
	}
	var res safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("JSON LOCK_BUSY receipt: %v\n%s", err, out.String())
	}
	if res.Reason != safecommit.ReasonLockBusy || res.LockWait == nil || res.LockWait.HolderPID != 4242 || res.LockWait.ElapsedNS != int64(10*time.Second) {
		t.Fatalf("JSON LOCK_BUSY receipt = %+v", res)
	}
}

func TestWaitForCommitLaneBoundsLiveHolderAndLetsDeadHolderThrough(t *testing.T) {
	oldBusy, oldNow, oldSleep := commitLaneBusyFn, commitLaneNow, commitLaneSleep
	t.Cleanup(func() {
		commitLaneBusyFn, commitLaneNow, commitLaneSleep = oldBusy, oldNow, oldSleep
	})

	clock := time.Unix(1_800_000_000, 0)
	commitLaneNow = func() time.Time { return clock }
	commitLaneSleep = func(d time.Duration) { clock = clock.Add(d) }
	commitLaneBusyFn = func(string) (bool, int) { return true, 4242 }
	ready, receipt := waitForCommitLane(".", time.Second)
	if ready {
		t.Fatal("live holder reported ready")
	}
	if time.Duration(receipt.ElapsedNS) != time.Second || receipt.HolderPID != 4242 || !receipt.HolderAlive {
		t.Fatalf("live-holder receipt = %+v, want 1s bounded wait and PID 4242 alive", receipt)
	}

	sleeps := 0
	commitLaneSleep = func(time.Duration) { sleeps++ }
	commitLaneBusyFn = func(string) (bool, int) { return false, 9191 }
	ready, receipt = waitForCommitLane(".", time.Second)
	if !ready || sleeps != 0 || receipt.HolderAlive || receipt.HolderPID != 9191 {
		t.Fatalf("dead-holder precheck ready=%t sleeps=%d receipt=%+v; want immediate guarded-reaper handoff", ready, sleeps, receipt)
	}
}

func TestRunCommitLaneClearStillBuildChecksAndCommits(t *testing.T) {
	oldBusy, oldBuild, oldCommit, oldAssess := commitLaneBusyFn, commitBuildCheckGate, commitFn, syncAssess
	t.Cleanup(func() {
		commitLaneBusyFn, commitBuildCheckGate, commitFn, syncAssess = oldBusy, oldBuild, oldCommit, oldAssess
	})

	commitLaneBusyFn = func(string) (bool, int) { return false, 0 }
	buildCalled := false
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) {
		buildCalled = true
		return safecommit.BuildCheckPassed, ""
	}
	commitCalled := false
	commitFn = func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		commitCalled = true
		return safecommit.Result{Committed: true, SHA: "abc123", Paths: opts.Paths}, nil
	}
	syncAssess = func(context.Context, safesync.Options) (safesync.Assessment, error) {
		return safesync.Assessment{OK: true, State: safesync.StateInSync, TargetRef: "origin/main", Branch: "main"}, nil
	}

	var out, errOut bytes.Buffer
	code := runCommit(&out, &errOut, []string{"--path", "cmd/fak/main.go", "-m", "fix(cmd): retain clear-lane checks (fak cmd)"})
	if code != 0 {
		t.Fatalf("code = %d, want 0; stderr=%s", code, errOut.String())
	}
	if !buildCalled || !commitCalled {
		t.Fatalf("clear lane called build=%v commit=%v, want both", buildCalled, commitCalled)
	}
}

func TestRunCommitProspectiveValidationRefusalNeverCallsCommit(t *testing.T) {
	oldWait, oldBuild, oldCommit := commitLaneWaitFn, commitBuildCheckGate, commitFn
	t.Cleanup(func() {
		commitLaneWaitFn, commitBuildCheckGate, commitFn = oldWait, oldBuild, oldCommit
	})
	commitLaneWaitFn = func(string, time.Duration) (bool, safecommit.LockWaitReceipt) {
		return true, safecommit.LockWaitReceipt{}
	}
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckFailed, "prospective validation failed\n  test: deliberate failure\nnext: fak validate --ref HEAD --mine p/p_test.go"
	}
	commitCalled := false
	commitFn = func(context.Context, safecommit.Options) (safecommit.Result, error) {
		commitCalled = true
		return safecommit.Result{}, nil
	}

	var stdout, stderr bytes.Buffer
	code := runCommit(&stdout, &stderr, []string{
		"--json", "--dir", t.TempDir(), "--path", "p/p_test.go",
		"-m", "test(cmd): prove prospective validation refusal (fak cmd)",
	})
	if code != safecommit.ExitRefused {
		t.Fatalf("code=%d, want %d; stdout=%s stderr=%s", code, safecommit.ExitRefused, stdout.String(), stderr.String())
	}
	if commitCalled {
		t.Fatal("validation refusal called safecommit.Commit")
	}
	var res safecommit.Result
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("decode typed refusal: %v\n%s", err, stdout.String())
	}
	if res.Reason != safecommit.ReasonCommittedRed || res.BuildCheck == nil || res.BuildCheck.Outcome != safecommit.BuildCheckFailed {
		t.Fatalf("typed refusal=%+v", res)
	}
	for _, want := range []string{"COMMITTED_RED", "deliberate failure", "fak validate --ref HEAD --mine p/p_test.go"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr=%q; want %q", stderr.String(), want)
		}
	}
}

func TestRunCommit_noPathsIsUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"-m", "msg"})
	if code != 2 {
		t.Fatalf("want exit 2 for no paths, got %d (stderr=%q)", code, errb.String())
	}
}

func TestRunCommitHelpDocumentsFiniteLockDeadline(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--help"})
	if code != 2 {
		t.Fatalf("help exit = %d, want usage exit 2", code)
	}
	for _, want := range []string{"lock-timeout", "finite deadline", "default 10s"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("commit help = %q, want %q", errb.String(), want)
		}
	}
}

func TestRunCommitRejectsNonPositiveLockDeadline(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--lock-timeout=0", "--path", "a.go", "-m", "msg"})
	if code != 2 || !strings.Contains(errb.String(), "must be greater than zero") {
		t.Fatalf("code=%d stderr=%q, want finite-deadline usage refusal", code, errb.String())
	}
}

func TestRunCommit_noMessageIsUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go"})
	if code != 2 {
		t.Fatalf("want exit 2 for no message, got %d (stderr=%q)", code, errb.String())
	}
}

func TestRunCommit_dashMAndDashFAreExclusive(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "x", "-F", "f.txt"})
	if code != 2 {
		t.Fatalf("want exit 2 for -m+-F, got %d", code)
	}
	if !strings.Contains(errb.String(), "mutually exclusive") {
		t.Fatalf("stderr should explain the conflict, got %q", errb.String())
	}
}

func TestRunCommit_derivesMissingSingleLaneStamp(t *testing.T) {
	tmp := t.TempDir()
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc", Paths: o.Paths}, nil
	})

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--dir", tmp,
		"--path", "internal/gateway/server.go",
		"-m", "feat(gateway): add stamp derivation\n\nBody stays here.",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	if !strings.HasPrefix(got.Message, "feat(gateway): add stamp derivation (fak gateway)") {
		t.Fatalf("commit message did not get derived trailer: %q", got.Message)
	}
	if !strings.Contains(got.Message, "Body stays here.") {
		t.Fatalf("commit body should be preserved, got %q", got.Message)
	}
}

// TestRunCommit_multipleDashMJoinsParagraphs locks the git-parity fix: `fak commit -m A -m B`
// must join A and B as separate paragraphs (subject + body), NOT silently keep only the last -m.
// Before the fix, Go's last-wins fs.String dropped the real subject AND its `(fak leaf)` stamp on
// this common muscle-memory call; deriveCommitMessageStamp then re-derived a different subject.
func TestRunCommit_multipleDashMJoinsParagraphs(t *testing.T) {
	tmp := t.TempDir()
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc", Paths: o.Paths}, nil
	})

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--dir", tmp,
		"--path", "internal/gateway/server.go",
		"-m", "feat(gateway): add real thing (fak gateway)",
		"-m", "First body paragraph.",
		"-m", "Second body paragraph.",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	want := "feat(gateway): add real thing (fak gateway)\n\nFirst body paragraph.\n\nSecond body paragraph."
	if got.Message != want {
		t.Fatalf("multiple -m should join as paragraphs (subject preserved)\n got %q\nwant %q", got.Message, want)
	}
}

func TestMessageList_JoinedEmptyIsBlank(t *testing.T) {
	var m messageList
	if m.Joined() != "" {
		t.Fatalf("empty messageList must join to \"\", got %q", m.Joined())
	}
	_ = m.Set("a")
	_ = m.Set("b")
	if got := m.Joined(); got != "a\n\nb" {
		t.Fatalf("join want %q got %q", "a\n\nb", got)
	}
}

func TestRunCommit_multiLaneMissingStampDoesNotGuess(t *testing.T) {
	tmp := t.TempDir()
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc", Paths: o.Paths}, nil
	})

	const msg = "feat(kernel): add cross-lane routing"
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--dir", tmp,
		"--path", "internal/gateway/server.go",
		"--path", "internal/policy/rules.go",
		"-m", msg,
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	if got.Message != msg {
		t.Fatalf("multi-lane path set should not guess a stamp, got %q", got.Message)
	}
}

func TestRunCommitSubmit_jsonPersistsIntentWithoutGit(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"submit",
		"--json",
		"--queue-dir", queueDir,
		"--id", "issue-1788-cli",
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"--diff-digest", "SHA256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"--path", `internal\commitintent\store.go`,
		"-m", "feat(commitintent): add submit cli (#1788) (fak commitintent)",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitSubmitResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("submit --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if !res.Queued || res.IntentID != "issue-1788-cli" || res.Sequence != 1 || res.QueueSize != 1 {
		t.Fatalf("submit result = %+v", res)
	}
	if got := res.Record.Intent.Paths; len(got) != 1 || got[0] != "internal/commitintent/store.go" {
		t.Fatalf("paths not normalized: %+v", got)
	}
	if got := res.Record.Intent.DiffDigest; got != "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" {
		t.Fatalf("diff digest = %q", got)
	}
	if _, err := os.Stat(filepath.Join(queueDir, "queue.json")); err != nil {
		t.Fatalf("queue file was not written: %v", err)
	}
}

func TestRunCommitSubmit_refusesMissingStampBeforeWritingQueue(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"submit",
		"--queue-dir", queueDir,
		"--id", "issue-1788-cli",
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"--path", "internal/commitintent/store.go",
		"-m", "feat(commitintent): add submit cli",
	})
	if code != safecommit.ExitRefused {
		t.Fatalf("want validation refusal exit %d, got %d stderr=%q stdout=%q", safecommit.ExitRefused, code, errb.String(), out.String())
	}
	if _, err := os.Stat(filepath.Join(queueDir, "queue.json")); !os.IsNotExist(err) {
		t.Fatalf("queue should not be written on refusal, stat err=%v", err)
	}
}

func TestRunCommitDrainDryRunPlansRollup(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "intent-a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{`internal\commitintent\a.go`}, "worker-a")
	submitDrainIntent(t, queueDir, "intent-b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/b.go"}, "worker-b")

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--dry-run",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if !res.DryRun || res.Drained || !res.Plan.OK {
		t.Fatalf("drain result = %+v", res)
	}
	assertStrings(t, res.Plan.IntentIDs, []string{"intent-a", "intent-b"})
	assertStrings(t, res.Plan.Submitters, []string{"worker-a", "worker-b"})
	assertStrings(t, res.Plan.UnionPaths, []string{"internal/commitintent/a.go", "internal/commitintent/b.go"})
	if res.Commit != nil {
		t.Fatalf("dry-run should not call commit, got %+v", res.Commit)
	}
	if !strings.Contains(res.Plan.Subject, "intent-a, intent-b") || !strings.Contains(res.Plan.Subject, "(fak commitintent)") {
		t.Fatalf("subject should include ids and stamp, got %q", res.Plan.Subject)
	}
	states := drainQueueStates(t, queueDir)
	if states["intent-a"] != commitintent.StatePending || states["intent-b"] != commitintent.StatePending {
		t.Fatalf("dry-run should leave intents pending, states=%+v", states)
	}
}

func TestRunCommitDrainExecutesRollupWithUnionPathsAndMarksDone(t *testing.T) {
	withPassedCommitBuildCheck(t)
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "intent-a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")
	submitDrainIntent(t, queueDir, "intent-b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/b.go"}, "worker-b")

	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	assertStrings(t, got.Paths, []string{"internal/commitintent/a.go", "internal/commitintent/b.go"})
	if !got.SignOff {
		t.Fatalf("drain commit should sign off by default")
	}
	if !strings.Contains(got.Message, "intent-a, intent-b") || !strings.Contains(got.Message, "(fak commitintent)") {
		t.Fatalf("commit message should include rollup ids and stamp, got %q", got.Message)
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if !res.Drained || res.Pathset == nil || !res.Pathset.OK {
		t.Fatalf("result should be drained with pathset witness, got %+v", res)
	}
	if res.Commit == nil || !res.Commit.DeliveryVerified() || res.Commit.Evidence == nil || res.Commit.Evidence.ClosureBound.Outcome != safecommit.EvidencePassed {
		t.Fatalf("green drain should carry closure-qualified delivery evidence, got %+v", res.Commit)
	}
	assertStrings(t, res.MarkedDone, []string{"intent-a", "intent-b"})
	states := drainQueueStates(t, queueDir)
	if states["intent-a"] != commitintent.StateDone || states["intent-b"] != commitintent.StateDone {
		t.Fatalf("successful drain should mark intents done, states=%+v", states)
	}
}

func TestRunCommitDrainSkippedValidationDoesNotMarkDone(t *testing.T) {
	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	var checked []string
	commitBuildCheckGate = func(_ io.Writer, _ string, paths []string) (safecommit.BuildCheckOutcome, string) {
		checked = append([]string(nil), paths...)
		return safecommit.BuildCheckSkippedInfra, "go toolchain unavailable"
	}
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "intent-a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")

	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain", "--json", "--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("unchecked recording should return its commit result, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	assertStrings(t, checked, []string{"internal/commitintent/a.go"})
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if res.Drained || res.Commit == nil || res.Commit.Verified || res.Commit.DeliveryVerified() {
		t.Fatalf("skipped validation advanced drain state: %+v", res)
	}
	if res.Commit.Evidence == nil || res.Commit.Evidence.Compiled.Outcome != safecommit.EvidenceSkipped || res.Commit.Evidence.ClosureBound.Outcome != safecommit.EvidencePassed {
		t.Fatalf("skipped drain lost typed evidence: %+v", res.Commit)
	}
	states := drainQueueStates(t, queueDir)
	if states["intent-a"] != commitintent.StatePending {
		t.Fatalf("skipped validation should leave intent pending, states=%+v", states)
	}
}

func TestRunCommitDrainDryRunRefusesStaleAndOverlap(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")
	submitDrainIntent(t, queueDir, "overlap", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-b")
	submitDrainIntent(t, queueDir, "stale", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", []string{"internal/commitintent/stale.go"}, "worker-c")

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--dry-run",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want dry-run exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	assertStrings(t, res.Plan.IntentIDs, []string{"base"})
	if len(res.Stale) != 1 || res.Stale[0].Intent.ID != "stale" {
		t.Fatalf("stale records = %+v", res.Stale)
	}
	if !drainHasRefusal(res, "overlap", "OVERLAPPING_PATH") || !drainHasRefusal(res, "stale", "STALE_INPUT") {
		t.Fatalf("refusals = %+v", res.Plan.Refusals)
	}
}

func TestRunCommitDrainNoRollupKeepsOneIntentMode(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "first", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")
	submitDrainIntent(t, queueDir, "second", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/b.go"}, "worker-b")

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--dry-run",
		"--no-rollup",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want dry-run exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if res.Plan.RollupEnabled {
		t.Fatalf("--no-rollup should disable rollup: %+v", res.Plan)
	}
	assertStrings(t, res.Plan.IntentIDs, []string{"first"})
	if !drainHasRefusal(res, "second", "ROLLUP_DISABLED") {
		t.Fatalf("refusals = %+v", res.Plan.Refusals)
	}
}

func TestRunCommitDrainPathsetMismatchDoesNotMarkDone(t *testing.T) {
	withPassedCommitBuildCheck(t)
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "intent-a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")

	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed: true,
			Verified:  true,
			SHA:       "abc123",
			Paths:     append(append([]string(nil), o.Paths...), "internal/commitintent/extra.go"),
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 1 {
		t.Fatalf("pathset mismatch should exit 1, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if res.Drained || res.Pathset == nil || res.Pathset.OK {
		t.Fatalf("pathset mismatch should block drain, got %+v", res)
	}
	states := drainQueueStates(t, queueDir)
	if states["intent-a"] != commitintent.StatePending {
		t.Fatalf("mismatched pathset should leave intent pending, states=%+v", states)
	}
}

func TestRunCommit_positionalPathsAfterDashDash(t *testing.T) {
	var gotOpts safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		gotOpts = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"-m", "msg", "--", "internal/x.go", "internal/y.go"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
	}
	if len(gotOpts.Paths) != 2 {
		t.Fatalf("positional paths after -- should reach Options, got %v", gotOpts.Paths)
	}
	if !gotOpts.SignOff {
		t.Fatalf("sign-off should default on")
	}
}

func TestRunCommit_jsonShapeAndRaceExitCode(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed:  true,
			Verified:   false,
			SHA:        "deadbeefcafe",
			Paths:      o.Paths,
			Reason:     safecommit.ReasonPathspecRace,
			RacedExtra: []string{"internal/peer/swept.go"},
			HeadBefore: "0000111122223333",
			LockHoldNS: 17_000_000,
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--json", "--path", "a.go", "-m", "msg"})
	// PATHSPEC_RACE: the commit ran but is bad -> exit 1 (halt).
	if code != 1 {
		t.Fatalf("race should exit 1, got %d", code)
	}
	var res safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("--json must emit a valid Result: %v\noutput=%q", err, out.String())
	}
	if res.Reason != safecommit.ReasonPathspecRace || len(res.RacedExtra) != 1 {
		t.Fatalf("json result lost the race evidence: %+v", res)
	}
	if res.Score == 0 || res.Grade == "" {
		t.Fatalf("json result should include scored outcome, got %+v", res)
	}
	if res.LockHoldNS != 17_000_000 {
		t.Fatalf("json result lost lock hold duration: %+v", res)
	}
}

func TestRunCommit_humanOutputShowsLockHoldDuration(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed:  true,
			Verified:   true,
			SHA:        "deadbeefcafe",
			Paths:      o.Paths,
			LockHoldNS: int64(23 * time.Millisecond),
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "lock hold: 23ms") {
		t.Fatalf("human output should expose lock hold duration, got %q", out.String())
	}
}

// TestRunCommit_offTrunkIsRefusedNotContention pins OFF_TRUNK to the verdict exit (4), not
// the retryable contention exit (3): HEAD stays off-trunk however long you back off, so a
// lander must replan rather than retry (#5505 W4).
func TestRunCommit_offTrunkIsRefusedNotContention(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Reason: safecommit.ReasonOffTrunk, Detail: "on feature/x, expected development branch main", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != safecommit.ExitRefused {
		t.Fatalf("a refusal on the merits should exit %d, got %d", safecommit.ExitRefused, code)
	}
	if !strings.Contains(out.String(), "score:") {
		t.Fatalf("human refusal output should include score, got %q", out.String())
	}
}

// TestRunCommit_lockBusyKeepsExit3 pins the OTHER half of the #5505 W4 split at the CLI
// boundary: LOCK_BUSY keeps exit 3, the code every existing retry loop already backs off
// on. The split moved the verdicts out to 4; it did not repurpose 3.
func TestRunCommit_lockBusyKeepsExit3(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Reason: safecommit.ReasonLockBusy, Detail: "held by pid 1234", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 3 {
		t.Fatalf("LOCK_BUSY must keep exit 3 (retryable contention), got %d", code)
	}
}

func TestRunCommit_infraErrorExit1(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{}, errTest
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 1 {
		t.Fatalf("an infra error should exit 1, got %d", code)
	}
}

func TestRunCommit_messageFromStdin(t *testing.T) {
	var gotMsg string
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		gotMsg = o.Message
		return safecommit.Result{Committed: true, Verified: true, Paths: o.Paths}, nil
	})
	prev := stdin
	stdin = func() io.Reader { return strings.NewReader("from stdin — em-dash ok\n") }
	t.Cleanup(func() { stdin = prev })

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-F", "-"})
	if code != 0 {
		t.Fatalf("want 0, got %d (stderr=%q)", code, errb.String())
	}
	if !strings.Contains(gotMsg, "from stdin") {
		t.Fatalf("message should come from stdin, got %q", gotMsg)
	}
}

func TestRunCommit_reviewModelWiresSafecommitReview(t *testing.T) {
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--path", "a.go",
		"-m", "feat(loop): add review (#1185) (fak cmd)",
		"--review-model", "cheap-scout",
		"--review-objective", "ship issue 1185",
	})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q", code, errb.String())
	}
	if got.Review == nil {
		t.Fatal("--review-model did not wire safecommit Review")
	}
	if got.Review.Model != "cheap-scout" || got.Review.Objective != "ship issue 1185" {
		t.Fatalf("review options = %+v", got.Review)
	}
}

func TestRunCommit_reviewModelCSVWiresQuorum(t *testing.T) {
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--path", "a.go",
		"-m", "feat(loop): add review quorum (#1185) (fak cmd)",
		"--review-model", "cheap-scout,frontier-scout,cheap-scout",
		"--review-min-models", "2",
	})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q", code, errb.String())
	}
	if got.Review == nil {
		t.Fatal("--review-model did not wire safecommit Review")
	}
	if got.Review.Model != "cheap-scout,frontier-scout" {
		t.Fatalf("review model list = %q", got.Review.Model)
	}
}

func TestRunCommit_reviewMinModelsRejectsNegative(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--path", "a.go",
		"-m", "feat(loop): add review quorum (#1185) (fak cmd)",
		"--review-min-models", "-1",
	})
	if code != 2 {
		t.Fatalf("negative --review-min-models should be usage, got %d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}

func TestRunCommit_coreLockMaintenanceWitnessWiresSafecommit(t *testing.T) {
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--path", "internal/corelocks/corelocks.go",
		"-m", "feat(corelocks): tighten hard-self enforcement (#1683) (fak corelocks)",
		"--core-lock-maintenance-witness", "ancestor:reviewed-maintenance-sha",
	})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q", code, errb.String())
	}
	if got.CoreLockMaintenanceWitness != "ancestor:reviewed-maintenance-sha" {
		t.Fatalf("core-lock witness did not reach safecommit: %+v", got)
	}
}

func TestParseCommitReviewScoutLabelAcceptsFencedJSON(t *testing.T) {
	label, err := parseCommitReviewScoutLabel("```json\n{\"verdict\":\"refute\",\"reason\":\"missing test\"}\n```")
	if err != nil {
		t.Fatalf("parseCommitReviewScoutLabel: %v", err)
	}
	if label.Labels["verdict"] != "refute" || label.Labels["reason"] != "missing test" {
		t.Fatalf("label = %+v", label)
	}
}

func TestRunCommitReviewRefuteRecordsLoopEvidenceAndGoalScratch(t *testing.T) {
	tmp := t.TempDir()
	ledger := filepath.Join(tmp, "loops.jsonl")
	goal := filepath.Join(tmp, "GOAL.md")
	if err := os.WriteFile(goal, []byte(`---
loop: issue-1185
witness: commit-audit
---
# Objective
Ship the review rung.

# Plan
- [ ] wire review
`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_GOAL_LOOP", "issue-1185")
	t.Setenv("FAK_GOAL_ITER", "2")
	t.Setenv("FAK_GOAL_SPEC", goal)
	t.Setenv("FAK_LOOP_LEDGER", ledger)

	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Paths:  o.Paths,
			Reason: safecommit.ReasonReviewRefuted,
			Detail: "missing regression test",
			Review: &modelroute.ReviewResult{
				Model:      "cheap-scout",
				Verdict:    modelroute.ReviewRefute,
				Reason:     "missing regression test",
				DiffSHA256: "abc123",
				ScoutCalls: 1,
			},
		}, nil
	})

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "feat(loop): add review (#1185) (fak cmd)", "--review-model", "cheap-scout"})
	if code != safecommit.ExitRefused {
		t.Fatalf("review refute should exit %d (a verdict, not contention), got %d stderr=%q stdout=%q",
			safecommit.ExitRefused, code, errb.String(), out.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load ledger: %v", err)
	}
	if len(events) != 1 || events[0].Kind != loopmgr.EventHeartbeat || events[0].Reason != "REVIEW_REFUTED" {
		t.Fatalf("review event = %+v", events)
	}
	if len(events[0].EvidenceRefs) != 1 || events[0].EvidenceRefs[0].Kind != "review" || events[0].EvidenceRefs[0].Ref != "refute" {
		t.Fatalf("review evidence = %+v", events[0].EvidenceRefs)
	}
	if events[0].EvidenceRefs[0].SHA256 != "abc123" || events[0].Metrics["scout_calls"] != 1 {
		t.Fatalf("review evidence lost digest/metrics: %+v", events[0])
	}
	raw, err := os.ReadFile(goal)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "NOT_YET review refuted") || !strings.Contains(string(raw), "missing regression test") {
		t.Fatalf("goal scratch missing critique:\n%s", string(raw))
	}
}

func TestRunCommitReviewPassRecordsLoopEvidence(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	t.Setenv("FAK_GOAL_LOOP", "issue-1185")
	t.Setenv("FAK_GOAL_RUN", "run-1")
	t.Setenv("FAK_LOOP_LEDGER", ledger)

	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed: true,
			Verified:  true,
			SHA:       "abc",
			Paths:     o.Paths,
			Review: &modelroute.ReviewResult{
				Model:      "cheap-scout",
				Verdict:    modelroute.ReviewPass,
				Reason:     "diff matches objective",
				DiffSHA256: "def456",
				ScoutCalls: 1,
			},
		}, nil
	})

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "feat(loop): add review (#1185) (fak cmd)"})
	if code != 0 {
		t.Fatalf("review pass should exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load ledger: %v", err)
	}
	if len(events) != 1 || events[0].Reason != "REVIEW_PASS" || events[0].EvidenceRefs[0].Ref != "pass" {
		t.Fatalf("review event = %+v", events)
	}
}

func submitDrainIntent(t *testing.T, queueDir, id, base string, paths []string, requester string) {
	t.Helper()
	store := commitintent.Store{
		Dir: queueDir,
		Now: func() time.Time { return time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC) },
	}
	if _, _, err := store.Submit(commitintent.Intent{
		ID:      id,
		BaseSHA: base,
		Paths:   paths,
		Subject: "feat(commitintent): add drain rollup (#1789) (fak commitintent)",
		Metadata: commitintent.StampMetadata{
			Issue:     1789,
			Requester: requester,
		},
	}); err != nil {
		t.Fatalf("submit drain intent %s: %v", id, err)
	}
}

func drainQueueStates(t *testing.T, queueDir string) map[string]commitintent.State {
	t.Helper()
	q, err := (commitintent.Store{Dir: queueDir}).Load()
	if err != nil {
		t.Fatalf("Load queue: %v", err)
	}
	out := map[string]commitintent.State{}
	for _, rec := range q.Records {
		out[rec.Intent.ID] = rec.State
	}
	return out
}

func drainHasRefusal(res commitDrainResult, id, reason string) bool {
	for _, refusal := range res.Plan.Refusals {
		if refusal.IntentID == id && string(refusal.Reason) == reason {
			return true
		}
	}
	return false
}

func assertStrings(t *testing.T, got, want []string) {
	t.Helper()
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("strings = %v, want %v", got, want)
	}
}

var errTest = errTestErr{}

type errTestErr struct{}

func (errTestErr) Error() string { return "test infra failure" }

func TestRunCommitRecordsTreeReceiptOnlyAfterSuccessfulBuildAndCommit(t *testing.T) {
	oldBusy, oldBuild, oldCommit, oldRecord, oldAssess := commitLaneBusyFn, commitBuildCheckGate, commitFn, commitRecordTreeReceipt, syncAssess
	t.Cleanup(func() {
		commitLaneBusyFn, commitBuildCheckGate, commitFn, commitRecordTreeReceipt, syncAssess = oldBusy, oldBuild, oldCommit, oldRecord, oldAssess
	})
	commitLaneBusyFn = func(string) (bool, int) { return false, 0 }
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckPassed, ""
	}
	commitFn = func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc", Paths: opts.Paths}, nil
	}
	syncAssess = func(context.Context, safesync.Options) (safesync.Assessment, error) {
		return safesync.Assessment{OK: true, State: safesync.StateInSync, TargetRef: "origin/main", Branch: "main"}, nil
	}
	var recorded int
	commitRecordTreeReceipt = func(string, time.Time) { recorded++ }
	var out, errOut bytes.Buffer
	if code := runCommit(&out, &errOut, []string{"-m", "fix(test): add receipt witness (fak cmd)", "--path", "cmd/fak/x.go"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if recorded != 1 {
		t.Fatalf("record calls=%d, want 1", recorded)
	}
}

// The human line is the live witness that qualifies exact-tree pre-push reuse.
func TestRenderCommitResultShowsBuildCheckOutcome(t *testing.T) {
	res := safecommit.Result{Committed: true, SHA: "abc", Paths: []string{"x.go"}, BuildCheck: &safecommit.BuildCheckResult{Outcome: safecommit.BuildCheckSkippedTimeout}}
	var out bytes.Buffer
	renderCommitResult(&out, res)
	if !strings.Contains(out.String(), "build check: skipped-timeout (compiled=false)") {
		t.Fatalf("output omitted build outcome: %q", out.String())
	}
}

func TestRunCommitJSONEmitsRecordingOnlyDeliveryReceipt(t *testing.T) {
	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckPassed, ""
	}
	withCommitFn(t, func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123", Paths: opts.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--json", "--path", "internal/workdelivery/adapters.go", "-m", "feat(workdelivery): record delivery receipt (fak workdelivery)"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
	var result safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Delivery == nil || result.Delivery.Receipt == nil {
		t.Fatalf("missing delivery receipt: %s", out.String())
	}
	transition := result.Delivery.Receipt.Transition
	if transition.Axis != "authoring" || transition.To != "recorded" {
		t.Fatalf("transition=%+v", transition)
	}
	if result.Delivery.Stage != "recording" {
		t.Fatalf("stage=%q", result.Delivery.Stage)
	}
	// The prospective build gate may be green, but commit still records no verification receipt.
	if result.Delivery.Receipt.Transition.Axis == "verification" {
		t.Fatal("commit inferred verification")
	}
}

func TestRunCommitJSONSkippedInfraDoesNotVerifyOrScoreVelocity(t *testing.T) {
	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckSkippedInfra, "go toolchain unavailable"
	}
	withCommitFn(t, func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		res := safecommit.Result{Committed: true, Verified: true, SHA: "abc123", Paths: opts.Paths}
		velocity := safecommit.ScoreCommitVelocity(res, time.Millisecond, 0, safecommit.DefaultVelocityBudgets)
		res.Velocity = &velocity
		return res, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--json", "--path", "internal/safecommit/x.go", "-m", "fix(safecommit): witness skipped check (fak safecommit)"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
	var result safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Verified || result.DeliveryVerified() || result.Score != 55 {
		t.Fatalf("skipped-infra inflated receipt: %s", out.String())
	}
	if result.Evidence == nil || result.Evidence.Compiled.Outcome != safecommit.EvidenceSkipped || result.Evidence.DiffWitnessed.Outcome != safecommit.EvidencePassed {
		t.Fatalf("missing separated evidence: %s", out.String())
	}
	if result.Velocity == nil || result.Velocity.Local.Status != safecommit.VelocityUnscored {
		t.Fatalf("unchecked commit retained velocity credit: %s", out.String())
	}
}

func TestRunCommitJSONExplicitRecordOnlyStaysTruthful(t *testing.T) {
	withCommitFn(t, func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123", Paths: opts.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--json", "--no-build-check", "--path", "README.md", "-m", "docs: record wording (fak docs)"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
	var result safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.Verified || result.DeliveryVerified() || !result.RecordOnlyVerified() {
		t.Fatalf("record-only class was promoted or rejected: %s", out.String())
	}
	if result.Score != 85 || result.Evidence == nil || result.Evidence.CompletionClass != safecommit.CompletionRecordOnly {
		t.Fatalf("record-only receipt is not explicit: %s", out.String())
	}
}

func TestCommitDrainMayMarkDoneUsesDeliveryContract(t *testing.T) {
	legacy := safecommit.Result{Committed: true, Verified: true}
	if !commitDrainMayMarkDone(legacy, true) {
		t.Fatal("schema-less result lost the legacy drain contract")
	}
	recordOnly := safecommit.FinalizeEvidence(legacy, safecommit.EvidenceContract{CompletionClass: safecommit.CompletionRecordOnly})
	if commitDrainMayMarkDone(recordOnly, true) {
		t.Fatal("record-only receipt advanced issue/action state")
	}
	skipped := legacy
	skipped.BuildCheck = &safecommit.BuildCheckResult{Outcome: safecommit.BuildCheckSkippedInfra, FailedOpen: true}
	skipped = safecommit.FinalizeEvidence(skipped, safecommit.EvidenceContract{})
	if commitDrainMayMarkDone(skipped, true) {
		t.Fatal("skipped-infra receipt advanced issue/action state")
	}
	green := legacy
	green.BuildCheck = &safecommit.BuildCheckResult{Outcome: safecommit.BuildCheckPassed, Compiled: true}
	green = safecommit.FinalizeEvidence(green, safecommit.EvidenceContract{})
	if !commitDrainMayMarkDone(green, true) || commitDrainMayMarkDone(green, false) {
		t.Fatal("delivery/pathset closure gate is inconsistent")
	}
}
