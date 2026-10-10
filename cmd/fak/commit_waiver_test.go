package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

const waiverTestSHA = "0123456789abcdef0123456789abcdef01234567"

// A host-configured waiver hook must never see the fake SHAs this package's commit tests land.
func init() {
	_ = os.Unsetenv(commitWaiverHookEnv)
	commitWaiverHookLookup = func(string) (string, error) { return "", nil }
}

type waiverHarness struct {
	message  string
	hookArgv []string
	hookRuns int
}

func withWaiverHarness(t *testing.T, gate safecommit.BuildCheckOutcome, hookErr error, configured string) *waiverHarness {
	t.Helper()
	h := &waiverHarness{}
	withCommitFn(t, func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		h.message = opts.Message
		return safecommit.Result{Committed: true, SHA: waiverTestSHA, Paths: opts.Paths}, nil
	})
	prevGate, prevLookup, prevRun := commitBuildCheckGate, commitWaiverHookLookup, commitWaiverHookRun
	t.Cleanup(func() {
		commitBuildCheckGate, commitWaiverHookLookup, commitWaiverHookRun = prevGate, prevLookup, prevRun
	})
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) { return gate, "" }
	commitWaiverHookLookup = func(string) (string, error) { return configured, nil }
	commitWaiverHookRun = func(_ context.Context, _ string, argv []string, _ io.Writer) error {
		h.hookRuns++
		h.hookArgv = append([]string(nil), argv...)
		return hookErr
	}
	t.Setenv("FAK_COMMIT_BUILD_CHECK", "")
	t.Setenv("FLEET_BUILDCHECK_GUARD", "")
	t.Setenv(commitWaiverHookEnv, "")
	return h
}

func runWaiverCommit(t *testing.T, extra ...string) (int, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	args := append([]string{"--dir", t.TempDir(), "--path", "cmd/fak/x.go", "-m", "feat(cmd): exercise waiver hook (fak cmd)"}, extra...)
	code := runCommit(&stdout, &stderr, args)
	return code, stderr.String()
}

// fak-test:runtime fast est=1s
func TestCommitWaiverStampsTrailerForEachKind(t *testing.T) {
	cases := []struct {
		name  string
		gate  safecommit.BuildCheckOutcome
		env   string
		flags []string
		want  string
	}{
		{"flag", safecommit.BuildCheckPassed, "", []string{"--no-build-check"}, waiverNoBuildCheck},
		{"env off", safecommit.BuildCheckPassed, "off", nil, waiverBuildCheckOff},
		{"timeout admitted", safecommit.BuildCheckSkippedTimeout, "", []string{"--allow-build-check-timeout"}, waiverBuildCheckTimeout},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := withWaiverHarness(t, tc.gate, nil, "")
			if tc.env != "" {
				t.Setenv("FAK_COMMIT_BUILD_CHECK", tc.env)
			}
			code, stderr := runWaiverCommit(t, tc.flags...)
			if code != 0 {
				t.Fatalf("code=%d, want 0; stderr=%s", code, stderr)
			}
			lines := strings.Split(strings.TrimSpace(h.message), "\n")
			if got, want := lines[len(lines)-1], verificationWaiverTrailerKey+": "+tc.want; got != want {
				t.Fatalf("last message line=%q, want %q", got, want)
			}
			if h.hookRuns != 0 {
				t.Fatalf("hook ran %d times with none configured", h.hookRuns)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestCommitWaiverHookArgShape(t *testing.T) {
	h := withWaiverHarness(t, safecommit.BuildCheckPassed, nil, "")
	t.Setenv(commitWaiverHookEnv, `record-waiver --root "C:/a b"`)
	if code, stderr := runWaiverCommit(t, "--no-build-check"); code != 0 {
		t.Fatalf("code=%d, want 0; stderr=%s", code, stderr)
	}
	want := []string{"record-waiver", "--root", "C:/a b", "--commit", waiverTestSHA, "--waiver", waiverNoBuildCheck}
	if !reflect.DeepEqual(h.hookArgv, want) {
		t.Fatalf("hook argv=%q, want %q", h.hookArgv, want)
	}
}

// fak-test:runtime fast est=1s
func TestCommitWaiverHookFromGitConfig(t *testing.T) {
	h := withWaiverHarness(t, safecommit.BuildCheckPassed, nil, "cfg-hook")
	t.Setenv("FAK_COMMIT_BUILD_CHECK", "off")
	if code, stderr := runWaiverCommit(t); code != 0 {
		t.Fatalf("code=%d, want 0; stderr=%s", code, stderr)
	}
	want := []string{"cfg-hook", "--commit", waiverTestSHA, "--waiver", waiverBuildCheckOff}
	if !reflect.DeepEqual(h.hookArgv, want) {
		t.Fatalf("hook argv=%q, want %q", h.hookArgv, want)
	}
}

// fak-test:runtime fast est=1s
func TestCommitWaiverHookFailureFailsClosed(t *testing.T) {
	withWaiverHarness(t, safecommit.BuildCheckPassed, errors.New("exit status 1"), "cfg-hook")
	code, stderr := runWaiverCommit(t, "--no-build-check")
	if code != safecommit.ExitPostCommitFailure {
		t.Fatalf("code=%d, want %d; stderr=%s", code, safecommit.ExitPostCommitFailure, stderr)
	}
	if !strings.Contains(stderr, ReasonVerificationWaiverUnrecorded) || !strings.Contains(stderr, waiverTestSHA) {
		t.Fatalf("stderr lacks typed code and sha: %s", stderr)
	}

	err := recordCommitWaiver(io.Discard, "fak commit", t.TempDir(), safecommit.Result{Committed: true, SHA: waiverTestSHA}, waiverNoBuildCheck)
	if !errors.Is(err, errVerificationWaiverUnrecorded) {
		t.Fatalf("err=%v, want errors.Is errVerificationWaiverUnrecorded", err)
	}
}

// fak-test:runtime fast est=1s
func TestCommitWithoutWaiverHasNoTrailerAndNoHook(t *testing.T) {
	h := withWaiverHarness(t, safecommit.BuildCheckPassed, nil, "cfg-hook")
	t.Setenv(commitWaiverHookEnv, "env-hook")
	if code, stderr := runWaiverCommit(t); code != 0 {
		t.Fatalf("code=%d, want 0; stderr=%s", code, stderr)
	}
	if strings.Contains(h.message, verificationWaiverTrailerKey) {
		t.Fatalf("unwaived message carries a waiver trailer: %q", h.message)
	}
	if h.hookRuns != 0 {
		t.Fatalf("hook ran %d times for an unwaived commit", h.hookRuns)
	}
}

// fak-test:runtime fast est=1s
func TestCommitVerificationWaiverIgnoresUnwaivedOutcomes(t *testing.T) {
	for _, bc := range []safecommit.BuildCheckResult{
		{Outcome: safecommit.BuildCheckPassed},
		{Outcome: safecommit.BuildCheckNotApplicable},
		{Outcome: safecommit.BuildCheckSkippedInfra, FailedOpen: true},
	} {
		if got := commitVerificationWaiver(false, false, bc); got != "" {
			t.Fatalf("outcome %s: waiver=%q, want none", bc.Outcome, got)
		}
	}
}

// fak-test:runtime fast est=1s
func TestWithVerificationWaiverTrailerJoinsTrailerBlock(t *testing.T) {
	got := withVerificationWaiverTrailer("feat(x): y (fak x)\n\nwhy\n\nRef: none\n", waiverBuildCheckTimeout)
	want := "feat(x): y (fak x)\n\nwhy\n\nRef: none\nVerification-Waiver: build-check-timeout\n"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if again := withVerificationWaiverTrailer(got, waiverBuildCheckTimeout); again != got {
		t.Fatalf("re-stamp duplicated the trailer: %q", again)
	}
}
