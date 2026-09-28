package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func TestValidateCommitTimeout(t *testing.T) {
	cases := []struct {
		name    string
		in      time.Duration
		wantErr string
	}{
		{name: "unset keeps the library default", in: 0},
		{name: "a hook-sized budget", in: 7 * time.Minute},
		{name: "the cap itself", in: safecommit.MaxPostValidationTimeout},
		{name: "negative", in: -time.Second, wantErr: "--commit-timeout must not be negative"},
		{name: "above the stale-index cap", in: 20 * time.Minute, wantErr: "exceeds the " + safecommit.MaxPostValidationTimeout.String() + " maximum"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateCommitTimeout(tc.in)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("validateCommitTimeout(%s) = %v, want nil", tc.in, err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("validateCommitTimeout(%s) = %v, want error containing %q", tc.in, err, tc.wantErr)
			}
		})
	}
}

// stubStalledCommit bypasses the phantom-deletion guard and the build gate (the operator's
// `--no-build-check` shape), guards the env the --no-build-check branch rewrites, and stubs
// commitFn to record the Options it receives and report COMMIT_STALLED.
func stubStalledCommit(t *testing.T) *[]safecommit.Options {
	t.Helper()
	t.Setenv("FAK_PHANTOM_DELETION_CHECK", "off")
	t.Setenv("FLEET_BUILDCHECK_GUARD", "")
	t.Setenv("FAK_COMMIT_BUILD_CHECK", "")
	var seen []safecommit.Options
	withCommitFn(t, func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		seen = append(seen, opts)
		return safecommit.Result{
			Paths:  opts.Paths,
			Reason: safecommit.ReasonCommitStalled,
			Detail: "git mutation timed out: context deadline exceeded",
		}, nil
	})
	return &seen
}

// TestRunCommitCommitTimeoutReachesPostValidationTimeout: --commit-timeout is the knob that
// makes a normal commit possible on a host whose hooks outlast the default, so it must reach
// safecommit unchanged, and a stall must name it.
func TestRunCommitCommitTimeoutReachesPostValidationTimeout(t *testing.T) {
	cases := []struct {
		name  string
		flags []string
		want  time.Duration
	}{
		{name: "flag", flags: []string{"--commit-timeout", "7m"}, want: 7 * time.Minute},
		{name: "unset leaves library default", want: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seen := stubStalledCommit(t)
			args := append([]string{"--path", "a.go", "-m", "fix(test): commit timeout (fak cmd)", "--no-build-check"}, tc.flags...)
			var out, errb bytes.Buffer
			code := runCommit(&out, &errb, args)
			if len(*seen) != 1 {
				t.Fatalf("commitFn calls = %d, want 1; code=%d stderr=%q", len(*seen), code, errb.String())
			}
			if got := (*seen)[0].PostValidationTimeout; got != tc.want {
				t.Fatalf("PostValidationTimeout = %v, want %v", got, tc.want)
			}
			if !strings.Contains(out.String(), safecommit.ReasonCommitStalled) || !strings.Contains(out.String(), "--commit-timeout") {
				t.Fatalf("COMMIT_STALLED render must name --commit-timeout; got:\n%s", out.String())
			}
		})
	}
}

func TestRunCommitCommitTimeoutUsageErrorsSkipCommit(t *testing.T) {
	cases := []struct {
		name    string
		flags   []string
		wantErr string
	}{
		{name: "negative flag", flags: []string{"--commit-timeout=-1s"}, wantErr: "fak commit: --commit-timeout must not be negative"},
		{name: "flag above the stale-index cap", flags: []string{"--commit-timeout", "30m"}, wantErr: "fak commit: --commit-timeout 30m0s exceeds the"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seen := stubStalledCommit(t)
			args := append([]string{"--path", "a.go", "-m", "fix(test): commit timeout (fak cmd)", "--no-build-check"}, tc.flags...)
			var out, errb bytes.Buffer
			code := runCommit(&out, &errb, args)
			if code != 2 || !strings.Contains(errb.String(), tc.wantErr) {
				t.Fatalf("code=%d stderr=%q; want exit 2 with %q", code, errb.String(), tc.wantErr)
			}
			if len(*seen) != 0 {
				t.Fatalf("commitFn must not run on a --commit-timeout usage error; calls=%d", len(*seen))
			}
		})
	}
}

func TestRunCommitHelpDocumentsCommitTimeout(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCommit(&out, &errb, []string{"--help"}); code != 2 {
		t.Fatalf("help exit = %d, want usage exit 2", code)
	}
	for _, want := range []string{"commit-timeout", "pre-commit/commit-msg hooks", safecommit.DefaultPostValidationTimeout.String(), safecommit.MaxPostValidationTimeout.String(), "COMMIT_STALLED"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("commit help missing %q; got:\n%s", want, errb.String())
		}
	}
}

func TestRenderCommitResultCommitTimeoutHintOnlyOnStall(t *testing.T) {
	var busy bytes.Buffer
	renderCommitResult(&busy, safecommit.Result{Reason: safecommit.ReasonLockBusy})
	if strings.Contains(busy.String(), "--commit-timeout") {
		t.Fatalf("LOCK_BUSY render must not suggest --commit-timeout; got:\n%s", busy.String())
	}
}
