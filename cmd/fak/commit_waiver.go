package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

const (
	waiverNoBuildCheck      = "no-build-check"
	waiverBuildCheckOff     = "build-check-off"
	waiverBuildCheckTimeout = "build-check-timeout"

	verificationWaiverTrailerKey = "Verification-Waiver"
	commitWaiverHookEnv          = "FAK_COMMIT_WAIVER_HOOK"
	commitWaiverHookGitConfig    = "fak.commit.waiverHook"
	commitWaiverHookTimeout      = 2 * time.Minute

	ReasonVerificationWaiverUnrecorded = "ERR_VERIFICATION_WAIVER_UNRECORDED"
)

var errVerificationWaiverUnrecorded = errors.New(ReasonVerificationWaiverUnrecorded)

var trailerLineRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9-]*: `)

// commitVerificationWaiver names the operator waiver a commit lands under, or "" when
// none was taken.
func commitVerificationWaiver(noBuildCheckFlag, buildCheckOff bool, bc safecommit.BuildCheckResult) string {
	switch {
	case noBuildCheckFlag:
		return waiverNoBuildCheck
	case buildCheckOff:
		return waiverBuildCheckOff
	case bc.FailedOpen && bc.Outcome == safecommit.BuildCheckSkippedTimeout:
		return waiverBuildCheckTimeout
	}
	return ""
}

func withVerificationWaiverTrailer(message, kind string) string {
	if kind == "" {
		return message
	}
	line := verificationWaiverTrailerKey + ": " + kind
	trimmed := strings.TrimRight(message, " \t\r\n")
	paras := strings.Split(strings.ReplaceAll(trimmed, "\r\n", "\n"), "\n\n")
	last := paras[len(paras)-1]
	if len(paras) > 1 && isTrailerBlock(last) {
		for _, l := range strings.Split(last, "\n") {
			if strings.TrimSpace(l) == line {
				return message
			}
		}
		return trimmed + "\n" + line + "\n"
	}
	return trimmed + "\n\n" + line + "\n"
}

func isTrailerBlock(para string) bool {
	for _, l := range strings.Split(para, "\n") {
		if !trailerLineRE.MatchString(l) {
			return false
		}
	}
	return true
}

type waiverUnrecordedError struct {
	SHA, Kind, Rerun string
	Err              error
}

func (e *waiverUnrecordedError) Error() string {
	return fmt.Sprintf("%s: commit %s exists but its %s waiver obligation is unrecorded: %v; rerun: %s",
		ReasonVerificationWaiverUnrecorded, e.SHA, e.Kind, e.Err, e.Rerun)
}

func (e *waiverUnrecordedError) Is(target error) bool {
	return target == errVerificationWaiverUnrecorded
}

func (e *waiverUnrecordedError) Unwrap() error { return e.Err }

// commitWaiverHookLookup reads git config; "" with a nil error means git answered with no
// readable value. Only a git that could not answer (spawn failure, timeout) is an error.
var commitWaiverHookLookup = func(root string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	args := []string{"config", "--get", commitWaiverHookGitConfig}
	if root != "" {
		args = append([]string{"-C", root}, args...)
	}
	out, err := exec.CommandContext(ctx, "git", args...).Output()
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) && ctx.Err() == nil {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("git config --get %s: %w", commitWaiverHookGitConfig, err)
	}
	return strings.TrimSpace(string(out)), nil
}

var commitWaiverHookRun = func(ctx context.Context, dir string, argv []string, out io.Writer) error {
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.WaitDelay = 5 * time.Second
	return cmd.Run()
}

// recordCommitWaiver runs the configured waiver hook for a commit that landed under
// a verification waiver. The commit already exists, so a hook failure is reported as
// an unrecorded obligation rather than undone.
func recordCommitWaiver(stderr io.Writer, verb, root string, res safecommit.Result, kind string) error {
	if kind == "" || !res.Committed || res.SHA == "" {
		return nil
	}
	hookArgs := []string{"--commit", res.SHA, "--waiver", kind}
	hook := strings.TrimSpace(os.Getenv(commitWaiverHookEnv))
	if hook == "" {
		configured, err := commitWaiverHookLookup(root)
		if err != nil {
			return &waiverUnrecordedError{SHA: res.SHA, Kind: kind, Err: err,
				Rerun: fmt.Sprintf("<waiver hook> %s (set %s or git config %s)", strings.Join(hookArgs, " "), commitWaiverHookEnv, commitWaiverHookGitConfig)}
		}
		hook = configured
	}
	if hook == "" {
		fmt.Fprintf(stderr, "%s: %s landed under verification waiver %s; no waiver hook configured (%s or git config %s)\n",
			verb, res.SHA, kind, commitWaiverHookEnv, commitWaiverHookGitConfig)
		return nil
	}
	rerun := hook + " " + strings.Join(hookArgs, " ")
	argv := splitCommandLine(hook)
	if len(argv) == 0 {
		return &waiverUnrecordedError{SHA: res.SHA, Kind: kind, Rerun: rerun, Err: errors.New("waiver hook command is empty")}
	}
	argv = append(argv, hookArgs...)
	ctx, cancel := context.WithTimeout(context.Background(), commitWaiverHookTimeout)
	defer cancel()
	if err := commitWaiverHookRun(ctx, root, argv, stderr); err != nil {
		if ctx.Err() != nil {
			err = fmt.Errorf("waiver hook timed out after %s: %w", commitWaiverHookTimeout, ctx.Err())
		}
		return &waiverUnrecordedError{SHA: res.SHA, Kind: kind, Rerun: rerun, Err: err}
	}
	return nil
}
