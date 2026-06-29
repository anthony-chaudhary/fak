package loopgate

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRealDOSCommitAuditDrivesGate(t *testing.T) {
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("dos CLI not on PATH")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.com", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.com")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	git("init", "-q")
	if err := os.WriteFile(filepath.Join(repo, "bug.go"), []byte("package bug\n\nfunc Fixed() bool { return true }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "bug.go")
	git("commit", "-q", "-m", "feat(loopgate): add code for witnessed exit gate")

	dec := Adjudicate(context.Background(), Turn{ClaimedDone: true}, func(ctx context.Context, req Request) (WitnessResult, error) {
		if req.Kind != CriterionCommitAudit {
			t.Fatalf("request kind = %s, want commit-audit", req.Kind)
		}
		cmd := exec.CommandContext(ctx, "dos", req.Argv()...)
		cmd.Dir = repo
		out, err := cmd.CombinedOutput()
		if err != nil {
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) && exitErr.ExitCode() == 1 {
				return CommitAuditResultFromJSON(out)
			}
			return WitnessResult{}, err
		}
		return CommitAuditResultFromJSON(out)
	})
	if dec.Verdict != VerdictWitnessed {
		t.Fatalf("real dos commit-audit decision = %+v", dec)
	}
}

func TestRealDOSVerifyNotShippedRearms(t *testing.T) {
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("dos CLI not on PATH")
	}
	dec := Adjudicate(context.Background(), Turn{
		ClaimedDone: true,
		Criterion: Criterion{
			Kind:  CriterionVerify,
			Plan:  "fak-loopgate-integration-never-shipped",
			Phase: "missing-phase",
		},
	}, func(ctx context.Context, req Request) (WitnessResult, error) {
		cmd := exec.CommandContext(ctx, "dos", req.Argv()...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) && exitErr.ExitCode() == 1 {
				return VerifyResultFromJSON(out)
			}
			return WitnessResult{}, err
		}
		return VerifyResultFromJSON(out)
	})
	if dec.Verdict != VerdictNotYet || dec.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("real dos verify decision = %+v, want NOT_YET/%s", dec, ReasonDoneUnwitnessed)
	}
	if !strings.Contains(dec.Summary, "not shipped") {
		t.Fatalf("summary = %q, want verify reason surfaced", dec.Summary)
	}
}
