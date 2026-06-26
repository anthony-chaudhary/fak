package witness

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestExecutionWitnessPassesOnlyOnRedToGreenWithStablePassToPass(t *testing.T) {
	dir := newExecutionRepo(t)
	ctx := context.Background()

	writeRepoFile(t, dir, "value.txt", "bad\n")
	writeRepoFile(t, dir, "stable.txt", "invariant\n")
	gitIn(t, dir, "add", "value.txt", "stable.txt")
	gitIn(t, dir, "commit", "-q", "-m", "parent red")

	writeRepoFile(t, dir, "value.txt", "good\n")
	gitIn(t, dir, "add", "value.txt")
	gitIn(t, dir, "commit", "-q", "-m", "child green")

	res := NewExecutionVerifier(dir).Verify(ctx, ExecutionSpec{
		Commit: "HEAD",
		FailToPass: []ExecutionSelector{{
			ID:      "value-good",
			Command: []string{"git", "grep", "-q", "good", "--", "value.txt"},
		}},
		PassToPass: []ExecutionSelector{{
			ID:      "stable-invariant",
			Command: []string{"git", "grep", "-q", "invariant", "--", "stable.txt"},
		}},
	})
	if res.Verdict != ExecPass {
		t.Fatalf("verdict = %s reason=%q evidence=%+v, want %s", res.Verdict, res.Reason, res.Evidence, ExecPass)
	}
	if got := res.WitnessOutcome(); got != abi.WitnessConfirmed {
		t.Fatalf("WitnessOutcome = %v, want confirmed", got)
	}
	assertRepoClean(t, dir)
}

func TestExecutionWitnessRejectsGreenNoOp(t *testing.T) {
	dir := newExecutionRepo(t)
	ctx := context.Background()

	writeRepoFile(t, dir, "value.txt", "good\n")
	gitIn(t, dir, "add", "value.txt")
	gitIn(t, dir, "commit", "-q", "-m", "already green")
	gitIn(t, dir, "commit", "--allow-empty", "-q", "-m", "no-op")

	res := NewExecutionVerifier(dir).Verify(ctx, ExecutionSpec{
		Commit: "HEAD",
		FailToPass: []ExecutionSelector{{
			ID:      "value-good",
			Command: []string{"git", "grep", "-q", "good", "--", "value.txt"},
		}},
	})
	if res.Verdict != ExecUnwitnessed {
		t.Fatalf("verdict = %s, want %s", res.Verdict, ExecUnwitnessed)
	}
	if res.Reason != "fail_to_pass_green_at_parent:value-good" {
		t.Fatalf("reason = %q, want parent-green rejection", res.Reason)
	}
	if got := res.WitnessOutcome(); got != abi.WitnessRefuted {
		t.Fatalf("WitnessOutcome = %v, want refuted", got)
	}
	assertRepoClean(t, dir)
}

func TestExecutionWitnessRejectsPassToPassRegression(t *testing.T) {
	dir := newExecutionRepo(t)
	ctx := context.Background()

	writeRepoFile(t, dir, "value.txt", "bad\n")
	writeRepoFile(t, dir, "stable.txt", "invariant\n")
	gitIn(t, dir, "add", "value.txt", "stable.txt")
	gitIn(t, dir, "commit", "-q", "-m", "parent red stable")

	writeRepoFile(t, dir, "value.txt", "good\n")
	writeRepoFile(t, dir, "stable.txt", "broken\n")
	gitIn(t, dir, "add", "value.txt", "stable.txt")
	gitIn(t, dir, "commit", "-q", "-m", "child green regression")

	res := NewExecutionVerifier(dir).Verify(ctx, ExecutionSpec{
		Commit: "HEAD",
		FailToPass: []ExecutionSelector{{
			ID:      "value-good",
			Command: []string{"git", "grep", "-q", "good", "--", "value.txt"},
		}},
		PassToPass: []ExecutionSelector{{
			ID:      "stable-invariant",
			Command: []string{"git", "grep", "-q", "invariant", "--", "stable.txt"},
		}},
	})
	if res.Verdict != ExecUnwitnessed {
		t.Fatalf("verdict = %s, want %s", res.Verdict, ExecUnwitnessed)
	}
	if res.Reason != "pass_to_pass_regressed:stable-invariant" {
		t.Fatalf("reason = %q, want PASS_TO_PASS regression", res.Reason)
	}
	assertRepoClean(t, dir)
}

func TestExecutionWitnessWithoutFailToPassAbstains(t *testing.T) {
	res := NewExecutionVerifierWithRunners((&fakeGit{code: 0, out: "abc\n"}).run, nilCommandRunner, "").Verify(context.Background(), ExecutionSpec{
		Commit: "HEAD",
		PassToPass: []ExecutionSelector{{
			ID:      "docs-only",
			Command: []string{"git", "status"},
		}},
	})
	if res.Verdict != ExecNotApplicable {
		t.Fatalf("verdict = %s, want %s", res.Verdict, ExecNotApplicable)
	}
	if got := res.WitnessOutcome(); got != abi.WitnessAbstain {
		t.Fatalf("WitnessOutcome = %v, want abstain", got)
	}
}

func TestResolverExecClaimMapsExecutionVerdicts(t *testing.T) {
	dir := newExecutionRepo(t)
	ctx := context.Background()

	writeRepoFile(t, dir, "value.txt", "bad\n")
	gitIn(t, dir, "add", "value.txt")
	gitIn(t, dir, "commit", "-q", "-m", "red")
	writeRepoFile(t, dir, "value.txt", "good\n")
	gitIn(t, dir, "add", "value.txt")
	gitIn(t, dir, "commit", "-q", "-m", "green")

	raw, err := json.Marshal(ExecutionSpec{
		Commit: "HEAD",
		FailToPass: []ExecutionSelector{{
			ID:      "value-good",
			Command: []string{"git", "grep", "-q", "good", "--", "value.txt"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "exec:"+string(raw)); got != abi.WitnessConfirmed {
		t.Fatalf("exec claim = %v, want confirmed", got)
	}

	raw, err = json.Marshal(ExecutionSpec{Commit: "HEAD"})
	if err != nil {
		t.Fatal(err)
	}
	if got := NewWithRunner(gitRunner, dir).Resolve(ctx, nil, "exec:"+string(raw)); got != abi.WitnessAbstain {
		t.Fatalf("exec claim without selector = %v, want abstain", got)
	}
}

func nilCommandRunner(ctx context.Context, dir string, argv ...string) (string, int, error) {
	return "", 0, nil
}

func newExecutionRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	gitIn(t, dir, "init", "-q")
	gitIn(t, dir, "config", "user.email", "t@t")
	gitIn(t, dir, "config", "user.name", "t")
	return dir
}

func gitIn(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.CommandContext(context.Background(), "git", args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func assertRepoClean(t *testing.T, dir string) {
	t.Helper()
	cmd := exec.CommandContext(context.Background(), "git", "status", "--porcelain")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(out)) != "" {
		t.Fatalf("source repo was mutated:\n%s", out)
	}
}

func writeRepoFile(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(dir+string(os.PathSeparator)+name, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
