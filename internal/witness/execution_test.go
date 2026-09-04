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

func TestFailToPassVerifier(t *testing.T) {
	t.Run("RedToGreenTransition", func(t *testing.T) {
		dir := newExecutionRepo(t)
		ctx := context.Background()

		writeRepoFile(t, dir, "value.txt", "bad\n")
		gitIn(t, dir, "add", "value.txt")
		gitIn(t, dir, "commit", "-q", "-m", "parent red")

		writeRepoFile(t, dir, "value.txt", "good\n")
		gitIn(t, dir, "add", "value.txt")
		gitIn(t, dir, "commit", "-q", "-m", "fix green")

		res, err := VerifyFailToPass(ctx, dir, "HEAD~1", "HEAD", "git grep -q good -- value.txt")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Verdict != ExecPass {
			t.Fatalf("verdict = %s, want %s (reason: %s)", res.Verdict, ExecPass, res.Reason)
		}
		if res.Reason != "" {
			t.Fatalf("reason = %q, want empty", res.Reason)
		}
		if len(res.Evidence) != 2 {
			t.Fatalf("evidence count = %d, want 2", len(res.Evidence))
		}
		if res.Evidence[0].Outcome != "fail" || res.Evidence[1].Outcome != "pass" {
			t.Fatalf("evidence outcomes = [%s, %s], want [fail, pass]", res.Evidence[0].Outcome, res.Evidence[1].Outcome)
		}
		if got := res.WitnessOutcome(); got != abi.WitnessConfirmed {
			t.Fatalf("WitnessOutcome = %v, want confirmed", got)
		}

		v := NewExecutionVerifier(dir)
		resRefs := v.VerifyFailToPassRefs(ctx, "HEAD~1", "HEAD", []string{"git", "grep", "-q", "good", "--", "value.txt"})
		if resRefs.Verdict != ExecPass {
			t.Fatalf("VerifyFailToPassRefs verdict = %s, want %s (reason: %s)", resRefs.Verdict, ExecPass, resRefs.Reason)
		}

		resDefaultParent := v.VerifyFailToPassRefs(ctx, "", "HEAD", []string{"git", "grep", "-q", "good", "--", "value.txt"})
		if resDefaultParent.Verdict != ExecPass {
			t.Fatalf("VerifyFailToPassRefs with default parent verdict = %s, want %s", resDefaultParent.Verdict, ExecPass)
		}

		assertRepoClean(t, dir)
	})

	t.Run("TautologicalTestRejection", func(t *testing.T) {
		dir := newExecutionRepo(t)
		ctx := context.Background()

		writeRepoFile(t, dir, "value.txt", "good\n")
		gitIn(t, dir, "add", "value.txt")
		gitIn(t, dir, "commit", "-q", "-m", "already good")

		gitIn(t, dir, "commit", "--allow-empty", "-q", "-m", "no-op fix")

		res, err := VerifyFailToPass(ctx, dir, "HEAD~1", "HEAD", "git grep -q good -- value.txt")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Verdict != ExecUnwitnessed {
			t.Fatalf("verdict = %s, want %s", res.Verdict, ExecUnwitnessed)
		}
		if !strings.HasPrefix(res.Reason, "fail_to_pass_green_at_parent") {
			t.Fatalf("reason = %q, want fail_to_pass_green_at_parent", res.Reason)
		}
		if got := res.WitnessOutcome(); got != abi.WitnessRefuted {
			t.Fatalf("WitnessOutcome = %v, want refuted", got)
		}

		v := NewExecutionVerifier(dir)
		resRefs := v.VerifyFailToPassRefs(ctx, "HEAD~1", "HEAD", []string{"git", "grep", "-q", "good", "--", "value.txt"})
		if resRefs.Verdict != ExecUnwitnessed {
			t.Fatalf("VerifyFailToPassRefs verdict = %s, want %s", resRefs.Verdict, ExecUnwitnessed)
		}
		if !strings.HasPrefix(resRefs.Reason, "fail_to_pass_green_at_parent") {
			t.Fatalf("VerifyFailToPassRefs reason = %q, want fail_to_pass_green_at_parent", resRefs.Reason)
		}

		assertRepoClean(t, dir)
	})

	t.Run("BrokenFixRejection", func(t *testing.T) {
		dir := newExecutionRepo(t)
		ctx := context.Background()

		writeRepoFile(t, dir, "value.txt", "bad\n")
		gitIn(t, dir, "add", "value.txt")
		gitIn(t, dir, "commit", "-q", "-m", "parent bad")

		writeRepoFile(t, dir, "value.txt", "still bad\n")
		gitIn(t, dir, "add", "value.txt")
		gitIn(t, dir, "commit", "-q", "-m", "broken fix")

		res, err := VerifyFailToPass(ctx, dir, "HEAD~1", "HEAD", "git grep -q good -- value.txt")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Verdict != ExecUnwitnessed {
			t.Fatalf("verdict = %s, want %s", res.Verdict, ExecUnwitnessed)
		}
		if !strings.HasPrefix(res.Reason, "fail_to_pass_still_red") {
			t.Fatalf("reason = %q, want fail_to_pass_still_red", res.Reason)
		}
		if got := res.WitnessOutcome(); got != abi.WitnessRefuted {
			t.Fatalf("WitnessOutcome = %v, want refuted", got)
		}

		v := NewExecutionVerifier(dir)
		resRefs := v.VerifyFailToPassRefs(ctx, "HEAD~1", "HEAD", []string{"git", "grep", "-q", "good", "--", "value.txt"})
		if resRefs.Verdict != ExecUnwitnessed {
			t.Fatalf("VerifyFailToPassRefs verdict = %s, want %s", resRefs.Verdict, ExecUnwitnessed)
		}
		if !strings.HasPrefix(resRefs.Reason, "fail_to_pass_still_red") {
			t.Fatalf("VerifyFailToPassRefs reason = %q, want fail_to_pass_still_red", resRefs.Reason)
		}

		assertRepoClean(t, dir)
	})

	t.Run("GoTestExecutionInRepo", func(t *testing.T) {
		requireGoAndGit(t)
		dir := newGoModuleRepo(t)
		ctx := context.Background()

		writeRepoFile(t, dir, "calc_test.go", "package m\n\nimport \"testing\"\n\nfunc TestCalc(t *testing.T) {\n\tt.Fatal(\"failing on parent\")\n}\n")
		gitIn(t, dir, "add", "calc_test.go")
		gitIn(t, dir, "commit", "-q", "-m", "parent with failing test")

		writeRepoFile(t, dir, "calc_test.go", "package m\n\nimport \"testing\"\n\nfunc TestCalc(t *testing.T) {\n\t// passing on fix\n}\n")
		gitIn(t, dir, "add", "calc_test.go")
		gitIn(t, dir, "commit", "-q", "-m", "fix test passes")

		res, err := VerifyFailToPass(ctx, dir, "HEAD~1", "HEAD", "./...")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Verdict != ExecPass {
			t.Fatalf("verdict = %s, want %s (reason: %s)", res.Verdict, ExecPass, res.Reason)
		}

		assertRepoClean(t, dir)
	})

	t.Run("InvalidInputsAndErrors", func(t *testing.T) {
		dir := newExecutionRepo(t)
		ctx := context.Background()

		writeRepoFile(t, dir, "value.txt", "bad\n")
		gitIn(t, dir, "add", "value.txt")
		gitIn(t, dir, "commit", "-q", "-m", "parent")
		writeRepoFile(t, dir, "value.txt", "good\n")
		gitIn(t, dir, "add", "value.txt")
		gitIn(t, dir, "commit", "-q", "-m", "fix")

		resEmpty, err := VerifyFailToPass(ctx, dir, "HEAD~1", "HEAD", "   ")
		if err == nil {
			t.Fatalf("expected error for empty selector, got nil")
		}
		if resEmpty.Verdict != ExecUnwitnessed {
			t.Fatalf("verdict = %s, want %s", resEmpty.Verdict, ExecUnwitnessed)
		}

		resBadCommit, err := VerifyFailToPass(ctx, dir, "HEAD~1", "nonexistent-ref", "git grep -q good -- value.txt")
		if err == nil {
			t.Fatalf("expected error for missing commit ref, got nil")
		}
		if resBadCommit.Verdict != ExecUnwitnessed {
			t.Fatalf("verdict = %s, want %s", resBadCommit.Verdict, ExecUnwitnessed)
		}

		resBadParent, err := VerifyFailToPass(ctx, dir, "nonexistent-parent", "HEAD", "git grep -q good -- value.txt")
		if err == nil {
			t.Fatalf("expected error for missing parent ref, got nil")
		}
		if resBadParent.Verdict != ExecUnwitnessed {
			t.Fatalf("verdict = %s, want %s", resBadParent.Verdict, ExecUnwitnessed)
		}

		resBadCmd, err := VerifyFailToPass(ctx, dir, "HEAD~1", "HEAD", "nonexistent-binary-command-xyz")
		if err == nil {
			t.Fatalf("expected error for invalid binary, got nil")
		}
		if resBadCmd.Verdict != ExecUnwitnessed {
			t.Fatalf("verdict = %s, want %s", resBadCmd.Verdict, ExecUnwitnessed)
		}
	})

	t.Run("SelectorParsing", func(t *testing.T) {
		cases := []struct {
			input string
			want  []string
		}{
			{
				input: "go test ./internal/witness",
				want:  []string{"go", "test", "./internal/witness"},
			},
			{
				input: "go test -run 'TestSomething' ./...",
				want:  []string{"go", "test", "-run", "TestSomething", "./..."},
			},
			{
				input: "-run TestSomething ./...",
				want:  []string{"go", "test", "-run", "TestSomething", "./..."},
			},
			{
				input: "./internal/witness",
				want:  []string{"go", "test", "./internal/witness"},
			},
			{
				input: "git grep -q good -- value.txt",
				want:  []string{"git", "grep", "-q", "good", "--", "value.txt"},
			},
			{
				input: "./test.sh",
				want:  []string{"./test.sh"},
			},
			{
				input: "   ",
				want:  nil,
			},
		}

		for _, tc := range cases {
			got := splitTestSelector(tc.input)
			if len(got) != len(tc.want) {
				t.Fatalf("splitTestSelector(%q) = %v, want %v", tc.input, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("splitTestSelector(%q)[%d] = %q, want %q", tc.input, i, got[i], tc.want[i])
				}
			}
		}
	})
}
