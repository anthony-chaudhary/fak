package livecodebench

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"testing"
	"time"
)

// twoSumProblem is the golden fixture: a single functional problem with one
// public and one private stdin test whose expected output is the sum of two
// space-separated integers.
func twoSumProblem() Problem {
	return Problem{
		QuestionID: "golden-sum",
		Scenario:   ScenarioCodeGeneration,
		PublicTests: []TestCase{
			{Input: "2 3\n", Output: "5\n", TestType: "stdin"},
		},
		PrivateTests: []TestCase{
			{Input: "10 20\n", Output: "30\n", TestType: "stdin"},
		},
	}
}

// sumRunner is a pure oracle standing in for a real sandbox: the "correct"
// program returns the true sum, so it accepts; any other code path returns a
// fixed wrong answer, times out, or errors, letting the golden test exercise
// every verdict without executing untrusted code.
func sumRunner(_ context.Context, code, stdin string, timeout time.Duration) (string, bool, error) {
	switch code {
	case "correct":
		var a, b int
		if _, err := fmt.Sscan(stdin, &a, &b); err != nil {
			return "", false, err
		}
		return strconv.Itoa(a+b) + "\n", false, nil
	case "wrong":
		return "0\n", false, nil
	case "slow":
		return "", true, nil
	case "boom":
		return "", false, errors.New("Traceback: ZeroDivisionError")
	default:
		return "", false, errors.New("unknown program")
	}
}

func TestGradeCode_GoldenCorrectAndWrong(t *testing.T) {
	p := twoSumProblem()

	got := GradeCode(context.Background(), p, "correct", sumRunner, GradeOptions{})
	if !got.Pass || got.Verdict != ExecAccepted {
		t.Fatalf("known-correct: got verdict=%q pass=%v, want AC/true", got.Verdict, got.Pass)
	}
	if !got.OfficialHarnessAvailable {
		t.Fatalf("known-correct: official_harness_available should be true with a live runner")
	}
	if got.TestsRun != 2 || got.TestsTotal != 2 {
		t.Fatalf("known-correct: ran %d/%d tests, want 2/2", got.TestsRun, got.TestsTotal)
	}

	bad := GradeCode(context.Background(), p, "wrong", sumRunner, GradeOptions{})
	if bad.Pass || bad.Verdict != ExecWrongAnswer {
		t.Fatalf("known-wrong: got verdict=%q pass=%v, want WA/false", bad.Verdict, bad.Pass)
	}
}

func TestGradeCode_TLEDistinctFromWA(t *testing.T) {
	p := twoSumProblem()
	got := GradeCode(context.Background(), p, "slow", sumRunner, GradeOptions{Timeout: 50 * time.Millisecond})
	if got.Pass || got.Verdict != ExecTimeLimit {
		t.Fatalf("slow program: got verdict=%q pass=%v, want TLE/false", got.Verdict, got.Pass)
	}
	if got.Verdict == ExecWrongAnswer {
		t.Fatalf("TLE must not be folded into WA")
	}
}

func TestGradeCode_RuntimeError(t *testing.T) {
	p := twoSumProblem()
	got := GradeCode(context.Background(), p, "boom", sumRunner, GradeOptions{})
	if got.Pass || got.Verdict != ExecRuntimeError {
		t.Fatalf("erroring program: got verdict=%q pass=%v, want RE/false", got.Verdict, got.Pass)
	}
}

func TestGradeCode_UnavailableSandboxNeverPasses(t *testing.T) {
	p := twoSumProblem()
	got := GradeCode(context.Background(), p, "correct", nil, GradeOptions{})
	if got.Pass {
		t.Fatalf("nil runner must never report a pass, got pass=%v", got.Pass)
	}
	if got.OfficialHarnessAvailable {
		t.Fatalf("nil runner must degrade to official_harness_available=false")
	}
	if got.Verdict != ExecUnavailable {
		t.Fatalf("nil runner verdict=%q, want %q", got.Verdict, ExecUnavailable)
	}
}

func TestGradeCode_ParallelEvalHonored(t *testing.T) {
	p := twoSumProblem()
	// NumProcessEvaluate>1 must still produce a deterministic AC on the correct
	// program and run every test.
	got := GradeCode(context.Background(), p, "correct", sumRunner, GradeOptions{NumProcessEvaluate: 4})
	if !got.Pass || got.TestsRun != 2 {
		t.Fatalf("parallel eval: got verdict=%q pass=%v run=%d, want AC/true/2", got.Verdict, got.Pass, got.TestsRun)
	}
}

func TestGradeCode_NoTestsAbstains(t *testing.T) {
	p := Problem{QuestionID: "empty", Scenario: ScenarioCodeGeneration}
	got := GradeCode(context.Background(), p, "correct", sumRunner, GradeOptions{})
	if got.Pass || got.Verdict != ExecUnavailable {
		t.Fatalf("no tests: got verdict=%q pass=%v, want UNAVAILABLE/false", got.Verdict, got.Pass)
	}
	if !got.OfficialHarnessAvailable {
		t.Fatalf("no tests but a live runner: official_harness_available should stay true")
	}
}

func TestSandboxCodegenGrader_AbstainsWithoutSandbox(t *testing.T) {
	p := twoSumProblem()

	live := SandboxCodegenGrader(sumRunner, GradeOptions{})
	pass, err := live(p, "correct")
	if err != nil || !pass {
		t.Fatalf("live grader on correct: pass=%v err=%v, want true/nil", pass, err)
	}

	unavailable := SandboxCodegenGrader(nil, GradeOptions{})
	pass, err = unavailable(p, "correct")
	if err == nil {
		t.Fatalf("nil-runner grader must error (abstain), not fabricate a miss; pass=%v", pass)
	}
	if pass {
		t.Fatalf("nil-runner grader must never report a pass")
	}
}
