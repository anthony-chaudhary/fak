package livecodebench

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// grade.go implements the sandboxed code-execution grader (#2100, epic #2085):
// it runs a candidate program against a problem's public + private test cases
// through an INJECTED sandbox runner, applies a per-test timeout and the
// upstream parallel-eval fan-out (mirroring lcb_runner's --timeout and
// --num_process_evaluate), and folds the per-test outcomes into a single
// verdict that keeps a time-limit-exceeded (TLE) distinct from a wrong answer
// (WA) and a runtime error (RE).
//
// The runner is injected so the whole grader is unit-tested without a live
// sandbox (the Windows dev box has none) and so the CLI can plug a Docker/POSIX
// executor behind it. The honesty fence is load-bearing: when no runner is
// available the grader degrades to official_harness.available=false and NEVER
// reports a pass — an unavailable sandbox can only abstain, never fabricate a
// green. A grader-produced number is a local, ungraded signal regardless; only
// the official lcb_runner grading (EvidenceOfficialLCBRunner) backs a claim.

// Execution verdicts. Upstream-aligned tokens: AC is the only passing verdict,
// and TLE is kept distinct from WA so a slow solution is never silently folded
// into a wrong answer. UNAVAILABLE is the abstention verdict — no sandbox, or no
// test cases to witness a pass — and always carries Pass=false.
const (
	ExecAccepted     = "AC"
	ExecWrongAnswer  = "WA"
	ExecTimeLimit    = "TLE"
	ExecRuntimeError = "RE"
	ExecUnavailable  = "UNAVAILABLE"
)

// DefaultGradeTimeout mirrors lcb_runner's per-test wall-clock default (6s).
const DefaultGradeTimeout = 6 * time.Second

// ExecRunner executes candidate code against one test's stdin under the given
// per-test timeout and returns the program's stdout. timedOut reports that the
// program exceeded the timeout (graded TLE); a non-nil err that is not a timeout
// is a runtime error (graded RE). It is injected: the CLI plugs a Docker/POSIX
// sandbox, tests plug a pure oracle.
type ExecRunner func(ctx context.Context, code, stdin string, timeout time.Duration) (stdout string, timedOut bool, err error)

// GradeOptions carries the lcb_runner-parity execution knobs.
type GradeOptions struct {
	// Timeout is the per-test wall-clock limit (upstream --timeout). Zero falls
	// back to DefaultGradeTimeout so a caller cannot accidentally run unbounded.
	Timeout time.Duration
	// NumProcessEvaluate mirrors upstream --num_process_evaluate: the maximum
	// number of test cases evaluated concurrently. Values <=1 grade serially.
	// Aggregation is index-ordered regardless, so the verdict is deterministic.
	NumProcessEvaluate int
}

// TestOutcome is one test case's execution result.
type TestOutcome struct {
	Index   int    `json:"index"`
	Verdict string `json:"verdict"`
	Detail  string `json:"detail,omitempty"`
}

// GradeResult is a candidate program's execution grade for one problem.
type GradeResult struct {
	QuestionID string        `json:"question_id"`
	Verdict    string        `json:"verdict"`
	Pass       bool          `json:"pass"`
	TestsRun   int           `json:"tests_run"`
	TestsTotal int           `json:"tests_total"`
	Outcomes   []TestOutcome `json:"outcomes,omitempty"`
	// OfficialHarnessAvailable is false when no sandbox runner was supplied; the
	// grader then abstains (Verdict=UNAVAILABLE, Pass=false). It is the
	// machine-readable honesty fence a report carries — an unavailable sandbox
	// degrades here, it never fabricates a pass.
	OfficialHarnessAvailable bool `json:"official_harness_available"`
}

// GradeCode runs code against p's public + private test cases through the
// injected sandbox runner and folds the per-test outcomes into one verdict:
// AC only when every test matches within the time limit, otherwise the first
// non-AC outcome in test-index order (RE / TLE / WA). When run is nil the
// sandbox is unavailable, so the grade abstains with
// OfficialHarnessAvailable=false and Pass=false — never a fabricated pass. A
// problem with no test cases likewise abstains: there is nothing to witness.
func GradeCode(ctx context.Context, p Problem, code string, run ExecRunner, opts GradeOptions) GradeResult {
	res := GradeResult{QuestionID: p.QuestionID}
	tests := make([]TestCase, 0, len(p.PublicTests)+len(p.PrivateTests))
	tests = append(tests, p.PublicTests...)
	tests = append(tests, p.PrivateTests...)
	res.TestsTotal = len(tests)

	if run == nil {
		res.Verdict = ExecUnavailable
		res.OfficialHarnessAvailable = false
		return res
	}
	res.OfficialHarnessAvailable = true
	if len(tests) == 0 {
		res.Verdict = ExecUnavailable
		return res
	}

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = DefaultGradeTimeout
	}
	workers := opts.NumProcessEvaluate
	if workers < 1 {
		workers = 1
	}
	if workers > len(tests) {
		workers = len(tests)
	}

	outcomes := make([]TestOutcome, len(tests))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for i := range tests {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, tc TestCase) {
			defer wg.Done()
			defer func() { <-sem }()
			outcomes[i] = gradeOneTest(ctx, code, tc, run, timeout, i)
		}(i, tests[i])
	}
	wg.Wait()

	verdict := ExecAccepted
	for _, o := range outcomes {
		if o.Verdict != ExecAccepted {
			verdict = o.Verdict
			break
		}
	}
	res.Outcomes = outcomes
	res.TestsRun = len(tests)
	res.Verdict = verdict
	res.Pass = verdict == ExecAccepted
	return res
}

func gradeOneTest(ctx context.Context, code string, tc TestCase, run ExecRunner, timeout time.Duration, idx int) TestOutcome {
	stdout, timedOut, err := run(ctx, code, tc.Input, timeout)
	switch {
	case timedOut:
		return TestOutcome{Index: idx, Verdict: ExecTimeLimit, Detail: fmt.Sprintf("exceeded %s", timeout)}
	case err != nil:
		return TestOutcome{Index: idx, Verdict: ExecRuntimeError, Detail: err.Error()}
	case normalizeExecOutput(stdout) != normalizeExecOutput(tc.Output):
		return TestOutcome{Index: idx, Verdict: ExecWrongAnswer}
	default:
		return TestOutcome{Index: idx, Verdict: ExecAccepted}
	}
}

// normalizeExecOutput matches lcb_runner's lenient stdout comparison: CRLF is
// folded to LF, trailing whitespace is stripped from every line, and trailing
// blank lines are dropped, so cosmetic newline/spacing differences never read as
// a wrong answer.
func normalizeExecOutput(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	lines := strings.Split(s, "\n")
	for i := range lines {
		lines[i] = strings.TrimRight(lines[i], " \t")
	}
	return strings.TrimRight(strings.Join(lines, "\n"), "\n")
}

// SandboxCodegenGrader adapts the sandbox execution grader to the injected
// CodegenGrader seam ScoreCodegen consumes: a sample passes only on an AC
// verdict from a live sandbox. When run is nil the sandbox is unavailable, so
// the adapter returns an error rather than a fabricated miss — the run aborts
// instead of silently scoring every sample zero, keeping the honesty fence
// intact from the grader through the scorer.
func SandboxCodegenGrader(run ExecRunner, opts GradeOptions) CodegenGrader {
	return func(p Problem, code string) (bool, error) {
		res := GradeCode(context.Background(), p, code, run, opts)
		if !res.OfficialHarnessAvailable {
			return false, fmt.Errorf("livecodebench grade: sandbox unavailable (official_harness.available=false); cannot grade %q", p.QuestionID)
		}
		return res.Pass, nil
	}
}
