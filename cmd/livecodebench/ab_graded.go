package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

// runABGraded implements `livecodebench ab-graded`: the GRADED companion to
// `livecodebench ab`. Where `ab` compares two arms on tokens/identity and
// deliberately carries NO pass-rate delta, this verb loads the normalized suite
// plus both arm reports, resolves a code-execution grader, and folds the arms
// into a raw-vs-fak pass@1/pass@5 delta (livecodebench.GradedArmDelta).
//
// The delta is a LOCAL-ungraded signal only: the artifact's evidence class is
// always local-ungraded and result_claim_allowed is always false. It is fenced
// twice inside the seam — a fairness fence (both arms provably ran the same
// problems) and a grader-availability fence (no sandbox ⇒ an honest
// GATED_UNGRADED abstain, never a fabricated zero). A Docker-isolated execution
// runner (execrunner.go) is wired in and gated on a preflight, so on a host with
// a working sandbox the seam grades a LOCAL pass-rate delta; on any other host
// it abstains and stderr prints the path to a claimable number.
func runABGraded(argv []string) int {
	fs := flag.NewFlagSet("livecodebench ab-graded", flag.ContinueOnError)
	suitePath := fs.String("suite", "", "normalized suite JSON defining the graded problems (required; the SAME suite both arms ran)")
	rawPath := fs.String("raw", "", "raw-arm report JSON (from `livecodebench raw --out`) (required)")
	fakPath := fs.String("fak", "", "fak-arm report JSON (from `livecodebench fak --out`) (required)")
	sandboxCmd := fs.String("sandbox-cmd", "docker", "executable on PATH treated as the code-execution sandbox")
	sandboxImage := fs.String("sandbox-image", "python:3.11-slim", "container image each candidate runs in (must be pre-pulled; grading runs it with --network none)")
	gradeWorkers := fs.Int("grade-concurrency", 4, "max test cases graded concurrently per problem (upstream --num_process_evaluate)")
	out := fs.String("out", "", "write the comparison JSON to this path (default: stdout)")
	mdPath := fs.String("md", "", "also write the comparison markdown to this path")
	check := fs.Bool("check", false, "exit nonzero unless the fairness fence (both arms ran the same problems) is witnessed")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench ab-graded: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*suitePath) == "" || strings.TrimSpace(*rawPath) == "" || strings.TrimSpace(*fakPath) == "" {
		fmt.Fprintln(os.Stderr, "livecodebench ab-graded: --suite, --raw, and --fak are all required")
		return 2
	}

	suite, err := livecodebench.LoadSuiteFile(*suitePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: --suite: %v\n", err)
		return 1
	}
	var raw livecodebench.RawArmReport
	if err := readJSON(*rawPath, &raw); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: --raw: %v\n", err)
		return 1
	}
	var fak livecodebench.FakArmReport
	if err := readJSON(*fakPath, &fak); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: --fak: %v\n", err)
		return 1
	}

	grade, graderNote := resolveCodegenGrader(*sandboxCmd, *sandboxImage, *gradeWorkers)
	fmt.Fprintf(os.Stderr, "livecodebench ab-graded: %s\n", graderNote)

	c, err := livecodebench.GradedArmDelta(suite, raw, fak, grade)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: %v\n", err)
		return 1
	}
	// Belt-and-suspenders: the seam already builds a valid artifact, but never
	// emit one that fails its own honesty invariant.
	if err := c.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: %v\n", err)
		return 1
	}

	if p := strings.TrimSpace(*mdPath); p != "" {
		if err := os.WriteFile(p, []byte(livecodebench.RenderGradedABMarkdown(c)), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench ab-graded: %v\n", err)
			return 1
		}
	}
	if err := writeJSON(*out, c); err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: %v\n", err)
		return 1
	}

	fmt.Fprintf(os.Stderr, "livecodebench ab-graded: verdict=%s fairness_witnessed=%t grader_available=%t delta=%t (evidence %s, result_claim_allowed=%t)\n",
		c.Verdict, c.FairnessWitnessed, c.GraderAvailable, c.Delta, c.EvidenceClass, c.ResultClaimAllowed)
	if !c.FairnessWitnessed {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: fairness fence unwitnessed: %s\n", c.FairnessReason)
	}
	if c.Delta {
		fmt.Fprintf(os.Stderr, "livecodebench ab-graded: LOCAL pass@1 delta (fak−raw) = %+.4f, pass@5 delta = %+.4f — a local sandbox signal, NOT a claimable result\n", c.Pass1Delta, c.Pass5Delta)
	}
	if *check && !c.FairnessWitnessed {
		fmt.Fprintln(os.Stderr, "livecodebench ab-graded: --check failed: the arms did not run the identical problems")
		return 1
	}
	return 0
}

// resolveCodegenGrader resolves the code-execution grader that grades both arms'
// saved generations. It probes for the sandbox, then gates on a preflight that
// proves the sandbox can actually execute a candidate: only then does it return
// a live SandboxCodegenGrader. When the sandbox is absent, or present but unable
// to run a smoke program (daemon down, image not pulled), it returns a nil
// grader so the seam emits the honest GATED_UNGRADED abstain rather than a
// fabricated delta. Either way the returned note names the path to a CLAIMABLE
// number, which this local delta can never be.
func resolveCodegenGrader(sandboxCmd, image string, workers int) (livecodebench.CodegenGrader, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()
	claimHandoff := "for a claimable number, `livecodebench export --format custom-evaluator` then grade with the official lcb_runner (this local delta is never claimable)"

	present, detail := probeExecVersion(ctx, sandboxCmd, "--version")
	if !present {
		return nil, fmt.Sprintf("no code-execution sandbox %q on PATH — grading abstains (GATED_UNGRADED); %s", sandboxCmd, claimHandoff)
	}

	runner := dockerExecRunner(sandboxCmd, image)
	if err := sandboxPreflight(runner); err != nil {
		return nil, fmt.Sprintf("sandbox %q (%s) present but could not execute a smoke program (%v) — grading abstains (GATED_UNGRADED); ensure `%s pull %s` has run; %s",
			sandboxCmd, detail, err, sandboxCmd, image, claimHandoff)
	}

	grader := livecodebench.SandboxCodegenGrader(runner, livecodebench.GradeOptions{
		Timeout:            livecodebench.DefaultGradeTimeout,
		NumProcessEvaluate: workers,
	})
	return grader, fmt.Sprintf("sandbox %q (%s) live with image %q — grading a LOCAL, public-tests-only pass-rate signal (evidence local-ungraded, never claimable); %s",
		sandboxCmd, detail, image, claimHandoff)
}
