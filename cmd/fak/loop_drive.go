package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopdrive"
)

type loopDriveOptions struct {
	GoalPath        string
	MaxIters        int
	Command         []string
	ReviewModel     string
	ReviewEndpoint  string
	ReviewAPIKeyEnv string
	Ledger          string
}

var loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return execLoopCommand{cmd: cmd}
}

func runLoopDrive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop drive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	goalPath := fs.String("goal", "GOAL.md", "GOAL.md goal-spec path")
	maxIters := fs.Int("max-iters", 0, "override max iterations (default: budget.max_iters from the goal spec, or 1)")
	reviewModel := fs.String("review-model", "", "optional scout model id exported to fak commit for per-turn diff review")
	reviewEndpoint := fs.String("review-endpoint", envOrDefault("FAK_REVIEW_ENDPOINT", "http://127.0.0.1:8080/v1"), "OpenAI-compatible base URL exported with --review-model")
	reviewAPIKeyEnv := fs.String("review-api-key-env", envOrDefault("FAK_REVIEW_API_KEY_ENV", "FAK_REVIEW_API_KEY"), "env var name exported with --review-model")
	ledger := fs.String("ledger", defaultLoopLedger(), "loop ledger path used when --review-model records review evidence")
	template := fs.Bool("template", false, "print a parseable GOAL.md template and exit")
	templateLoop := fs.String("loop", "goal", "loop id to place in --template output")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *template {
		if fs.NArg() != 0 {
			fmt.Fprintf(stderr, "fak loop drive: --template does not accept a command\n")
			return 2
		}
		_, _ = stdout.Write(loopdrive.Template(*templateLoop))
		return 0
	}
	cmdArgs := fs.Args()
	if len(cmdArgs) == 0 {
		fmt.Fprintln(stderr, "fak loop drive: command is required after --")
		return 2
	}
	return driveGoalSpec(stdout, stderr, loopDriveOptions{
		GoalPath:        *goalPath,
		MaxIters:        *maxIters,
		Command:         cmdArgs,
		ReviewModel:     *reviewModel,
		ReviewEndpoint:  *reviewEndpoint,
		ReviewAPIKeyEnv: *reviewAPIKeyEnv,
		Ledger:          *ledger,
	})
}

func driveGoalSpec(stdout, stderr io.Writer, opt loopDriveOptions) int {
	goalPath := strings.TrimSpace(opt.GoalPath)
	if goalPath == "" {
		goalPath = "GOAL.md"
	}
	iterations := 0
	for {
		spec, err := loadLoopGoal(goalPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
			return 2
		}
		planIndex, item, ok := spec.NextUnchecked()
		if !ok {
			fmt.Fprintf(stdout, "loop drive done: no unchecked plan items goal=%s\n", goalPath)
			return 0
		}
		limit := loopDriveLimit(opt.MaxIters, spec.Budget.MaxIters)
		if iterations >= limit {
			reason := fmt.Sprintf("NOT_YET budget exhausted after %d iteration(s); next plan[%d]=%s", iterations, planIndex+1, item.Text)
			if err := appendLoopGoalScratch(goalPath, reason); err != nil {
				fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", err)
				return 1
			}
			fmt.Fprintf(stderr, "fak loop drive: %s\n", reason)
			return 3
		}

		iterations++
		fmt.Fprintf(stdout, "loop drive turn %d loop=%s plan=%d witness=%s\n", iterations, spec.Loop, planIndex+1, spec.Witness)
		exitCode, err := runLoopDriveTurn(opt, goalPath, spec, planIndex, item, iterations, limit, stdout, stderr)
		if err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
		}
		if exitCode != 0 {
			reason := fmt.Sprintf("NOT_YET turn=%d exit=%d plan[%d]=%s", iterations, exitCode, planIndex+1, item.Text)
			if err != nil {
				reason += " reason=" + err.Error()
			}
			if scratchErr := appendLoopGoalScratch(goalPath, reason); scratchErr != nil {
				fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", scratchErr)
				return 1
			}
			return exitCode
		}
	}
}

func loadLoopGoal(path string) (loopdrive.Spec, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return loopdrive.Spec{}, fmt.Errorf("read goal spec %s: %w", path, err)
	}
	spec, err := loopdrive.Parse(b)
	if err != nil {
		return loopdrive.Spec{}, fmt.Errorf("parse goal spec %s: %w", path, err)
	}
	return spec, nil
}

func runLoopDriveTurn(opt loopDriveOptions, goalPath string, spec loopdrive.Spec, planIndex int, item loopdrive.PlanItem, iter, limit int, stdout, stderr io.Writer) (int, error) {
	env := loopDriveEnv(os.Environ(), goalPath, spec, planIndex, item, iter, limit)
	env = loopDriveReviewEnv(env, opt, spec, iter)
	cmd := loopDriveNewCommand(opt.Command, env, stdout, stderr)
	if err := cmd.Start(); err != nil {
		return 127, fmt.Errorf("start command: %w", err)
	}
	if err := cmd.Wait(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return exitErr.ExitCode(), nil
		}
		return 1, fmt.Errorf("wait command: %w", err)
	}
	return 0, nil
}

func loopDriveLimit(override, specLimit int) int {
	if override > 0 {
		return override
	}
	if specLimit > 0 {
		return specLimit
	}
	return 1
}

func loopDriveEnv(base []string, goalPath string, spec loopdrive.Spec, planIndex int, item loopdrive.PlanItem, iter, limit int) []string {
	env := append([]string(nil), base...)
	add := func(k, v string) {
		env = append(env, k+"="+v)
	}
	add("FAK_GOAL_SPEC", goalPath)
	add("FAK_GOAL_LOOP", spec.Loop)
	add("FAK_GOAL_WITNESS", spec.Witness)
	add("FAK_GOAL_OBJECTIVE", spec.Objective)
	add("FAK_GOAL_PLAN_INDEX", strconv.Itoa(planIndex+1))
	add("FAK_GOAL_PLAN_TOTAL", strconv.Itoa(len(spec.Plan)))
	add("FAK_GOAL_NEXT", item.Text)
	add("FAK_GOAL_ITER", strconv.Itoa(iter))
	add("FAK_GOAL_MAX_ITERS", strconv.Itoa(limit))
	return env
}

func loopDriveReviewEnv(env []string, opt loopDriveOptions, spec loopdrive.Spec, iter int) []string {
	model := strings.TrimSpace(opt.ReviewModel)
	if model == "" {
		return env
	}
	add := func(k, v string) {
		if strings.TrimSpace(v) != "" {
			env = append(env, k+"="+v)
		}
	}
	add("FAK_REVIEW_MODEL", model)
	add("FAK_REVIEW_OBJECTIVE", spec.Objective)
	add("FAK_REVIEW_ENDPOINT", opt.ReviewEndpoint)
	add("FAK_REVIEW_API_KEY_ENV", opt.ReviewAPIKeyEnv)
	add("FAK_LOOP_LEDGER", opt.Ledger)
	add("FAK_GOAL_RUN", fmt.Sprintf("%s-turn-%d", spec.Loop, iter))
	return env
}

func appendLoopGoalScratch(path, line string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	text := string(b)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	if !loopGoalHasScratch(text) {
		text += "\n# Scratch / last-refusal\n"
	}
	text += "- " + strings.TrimSpace(line) + "\n"
	return os.WriteFile(path, []byte(text), 0o644)
}

func loopGoalHasScratch(text string) bool {
	for _, line := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		line = strings.ToLower(strings.TrimSpace(line))
		if strings.HasPrefix(line, "#") && strings.HasPrefix(strings.TrimSpace(strings.TrimLeft(line, "#")), "scratch") {
			return true
		}
	}
	return false
}
