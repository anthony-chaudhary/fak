package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchpost"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/loopgate"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

type loopDriveOptions struct {
	GoalPath        string
	LedgerPath      string
	PolicyPath      string
	LoopID          string
	Source          string
	Principal       string
	WitnessOverride string
	MaxIters        int
	MaxTokens       int64
	Deadline        time.Time
	Command         []string
	Clock           func() time.Time
	ReviewModel     string
	ReviewEndpoint  string
	ReviewAPIKeyEnv string
}

type loopDriveWitnessResult struct {
	Status       loopmgr.RunStatus
	Reason       string
	Summary      string
	EvidenceRefs []loopmgr.EvidenceRef
	ExitCode     int
}

var loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return execLoopCommand{cmd: cmd}
}

var loopDriveRunWitness = defaultLoopDriveRunWitness

func runLoopDrive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop drive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	policy := fs.String("policy", defaultLoopPolicy(), "loop admission policy JSON path")
	loopID := fs.String("loop", "", "loop id override (default: frontmatter loop; with --template, template loop id)")
	goalPath := fs.String("goal", "GOAL.md", "GOAL.md goal-spec path")
	maxIters := fs.Int("max-iters", 0, "override max iterations (default: budget.max_iters from the goal spec, or 1)")
	maxTokens := fs.Int64("max-tokens", 0, "token budget exposed to the child; child reports deltas through FAK_GOAL_TOKEN_USAGE_FILE")
	deadlineRaw := fs.String("deadline", "", "wall-clock budget as RFC3339 timestamp or duration from now, such as 10m")
	source := fs.String("source", "manual", "trigger source, such as cron|launchd|task-scheduler|manual")
	principal := fs.String("principal", "", "authenticated principal or producer id")
	witnessOverride := fs.String("witness", "", "override GOAL.md witness criterion")
	reviewModel := fs.String("review-model", "", "optional scout model id exported to fak commit for per-turn diff review")
	reviewEndpoint := fs.String("review-endpoint", envOrDefault("FAK_REVIEW_ENDPOINT", "http://127.0.0.1:8080/v1"), "OpenAI-compatible base URL exported with --review-model")
	reviewAPIKeyEnv := fs.String("review-api-key-env", envOrDefault("FAK_REVIEW_API_KEY_ENV", "FAK_REVIEW_API_KEY"), "env var name exported with --review-model")
	template := fs.Bool("template", false, "print a parseable GOAL.md template and exit")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *template {
		if fs.NArg() != 0 {
			fmt.Fprintf(stderr, "fak loop drive: --template does not accept a command\n")
			return 2
		}
		templateLoop := strings.TrimSpace(*loopID)
		if templateLoop == "" {
			templateLoop = "goal"
		}
		_, _ = stdout.Write(loopdrive.Template(templateLoop))
		return 0
	}
	if *maxIters < 0 {
		fmt.Fprintln(stderr, "fak loop drive: --max-iters must be non-negative")
		return 2
	}
	if *maxTokens < 0 {
		fmt.Fprintln(stderr, "fak loop drive: --max-tokens must be non-negative")
		return 2
	}
	deadline, err := parseLoopDriveDeadline(*deadlineRaw, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
		return 2
	}
	cmdArgs := fs.Args()
	if len(cmdArgs) == 0 {
		fmt.Fprintln(stderr, "fak loop drive: command is required after --")
		return 2
	}
	return driveGoalSpec(stdout, stderr, loopDriveOptions{
		GoalPath:        *goalPath,
		LedgerPath:      *ledger,
		PolicyPath:      *policy,
		LoopID:          *loopID,
		Source:          *source,
		Principal:       *principal,
		WitnessOverride: *witnessOverride,
		MaxIters:        *maxIters,
		MaxTokens:       *maxTokens,
		Deadline:        deadline,
		Command:         cmdArgs,
		ReviewModel:     *reviewModel,
		ReviewEndpoint:  *reviewEndpoint,
		ReviewAPIKeyEnv: *reviewAPIKeyEnv,
	})
}

func driveGoalSpec(stdout, stderr io.Writer, opt loopDriveOptions) int {
	clock := opt.Clock
	if clock == nil {
		clock = time.Now
	}
	goalPath := strings.TrimSpace(opt.GoalPath)
	if goalPath == "" {
		goalPath = "GOAL.md"
	}
	iterations := 0
	var tokensUsed int64
	for {
		spec, err := loadLoopGoal(goalPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
			return 2
		}
		if strings.TrimSpace(opt.LoopID) != "" {
			spec.Loop = strings.TrimSpace(opt.LoopID)
		}
		if strings.TrimSpace(opt.WitnessOverride) != "" {
			spec.Witness = strings.TrimSpace(opt.WitnessOverride)
		}
		limit := loopDriveLimit(opt.MaxIters, spec.Budget.MaxIters)
		tokenLimit := loopDriveTokenLimit(opt.MaxTokens, spec.Budget.MaxTokens)
		decision := loopdrive.Decide(loopdrive.PolicyInput{
			Iterations:       iterations,
			MaxIters:         limit,
			TokensUsed:       tokensUsed,
			MaxTokens:        tokenLimit,
			NowUnixNano:      clock().UTC().UnixNano(),
			DeadlineUnixNano: deadlineUnixNano(opt.Deadline),
		})
		if decision.Action == loopdrive.ActionStopBudget {
			return stopLoopDriveBudget(stderr, opt, goalPath, spec, decision, iterations, tokensUsed)
		}

		admit, err := loopDriveAdmit(opt.LedgerPath, opt.PolicyPath, spec.Loop, clock())
		if err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
			return 1
		}
		if !admit.Admit {
			if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
				LoopID:    spec.Loop,
				Kind:      loopmgr.EventAdmit,
				Source:    opt.Source,
				Principal: opt.Principal,
				Status:    loopmgr.StatusRefused,
				Reason:    admit.Reason,
				Summary:   admit.Summary,
				EvidenceRefs: []loopmgr.EvidenceRef{
					{Kind: "goal", Ref: goalPath},
					{Kind: "policy", Ref: opt.PolicyPath},
				},
				Metrics: map[string]int64{"iterations": int64(iterations), "tokens_used": tokensUsed},
			}); err != nil {
				fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
				return 1
			}
			fmt.Fprintf(stderr, "fak loop drive: refused by loop governor: %s %s\n", admit.Reason, admit.Summary)
			return 3
		}

		planIndex, item, unchecked := spec.NextWork()
		turn := iterations + 1
		runID := fmt.Sprintf("%s-turn-%d", defaultLoopRunID(spec.Loop), turn)
		headBefore := dispatchpost.HeadSHA(ctx(), "")
		baseEvidence := loopDriveEvidence(goalPath, spec.Witness, opt.Command, headBefore, "")
		baseMetrics := loopDriveMetrics(turn, limit, planIndex, unchecked, tokensUsed, tokenLimit)

		if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
			LoopID:       spec.Loop,
			RunID:        runID,
			Kind:         loopmgr.EventFire,
			Source:       opt.Source,
			Principal:    opt.Principal,
			Summary:      "loop drive turn requested",
			EvidenceRefs: baseEvidence,
			Metrics:      cloneLoopMetrics(baseMetrics),
		}); err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
			return 1
		}
		if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
			LoopID:       spec.Loop,
			RunID:        runID,
			Kind:         loopmgr.EventAdmit,
			Source:       opt.Source,
			Principal:    opt.Principal,
			Status:       loopmgr.StatusAdmitted,
			Reason:       admit.Reason,
			Summary:      admit.Summary,
			EvidenceRefs: baseEvidence,
			Metrics:      cloneLoopMetrics(baseMetrics),
		}); err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
			return 1
		}

		fmt.Fprintf(stdout, "loop drive turn %d loop=%s plan=%d witness=%s\n", turn, spec.Loop, planIndex+1, spec.Witness)
		exitCode, tokenDelta, timedOut, err := runLoopDriveTurn(opt, goalPath, spec, planIndex, item, turn, limit, tokensUsed, tokenLimit, stdout, stderr, func(pid int) error {
			startMetrics := loopDriveMetrics(turn, limit, planIndex, unchecked, tokensUsed, tokenLimit)
			startMetrics["pid"] = int64(pid)
			return appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
				LoopID:       spec.Loop,
				RunID:        runID,
				Kind:         loopmgr.EventStart,
				Source:       opt.Source,
				Principal:    opt.Principal,
				Status:       loopmgr.StatusRunning,
				Reason:       "STARTED",
				Summary:      "child process started",
				EvidenceRefs: baseEvidence,
				Metrics:      startMetrics,
			})
		})
		iterations++
		tokensUsed += tokenDelta
		headAfter := dispatchpost.HeadSHA(ctx(), "")
		endMetrics := loopDriveMetrics(turn, limit, planIndex, unchecked, tokensUsed, tokenLimit)
		endMetrics["exit_code"] = int64(exitCode)
		endMetrics["token_delta"] = tokenDelta
		status := loopmgr.StatusClaimedDone
		reason := "EXIT_0"
		summary := fmt.Sprintf("child exited with code %d", exitCode)
		if timedOut {
			status = loopmgr.StatusCanceled
			reason = loopdrive.ReasonBudgetSpent
			summary = "deadline spent while child was running"
		} else if exitCode != 0 {
			status = loopmgr.StatusFailed
			reason = "EXIT_NONZERO"
		}
		if err != nil && !timedOut {
			summary = err.Error()
		}
		if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
			LoopID:       spec.Loop,
			RunID:        runID,
			Kind:         loopmgr.EventEnd,
			Source:       opt.Source,
			Principal:    opt.Principal,
			Status:       status,
			Reason:       reason,
			Summary:      summary,
			EvidenceRefs: loopDriveEvidence(goalPath, spec.Witness, opt.Command, headBefore, headAfter),
			Metrics:      endMetrics,
		}); err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
			if exitCode == 0 {
				return 1
			}
		}
		if timedOut {
			scratch := fmt.Sprintf("NOT_YET %s turn=%d deadline spent", loopdrive.ReasonBudgetSpent, turn)
			if scratchErr := appendGoalScratch(goalPath, scratch); scratchErr != nil {
				fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", scratchErr)
				return 1
			}
			fmt.Fprintf(stderr, "fak loop drive: %s\n", scratch)
			return 3
		}
		if exitCode != 0 {
			scratch := fmt.Sprintf("NOT_YET turn=%d exit=%d plan[%d]=%s", turn, exitCode, planIndex+1, item.Text)
			if err != nil {
				scratch += " reason=" + err.Error()
			}
			if scratchErr := appendGoalScratch(goalPath, scratch); scratchErr != nil {
				fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", scratchErr)
				return 1
			}
			return exitCode
		}

		witness := loopDriveRunWitness(spec, headBefore, headAfter)
		witnessMetrics := loopDriveMetrics(turn, limit, planIndex, unchecked, tokensUsed, tokenLimit)
		witnessMetrics["witness_exit_code"] = int64(witness.ExitCode)
		if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
			LoopID:       spec.Loop,
			RunID:        runID,
			Kind:         loopmgr.EventWitness,
			Source:       opt.Source,
			Principal:    opt.Principal,
			Status:       witness.Status,
			Reason:       witness.Reason,
			Summary:      witness.Summary,
			EvidenceRefs: append(loopDriveEvidence(goalPath, spec.Witness, opt.Command, headBefore, headAfter), witness.EvidenceRefs...),
			Metrics:      witnessMetrics,
		}); err != nil {
			fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
			return 1
		}
		if witness.Status == loopmgr.StatusWitnessedDone {
			fmt.Fprintf(stdout, "loop drive witnessed done: loop=%s turns=%d ledger=%s\n", spec.Loop, iterations, opt.LedgerPath)
			return 0
		}
		scratch := fmt.Sprintf("NOT_YET turn=%d witness=%s reason=%s %s", turn, witness.Status, witness.Reason, witness.Summary)
		if scratchErr := appendGoalScratch(goalPath, scratch); scratchErr != nil {
			fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", scratchErr)
			return 1
		}
		if witness.Status == loopmgr.StatusWitnessUnavailable {
			fmt.Fprintf(stderr, "fak loop drive: exit gate refused: %s %s\n", witness.Reason, witness.Summary)
			return 3
		}
	}
}

func stopLoopDriveBudget(stderr io.Writer, opt loopDriveOptions, goalPath string, spec loopdrive.Spec, decision loopdrive.Decision, iterations int, tokensUsed int64) int {
	reason := fmt.Sprintf("NOT_YET %s after %d iteration(s): %s", decision.Reason, iterations, decision.Summary)
	if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
		LoopID:    spec.Loop,
		Kind:      loopmgr.EventAdmit,
		Source:    opt.Source,
		Principal: opt.Principal,
		Status:    loopmgr.StatusRefused,
		Reason:    decision.Reason,
		Summary:   decision.Summary,
		EvidenceRefs: []loopmgr.EvidenceRef{
			{Kind: "goal", Ref: goalPath},
			{Kind: "witness", Ref: spec.Witness},
		},
		Metrics: map[string]int64{"iterations": int64(iterations), "tokens_used": tokensUsed},
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
		return 1
	}
	if err := appendGoalScratch(goalPath, reason); err != nil {
		fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "fak loop drive: %s\n", reason)
	return 3
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

func runLoopDriveTurn(opt loopDriveOptions, goalPath string, spec loopdrive.Spec, planIndex int, item loopdrive.PlanItem, iter, limit int, tokensUsed, tokenLimit int64, stdout, stderr io.Writer, onStart func(pid int) error) (int, int64, bool, error) {
	tokenFile, cleanup, err := loopDriveTokenFile(tokenLimit)
	if err != nil {
		return 1, 0, false, err
	}
	defer cleanup()
	env := loopDriveEnv(os.Environ(), goalPath, spec, planIndex, item, iter, limit, tokensUsed, tokenLimit, opt.Deadline, tokenFile)
	env = loopDriveReviewEnv(env, opt, spec, iter)
	cmd := loopDriveNewCommand(opt.Command, env, stdout, stderr)
	if err := cmd.Start(); err != nil {
		return 127, 0, false, fmt.Errorf("start command: %w", err)
	}
	if onStart != nil {
		if err := onStart(cmd.PID()); err != nil {
			_ = cmd.Kill()
			return 1, 0, false, err
		}
	}
	exitCode, timedOut, waitErr := waitLoopDriveCommand(cmd, opt.Deadline)
	tokenDelta, tokenErr := readLoopDriveTokenUsage(tokenFile)
	if tokenErr != nil && waitErr == nil {
		waitErr = tokenErr
	}
	return exitCode, tokenDelta, timedOut, waitErr
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

func loopDriveTokenLimit(override, specLimit int64) int64 {
	if override > 0 {
		return override
	}
	return specLimit
}

func loopDriveEnv(base []string, goalPath string, spec loopdrive.Spec, planIndex int, item loopdrive.PlanItem, iter, limit int, tokensUsed, tokenLimit int64, deadline time.Time, tokenFile string) []string {
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
	add("FAK_GOAL_TOKENS_USED", strconv.FormatInt(tokensUsed, 10))
	add("FAK_GOAL_MAX_TOKENS", strconv.FormatInt(tokenLimit, 10))
	add("FAK_GOAL_TOKEN_USAGE_FILE", tokenFile)
	add("FAK_GOAL_DEADLINE", formatLoopDriveDeadline(deadline))
	add("FAK_GOAL_LAST_REFUSAL", lastLoopGoalScratchLine(spec.Scratch))
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
	add("FAK_LOOP_LEDGER", opt.LedgerPath)
	add("FAK_GOAL_RUN", fmt.Sprintf("%s-turn-%d", spec.Loop, iter))
	return env
}

func loopDriveAdmit(ledger, policyPath, loopID string, now time.Time) (loopmgr.Decision, error) {
	policies, err := loopmgr.LoadPolicies(policyPath)
	if err != nil {
		return loopmgr.Decision{}, err
	}
	st, err := loopmgr.SnapshotFile(ledger, now)
	if err != nil {
		return loopmgr.Decision{}, err
	}
	return loopmgr.Admit(loopSnapshotForID(st, loopID), policies.PolicyFor(loopID), now), nil
}

func loopDriveEvidence(goalPath, witness string, cmdArgs []string, headBefore, headAfter string) []loopmgr.EvidenceRef {
	ev := []loopmgr.EvidenceRef{
		{Kind: "goal", Ref: goalPath},
		{Kind: "witness", Ref: witness},
	}
	if len(cmdArgs) > 0 {
		ev = append(ev, loopmgr.EvidenceRef{Kind: "command", Ref: filepath.Base(cmdArgs[0])})
	}
	if headBefore != "" {
		ev = append(ev, loopmgr.EvidenceRef{Kind: "head_before", Ref: headBefore})
	}
	if headAfter != "" {
		ev = append(ev, loopmgr.EvidenceRef{Kind: "head_after", Ref: headAfter})
	}
	return ev
}

func loopDriveMetrics(iter, limit, planIndex int, unchecked bool, tokensUsed, tokenLimit int64) map[string]int64 {
	m := map[string]int64{
		"iter":         int64(iter),
		"max_iters":    int64(limit),
		"plan_index":   int64(planIndex + 1),
		"tokens_used":  tokensUsed,
		"max_tokens":   tokenLimit,
		"plan_checked": 1,
	}
	if unchecked {
		m["plan_checked"] = 0
	}
	return m
}

func waitLoopDriveCommand(cmd loopCommand, deadline time.Time) (int, bool, error) {
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	if deadline.IsZero() {
		return loopDriveWaitResult(<-done)
	}
	remaining := time.Until(deadline)
	if remaining <= 0 {
		_ = cmd.Kill()
		return 3, true, fmt.Errorf("deadline spent")
	}
	timer := time.NewTimer(remaining)
	defer timer.Stop()
	select {
	case err := <-done:
		return loopDriveWaitResult(err)
	case <-timer.C:
		_ = cmd.Kill()
		return 3, true, fmt.Errorf("deadline spent")
	}
}

func loopDriveWaitResult(err error) (int, bool, error) {
	if err == nil {
		return 0, false, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), false, nil
	}
	return 1, false, fmt.Errorf("wait command: %w", err)
}

func loopDriveTokenFile(tokenLimit int64) (string, func(), error) {
	if tokenLimit <= 0 {
		return "", func() {}, nil
	}
	f, err := os.CreateTemp("", "fak-loop-drive-tokens-*")
	if err != nil {
		return "", func() {}, fmt.Errorf("create token usage file: %w", err)
	}
	path := f.Name()
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return "", func() {}, fmt.Errorf("close token usage file: %w", err)
	}
	return path, func() { _ = os.Remove(path) }, nil
}

func readLoopDriveTokenUsage(path string) (int64, error) {
	if path == "" {
		return 0, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read token usage file: %w", err)
	}
	raw := strings.TrimSpace(string(b))
	if raw == "" {
		return 0, nil
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("token usage file must contain a non-negative integer")
	}
	return n, nil
}

func defaultLoopDriveRunWitness(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
	criterion, err := loopDriveGateCriterion(spec.Witness)
	if err != nil {
		return loopDriveWitnessResult{
			Status:   loopmgr.StatusWitnessUnavailable,
			Reason:   loopgate.ReasonSchemaUnreadable,
			Summary:  err.Error(),
			ExitCode: 2,
		}
	}
	headRef := "HEAD"
	if criterion.Kind == "" || criterion.Kind == loopgate.CriterionCommitAudit {
		if strings.TrimSpace(criterion.Ref) != "" {
			headRef = strings.TrimSpace(criterion.Ref)
		} else {
			if headBefore == "" || headAfter == "" {
				return loopDriveWitnessResult{
					Status:  loopmgr.StatusWitnessUnavailable,
					Reason:  "GIT_HEAD_UNAVAILABLE",
					Summary: "cannot run dos commit-audit without git HEAD evidence",
				}
			}
			if headBefore == headAfter {
				return loopDriveWitnessResult{
					Status:  loopmgr.StatusWitnessRefused,
					Reason:  loopgate.ReasonDoneUnwitnessed,
					Summary: "exit gate refused: turn landed no new commit",
					EvidenceRefs: []loopmgr.EvidenceRef{
						{Kind: "head_before", Ref: headBefore},
						{Kind: "head_after", Ref: headAfter},
					},
					ExitCode: 1,
				}
			}
			headRef = headBefore + ".." + headAfter
		}
	}
	decision := loopgate.Adjudicate(ctx(), loopgate.Turn{
		ClaimedDone: true,
		Claim:       spec.Objective,
		HeadRef:     headRef,
		Criterion:   criterion,
	}, runDOSLoopGateWitness)
	return loopDriveWitnessFromGate(decision)
}

func loopDriveGateCriterion(raw string) (loopgate.Criterion, error) {
	fields := strings.Fields(strings.TrimSpace(raw))
	if len(fields) == 0 {
		return loopgate.Criterion{}, fmt.Errorf("goal witness criterion is empty")
	}
	switch fields[0] {
	case "commit-audit":
		if len(fields) > 2 {
			return loopgate.Criterion{}, fmt.Errorf("commit-audit witness must be: commit-audit [REF]")
		}
		c := loopgate.Criterion{Kind: loopgate.CriterionCommitAudit}
		if len(fields) == 2 {
			c.Ref = fields[1]
		}
		return c, nil
	case "verify":
		if len(fields) != 3 {
			return loopgate.Criterion{}, fmt.Errorf("verify witness must be: verify PLAN PHASE")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionVerify, Plan: fields[1], Phase: fields[2]}, nil
	case "test-witness":
		if len(fields) != 3 {
			return loopgate.Criterion{}, fmt.Errorf("test-witness criterion requires baseline and candidate outcomes")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionTestWitness, Baseline: fields[1], Candidate: fields[2]}, nil
	case "citation-resolve":
		if len(fields) < 2 {
			return loopgate.Criterion{}, fmt.Errorf("citation-resolve criterion requires a subject citation")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionCitationResolve, Subject: strings.Join(fields[1:], " ")}, nil
	case "witness":
		if len(fields) < 3 {
			return loopgate.Criterion{}, fmt.Errorf("witness criterion requires source and subject")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionWitness, Source: fields[1], Subject: strings.Join(fields[2:], " ")}, nil
	case "metric":
		if len(fields) < 2 {
			return loopgate.Criterion{}, fmt.Errorf("metric criterion requires a subject")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionMetric, Subject: strings.Join(fields[1:], " ")}, nil
	case "none":
		return loopgate.Criterion{Kind: loopgate.CriterionMetric, Subject: "none"}, nil
	case "dos":
		return loopDriveDOSGateCriterion(fields[1:])
	default:
		return loopgate.Criterion{}, fmt.Errorf("unsupported witness criterion: %s", raw)
	}
}

func loopDriveDOSGateCriterion(fields []string) (loopgate.Criterion, error) {
	if len(fields) == 0 {
		return loopgate.Criterion{}, fmt.Errorf("dos witness must include a dos subcommand")
	}
	switch fields[0] {
	case "commit-audit":
		if len(fields) > 2 {
			return loopgate.Criterion{}, fmt.Errorf("dos commit-audit witness must be: dos commit-audit [REF]")
		}
		c := loopgate.Criterion{Kind: loopgate.CriterionCommitAudit}
		if len(fields) == 2 {
			c.Ref = fields[1]
		}
		return c, nil
	case "verify":
		if len(fields) != 3 {
			return loopgate.Criterion{}, fmt.Errorf("dos verify witness must be: dos verify PLAN PHASE")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionVerify, Plan: fields[1], Phase: fields[2]}, nil
	case "test-witness":
		if len(fields) != 3 {
			return loopgate.Criterion{}, fmt.Errorf("dos test-witness criterion requires baseline and candidate outcomes")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionTestWitness, Baseline: fields[1], Candidate: fields[2]}, nil
	case "citation-resolve":
		if len(fields) < 2 {
			return loopgate.Criterion{}, fmt.Errorf("dos citation-resolve criterion requires a subject citation")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionCitationResolve, Subject: strings.Join(fields[1:], " ")}, nil
	case "witness":
		if len(fields) < 3 {
			return loopgate.Criterion{}, fmt.Errorf("dos witness criterion requires source and subject")
		}
		return loopgate.Criterion{Kind: loopgate.CriterionWitness, Source: fields[1], Subject: strings.Join(fields[2:], " ")}, nil
	default:
		return loopgate.Criterion{}, fmt.Errorf("unsupported dos witness subcommand: %s", strings.Join(fields, " "))
	}
}

func runDOSLoopGateWitness(ctx context.Context, req loopgate.Request) (loopgate.WitnessResult, error) {
	cmd := exec.CommandContext(ctx, "dos", req.Argv()...)
	var out, errOut bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errOut
	runErr := cmd.Run()
	result, parseErr := parseDOSLoopGateWitness(req, out.Bytes())
	if parseErr == nil {
		return result, nil
	}
	if runErr != nil {
		return loopgate.WitnessResult{}, fmt.Errorf("%s exited non-zero: %s", strings.Join(req.Argv(), " "), trimLoopDriveSummary(errOut.String()))
	}
	return loopgate.WitnessResult{}, parseErr
}

func parseDOSLoopGateWitness(req loopgate.Request, data []byte) (loopgate.WitnessResult, error) {
	switch req.Kind {
	case loopgate.CriterionVerify:
		return loopgate.VerifyResultFromJSON(data)
	case loopgate.CriterionTestWitness:
		return loopgate.TestWitnessResultFromJSON(data)
	case loopgate.CriterionCitationResolve, loopgate.CriterionWitness:
		return loopgate.GenericWitnessResultFromJSON(data)
	default:
		return loopgate.CommitAuditResultFromJSON(data)
	}
}

func loopDriveWitnessFromGate(decision loopgate.Decision) loopDriveWitnessResult {
	status := loopmgr.StatusWitnessUnavailable
	exitCode := 2
	switch decision.Verdict {
	case loopgate.VerdictWitnessed:
		status = loopmgr.StatusWitnessedDone
		exitCode = 0
	case loopgate.VerdictNotYet:
		status = loopmgr.StatusWitnessRefused
		exitCode = 1
	case loopgate.VerdictRefused:
		status = loopmgr.StatusWitnessUnavailable
		exitCode = 2
	}
	reason := strings.TrimSpace(decision.Reason)
	if reason == "" {
		reason = string(decision.Verdict)
	}
	summary := strings.TrimSpace(decision.Summary)
	if summary == "" {
		summary = "loop exit gate " + strings.ToLower(string(decision.Verdict))
	}
	return loopDriveWitnessResult{
		Status:       status,
		Reason:       reason,
		Summary:      summary,
		EvidenceRefs: loopDriveGateEvidence(decision),
		ExitCode:     exitCode,
	}
}

func loopDriveGateEvidence(decision loopgate.Decision) []loopmgr.EvidenceRef {
	var refs []loopmgr.EvidenceRef
	if decision.Request.Kind != "" {
		refs = append(refs, loopmgr.EvidenceRef{
			Kind:    "loopgate",
			Ref:     "dos " + strings.Join(decision.Request.Argv(), " "),
			Summary: decision.Summary,
		})
	}
	if strings.TrimSpace(decision.Witness) != "" {
		refs = append(refs, loopmgr.EvidenceRef{Kind: "witness_rung", Ref: decision.Witness})
	}
	return refs
}

func trimLoopDriveSummary(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 300 {
		return s[:297] + "..."
	}
	return s
}

func parseLoopDriveDeadline(raw string, now time.Time) (time.Time, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, nil
	}
	if d, err := time.ParseDuration(raw); err == nil {
		if d < 0 {
			return time.Time{}, fmt.Errorf("--deadline duration must be non-negative")
		}
		return now.Add(d), nil
	}
	if ts, err := time.Parse(time.RFC3339, raw); err == nil {
		return ts, nil
	}
	return time.Time{}, fmt.Errorf("--deadline must be an RFC3339 timestamp or duration")
}

func deadlineUnixNano(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UTC().UnixNano()
}

func formatLoopDriveDeadline(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339)
}

func lastLoopGoalScratchLine(s string) string {
	lines := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line != "" {
			return strings.TrimPrefix(line, "- ")
		}
	}
	return ""
}

// appendGoalScratch / goalHasScratch are shared with the commit gate (see commit.go).
