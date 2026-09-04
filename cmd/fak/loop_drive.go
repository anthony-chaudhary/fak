package main

import (
	"bytes"
	"context"
	"encoding/json"
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
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// loopDriveHandoffFileEnv is the model-facing path a wrapped agent writes a
// fak.task-handoff.v1 record to before a witnessed-done loop-drive completion.
// It matches the alias the guard Stop hook exposes so an agent that already
// knows how to hand off under guard needs no new convention here.
const loopDriveHandoffFileEnv = "FAK_TASK_HANDOFF_FILE"

type loopDriveOptions struct {
	GoalPath          string
	LedgerPath        string
	PolicyPath        string
	LoopID            string
	Source            string
	Principal         string
	WitnessOverride   string
	MaxIters          int
	MaxTokens         int64
	Deadline          time.Time
	Command           []string
	Clock             func() time.Time
	Sleep             func(time.Duration)
	MinIterationFloor time.Duration
	ReviewModel       string
	ReviewEndpoint    string
	ReviewAPIKeyEnv   string
	// HandoffFile is the path the wrapped agent writes a fak.task-handoff.v1
	// record to before a witnessed-done completion. Empty means the loop driver
	// allocates a private per-session file and exposes it to the child.
	HandoffFile string
	// Lane / Region override the GOAL.md lane:/region: frontmatter and arm the
	// drive's region admission (see loop_drive_region.go). Both empty and no
	// spec declaration = the historical uncoordinated drive.
	Lane   string
	Region []string
	// ArmedFile is a filesystem sentinel checked at the top of every turn: while
	// it exists the loop keeps driving; once it is removed out-of-band (an
	// operator `rm`) the loop takes a clean cancel between turns (issue #4059).
	// Empty means no sentinel is armed and the loop never disarms on this axis.
	ArmedFile string
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
	windowgate.ConfigureBackgroundCommand(cmd)
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
	handoffFile := fs.String("task-handoff-file", "", "path the wrapped agent writes a fak.task-handoff.v1 record to before a witnessed-done completion; required for the handoff gate (default: a private per-session file exposed as FAK_TASK_HANDOFF_FILE). A missing/empty file is an ordinary non-agent stop and fails open.")
	lane := fs.String("lane", "", "dos.toml lane this loop's writes stay inside (overrides the GOAL.md lane: field); arms region admission against the live lease fabric")
	armedFile := fs.String("armed-file", "", "filesystem sentinel: while this path exists the loop keeps driving; remove it out-of-band to take a clean cancel between turns (empty = never disarm on this axis)")
	var region repeatedString
	fs.Var(&region, "tree", "region glob this loop's writes stay inside (repeatable; overrides the GOAL.md region: field); arms region admission against the live lease fabric")
	template := fs.Bool("template", false, "print a parseable GOAL.md template and exit")
	if !parseFlags(fs, argv) {
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
		HandoffFile:     *handoffFile,
		Lane:            *lane,
		Region:          region,
		ArmedFile:       *armedFile,
	})
}

func driveGoalSpec(stdout, stderr io.Writer, opt loopDriveOptions) int {
	runtime, err := prepareLoopDriveRuntime(&opt)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
		return 1
	}
	defer runtime.cleanup()
	clock, sleep, floor, goalPath := runtime.clock, runtime.sleep, runtime.floor, runtime.goalPath
	iterations := 0
	var tokensUsed int64
	var regionHold *loopDriveRegionHold
	defer func() { regionHold.release() }()
	for {
		iterationStarted := clock()
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
		// Filesystem-sentinel disarm (#4059): an armed sentinel removed
		// out-of-band during the previous turn stops the loop cleanly here,
		// before this turn spawns anything.
		if loopDriveDisarmed(opt.ArmedFile) {
			return stopLoopDriveDisarmed(stderr, opt, goalPath, spec, iterations, tokensUsed)
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
			return refuseLoopDriveAdmit(stderr, opt, goalPath, spec, admit, iterations, tokensUsed)
		}

		// Region admission (loop_drive_region.go): decide against the live
		// lease fabric and hold/renew the region lease before this turn runs.
		// A witnessed collision refuses (exit 3); an infra error fails open
		// with a warning, the dispatch tick's posture.
		if regionHold == nil {
			// announceOn is nil-safe: an undeclared region yields no hold at all,
			// and a hold that has one announces its nonfatal lease-ref sync
			// degradations on this drive's stderr (#5571).
			regionHold = newLoopDriveRegionHold(opt, spec).announceOn(stderr)
		}
		if refuse, err := regionHold.ensure(clock()); err != nil {
			fmt.Fprintf(stderr, "fak loop drive: region admission unavailable (fail-open): %v\n", err)
		} else if refuse != nil {
			return refuseLoopDriveRegion(stderr, opt, goalPath, spec, regionHold, refuse, iterations, tokensUsed)
		}

		planIndex, item, unchecked := spec.NextWork()
		turn := iterations + 1
		runID := fmt.Sprintf("%s-turn-%d", defaultLoopRunID(spec.Loop), turn)
		headBefore := dispatchpost.HeadSHA(ctx(), "")
		baseEvidence := append(loopDriveEvidence(goalPath, spec.Witness, opt.Command, headBefore, ""), regionHold.evidence()...)
		baseMetrics := loopDriveMetrics(turn, limit, planIndex, unchecked, tokensUsed, tokenLimit)

		if err := emitLoopDriveFireAndAdmit(opt, spec, admit, runID, baseEvidence, baseMetrics); err != nil {
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
		status, reason, summary := loopDriveEndStatus(exitCode, timedOut, err)
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
		if code, done := handleLoopDriveNonSuccess(stderr, goalPath, turn, exitCode, planIndex, item, timedOut, err); done {
			return code
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
			gate := loopDriveReviewHandoff(opt.HandoffFile)
			if !gate.OK() {
				handoffMetrics := loopDriveMetrics(turn, limit, planIndex, unchecked, tokensUsed, tokenLimit)
				if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
					LoopID:       spec.Loop,
					RunID:        runID,
					Kind:         loopmgr.EventAdmit,
					Source:       opt.Source,
					Principal:    opt.Principal,
					Status:       loopmgr.StatusRefused,
					Reason:       gate.Reason,
					Summary:      gate.Summary,
					EvidenceRefs: []loopmgr.EvidenceRef{{Kind: "task-handoff", Ref: opt.HandoffFile}},
					Metrics:      handoffMetrics,
				}); err != nil {
					fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
					return 1
				}
				scratch := fmt.Sprintf("NOT_YET turn=%d witnessed done but %s", turn, gate.Summary)
				if scratchErr := appendGoalScratch(goalPath, scratch); scratchErr != nil {
					fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", scratchErr)
					return 1
				}
				fmt.Fprintf(stderr, "fak loop drive: %s\n", scratch)
				return 3
			}
			fmt.Fprintf(stdout, "loop drive witnessed done: loop=%s turns=%d ledger=%s handoff=%s\n", spec.Loop, iterations, opt.LedgerPath, gate.Reason)
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
		if remain := floor - clock().Sub(iterationStarted); remain > 0 {
			sleep(remain)
		}
	}
}

// refuseLoopDriveAdmit records the governor refusal for a turn the loop policy
// declined and reports the exit code to return (1 if the ledger append fails,
// otherwise 3 for the refusal).
func refuseLoopDriveAdmit(stderr io.Writer, opt loopDriveOptions, goalPath string, spec loopdrive.Spec, admit loopmgr.Decision, iterations int, tokensUsed int64) int {
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

// emitLoopDriveFireAndAdmit appends the per-turn Fire and Admit ledger events.
// It returns the first append error, letting the caller report it once.
func emitLoopDriveFireAndAdmit(opt loopDriveOptions, spec loopdrive.Spec, admit loopmgr.Decision, runID string, baseEvidence []loopmgr.EvidenceRef, baseMetrics map[string]int64) error {
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
		return err
	}
	return appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
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
	})
}

// loopDriveEndStatus maps a turn's exit code / timeout / wait error onto the
// ledger End event's status, reason, and summary.
func loopDriveEndStatus(exitCode int, timedOut bool, err error) (loopmgr.RunStatus, string, string) {
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
	return status, reason, summary
}

// handleLoopDriveNonSuccess records the NOT_YET scratch line for a turn that
// timed out or exited non-zero and reports the exit code to return. done is
// false only when the turn neither timed out nor failed, leaving the caller to
// proceed to the witness gate.
func handleLoopDriveNonSuccess(stderr io.Writer, goalPath string, turn, exitCode, planIndex int, item loopdrive.PlanItem, timedOut bool, err error) (int, bool) {
	if timedOut {
		scratch := fmt.Sprintf("NOT_YET %s turn=%d deadline spent", loopdrive.ReasonBudgetSpent, turn)
		if scratchErr := appendGoalScratch(goalPath, scratch); scratchErr != nil {
			fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", scratchErr)
			return 1, true
		}
		fmt.Fprintf(stderr, "fak loop drive: %s\n", scratch)
		return 3, true
	}
	if exitCode != 0 {
		scratch := fmt.Sprintf("NOT_YET turn=%d exit=%d plan[%d]=%s", turn, exitCode, planIndex+1, item.Text)
		if err != nil {
			scratch += " reason=" + err.Error()
		}
		if scratchErr := appendGoalScratch(goalPath, scratch); scratchErr != nil {
			fmt.Fprintf(stderr, "fak loop drive: append scratch: %v\n", scratchErr)
			return 1, true
		}
		return exitCode, true
	}
	return 0, false
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

// loopDriveDisarmed reports whether an armed sentinel file was configured and has
// since been removed out-of-band — a clean between-turns stop signal (#4059). An
// empty ArmedFile means no sentinel is armed. Only a confirmed absence
// (os.ErrNotExist) disarms; any other stat error (e.g. a transient permission
// blip) keeps the loop running, so the sentinel fails safe toward continuing.
func loopDriveDisarmed(armedFile string) bool {
	armedFile = strings.TrimSpace(armedFile)
	if armedFile == "" {
		return false
	}
	_, err := os.Stat(armedFile)
	return errors.Is(err, os.ErrNotExist)
}

// stopLoopDriveDisarmed records the clean cancel for a loop whose armed sentinel
// was removed and returns exit 0 (a disarm is a success, not a refusal). It
// writes a terminal End event carrying the structured reason LOOP_DISARMED and
// StatusCanceled so a ledger reader sees the loop stopped on operator signal.
func stopLoopDriveDisarmed(stderr io.Writer, opt loopDriveOptions, goalPath string, spec loopdrive.Spec, iterations int, tokensUsed int64) int {
	summary := fmt.Sprintf("armed sentinel %s removed; clean stop after %d turn(s)", opt.ArmedFile, iterations)
	if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
		LoopID:    spec.Loop,
		Kind:      loopmgr.EventEnd,
		Source:    opt.Source,
		Principal: opt.Principal,
		Status:    loopmgr.StatusCanceled,
		Reason:    "LOOP_DISARMED",
		Summary:   summary,
		EvidenceRefs: []loopmgr.EvidenceRef{
			{Kind: "goal", Ref: goalPath},
			{Kind: "armed-file", Ref: opt.ArmedFile},
		},
		Metrics: map[string]int64{"iterations": int64(iterations), "tokens_used": tokensUsed},
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "fak loop drive: LOOP_DISARMED %s\n", summary)
	return 0
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
	if h := strings.TrimSpace(opt.HandoffFile); h != "" {
		env = append(env, loopDriveHandoffFileEnv+"="+h)
	}
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
	// The drive loop is SELF-paced (an in-process Ralph loop that re-drives a goal
	// as fast as each turn completes), NOT externally scheduled — so it must keep
	// the permissive-when-unconfigured contract: a fast successful turn is not a
	// storm, and the embedded default's cadence floor (aimed at doubled schedulers)
	// would wrongly refuse a legitimate back-to-back turn. An operator who wants to
	// brake a driven loop still writes an explicit policy file.
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

// loopDriveHandoffFile resolves the path the wrapped agent writes its
// fak.task-handoff.v1 record to. An explicit path is used as-is (and not
// removed); otherwise a private per-session file is allocated so the child can
// see a stable path through loopDriveHandoffFileEnv.
func loopDriveHandoffFile(explicit string) (string, func(), error) {
	if p := strings.TrimSpace(explicit); p != "" {
		return p, func() {}, nil
	}
	dir, err := os.MkdirTemp("", "fak-loop-drive-handoff-*")
	if err != nil {
		return "", func() {}, fmt.Errorf("create task handoff dir: %w", err)
	}
	path := filepath.Join(dir, "task-handoff.json")
	return path, func() { _ = os.RemoveAll(dir) }, nil
}

// loopDriveReviewHandoff reads and grades the task handoff at the witnessed-done
// completion boundary. A missing or empty file is an ordinary non-agent stop:
// the gate fails open. A present-but-malformed record is treated as present and
// refused so a half-written handoff cannot slip a clean completion through.
func loopDriveReviewHandoff(file string) loopdrive.HandoffGateResult {
	file = strings.TrimSpace(file)
	if file == "" {
		return loopdrive.HandoffGate(false, taskmgr.Handoff{})
	}
	b, err := os.ReadFile(file)
	if err != nil || len(bytes.TrimSpace(b)) == 0 {
		return loopdrive.HandoffGate(false, taskmgr.Handoff{})
	}
	var h taskmgr.Handoff
	if err := json.Unmarshal(b, &h); err != nil {
		return loopdrive.HandoffGateResult{
			Outcome: loopdrive.HandoffGated,
			Reason:  loopdrive.ReasonHandoffRefused,
			Summary: "task handoff present but unparseable: " + trimLoopDriveSummary(err.Error()),
			Reasons: []string{"UNPARSEABLE_HANDOFF"},
		}
	}
	return loopdrive.HandoffGate(true, h)
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
	if c, handled, err := parseSharedGateCriterion(fields, ""); handled {
		return c, err
	}
	switch fields[0] {
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
	if c, handled, err := parseSharedGateCriterion(fields, "dos "); handled {
		return c, err
	}
	return loopgate.Criterion{}, fmt.Errorf("unsupported dos witness subcommand: %s", strings.Join(fields, " "))
}

// parseSharedGateCriterion parses the witness-criterion kinds common to both the bare
// and the `dos`-prefixed forms (commit-audit / verify / test-witness / citation-resolve
// / witness). msgPrefix is prepended to each shape error ("" for the bare form, "dos "
// for the dos form) so the messages are byte-identical to the hand-written originals.
// handled is false when fields[0] is not one of these shared kinds, leaving the caller
// to fall through to its own extra cases (metric/none/dos for the bare form).
func parseSharedGateCriterion(fields []string, msgPrefix string) (loopgate.Criterion, bool, error) {
	switch fields[0] {
	case "commit-audit":
		if len(fields) > 2 {
			return loopgate.Criterion{}, true, fmt.Errorf("%[1]scommit-audit witness must be: %[1]scommit-audit [REF]", msgPrefix)
		}
		c := loopgate.Criterion{Kind: loopgate.CriterionCommitAudit}
		if len(fields) == 2 {
			c.Ref = fields[1]
		}
		return parsedGateCriterion(c)
	case "verify":
		if err := requireCriterionFields(fields, 3, "%[1]sverify witness must be: %[1]sverify PLAN PHASE", msgPrefix); err != nil {
			return loopgate.Criterion{}, true, err
		}
		return parsedGateCriterion(loopgate.Criterion{Kind: loopgate.CriterionVerify, Plan: fields[1], Phase: fields[2]})
	case "test-witness":
		if err := requireCriterionFields(fields, 3, "%stest-witness criterion requires baseline and candidate outcomes", msgPrefix); err != nil {
			return loopgate.Criterion{}, true, err
		}
		return parsedGateCriterion(loopgate.Criterion{Kind: loopgate.CriterionTestWitness, Baseline: fields[1], Candidate: fields[2]})
	case "citation-resolve":
		if len(fields) < 2 {
			return loopgate.Criterion{}, true, fmt.Errorf("%scitation-resolve criterion requires a subject citation", msgPrefix)
		}
		return parsedGateCriterion(loopgate.Criterion{Kind: loopgate.CriterionCitationResolve, Subject: strings.Join(fields[1:], " ")})
	case "witness":
		if len(fields) < 3 {
			return loopgate.Criterion{}, true, fmt.Errorf("%switness criterion requires source and subject", msgPrefix)
		}
		return parsedGateCriterion(loopgate.Criterion{Kind: loopgate.CriterionWitness, Source: fields[1], Subject: strings.Join(fields[2:], " ")})
	default:
		return loopgate.Criterion{}, false, nil
	}
}

func parsedGateCriterion(criterion loopgate.Criterion) (loopgate.Criterion, bool, error) {
	return criterion, true, nil
}

func requireCriterionFields(fields []string, count int, format, prefix string) error {
	if len(fields) != count {
		return fmt.Errorf(format, prefix)
	}
	return nil
}

func runDOSLoopGateWitness(ctx context.Context, req loopgate.Request) (loopgate.WitnessResult, error) {
	cmd := exec.CommandContext(ctx, "dos", req.Argv()...)
	// ⛔ Pin the workspace (#5933). The kernel finds its lease journal by an
	// UPWARD POSITIONAL WALK for a `.dos` directory, and there are `.dos` roots
	// nested inside this repository — so an unpinned child inherits whatever cwd
	// the loop happens to run from and can resolve a SHADOW journal without ever
	// leaving the tree. This verdict gates whether a loop may proceed, so a read
	// against the wrong workspace grants a witness this repository never
	// produced. repoRoot() resolves the go.mod module root, which is the
	// repository root by construction and is not confusable with a shadow `.dos`.
	cmd.Dir = repoRoot()
	windowgate.ConfigureBackgroundCommand(cmd)
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
