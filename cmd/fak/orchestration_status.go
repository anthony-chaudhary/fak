package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

const orchestrationStatusSchema = "fak.codex_orchestration_status.v1"

const orchestrationEvidenceNotObserved = "not_observed"

var orchestrationStatusNow = time.Now

type orchestrationOutcomeStatus struct {
	Verdict            string `json:"verdict"`
	EffectReadback     string `json:"effect_readback"`
	IndependentWitness string `json:"independent_witness"`
	Reconciliation     string `json:"reconciliation"`
	Reason             string `json:"reason"`
}

type orchestrationWorkerStatus struct {
	RoleID            string                               `json:"role_id"`
	OutputProfile     string                               `json:"output_profile"`
	WorkProfile       string                               `json:"work_profile"`
	PID               int                                  `json:"pid,omitempty"`
	State             string                               `json:"state"`
	ProcessAlive      bool                                 `json:"process_alive"`
	LogPath           string                               `json:"log_path,omitempty"`
	Model             string                               `json:"model"`
	StartedAt         time.Time                            `json:"started_at,omitempty"`
	Attempt           int                                  `json:"attempt"`
	RecoveryAttempts  int                                  `json:"recovery_attempts"`
	LogBytes          int64                                `json:"log_bytes"`
	LogReadable       bool                                 `json:"log_readable"`
	LogParseErrors    int                                  `json:"log_parse_errors"`
	UpdatedAt         time.Time                            `json:"updated_at,omitempty"`
	TurnsStarted      int                                  `json:"turns_started"`
	TurnsDone         int                                  `json:"turns_completed"`
	LastEvent         string                               `json:"last_event,omitempty"`
	InputTokens       int64                                `json:"input_tokens,omitempty"`
	CachedInputTokens int64                                `json:"cached_input_tokens,omitempty"`
	OutputTokens      int64                                `json:"output_tokens,omitempty"`
	ProviderTokens    int64                                `json:"provider_tokens,omitempty"`
	UsageCovered      bool                                 `json:"usage_covered"`
	UsageAuthority    string                               `json:"usage_authority"`
	EmptyUsage        *trajectory.QwenEmptyUsageAssessment `json:"empty_usage_assessment,omitempty"`
	Terminal          *qwenEmptyUsageTerminalReceipt       `json:"terminal,omitempty"`
}

type orchestrationRunStatus struct {
	Schema           string                        `json:"schema"`
	SessionID        string                        `json:"session_id"`
	RunID            string                        `json:"run_id"`
	LaunchedAt       time.Time                     `json:"launched_at"`
	RequestedProfile string                        `json:"requested_profile"`
	ResolvedProfile  string                        `json:"resolved_profile"`
	WorkClass        string                        `json:"work_class"`
	OutputProfile    string                        `json:"output_profile"`
	WorkProfile      string                        `json:"work_profile"`
	ProfileSource    string                        `json:"profile_source"`
	State            string                        `json:"state"`
	Running          int                           `json:"running"`
	Completed        int                           `json:"completed"`
	Exited           int                           `json:"exited"`
	Excluded         int                           `json:"excluded"`
	Terminal         int                           `json:"terminal"`
	Outcome          orchestrationOutcomeStatus    `json:"outcome"`
	Workload         *orchestrationWorkloadReceipt `json:"workload,omitempty"`
	EmptyUsagePolicy *qwenEmptyUsagePolicyReceipt  `json:"empty_usage_policy,omitempty"`
	Workers          []orchestrationWorkerStatus   `json:"workers"`
}

func runOrchestrationStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("orchestration status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak orchestration status [--session ID] [--home DIR] [--json]")
		fs.PrintDefaults()
	}
	home := fs.String("home", "", "state root (default: current directory)")
	sessionID := fs.String("session", "", "specific launch session id (default: newest)")
	asJSON := fs.Bool("json", false, "emit versioned machine-readable status")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak orchestration status [--session ID] [--home DIR] [--json]")
		return 2
	}
	root := *home
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak orchestration status: %v\n", err)
			return 1
		}
	}
	receipt, err := newestOrchestrationReceipt(root, *sessionID)
	if err != nil {
		fmt.Fprintf(stderr, "fak orchestration status: %v\n", err)
		return 1
	}
	status := inspectOrchestrationRun(root, receipt)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(status); err != nil {
			fmt.Fprintf(stderr, "fak orchestration status: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "Orchestration %s - %s (worker execution)\n", status.RunID, status.State)
	fmt.Fprintf(stdout, "  Outcome %s | effects=%s | witness=%s | reconciliation=%s\n", status.Outcome.Verdict, status.Outcome.EffectReadback, status.Outcome.IndependentWitness, status.Outcome.Reconciliation)
	fmt.Fprintf(stdout, "  %s\n", status.Outcome.Reason)
	fmt.Fprintf(stdout, "  %d running | %d completed | %d exited | %d excluded | %d terminal | launched %s\n", status.Running, status.Completed, status.Exited, status.Excluded, status.Terminal, status.LaunchedAt.Local().Format("2006-01-02 15:04:05"))
	for _, worker := range status.Workers {
		activity := "no events"
		if worker.LastEvent != "" {
			activity = fmt.Sprintf("%s | turns %d/%d", worker.LastEvent, worker.TurnsDone, worker.TurnsStarted)
		}
		if worker.Terminal != nil {
			activity = fmt.Sprintf("%s | %s", worker.Terminal.Reason, worker.Terminal.Assessment.Reason)
		}
		fmt.Fprintf(stdout, "  %-10s %-9s pid=%-7d %s | %s\n", worker.RoleID, worker.State, worker.PID, humanBytes(worker.LogBytes), activity)
	}
	fmt.Fprintf(stdout, "  machine: fak orchestration status --session %s --json\n", status.SessionID)
	return 0
}

func newestOrchestrationReceipt(home, sessionID string) (codexOrchestrationLaunchReceipt, error) {
	if sessionID != "" {
		if r, ok := readCodexOrchestrationLaunchReceipt(home, sessionID); ok {
			return r, nil
		}
		return codexOrchestrationLaunchReceipt{}, fmt.Errorf("launch receipt %q not found", sessionID)
	}
	paths, err := filepath.Glob(filepath.Join(home, "fak-orchestration-launches", "*.json"))
	if err != nil {
		return codexOrchestrationLaunchReceipt{}, err
	}
	sort.Slice(paths, func(i, j int) bool {
		ai, _ := os.Stat(paths[i])
		aj, _ := os.Stat(paths[j])
		return ai.ModTime().After(aj.ModTime())
	})
	for _, p := range paths {
		id := filepath.Base(p)
		id = id[:len(id)-len(filepath.Ext(id))]
		if r, ok := readCodexOrchestrationLaunchReceipt(home, id); ok {
			return r, nil
		}
	}
	return codexOrchestrationLaunchReceipt{}, fmt.Errorf("no launch receipts under %s; pass --session with a session id from a completed launch", filepath.Join(home, "fak-orchestration-launches"))
}

func inspectOrchestrationRun(home string, receipt codexOrchestrationLaunchReceipt) orchestrationRunStatus {
	out := orchestrationRunStatus{Schema: orchestrationStatusSchema, SessionID: receipt.SessionID, RunID: receipt.RunID, LaunchedAt: receipt.LaunchedAt, RequestedProfile: receipt.RequestedProfile, ResolvedProfile: receipt.ResolvedProfile, WorkClass: receipt.WorkClass, OutputProfile: receipt.OutputProfile, WorkProfile: receipt.WorkProfile, ProfileSource: receipt.ProfileSource, State: "complete", Outcome: unverifiedOrchestrationOutcome(), Workload: receipt.Workload, EmptyUsagePolicy: receipt.EmptyUsagePolicy, Workers: make([]orchestrationWorkerStatus, 0, len(receipt.Workers))}
	for _, worker := range receipt.Workers {
		ws := orchestrationWorkerStatus{
			RoleID: worker.RoleID, OutputProfile: worker.OutputProfile, WorkProfile: worker.WorkProfile,
			PID: worker.PID, LogPath: worker.LogPath, ProcessAlive: processalive.Check(worker.PID),
			Model: worker.Model, StartedAt: worker.StartedAt, Attempt: worker.Attempt,
			RecoveryAttempts: worker.RecoveryAttempts, Terminal: worker.Terminal,
		}
		path := worker.LogPath
		if !filepath.IsAbs(path) {
			path = filepath.Join(home, path)
		}
		inspectOrchestrationLog(path, &ws)
		if receipt.Workload != nil {
			window := time.Duration(0)
			if receipt.EmptyUsagePolicy != nil {
				window, _ = time.ParseDuration(receipt.EmptyUsagePolicy.Window)
			}
			assessment := trajectory.AssessQwenEmptyUsage(trajectory.QwenEmptyUsageInput{
				WorkloadKind: receipt.Workload.Kind, TargetModelFamily: receipt.Workload.TargetModelFamily,
				WorkerKind: receipt.Workload.WorkerKind, UsageExpectation: receipt.Workload.UsageExpectation,
				WorkerModel: worker.Model, LaunchStatus: worker.Status, PID: worker.PID,
				StartedAt: worker.StartedAt, ObservedAt: orchestrationStatusNow().UTC(), Window: window,
				ProcessAlive: ws.ProcessAlive,
				Usage: trajectory.CodexExecUsage{
					LogReadable: ws.LogReadable, ParseErrors: ws.LogParseErrors,
					TurnsStarted: ws.TurnsStarted, TurnsCompleted: ws.TurnsDone,
					LastEvent: ws.LastEvent, InputTokens: ws.InputTokens, CachedInputTokens: ws.CachedInputTokens,
					OutputTokens: ws.OutputTokens, ProviderTokens: ws.ProviderTokens, UsageCovered: ws.UsageCovered,
				},
			})
			ws.EmptyUsage = &assessment
		}
		switch {
		case ws.Terminal != nil:
			ws.State = "terminal"
			out.Terminal++
		case ws.EmptyUsage != nil && ws.EmptyUsage.State == trajectory.QwenUsageStateEmpty:
			ws.State = "empty_usage"
			out.Exited++
		case ws.EmptyUsage != nil && ws.EmptyUsage.State == trajectory.QwenUsageStateUnobservable:
			ws.State = "unobservable"
			out.Exited++
		case ws.EmptyUsage != nil && ws.EmptyUsage.State == trajectory.QwenUsageStateExcluded:
			ws.State = "excluded"
			out.Excluded++
		case ws.TurnsStarted > 0 && ws.TurnsDone >= ws.TurnsStarted:
			ws.State = "completed"
			out.Completed++
		case ws.ProcessAlive:
			ws.State = "running"
			out.Running++
		default:
			ws.State = "exited"
			out.Exited++
		}
		out.Workers = append(out.Workers, ws)
	}
	if out.Running > 0 {
		out.State = "running"
	} else if out.Exited > 0 || out.Terminal > 0 {
		out.State = "attention"
	}
	return out
}

func unverifiedOrchestrationOutcome() orchestrationOutcomeStatus {
	return orchestrationOutcomeStatus{
		Verdict:            "unverified",
		EffectReadback:     orchestrationEvidenceNotObserved,
		IndependentWitness: orchestrationEvidenceNotObserved,
		Reconciliation:     orchestrationEvidenceNotObserved,
		Reason:             "worker and turn events do not prove effects, independent witness acceptance, or reconciliation",
	}
}

func inspectOrchestrationLog(path string, ws *orchestrationWorkerStatus) {
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	ws.LogBytes = info.Size()
	ws.LogReadable = true
	ws.UpdatedAt = info.ModTime().UTC()
	usage, err := trajectory.InspectCodexExecUsage(path)
	if err != nil {
		return
	}
	ws.LastEvent = usage.LastEvent
	ws.LogReadable = usage.LogReadable
	ws.LogParseErrors = usage.ParseErrors
	ws.TurnsStarted = usage.TurnsStarted
	ws.TurnsDone = usage.TurnsCompleted
	ws.InputTokens = usage.InputTokens
	ws.CachedInputTokens = usage.CachedInputTokens
	ws.OutputTokens = usage.OutputTokens
	ws.ProviderTokens = usage.ProviderTokens
	ws.UsageCovered = usage.UsageCovered
	if usage.UsageCovered {
		ws.UsageAuthority = orchestration.UltracodeBudgetAuthorityProvider
	}
}
