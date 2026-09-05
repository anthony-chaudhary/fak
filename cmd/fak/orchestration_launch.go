package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

const codexOrchestrationLaunchSchema = "fak.codex_orchestration_launch.v1"

const defaultUltracodeWallBudget = 30 * time.Minute

const (
	maxQwenEmptyUsageRecoveryAttempts = 1
	orchestrationWorkloadKindEnv      = "FAK_ORCHESTRATION_WORKLOAD_KIND"
	orchestrationTargetModelFamilyEnv = "FAK_ORCHESTRATION_TARGET_MODEL_FAMILY"
	orchestrationUsageExpectationEnv  = "FAK_ORCHESTRATION_USAGE_EXPECTATION"
	qwenEmptyUsageWindowEnv           = "FAK_QWEN_EMPTY_USAGE_WINDOW"
	qwenEmptyUsageRecoveryAttemptsEnv = "FAK_QWEN_EMPTY_USAGE_RECOVERY_ATTEMPTS"
	qwenEmptyUsageTerminalSchema      = "fak.qwen_empty_usage_terminal.v1"
	qwenEmptyUsageTerminalReason      = "QWEN_EMPTY_USAGE"
	qwenEmptyUsageStopChecks          = 20
	qwenEmptyUsageStopInterval        = 100 * time.Millisecond
)

var orchestrationLaunchNow = time.Now
var orchestrationWorkerMonitorSleep = time.Sleep

func validateCodexOrchestrationArtifactHome(codexHome string) error {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return err
	}
	absHome, err := filepath.Abs(home)
	if err != nil {
		return fmt.Errorf("resolve Codex home: %w", err)
	}
	probe := absHome
	for {
		if _, statErr := os.Stat(probe); statErr == nil {
			break
		} else if !os.IsNotExist(statErr) {
			return fmt.Errorf("inspect Codex home %q: %w", absHome, statErr)
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			return nil
		}
		probe = parent
	}
	cmd := exec.Command("git", "-C", probe, "rev-parse", "--show-toplevel")
	configureDispatchHelperCommand(cmd)
	root, err := cmd.Output()
	if err != nil {
		return nil
	}
	worktree := strings.TrimSpace(string(root))
	if worktree == "" {
		return nil
	}
	return fmt.Errorf("unsafe Codex home %q is inside Git worktree %q; omit --codex-home to use $CODEX_HOME, or choose an external path allocated for scratch/runtime state", absHome, worktree)
}

const orchestrationChildEnv = "FAK_ORCHESTRATION_CHILD"

type codexOrchestrationWorkerLaunch struct {
	RoleID           string                               `json:"role_id"`
	OutputProfile    string                               `json:"output_profile"`
	WorkProfile      string                               `json:"work_profile"`
	PID              int                                  `json:"pid,omitempty"`
	Status           string                               `json:"status"`
	LogPath          string                               `json:"log_path,omitempty"`
	StartedAt        time.Time                            `json:"started_at,omitempty"`
	Attempt          int                                  `json:"attempt"`
	RecoveryAttempts int                                  `json:"recovery_attempts"`
	AttemptLogs      []string                             `json:"attempt_logs,omitempty"`
	Model            string                               `json:"model"`
	Mode             string                               `json:"mode"`
	Effort           string                               `json:"reasoning_effort"`
	AccessMode       string                               `json:"access_mode,omitempty"`
	ReadOnly         bool                                 `json:"read_only"`
	WriteTree        string                               `json:"write_tree,omitempty"`
	PolicyPath       string                               `json:"policy_path,omitempty"`
	ReservedTokens   int64                                `json:"reserved_tokens,omitempty"`
	DeadlineAt       time.Time                            `json:"deadline_at,omitempty"`
	Refusal          string                               `json:"refusal,omitempty"`
	Usage            *trajectory.QwenEmptyUsageAssessment `json:"empty_usage_assessment,omitempty"`
	Terminal         *qwenEmptyUsageTerminalReceipt       `json:"terminal,omitempty"`
}

type qwenEmptyUsagePolicyReceipt struct {
	Window              string   `json:"window"`
	MaxRecoveryAttempts int      `json:"max_recovery_attempts"`
	ValidExclusions     []string `json:"valid_exclusions"`
}

type orchestrationWorkloadReceipt struct {
	Kind              string `json:"kind"`
	TargetModelFamily string `json:"target_model_family"`
	WorkerKind        string `json:"worker_kind"`
	UsageExpectation  string `json:"usage_expectation"`
}

type qwenEmptyUsageTerminalReceipt struct {
	Schema              string                              `json:"schema"`
	Reason              string                              `json:"reason"`
	RunID               string                              `json:"run_id"`
	RoleID              string                              `json:"role_id"`
	WorkerModel         string                              `json:"worker_model"`
	TargetModelFamily   string                              `json:"target_model_family"`
	Attempts            int                                 `json:"attempts"`
	RecoveryAttempts    int                                 `json:"recovery_attempts"`
	MaxRecoveryAttempts int                                 `json:"max_recovery_attempts"`
	EmittedAt           time.Time                           `json:"emitted_at"`
	Assessment          trajectory.QwenEmptyUsageAssessment `json:"assessment"`
}

type codexOrchestrationLaunchReceipt struct {
	Schema            string                                 `json:"schema"`
	SessionID         string                                 `json:"session_id"`
	RunID             string                                 `json:"run_id"`
	LaunchedAt        time.Time                              `json:"launched_at"`
	TaskID            string                                 `json:"task_id"`
	RequestedProfile  string                                 `json:"requested_profile"`
	ResolvedProfile   string                                 `json:"resolved_profile"`
	WorkClass         string                                 `json:"work_class"`
	OutputProfile     string                                 `json:"output_profile"`
	WorkProfile       string                                 `json:"work_profile"`
	ProfileSource     string                                 `json:"profile_source"`
	CapabilityProfile string                                 `json:"capability_profile"`
	Degradations      []string                               `json:"degradations"`
	Status            string                                 `json:"status"`
	DeclineReason     string                                 `json:"decline_reason,omitempty"`
	Workers           []codexOrchestrationWorkerLaunch       `json:"workers"`
	Activations       []ultracodebench.ActivationReceipt     `json:"activations"`
	Budget            orchestration.UltracodeEnvelopeReceipt `json:"budget"`
	Workload          *orchestrationWorkloadReceipt          `json:"workload,omitempty"`
	EmptyUsagePolicy  *qwenEmptyUsagePolicyReceipt           `json:"empty_usage_policy,omitempty"`
	Graph             *ultracodeNodeGraphReceipt             `json:"graph,omitempty"`
}

func orchestrationDegradationNames(items []orchestration.Degradation) []string {
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, item.Capability+":"+item.Reason)
	}
	return out
}

func orchestrationWorkloadFromEnv() (*orchestrationWorkloadReceipt, error) {
	kind := strings.ToLower(strings.TrimSpace(os.Getenv(orchestrationWorkloadKindEnv)))
	family := strings.ToLower(strings.TrimSpace(os.Getenv(orchestrationTargetModelFamilyEnv)))
	expectation := strings.ToLower(strings.TrimSpace(os.Getenv(orchestrationUsageExpectationEnv)))
	if kind == "" && family == "" && expectation == "" {
		return nil, nil
	}
	if kind == "" || family == "" || expectation == "" {
		return nil, fmt.Errorf("%s, %s, and %s must be configured together", orchestrationWorkloadKindEnv, orchestrationTargetModelFamilyEnv, orchestrationUsageExpectationEnv)
	}
	if expectation != trajectory.QwenUsageExpectationProvider && expectation != trajectory.QwenUsageExpectationNone {
		return nil, fmt.Errorf("%s must be %q or %q", orchestrationUsageExpectationEnv, trajectory.QwenUsageExpectationProvider, trajectory.QwenUsageExpectationNone)
	}
	return &orchestrationWorkloadReceipt{
		Kind: kind, TargetModelFamily: family,
		WorkerKind: trajectory.QwenWorkerKindExecution, UsageExpectation: expectation,
	}, nil
}

func qwenUsageMonitoringEnabled(workload *orchestrationWorkloadReceipt) bool {
	return workload != nil &&
		workload.Kind == trajectory.QwenWorkloadKindModelPerformance &&
		workload.TargetModelFamily == trajectory.QwenTargetModelFamily &&
		workload.WorkerKind == trajectory.QwenWorkerKindExecution &&
		workload.UsageExpectation == trajectory.QwenUsageExpectationProvider
}

func qwenEmptyUsageGuardFromEnv() (qwenEmptyUsagePolicyReceipt, time.Duration, error) {
	windowRaw := strings.TrimSpace(os.Getenv(qwenEmptyUsageWindowEnv))
	window, err := time.ParseDuration(windowRaw)
	if windowRaw == "" || err != nil || window <= 0 {
		return qwenEmptyUsagePolicyReceipt{}, 0, fmt.Errorf("%s must be configured as a positive duration", qwenEmptyUsageWindowEnv)
	}
	recoveryRaw := strings.TrimSpace(os.Getenv(qwenEmptyUsageRecoveryAttemptsEnv))
	recoveryAttempts, err := strconv.Atoi(recoveryRaw)
	if recoveryRaw == "" || err != nil || recoveryAttempts < 0 || recoveryAttempts > maxQwenEmptyUsageRecoveryAttempts {
		return qwenEmptyUsagePolicyReceipt{}, 0, fmt.Errorf("%s must be configured as 0 or 1", qwenEmptyUsageRecoveryAttemptsEnv)
	}
	return qwenEmptyUsagePolicyReceipt{
		Window:              window.String(),
		MaxRecoveryAttempts: recoveryAttempts,
		ValidExclusions: []string{
			trajectory.QwenUsageReasonNotApplicable,
			trajectory.QwenUsageReasonUsageNotExpected,
			trajectory.QwenUsageReasonLaunchNotStarted,
		},
	}, window, nil
}

type orchestrationWorkerLaunchRequest struct {
	Role          orchestration.Role
	Access        orchestrationCompiledChildAccess
	WorkClass     orchestration.WorkClass
	TaskText      string
	Root          string
	RunDir        string
	Model         string
	Mode          orchestration.SOLMode
	Effort        string
	TokenBudget   int64
	DeadlineAt    time.Time
	RemainingWall time.Duration
	RunID         string
	OutputProfile string
	WorkProfile   string
	Attempt       int
	RecordStarted func(codexOrchestrationWorkerLaunch) error
}

var orchestrationWorkerLauncher = launchGuardedCodexOrchestrationWorker
var orchestrationWorkerUsageMonitor = monitorQwenOrchestrationWorker
var orchestrationWorkerStopper = stopQwenOrchestrationWorker

func launchCodexOrchestrationWorkers(home, sessionID, requestedProfile, capabilityProfile, taskText string, resolution orchestration.Resolution, wallLimitArg ...time.Duration) (codexOrchestrationLaunchReceipt, error) {
	return launchCodexOrchestrationWorkersWithProfiles(home, sessionID, requestedProfile, capabilityProfile, taskText, agentDefaultOutputStyle, agentDefaultWorkProfile, "shipped-default", resolution, wallLimitArg...)
}

func launchCodexOrchestrationWorkersWithProfiles(home, sessionID, requestedProfile, capabilityProfile, taskText, outputProfile, workProfile, profileSource string, resolution orchestration.Resolution, wallLimitArg ...time.Duration) (codexOrchestrationLaunchReceipt, error) {
	runID, err := newCodexOrchestrationRunID()
	if err != nil {
		return codexOrchestrationLaunchReceipt{}, err
	}
	launchedAt := orchestrationLaunchNow().UTC()
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: sessionID, RunID: runID,
		LaunchedAt: launchedAt, TaskID: resolution.Resolved.TaskID,
		RequestedProfile: requestedProfile, ResolvedProfile: string(resolution.Resolved.Profile),
		WorkClass: string(resolution.Resolved.WorkClass), CapabilityProfile: capabilityProfile,
		Degradations: orchestrationDegradationNames(resolution.Degradations), Workers: []codexOrchestrationWorkerLaunch{},
		Activations: []ultracodebench.ActivationReceipt{}, Graph: &ultracodeNodeGraphReceipt{Plan: &resolution.Resolved},
	}
	if resolution.Resolved.Profile != orchestration.ProfileUltracode || resolution.Resolved.Budget.MaxWorkers <= 1 {
		receipt.Status = "declined"
		receipt.DeclineReason = "resolved-direct"
		return receipt, persistCodexOrchestrationLaunchReceipt(home, receipt)
	}
	wallLimit := defaultUltracodeWallBudget
	if len(wallLimitArg) > 0 {
		wallLimit = wallLimitArg[0]
	}
	childIDs := make([]string, 0, len(resolution.Resolved.Roles)-1)
	for _, role := range resolution.Resolved.Roles {
		if role.ID != "lead" {
			childIDs = append(childIDs, role.ID)
		}
	}
	receipt.Budget, err = orchestration.NewUltracodeEnvelopeReceipt(resolution.Resolved.Budget.MaxTokens, wallLimit, launchedAt, childIDs)
	if err != nil {
		receipt.Status = "declined"
		receipt.DeclineReason = err.Error()
		if persistErr := persistCodexOrchestrationLaunchReceipt(home, receipt); persistErr != nil {
			return receipt, persistErr
		}
		return receipt, err
	}
	childBudgets := make(map[string]orchestration.UltracodeChildBudget, len(receipt.Budget.Children))
	for _, child := range receipt.Budget.Children {
		childBudgets[child.ChildID] = child
	}
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o700); err != nil {
		return receipt, err
	}
	root, err := os.Getwd()
	if err != nil {
		return receipt, err
	}
	snapshot, err := orchestrationLaunchAccessSnapshot(root, resolution.Resolved.Roles)
	if err != nil {
		return receipt, fmt.Errorf("child access snapshot: %w", err)
	}
	live := append([]laneadmit.Lease(nil), snapshot.Live...)
	route := resolution.Resolved.SOLRoute
	if route.Model == "" {
		route = orchestration.SelectSOLRoute(taskText, resolution.Resolved.Profile, resolution.Resolved.WorkClass, guardCodexDefaultModelID)
	}
	if route.ConsultOnly {
		return receipt, fmt.Errorf("SOL_ROUTE_PRO_CONSULT_ONLY: Codex cannot transmit reasoning.mode=pro; launch a separately metered Pro consultation instead")
	}
	managerModel := route.Model
	if managerModel == "" {
		managerModel = guardCodexDefaultModelID
	}
	workerModel, workerEffort := orchestration.ChildWorkerRoute(managerModel, route.WorkerModel, route.WorkerReasoningEffort)
	if envModel := strings.TrimSpace(os.Getenv("FAK_ORCHESTRATION_WORKER_MODEL")); envModel != "" {
		workerModel = envModel
	}
	if envEffort := strings.TrimSpace(os.Getenv("FAK_ORCHESTRATION_WORKER_EFFORT")); envEffort != "" {
		workerEffort = strings.ToLower(envEffort)
	}
	var emptyUsageWindow time.Duration
	workload, workloadErr := orchestrationWorkloadFromEnv()
	if workloadErr != nil {
		receipt.Status = "declined"
		receipt.DeclineReason = workloadErr.Error()
		_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
		return receipt, workloadErr
	}
	receipt.Workload = workload
	if qwenUsageMonitoringEnabled(workload) {
		guardReceipt, configuredWindow, policyErr := qwenEmptyUsageGuardFromEnv()
		if policyErr != nil {
			receipt.Status = "declined"
			receipt.DeclineReason = policyErr.Error()
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, policyErr
		}
		receipt.EmptyUsagePolicy = &guardReceipt
		emptyUsageWindow = configuredWindow
	}
	for _, role := range resolution.Resolved.Roles {
		if role.ID == "lead" {
			continue
		}
		access, compileErr := compileOrchestrationChildAccess(role, snapshot.Parent, laneadmit.Request{})
		if compileErr != nil {
			receipt.Workers = append(receipt.Workers, refusedOrchestrationWorker(role, access, compileErr))
			receipt.Status = "partial"
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, compileErr
		}
		admission := laneadmit.Decide(access.Admission, live, snapshot.Taxonomy)
		if !admission.Admit {
			admitErr := fmt.Errorf("%s: child %q: %s", admission.Reason, role.ID, admission.Detail)
			receipt.Workers = append(receipt.Workers, refusedOrchestrationWorker(role, access, admitErr))
			receipt.Status = "partial"
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, admitErr
		}
		access.PolicyPath, err = persistOrchestrationChildEnvelope(runDir, role.ID, access.ManifestJSON)
		if err != nil {
			return receipt, fmt.Errorf("persist %s child policy: %w", role.ID, err)
		}
		activation, activationErr := codexOrchestrationActivation(receipt, role.ID)
		if activationErr != nil {
			return receipt, fmt.Errorf("activation receipt %s: %w", role.ID, activationErr)
		}
		receipt.Activations = append(receipt.Activations, activation)
		remainingWall := receipt.Budget.DeadlineAt.Sub(orchestrationLaunchNow())
		if remainingWall <= 0 {
			deadlineErr := fmt.Errorf("%s: parent wall deadline elapsed before child %q launch", orchestration.UltracodeBudgetReasonWallOverrun, role.ID)
			receipt.Workers = append(receipt.Workers, refusedOrchestrationWorker(role, access, deadlineErr))
			receipt.Status = "invalid"
			receipt.DeclineReason = orchestration.UltracodeBudgetReasonWallOverrun
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, deadlineErr
		}
		request := orchestrationWorkerLaunchRequest{
			Role: role, Access: access, WorkClass: resolution.Resolved.WorkClass, TaskText: taskText,
			Root: root, RunDir: runDir, Model: workerModel, Mode: route.Mode, Effort: workerEffort,
			TokenBudget: childBudgets[role.ID].ReservedTokens, DeadlineAt: receipt.Budget.DeadlineAt,
			RemainingWall: remainingWall, RunID: receipt.RunID, OutputProfile: outputProfile, WorkProfile: workProfile,
		}
		joinWorker := func(launched codexOrchestrationWorkerLaunch) codexOrchestrationWorkerLaunch {
			launched.OutputProfile = outputProfile
			launched.WorkProfile = workProfile
			launched.Model = workerModel
			launched.Mode = string(route.Mode)
			launched.Effort = workerEffort
			launched.AccessMode = string(access.Mode)
			launched.ReadOnly = access.Admission.ReadOnly
			launched.PolicyPath = access.PolicyPath
			launched.ReservedTokens = request.TokenBudget
			launched.DeadlineAt = request.DeadlineAt
			if launched.RoleID == "" {
				launched.RoleID = role.ID
			}
			if len(access.Admission.Tree) > 0 {
				launched.WriteTree = access.Admission.Tree[0]
			}
			receipt.Workers = joinCodexOrchestrationWorker(receipt.Workers, launched)
			return launched
		}
		failWorker := func(launched codexOrchestrationWorkerLaunch, attempt int, stage string, cause error) (codexOrchestrationLaunchReceipt, error) {
			launched.Status = "failed"
			joinWorker(launched)
			receipt.Status = "partial"
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, fmt.Errorf("%s %s attempt %d: %w", stage, role.ID, attempt, cause)
		}
		maxRecoveryAttempts := 0
		if receipt.EmptyUsagePolicy != nil {
			maxRecoveryAttempts = receipt.EmptyUsagePolicy.MaxRecoveryAttempts
		}
		attemptLogs := []string{}
		for attempt := 1; attempt <= maxRecoveryAttempts+1; attempt++ {
			request.Attempt = attempt
			request.RemainingWall = receipt.Budget.DeadlineAt.Sub(orchestrationLaunchNow())
			if request.RemainingWall <= 0 {
				deadlineErr := fmt.Errorf("%s: parent wall deadline elapsed before child %q attempt %d", orchestration.UltracodeBudgetReasonWallOverrun, role.ID, attempt)
				receipt.Status = "invalid"
				receipt.DeclineReason = orchestration.UltracodeBudgetReasonWallOverrun
				_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
				return receipt, deadlineErr
			}
			logPath := orchestrationWorkerLogPath(request)
			attemptLogs = append(attemptLogs, logPath)
			joinWorker(codexOrchestrationWorkerLaunch{
				RoleID: role.ID, Status: "starting", LogPath: logPath,
				Attempt: attempt, RecoveryAttempts: attempt - 1, AttemptLogs: append([]string(nil), attemptLogs...),
			})
			receipt.Status = "launching"
			if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
				return receipt, fmt.Errorf("persist pre-spawn child %s attempt %d: %w", role.ID, attempt, err)
			}
			request.RecordStarted = func(started codexOrchestrationWorkerLaunch) error {
				if started.RoleID != "" && started.RoleID != role.ID {
					return fmt.Errorf("started child identity %q does not match launch role %q", started.RoleID, role.ID)
				}
				if started.PID <= 0 {
					return fmt.Errorf("started child %q has no process id", role.ID)
				}
				if started.Status == "" {
					started.Status = "started"
				}
				if started.StartedAt.IsZero() {
					started.StartedAt = orchestrationLaunchNow().UTC()
				}
				started.Attempt = attempt
				started.RecoveryAttempts = attempt - 1
				started.AttemptLogs = append([]string(nil), attemptLogs...)
				if started.LogPath == "" {
					started.LogPath = logPath
				}
				joinWorker(started)
				receipt.Status = "launching"
				if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
					return fmt.Errorf("persist started child %s attempt %d: %w", role.ID, attempt, err)
				}
				return nil
			}
			attemptStartedAt := orchestrationLaunchNow().UTC()
			launched, launchErr := orchestrationWorkerLauncher(request)
			if launched.RoleID == "" {
				launched.RoleID = role.ID
			}
			if launched.Status == "" {
				launched.Status = "started"
			}
			if launched.StartedAt.IsZero() {
				launched.StartedAt = attemptStartedAt
			}
			if launched.LogPath == "" {
				launched.LogPath = logPath
			}
			launched.Attempt = attempt
			launched.RecoveryAttempts = attempt - 1
			launched.AttemptLogs = append([]string(nil), attemptLogs...)
			launched = joinWorker(launched)
			if launchErr != nil {
				return failWorker(launched, attempt, "launch", launchErr)
			}
			if receipt.EmptyUsagePolicy == nil {
				break
			}
			assessment, monitorErr := orchestrationWorkerUsageMonitor(request, launched, emptyUsageWindow, receipt.Workload)
			launched.Usage = &assessment
			launched = joinWorker(launched)
			if monitorErr != nil {
				_ = orchestrationWorkerStopper(launched.PID)
				return failWorker(launched, attempt, "monitor", monitorErr)
			}
			if assessment.State == trajectory.QwenUsageStateUnobservable {
				_ = orchestrationWorkerStopper(launched.PID)
				launched.Status = "failed"
				joinWorker(launched)
				receipt.Status = "partial"
				receipt.DeclineReason = trajectory.QwenUsageReasonEvidenceUnobservable
				_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
				return receipt, fmt.Errorf("%s: child %q attempt %d has unreadable or malformed usage evidence", trajectory.QwenUsageReasonEvidenceUnobservable, role.ID, attempt)
			}
			if assessment.State != trajectory.QwenUsageStateEmpty {
				if attempt > 1 && assessment.State == trajectory.QwenUsageStateHealthy {
					launched.Status = "recovered"
					joinWorker(launched)
				}
				break
			}
			if stopErr := orchestrationWorkerStopper(launched.PID); stopErr != nil {
				launched.Status = "failed"
				joinWorker(launched)
				receipt.Status = "partial"
				_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
				return receipt, fmt.Errorf("stop empty-usage child %s attempt %d: %w", role.ID, attempt, stopErr)
			}
			if attempt <= maxRecoveryAttempts {
				launched.Status = "recovering"
				joinWorker(launched)
				receipt.Status = "recovering"
				if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
					return receipt, fmt.Errorf("persist recovery child %s attempt %d: %w", role.ID, attempt, err)
				}
				continue
			}
			launched.Status = "terminal"
			launched.Terminal = &qwenEmptyUsageTerminalReceipt{
				Schema: qwenEmptyUsageTerminalSchema, Reason: qwenEmptyUsageTerminalReason,
				RunID: receipt.RunID, RoleID: role.ID, WorkerModel: workerModel,
				TargetModelFamily: receipt.Workload.TargetModelFamily,
				Attempts:          attempt, RecoveryAttempts: attempt - 1, MaxRecoveryAttempts: maxRecoveryAttempts,
				EmittedAt: orchestrationLaunchNow().UTC(), Assessment: assessment,
			}
			joinWorker(launched)
			receipt.Status = "terminal"
			receipt.DeclineReason = qwenEmptyUsageTerminalReason
			if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
				return receipt, fmt.Errorf("persist terminal child %s: %w", role.ID, err)
			}
			return receipt, fmt.Errorf("%s: child %q produced no provider usage after %d attempt(s)", qwenEmptyUsageTerminalReason, role.ID, attempt)
		}
		if !access.Admission.ReadOnly {
			live = append(live, laneadmit.Lease{
				ID: "orchestration-child-" + role.ID, Lane: access.Admission.Lane,
				Tree: append([]string(nil), access.Admission.Tree...), Holder: role.ID,
			})
		}
	}
	receipt.Status = "launched"
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
}

// joinCodexOrchestrationWorker replaces a launch-time worker observation with
// the launcher's final joined row. Appending only on first sight keeps a
// controller-crash receipt truthful without duplicating the child after the
// controller survives to collect its final result.
func joinCodexOrchestrationWorker(workers []codexOrchestrationWorkerLaunch, joined codexOrchestrationWorkerLaunch) []codexOrchestrationWorkerLaunch {
	for i := range workers {
		if workers[i].RoleID == joined.RoleID {
			workers[i] = joined
			return workers
		}
	}
	return append(workers, joined)
}

func orchestrationWorkerArgs(req orchestrationWorkerLaunchRequest, auditPath string) []string {
	args := []string{
		"guard", "--codex-loop-gate", "off", "--provider", "openai-responses", "--audit", auditPath, "--expose-profile", "headless",
		"--output-profile", req.OutputProfile, "--work-profile", req.WorkProfile,
	}
	if req.Access.PolicyPath != "" {
		args = append(args, "--policy", req.Access.PolicyPath)
	}
	if !req.Access.Admission.ReadOnly && len(req.Access.Admission.Tree) > 0 {
		lease := "mode=enforce"
		if req.Access.Admission.Lane != "" {
			lease += ",lane=" + req.Access.Admission.Lane
		}
		for _, tree := range req.Access.Admission.Tree {
			lease += ",tree=" + tree
		}
		args = append(args, "--lease", lease)
	}
	if req.TokenBudget > 0 {
		args = append(args, "--context-budget-tokens", strconv.FormatInt(req.TokenBudget, 10))
	}
	if req.RemainingWall > 0 {
		args = append(args, "--max-duration", req.RemainingWall.String())
	}
	if req.RunID != "" && req.Role.ID != "" {
		args = append(args, "--session-id", req.RunID+"-"+req.Role.ID)
	}
	return append(args,
		"--", "codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json",
		"-c", "model="+strconv.Quote(req.Model), "-c", "model_reasoning_effort="+strconv.Quote(req.Effort), "-",
	)
}

func orchestrationWorkerAttemptPath(req orchestrationWorkerLaunchRequest, suffix string) string {
	name := req.Role.ID
	if req.Attempt > 1 {
		name = fmt.Sprintf("%s.attempt-%d", req.Role.ID, req.Attempt)
	}
	return filepath.Join(req.RunDir, name+suffix)
}

func orchestrationWorkerLogPath(req orchestrationWorkerLaunchRequest) string {
	return orchestrationWorkerAttemptPath(req, ".jsonl")
}

func orchestrationWorkerAuditPath(req orchestrationWorkerLaunchRequest) string {
	return orchestrationWorkerAttemptPath(req, "-guard.audit.jsonl")
}

func monitorQwenOrchestrationWorker(req orchestrationWorkerLaunchRequest, launched codexOrchestrationWorkerLaunch, window time.Duration, workload *orchestrationWorkloadReceipt) (trajectory.QwenEmptyUsageAssessment, error) {
	if workload == nil {
		return trajectory.AssessQwenEmptyUsage(trajectory.QwenEmptyUsageInput{
			WorkerModel: req.Model, LaunchStatus: launched.Status, PID: launched.PID,
			StartedAt: launched.StartedAt, ObservedAt: orchestrationLaunchNow().UTC(),
			Window: window, ProcessAlive: dispatchPIDAlive(launched.PID),
		}), nil
	}
	effectiveWindow := window
	if untilParentDeadline := req.DeadlineAt.Sub(launched.StartedAt); untilParentDeadline < effectiveWindow {
		effectiveWindow = untilParentDeadline
	}
	if effectiveWindow <= 0 {
		effectiveWindow = time.Nanosecond
	}
	for {
		usage, err := trajectory.InspectCodexExecUsage(launched.LogPath)
		if err != nil {
			return trajectory.QwenEmptyUsageAssessment{}, err
		}
		now := orchestrationLaunchNow().UTC()
		assessment := trajectory.AssessQwenEmptyUsage(trajectory.QwenEmptyUsageInput{
			WorkloadKind: workload.Kind, TargetModelFamily: workload.TargetModelFamily,
			WorkerKind: workload.WorkerKind, UsageExpectation: workload.UsageExpectation,
			WorkerModel: req.Model, LaunchStatus: launched.Status, PID: launched.PID,
			StartedAt: launched.StartedAt, ObservedAt: now, Window: effectiveWindow,
			ProcessAlive: dispatchPIDAlive(launched.PID), Usage: usage,
		})
		if assessment.State != trajectory.QwenUsageStatePending {
			return assessment, nil
		}
		wait := 250 * time.Millisecond
		if remaining := assessment.WindowEndsAt.Sub(now); remaining < wait {
			wait = remaining
		}
		if wait <= 0 {
			continue
		}
		orchestrationWorkerMonitorSleep(wait)
	}
}

func stopQwenOrchestrationWorker(pid int) error {
	if pid <= 0 || !dispatchPIDAlive(pid) {
		return nil
	}
	killed, detail := procguard.KillPID(pid)
	if killed {
		for i := 0; i < qwenEmptyUsageStopChecks && dispatchPIDAlive(pid); i++ {
			orchestrationWorkerMonitorSleep(qwenEmptyUsageStopInterval)
		}
	}
	if dispatchPIDAlive(pid) {
		if strings.TrimSpace(detail) == "" {
			detail = "process remained alive"
		}
		return errors.New(detail)
	}
	return nil
}

func launchGuardedCodexOrchestrationWorker(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
	fakBin, err := os.Executable()
	if err != nil {
		return codexOrchestrationWorkerLaunch{}, err
	}
	logPath := orchestrationWorkerLogPath(req)
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o600)
	if err != nil {
		return codexOrchestrationWorkerLaunch{}, err
	}
	auditPath := orchestrationWorkerAuditPath(req)
	cmd := exec.Command(fakBin, orchestrationWorkerArgs(req, auditPath)...)
	cmd.Dir = req.Root
	cmd.Stdin = strings.NewReader(orchestrationWorkerPrompt(req))
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.Env = orchestrationWorkerEnv(os.Environ())
	configureDispatchSpawn(cmd)
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, LogPath: logPath}, err
	}
	_ = logFile.Close()
	if cmd.Process == nil {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, LogPath: logPath}, fmt.Errorf("worker started without process")
	}
	pid := cmd.Process.Pid
	started := codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: pid, Status: "started", LogPath: logPath, StartedAt: orchestrationLaunchNow().UTC(), Attempt: req.Attempt}
	if req.RecordStarted != nil {
		if err := req.RecordStarted(started); err != nil {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
			return started, err
		}
	}
	time.Sleep(3 * time.Second)
	if !dispatchPIDAlive(pid) {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: pid, Status: "failed", LogPath: logPath}, fmt.Errorf("worker exited during launch probe; inspect %s", logPath)
	}
	if err := cmd.Process.Release(); err != nil {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: pid, Status: "failed", LogPath: logPath}, fmt.Errorf("release worker process handle: %w", err)
	}
	return started, nil
}

func orchestrationWorkerPrompt(req orchestrationWorkerLaunchRequest) string {
	if req.Access.Mode == orchestration.ChildAccessEffect {
		return fmt.Sprintf("You are %s in a fak ultracode workflow. Work only through the compiled effect envelope: lane %s, write tree %s, tools %s. Do not widen the region, change policy, commit, push, or launch more workers. Return a concise evidence-linked report to the lead.\n\nTask:\n%s\n",
			req.Role.Purpose, req.Access.Admission.Lane, strings.Join(req.Access.Admission.Tree, ","), strings.Join(req.Role.Access.Tools, ","), strings.TrimSpace(req.TaskText))
	}
	return fmt.Sprintf("You are %s in a fak ultracode workflow. Work read-only: inspect evidence for the task, identify concrete implementation or verification findings, and return a concise evidence-linked report to the lead. Do not edit files, commit, push, or launch more workers.\n\nTask:\n%s\n", req.Role.Purpose, strings.TrimSpace(req.TaskText))
}

func orchestrationLaunchAccessSnapshot(root string, roles []orchestration.Role) (orchestrationChildAccessSnapshot, error) {
	for _, role := range roles {
		if strings.EqualFold(strings.TrimSpace(string(role.Access.Mode)), string(orchestration.ChildAccessEffect)) {
			return orchestrationChildAccessSnapshotLoader(root)
		}
	}
	parent, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		return orchestrationChildAccessSnapshot{}, err
	}
	return orchestrationChildAccessSnapshot{Parent: parent, Taxonomy: laneadmit.Taxonomy{Loaded: true, Exclusive: map[string]bool{}, Trees: map[string][]string{}}}, nil
}

func persistOrchestrationChildEnvelope(runDir, roleID string, raw []byte) (string, error) {
	path := filepath.Join(runDir, toolcallFileStem(roleID)+"-policy.json")
	if err := writeFileAtomic(path, raw, 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func refusedOrchestrationWorker(role orchestration.Role, access orchestrationCompiledChildAccess, err error) codexOrchestrationWorkerLaunch {
	row := codexOrchestrationWorkerLaunch{RoleID: role.ID, Status: "refused", AccessMode: string(access.Mode), ReadOnly: access.Admission.ReadOnly}
	if len(access.Admission.Tree) > 0 {
		row.WriteTree = access.Admission.Tree[0]
	}
	if err != nil {
		row.Refusal = err.Error()
	}
	return row
}

func orchestrationWorkerEnv(env []string) []string {
	out := make([]string, 0, len(env)+1)
	for _, item := range env {
		key, _, _ := strings.Cut(item, "=")
		if strings.EqualFold(key, "CODEX_THREAD_ID") || strings.EqualFold(key, orchestrationChildEnv) || strings.HasPrefix(strings.ToUpper(key), "FAK_GUARD_") {
			continue
		}
		out = append(out, item)
	}
	return append(out, orchestrationChildEnv+"=1")
}

func newCodexOrchestrationRunID() (string, error) {
	var raw [12]byte
	if _, err := io.ReadFull(rand.Reader, raw[:]); err != nil {
		return "", err
	}
	return "orch-" + hex.EncodeToString(raw[:]), nil
}

func persistCodexOrchestrationLaunchReceipt(home string, receipt codexOrchestrationLaunchReceipt) error {
	dir := filepath.Join(home, "fak-orchestration-launches")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return writeFileAtomic(filepath.Join(dir, receipt.SessionID+".json"), raw, 0o600)
}

func readCodexOrchestrationLaunchReceipt(home, sessionID string) (codexOrchestrationLaunchReceipt, bool) {
	raw, err := os.ReadFile(filepath.Join(home, "fak-orchestration-launches", sessionID+".json"))
	if err != nil {
		return codexOrchestrationLaunchReceipt{}, false
	}
	var receipt codexOrchestrationLaunchReceipt
	if json.Unmarshal(raw, &receipt) != nil || receipt.Schema != codexOrchestrationLaunchSchema || receipt.SessionID != sessionID || receipt.RunID == "" {
		return codexOrchestrationLaunchReceipt{}, false
	}
	return receipt, true
}
