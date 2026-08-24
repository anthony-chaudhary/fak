package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

const ultracodeStatusSchema = "fak.ultracode_status.v1"

const (
	ultracodeActivationVerdictPending          = "pending"
	ultracodeActivationVerdictVerifiedActive   = "verified-active"
	ultracodeActivationVerdictVerifiedInactive = "verified-inactive"
	ultracodeActivationVerdictDegraded         = "degraded"
	ultracodeActivationVerdictFailed           = "failed"
	ultracodeActivationVerdictUnknown          = "unknown"

	ultracodeActivationReasonPending          = "ACTIVATION_EVIDENCE_PENDING"
	ultracodeActivationReasonDeadlineExceeded = "ACTIVATION_EVIDENCE_DEADLINE_EXCEEDED"
	ultracodeActivationReasonChildExited      = "CHILD_EXITED_BEFORE_ACTIVATION_EVIDENCE"
	ultracodeActivationReasonReceiptMissing   = "ACTIVATION_RECEIPT_MISSING"
	ultracodeActivationReasonDegraded         = "ACTIVATION_DEGRADED"

	ultracodeBudgetPhaseProvisional = "provisional"
	ultracodeBudgetPhaseFinal       = "final"
	ultracodeBudgetPhaseInvalid     = "invalid"
	ultracodeBudgetPhaseMissing     = "missing"
)

var ultracodeStatusNow = time.Now

type ultracodeWorkerStatus struct {
	ChildID            string                         `json:"child_id"`
	State              string                         `json:"state"`
	TurnsStarted       int                            `json:"turns_started"`
	TurnsCompleted     int                            `json:"turns_completed"`
	LastEvent          string                         `json:"last_event,omitempty"`
	Activation         ultracodebench.ActivationState `json:"activation"`
	ActivationVerdict  string                         `json:"activation_verdict"`
	ActivationReason   string                         `json:"activation_reason,omitempty"`
	ActivationAgeMS    int64                          `json:"activation_age_ms"`
	ActivationDeadline time.Time                      `json:"activation_deadline_at,omitempty"`
}

type ultracodeStatus struct {
	Schema           string                                 `json:"schema"`
	SessionID        string                                 `json:"session_id"`
	RunID            string                                 `json:"run_id"`
	RequestedProfile string                                 `json:"requested_profile"`
	ResolvedProfile  string                                 `json:"resolved_profile"`
	State            string                                 `json:"state"`
	Outcome          orchestrationOutcomeStatus             `json:"outcome"`
	Activation       ultracodebench.ActivationSummary       `json:"activation"`
	Budget           orchestration.UltracodeEnvelopeReceipt `json:"budget"`
	BudgetPhase      string                                 `json:"budget_phase"`
	Workers          []ultracodeWorkerStatus                `json:"workers"`
}

func runUltracodeStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("ultracode status", flag.ContinueOnError)
	fs.SetOutput(stderr)
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
			fmt.Fprintf(stderr, "fak ultracode status: %v\n", err)
			return 1
		}
	}
	receipt, err := newestOrchestrationReceipt(root, *sessionID)
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode status: %v\n", err)
		return 1
	}
	status, err := projectUltracodeStatus(root, receipt)
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode status: %v\n", err)
		return 1
	}
	receipt.Budget = status.Budget
	if err := persistCodexOrchestrationLaunchReceipt(root, receipt); err != nil {
		fmt.Fprintf(stderr, "fak ultracode status: persist budget receipt: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(status); err != nil {
			fmt.Fprintf(stderr, "fak ultracode status: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "Ultracode %s - %s\n", status.RunID, status.State)
	fmt.Fprintf(stdout, "  activation %.1f%% verified | active=%d inactive=%d degraded=%d unknown=%d\n",
		status.Activation.Ratio*100, status.Activation.Active, status.Activation.Inactive, status.Activation.Degraded, status.Activation.Unknown)
	fmt.Fprintf(stdout, "  budget %s tokens=%d/%d remaining=%d | children=%d/%d | wall=%d/%dms remaining=%dms | authority=%s | overrun=%t | admitted=%t\n",
		status.BudgetPhase,
		status.Budget.ConsumedTokens, status.Budget.DeclaredTokens, status.Budget.RemainingTokens,
		status.Budget.CoveredChildren, status.Budget.TotalChildren, status.Budget.ConsumedWallMS,
		status.Budget.WallBudgetMS, status.Budget.RemainingWallMS, status.Budget.Authority,
		status.Budget.Overrun, status.Budget.Admitted)
	for _, worker := range status.Workers {
		fmt.Fprintf(stdout, "  %-12s %-9s activation=%s raw=%s reason=%s age=%dms turns=%d/%d\n", worker.ChildID, worker.State, worker.ActivationVerdict, worker.Activation, worker.ActivationReason, worker.ActivationAgeMS, worker.TurnsCompleted, worker.TurnsStarted)
	}
	return 0
}

func projectUltracodeStatus(root string, receipt codexOrchestrationLaunchReceipt) (ultracodeStatus, error) {
	runStatus := inspectOrchestrationRun(root, receipt)
	now := ultracodeStatusNow().UTC()
	coverage, err := ultracodebench.SummarizeActivation(receipt.Activations)
	if err != nil {
		return ultracodeStatus{}, err
	}
	states := make(map[string]ultracodebench.ActivationState, len(coverage.Children))
	activationReceipts := make(map[string]ultracodebench.ActivationReceipt, len(receipt.Activations))
	launchWorkers := make(map[string]codexOrchestrationWorkerLaunch, len(receipt.Workers))
	for _, activation := range receipt.Activations {
		activationReceipts[activation.ChildID] = activation
	}
	for _, worker := range receipt.Workers {
		launchWorkers[worker.RoleID] = worker
	}
	for _, child := range coverage.Children {
		states[child.ChildID] = child.State
	}
	for _, worker := range runStatus.Workers {
		if _, ok := states[worker.RoleID]; ok {
			continue
		}
		states[worker.RoleID] = ultracodebench.ActivationUnknown
		coverage.Total++
		coverage.Unknown++
		coverage.Children = append(coverage.Children, ultracodebench.ChildActivationStatus{
			RunID: receipt.RunID, ChildID: worker.RoleID, Harness: "codex", State: ultracodebench.ActivationUnknown,
		})
	}
	coverage.Verified = coverage.Active + coverage.Inactive
	coverage.Ratio = 0
	if coverage.Total > 0 {
		coverage.Ratio = float64(coverage.Verified) / float64(coverage.Total)
	}
	budget := receipt.Budget
	if budget.Schema == orchestration.UltracodeEnvelopeReceiptSchema && budget.DeclaredTokens > 0 && budget.WallBudgetMS > 0 {
		usage := make([]orchestration.UltracodeChildUsage, 0, len(runStatus.Workers))
		for _, worker := range runStatus.Workers {
			if worker.UsageCovered {
				usage = append(usage, orchestration.UltracodeChildUsage{
					ChildID: worker.RoleID, ProviderTokens: worker.ProviderTokens, Authority: worker.UsageAuthority,
				})
			}
		}
		var err error
		budget, err = orchestration.FoldUltracodeEnvelopeReceipt(budget, usage, now)
		if err != nil {
			return ultracodeStatus{}, err
		}
	} else {
		budget = legacyIncompleteUltracodeBudget(receipt)
	}
	out := ultracodeStatus{
		Schema: ultracodeStatusSchema, SessionID: receipt.SessionID, RunID: receipt.RunID,
		RequestedProfile: receipt.RequestedProfile, ResolvedProfile: receipt.ResolvedProfile,
		State: runStatus.State, Outcome: runStatus.Outcome, Activation: coverage, Budget: budget,
		BudgetPhase: ultracodeBudgetPhaseProvisional,
		Workers:     make([]ultracodeWorkerStatus, 0, len(runStatus.Workers)),
	}
	if receipt.Budget.Schema != orchestration.UltracodeEnvelopeReceiptSchema {
		out.BudgetPhase = ultracodeBudgetPhaseMissing
	} else if runStatus.Running == 0 && budget.Complete {
		out.BudgetPhase = ultracodeBudgetPhaseFinal
	}
	launchPending := false
	for _, worker := range receipt.Workers {
		if worker.Status == "starting" {
			launchPending = true
			break
		}
	}
	if launchPending && runStatus.Running == 0 {
		out.State = "launching"
	}
	if budget.Overrun || (!launchPending && runStatus.Running == 0 && !budget.Complete) {
		out.State = "invalid"
		out.Outcome.Verdict = "invalid"
		out.Outcome.Reason = budget.Reason
		out.BudgetPhase = ultracodeBudgetPhaseInvalid
	}
	for _, worker := range runStatus.Workers {
		workerStatus := ultracodeWorkerStatus{
			ChildID: worker.RoleID, State: worker.State, TurnsStarted: worker.TurnsStarted,
			TurnsCompleted: worker.TurnsDone, LastEvent: worker.LastEvent, Activation: states[worker.RoleID],
		}
		if launchWorkers[worker.RoleID].Status == "starting" {
			workerStatus.State = "starting"
		}
		projectUltracodeActivationVerdict(&workerStatus, receipt, launchWorkers[worker.RoleID], activationReceipts[worker.RoleID], now)
		if states[worker.RoleID] == ultracodebench.ActivationUnknown && workerStatus.Activation == ultracodebench.ActivationDegraded {
			markUltracodeActivationDegraded(&out.Activation, worker.RoleID)
		}
		out.Workers = append(out.Workers, workerStatus)
	}
	return out, nil
}

func projectUltracodeActivationVerdict(status *ultracodeWorkerStatus, receipt codexOrchestrationLaunchReceipt, launched codexOrchestrationWorkerLaunch, activation ultracodebench.ActivationReceipt, now time.Time) {
	if !receipt.LaunchedAt.IsZero() && now.After(receipt.LaunchedAt) {
		status.ActivationAgeMS = now.Sub(receipt.LaunchedAt).Milliseconds()
	}
	status.ActivationDeadline = launched.DeadlineAt
	if status.ActivationDeadline.IsZero() {
		status.ActivationDeadline = receipt.Budget.DeadlineAt
	}
	switch status.Activation {
	case ultracodebench.ActivationActive:
		status.ActivationVerdict = ultracodeActivationVerdictVerifiedActive
		return
	case ultracodebench.ActivationInactive:
		status.ActivationVerdict = ultracodeActivationVerdictVerifiedInactive
		return
	case ultracodebench.ActivationDegraded:
		status.ActivationVerdict = ultracodeActivationVerdictDegraded
		status.ActivationReason = ultracodeActivationReasonDegraded
		return
	}
	if activation.Schema != ultracodebench.ActivationSchema && status.ActivationDeadline.IsZero() {
		status.ActivationVerdict = ultracodeActivationVerdictUnknown
		status.ActivationReason = ultracodeActivationReasonReceiptMissing
		return
	}
	if status.State != "running" && status.State != "starting" {
		status.Activation = ultracodebench.ActivationDegraded
		status.ActivationVerdict = ultracodeActivationVerdictFailed
		status.ActivationReason = ultracodeActivationReasonChildExited
		return
	}
	if !status.ActivationDeadline.IsZero() && !now.Before(status.ActivationDeadline) {
		status.Activation = ultracodebench.ActivationDegraded
		status.ActivationVerdict = ultracodeActivationVerdictFailed
		status.ActivationReason = ultracodeActivationReasonDeadlineExceeded
		return
	}
	if activation.Schema != ultracodebench.ActivationSchema {
		status.ActivationVerdict = ultracodeActivationVerdictPending
		status.ActivationReason = ultracodeActivationReasonReceiptMissing
		return
	}
	status.ActivationVerdict = ultracodeActivationVerdictPending
	status.ActivationReason = ultracodeActivationReasonPending
}

func markUltracodeActivationDegraded(summary *ultracodebench.ActivationSummary, childID string) {
	if summary.Unknown > 0 {
		summary.Unknown--
	}
	summary.Degraded++
	for i := range summary.Children {
		if summary.Children[i].ChildID == childID {
			summary.Children[i].State = ultracodebench.ActivationDegraded
			break
		}
	}
}

func legacyIncompleteUltracodeBudget(receipt codexOrchestrationLaunchReceipt) orchestration.UltracodeEnvelopeReceipt {
	children := make([]orchestration.UltracodeChildBudget, 0, len(receipt.Workers))
	for _, worker := range receipt.Workers {
		children = append(children, orchestration.UltracodeChildBudget{
			ChildID: worker.RoleID, Authority: orchestration.UltracodeBudgetAuthorityIncomplete,
		})
	}
	return orchestration.UltracodeEnvelopeReceipt{
		Schema:        orchestration.UltracodeEnvelopeReceiptSchema,
		Authority:     orchestration.UltracodeBudgetAuthorityIncomplete,
		TotalChildren: len(children), Reason: orchestration.UltracodeBudgetReasonIncomplete,
		Children: children,
	}
}

func codexOrchestrationActivation(launch codexOrchestrationLaunchReceipt, childID string) (ultracodebench.ActivationReceipt, error) {
	requested, err := parseActivationSetting(launch.RequestedProfile)
	if err != nil {
		return ultracodebench.ActivationReceipt{}, err
	}
	resolved, err := parseActivationSetting(launch.ResolvedProfile)
	if err != nil {
		return ultracodebench.ActivationReceipt{}, err
	}
	degradations := make([]string, 0, len(launch.Degradations))
	for _, degradation := range launch.Degradations {
		capability, _, _ := strings.Cut(degradation, ":")
		capability = strings.ToLower(strings.TrimSpace(capability))
		capability = strings.NewReplacer("-", "_", " ", "_").Replace(capability)
		if capability != "" {
			degradations = append(degradations, "capability_"+capability)
		}
	}
	return ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{
		RunID: launch.RunID, ChildID: childID, Harness: "codex", Requested: requested, Resolved: resolved,
		Injected: resolved == ultracodebench.SettingOn, Degradations: degradations,
	})
}

func persistAccountsUltracodeActivation(root, runID, command, requestedWord string, resolvedOn bool) error {
	requested, err := parseActivationSetting(ultracodePostureWord(requestedWord))
	if err != nil {
		return err
	}
	resolved := ultracodebench.SettingOff
	if resolvedOn {
		resolved = ultracodebench.SettingOn
	}
	harness := strings.ToLower(strings.TrimSpace(guardAgentBaseName(command)))
	if harness == "claude-code" {
		harness = "claude"
	}
	injected := resolvedOn && harness == "claude"
	var degradations []string
	if resolvedOn && !injected {
		degradations = []string{"harness_cannot_inject"}
	}
	receipt, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{
		RunID: runID, ChildID: "agent", Harness: harness, Requested: requested, Resolved: resolved,
		Injected: injected, Degradations: degradations,
	})
	if err != nil {
		return err
	}
	root = strings.TrimSpace(root)
	if root == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return err
		}
		root = filepath.Join(cache, "fak")
	}
	return ultracodebench.WriteActivation(root, receipt)
}

func parseActivationSetting(value string) (ultracodebench.ActivationSetting, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "auto":
		return ultracodebench.SettingAuto, nil
	case "on", "true", "ultracode":
		return ultracodebench.SettingOn, nil
	case "off", "false", "direct":
		return ultracodebench.SettingOff, nil
	default:
		return "", fmt.Errorf("unknown Ultracode setting %q", value)
	}
}
