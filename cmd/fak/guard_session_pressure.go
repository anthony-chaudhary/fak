package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

type guardSessionPressureGateConfig struct {
	Threshold       string
	SinceDays       float64
	Max             int
	NamespacePrefix string
	Roots           []string
	Quiet           bool
	ReportPath      string
	LaunchModel     string
}

func runGuardSessionPressureGate(stderr io.Writer, cfg guardSessionPressureGateConfig) int {
	threshold := strings.ToLower(strings.TrimSpace(cfg.Threshold))
	if threshold == "" || threshold == "off" || threshold == "none" {
		return 0
	}
	nsPrefix := strings.TrimSpace(cfg.NamespacePrefix)
	if nsPrefix == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak guard: --session-pressure-gate: current workspace namespace: %v\n", err)
			return 1
		}
		nsPrefix = sessionaudit.ProjectNamespace(cwd)
	}
	var since *float64
	if cfg.SinceDays >= 0 {
		v := cfg.SinceDays
		since = &v
	}
	rep, rc := buildSessionAuditCompactReport(stderr, "actions", sessionaudit.DiscoverOptions{
		Roots:           cfg.Roots,
		SinceDays:       since,
		NamespacePrefix: nsPrefix,
	}, false, cfg.Max)
	if rc != 0 {
		return rc
	}
	plan := sessionaudit.BuildCompactActionPlan(rep)
	var ok bool
	plan, ok = sessionaudit.ApplyCompactActionGate(plan, threshold)
	if !ok {
		fmt.Fprintf(stderr, "fak guard: invalid --session-pressure-gate %q (want high, medium, none, or off)\n", cfg.Threshold)
		return 2
	}
	plan = applyGuardSessionPressureLaunchContext(plan, cfg.LaunchModel)
	if cfg.ReportPath != "" {
		if err := writeGuardSessionPressureReport(cfg.ReportPath, plan); err != nil {
			fmt.Fprintf(stderr, "fak guard: --session-pressure-report: %v\n", err)
			return 1
		}
	}
	if plan.Gate.Verdict == "refuse" {
		fmt.Fprintf(stderr, "fak guard: session pressure gate REFUSE threshold=%s refused=%d scope=%s\n",
			plan.Gate.Threshold, plan.Gate.Refused, plan.Scope.NamespaceFilter)
		for _, action := range plan.Actions {
			rank, _ := sessionaudit.CompactActionSeverityRank(action.Severity)
			thresholdRank, _ := sessionaudit.CompactActionSeverityRank(plan.Gate.Threshold)
			if rank >= thresholdRank {
				fmt.Fprintf(stderr, "  - %s %s [%s] target=%s: %s (%s)\n",
					action.ID, action.Kind, action.Severity, action.Target, action.Command, action.Evidence)
			}
		}
		return 1
	}
	if !cfg.Quiet {
		fmt.Fprintf(stderr, "fak guard: session pressure gate allow threshold=%s actions=%d scope=%s\n",
			plan.Gate.Threshold, plan.Counts.Total, plan.Scope.NamespaceFilter)
	}
	return 0
}

func applyGuardSessionPressureLaunchContext(plan sessionaudit.CompactActionPlan, launchModel string) sessionaudit.CompactActionPlan {
	launchModel = strings.TrimSpace(launchModel)
	if plan.Gate.Verdict != "refuse" || launchModel == "" || sessionaudit.ModelTier(launchModel) != "fable" {
		return plan
	}
	thresholdRank, ok := sessionaudit.CompactActionSeverityRank(plan.Gate.Threshold)
	if !ok || thresholdRank == 0 {
		return plan
	}
	refused := 0
	for _, action := range plan.Actions {
		rank, ok := sessionaudit.CompactActionSeverityRank(action.Severity)
		if !ok || rank < thresholdRank {
			continue
		}
		if guardSessionPressureActionAppliesToFableLaunch(action) {
			refused++
		}
	}
	plan.Gate.Refused = refused
	if refused == 0 {
		plan.Gate.Verdict = "allow"
		plan.Gate.Reason = fmt.Sprintf("explicit Fable launch model %q satisfies the current high-pressure routing/context actions; no action at the threshold applies to this launch", launchModel)
	}
	return plan
}

func guardSessionPressureActionAppliesToFableLaunch(action sessionaudit.CompactAction) bool {
	switch action.Kind {
	case "opus_cost_pressure", "long_context_pressure":
		return false
	default:
		return true
	}
}

func writeGuardSessionPressureReport(path string, plan sessionaudit.CompactActionPlan) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	if path == "-" {
		return sessionaudit.WriteJSON(os.Stdout, plan)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	err = sessionaudit.WriteJSON(f, plan)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
