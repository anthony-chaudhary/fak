package main

import (
	"fmt"
	"io"
	"os"
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
