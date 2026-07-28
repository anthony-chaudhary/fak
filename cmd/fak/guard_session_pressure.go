package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
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
	Justification   string
}

const (
	// guardSessionPressureDefaultDays/Max are the audit window the retired
	// --session-pressure-days/--session-pressure-max flags defaulted to, held
	// identical here so a spec that names only a threshold behaves exactly as the
	// five-flag form did.
	guardSessionPressureDefaultDays = 7.0
	guardSessionPressureDefaultMax  = 40
)

// guardSessionPressureSpec is one --session-pressure-gate value, parsed.
//
// The launch gate used to own FIVE front-door flags, four of which existed only to
// qualify the fifth — their help text literally began "with --session-pressure-gate",
// and none of them did anything on their own. That is the flag-per-knob shape
// internal/heavinessscore's front-door burden KPI refuses past its hard ceiling on
// `fak guard`, the one verb most operators actually type. They fold into one spec
// string here, the same idiom --budget-envelope already uses on this FlagSet.
type guardSessionPressureSpec struct {
	Threshold     string
	SinceDays     float64
	Max           int
	ReportPath    string
	Justification string
}

// parseGuardSessionPressureSpec reads `THRESHOLD[,days=N][,max=N][,report=PATH][,justify=TEXT]`.
//
// The leading bare word is the severity threshold (high|medium|none|off), so the old
// `--session-pressure-gate high` keeps working verbatim; every later field is key=value.
// `justify=` deliberately consumes the REST of the spec, unsplit, because a real
// justification is prose and prose contains commas — which means it has to come last.
// An unknown key REFUSES rather than being ignored: a mistyped knob that silently
// no-ops would leave the operator believing they had armed a gate they had not.
func parseGuardSessionPressureSpec(spec string) (guardSessionPressureSpec, error) {
	out := guardSessionPressureSpec{
		SinceDays: guardSessionPressureDefaultDays,
		Max:       guardSessionPressureDefaultMax,
	}
	rest := strings.TrimSpace(spec)
	fields := 0
	for rest != "" {
		rest = strings.TrimLeft(rest, " \t")
		if text, ok := strings.CutPrefix(rest, "justify="); ok {
			out.Justification = strings.TrimSpace(text)
			return out, nil
		}
		field, tail, _ := strings.Cut(rest, ",")
		rest = tail
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		fields++
		key, value, hasValue := strings.Cut(field, "=")
		if !hasValue {
			if fields != 1 {
				return out, fmt.Errorf("threshold %q must come first, before the key=value fields", field)
			}
			out.Threshold = strings.ToLower(field)
			continue
		}
		value = strings.TrimSpace(value)
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "days":
			days, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return out, fmt.Errorf("days=%q: want a number of days", value)
			}
			out.SinceDays = days
		case "max":
			count, err := strconv.Atoi(value)
			if err != nil {
				return out, fmt.Errorf("max=%q: want a transcript count", value)
			}
			out.Max = count
		case "report":
			out.ReportPath = value
		default:
			return out, fmt.Errorf("unknown key %q (want days, max, report, or justify)", key)
		}
	}
	return out, nil
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
	plan = applyGuardSessionPressureLaunchContext(plan, cfg.LaunchModel, cfg.Justification)
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

func applyGuardSessionPressureLaunchContext(plan sessionaudit.CompactActionPlan, launchModel string, justification string) sessionaudit.CompactActionPlan {
	launchModel = strings.TrimSpace(launchModel)
	launchTier := sessionaudit.ModelTier(launchModel)
	if plan.Gate.Verdict != "refuse" || launchModel == "" || (launchTier != "fable" && launchTier != "opus") {
		return plan
	}
	thresholdRank, ok := sessionaudit.CompactActionSeverityRank(plan.Gate.Threshold)
	if !ok || thresholdRank == 0 {
		return plan
	}
	hasJustification := strings.TrimSpace(justification) != ""
	refused := 0
	for _, action := range plan.Actions {
		rank, ok := sessionaudit.CompactActionSeverityRank(action.Severity)
		if !ok || rank < thresholdRank {
			continue
		}
		if guardSessionPressureActionAppliesToLaunch(action, launchTier, hasJustification) {
			refused++
		}
	}
	plan.Gate.Refused = refused
	if refused == 0 {
		plan.Gate.Verdict = "allow"
		switch launchTier {
		case "fable":
			plan.Gate.Reason = fmt.Sprintf("explicit Fable launch model %q satisfies the current high-pressure routing/context actions; no action at the threshold applies to this launch", launchModel)
		case "opus":
			plan.Gate.Reason = fmt.Sprintf("explicit Opus launch model %q supplied a session-pressure justification; no action at the threshold applies to this launch", launchModel)
		}
	}
	return plan
}

func guardSessionPressureActionAppliesToLaunch(action sessionaudit.CompactAction, launchTier string, hasJustification bool) bool {
	if launchTier == "opus" && hasJustification {
		switch action.Kind {
		case "opus_cost_pressure", "long_context_pressure":
			return false
		}
	}
	if launchTier != "fable" {
		return true
	}
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
