package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const hookRecurrenceSchema = "fak/codex-hook-recurrence-gate/v1"

type recurrenceThresholds struct {
	MaxHookFailures       int    `json:"max_hook_failures"`
	MaxUnexpectedOutcomes int    `json:"max_unexpected_outcomes"`
	MaxTelemetryAge       string `json:"max_telemetry_age"`
}
type recurrenceReport struct {
	Schema                string               `json:"schema"`
	GeneratedAt           time.Time            `json:"generated_at"`
	Window                string               `json:"window"`
	Thresholds            recurrenceThresholds `json:"thresholds"`
	Census                hookCensusReport     `json:"census"`
	Profile               hookProfileReport    `json:"profile"`
	UnexpectedNumerator   int                  `json:"unexpected_numerator"`
	UnexpectedDenominator int                  `json:"unexpected_denominator"`
	TopCauses             []string             `json:"top_causes,omitempty"`
	Verdict               string               `json:"verdict"`
	Reasons               []string             `json:"reasons,omitempty"`
}

func RunCodexHookRecurrence(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-hook-gate", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	home := fs.String("codex-home", "", "active Codex home")
	workspace := fs.String("workspace", ".", "workspace")
	observations := fs.String("observations", filepath.Join(".dos", "metrics", "observations.jsonl"), "hook receipts")
	since := fs.Duration("since", 15*time.Minute, "bounded soak window")
	maxFailures := fs.Int("max-hook-failures", 0, "permitted real hook failures")
	maxUnexpected := fs.Int("max-unexpected", 0, "permitted unexpected outcomes")
	jsonOut := fs.Bool("json", false, "emit JSON")
	if fs.Parse(args) != nil || fs.NArg() != 0 || *since <= 0 || *maxFailures < 0 || *maxUnexpected < 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-hook-gate [--since 15m] [--max-hook-failures 0] [--max-unexpected 0] [--json]")
		return 2
	}
	if *home == "" {
		*home = os.Getenv("CODEX_HOME")
	}
	if *home == "" {
		h, e := os.UserHomeDir()
		if e != nil {
			return 1
		}
		*home = filepath.Join(h, ".codex")
	}
	now := time.Now().UTC()
	c, e := buildHookCensus(*home, *home, *workspace, os.Getenv("CODEX_THREAD_ID"), *observations, *since, now)
	if e != nil {
		fmt.Fprintf(stderr, "codex-hook-gate: census: %v\n", e)
		return 1
	}
	p, e := inspectCodexHookProfile(*home, *workspace, "")
	if e != nil {
		fmt.Fprintf(stderr, "codex-hook-gate: profile: %v\n", e)
		return 1
	}
	outcomes, e := queryCodexToolErrors(filepath.Join(*home, "logs_2.sqlite"), now.Add(-*since))
	if e != nil {
		fmt.Fprintf(stderr, "codex-hook-gate: outcomes: %v\n", e)
		return 1
	}
	g := evaluateHookRecurrence(c, p, outcomes, *maxFailures, *maxUnexpected)
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(g)
	} else {
		writeHookRecurrence(stdout, g)
	}
	if g.Verdict != "PASS" {
		return 1
	}
	return 0
}
func evaluateHookRecurrence(c hookCensusReport, p hookProfileReport, o codexToolErrorSummary, maxFailures, maxUnexpected int) recurrenceReport {
	g := recurrenceReport{Schema: hookRecurrenceSchema, GeneratedAt: c.GeneratedAt, Window: c.Window, Thresholds: recurrenceThresholds{MaxHookFailures: maxFailures, MaxUnexpectedOutcomes: maxUnexpected, MaxTelemetryAge: "5m"}, Census: c, Profile: p, UnexpectedNumerator: o.OutcomeErrors + o.ContractErrors + o.OtherErrors, UnexpectedDenominator: c.DispatchedCalls, Verdict: "PASS"}
	for k, n := range o.Categories {
		g.TopCauses = append(g.TopCauses, fmt.Sprintf("%d %s", n, k))
	}
	sort.Slice(g.TopCauses, func(i, j int) bool { return g.TopCauses[i] > g.TopCauses[j] })
	if len(g.TopCauses) > 5 {
		g.TopCauses = g.TopCauses[:5]
	}
	failures := c.PreToolUse.Failed + c.PostToolUse.Failed
	switch {
	case !c.ProfileMatch:
		g.Reasons = append(g.Reasons, "PROFILE_MISMATCH")
	case p.Verdict != "HEALTHY":
		g.Reasons = append(g.Reasons, "EFFECTIVE_PROFILE_UNHEALTHY")
	}
	if !c.TelemetryFresh {
		g.Reasons = append(g.Reasons, "STALE_TELEMETRY")
	}
	if c.DispatchedCalls == 0 {
		g.Reasons = append(g.Reasons, "DENOMINATOR_ZERO")
	}
	if c.PreToolUse.Unknown > 0 || c.PostToolUse.Unknown > 0 {
		g.Reasons = append(g.Reasons, "UNKNOWN_LIFECYCLE_ROWS")
	}
	if c.PreToolUse.Disabled > 0 || c.PostToolUse.Disabled > 0 {
		g.Reasons = append(g.Reasons, "EXPECTED_PHASE_DISABLED")
	}
	if failures > maxFailures {
		g.Reasons = append(g.Reasons, "HOOK_FAILURE_BUDGET_EXCEEDED")
	}
	if g.UnexpectedNumerator > maxUnexpected {
		g.Reasons = append(g.Reasons, "UNEXPECTED_OUTCOME_BUDGET_EXCEEDED")
	}
	if len(g.Reasons) > 0 {
		g.Verdict = "FAIL"
	}
	return g
}
func writeHookRecurrence(w io.Writer, g recurrenceReport) {
	fmt.Fprintf(w, "Codex hook recurrence gate: %s\nwindow=%s unexpected=%d/%d max=%d hook-failures=%d/%d max=%d telemetry-fresh=%t\n", g.Verdict, g.Window, g.UnexpectedNumerator, g.UnexpectedDenominator, g.Thresholds.MaxUnexpectedOutcomes, g.Census.PreToolUse.Failed+g.Census.PostToolUse.Failed, g.Census.DispatchedCalls, g.Thresholds.MaxHookFailures, g.Census.TelemetryFresh)
	fmt.Fprintf(w, "pre: attempted=%d succeeded=%d failed=%d skipped=%d disabled=%d unknown=%d denominator=%d\n", g.Census.PreToolUse.Attempted, g.Census.PreToolUse.Succeeded, g.Census.PreToolUse.Failed, g.Census.PreToolUse.Skipped, g.Census.PreToolUse.Disabled, g.Census.PreToolUse.Unknown, g.Census.PreToolUse.Denominator)
	fmt.Fprintf(w, "post: attempted=%d succeeded=%d failed=%d skipped=%d disabled=%d unknown=%d denominator=%d\n", g.Census.PostToolUse.Attempted, g.Census.PostToolUse.Succeeded, g.Census.PostToolUse.Failed, g.Census.PostToolUse.Skipped, g.Census.PostToolUse.Disabled, g.Census.PostToolUse.Unknown, g.Census.PostToolUse.Denominator)
	if len(g.TopCauses) > 0 {
		fmt.Fprintf(w, "top causes: %s\n", strings.Join(g.TopCauses, "; "))
	}
	if len(g.Reasons) > 0 {
		fmt.Fprintf(w, "reasons: %s\n", strings.Join(g.Reasons, ", "))
	}
}
