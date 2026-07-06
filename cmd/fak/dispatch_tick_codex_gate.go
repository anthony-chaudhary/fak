package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func dispatchCodexLoopGateForTick(opts dispatchTickOptions, account dispatchtick.Account) (map[string]any, bool, error) {
	if !opts.Live || opts.Backend != "codex" {
		return nil, false, nil
	}
	threshold := strings.ToLower(strings.TrimSpace(opts.CodexLoopGate))
	if threshold == "" {
		threshold = dispatchCodexLoopGateDefaultThreshold()
	}
	if threshold == "" || threshold == "off" || threshold == "none" || threshold == "false" || threshold == "0" {
		return nil, false, nil
	}
	if _, ok := codexLoopFailOnRank(threshold); !ok {
		return nil, false, fmt.Errorf("invalid --codex-loop-gate %q (want loop, action, or off)", opts.CodexLoopGate)
	}
	if d, ok, err := diagnoseCurrentCodexLoop(account.Dir); ok {
		if err != nil {
			return nil, false, fmt.Errorf("Codex current-thread gate audit failed: %w", err)
		}
		if codexLoopDiagnosisUnguarded(d) {
			return dispatchCodexLoopCurrentThreadPayload(d, threshold), true, nil
		}
	}
	limit := opts.CodexLoopGateLimit
	if limit <= 0 {
		limit = dispatchCodexLoopGateDefaultLimitValue()
	}
	rep, err := diagnoseRecentCodexLoops(account.Dir, opts.CodexLoopGateSinceHours, limit)
	if err != nil {
		return nil, false, fmt.Errorf("Codex loop gate audit failed: %w", err)
	}
	gateCode, _ := codexLoopFailOnExitCode(rep.Verdict, threshold)
	return dispatchCodexLoopGatePayload(rep, threshold), gateCode != 0, nil
}

func dispatchCodexLoopGateDefaultThreshold() string {
	return firstString(strings.TrimSpace(os.Getenv("FLEET_CODEX_LOOP_GATE")), "loop")
}

func dispatchCodexLoopGateDefaultSinceHoursValue() float64 {
	raw := strings.TrimSpace(os.Getenv("FLEET_CODEX_LOOP_GATE_SINCE_HOURS"))
	if raw == "" {
		return dispatchCodexLoopGateDefaultSinceHours
	}
	if n, err := strconv.ParseFloat(raw, 64); err == nil && n >= 0 {
		return n
	}
	return dispatchCodexLoopGateDefaultSinceHours
}

func dispatchCodexLoopGateDefaultLimitValue() int {
	raw := strings.TrimSpace(os.Getenv("FLEET_CODEX_LOOP_GATE_LIMIT"))
	if raw == "" {
		return dispatchCodexLoopGateDefaultLimit
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchCodexLoopGateDefaultLimit
}

func dispatchCodexLoopGatePayload(rep codexLoopRecentReport, threshold string) map[string]any {
	return map[string]any{
		"schema":       rep.Schema,
		"source":       "recent",
		"codex_home":   rep.CodexHome,
		"fail_on":      codexLoopFailOnName(threshold),
		"verdict":      rep.Verdict,
		"reason":       rep.Reason,
		"since_hours":  rep.SinceHours,
		"limit":        rep.Limit,
		"scanned":      rep.Scanned,
		"loop_count":   rep.LoopCount,
		"action_count": rep.ActionCount,
		"ok_count":     rep.OKCount,
		"tool_calls":   rep.ToolCalls,
		"tool_outputs": rep.ToolOutputs,
		"top_repeated": rep.TopRepeated,
	}
}

func dispatchCodexLoopCurrentThreadPayload(d codexLoopDiagnosis, threshold string) map[string]any {
	return map[string]any{
		"schema":           d.Schema,
		"source":           "current_thread",
		"fail_on":          "unguarded",
		"threshold":        codexLoopFailOnName(threshold),
		"verdict":          d.Verdict,
		"reason":           codexLoopDiagnosisGateReason(d, "unguarded"),
		"path":             d.Path,
		"session_id":       d.SessionID,
		"model_provider":   d.ModelProvider,
		"tool_calls":       d.ToolCalls,
		"tool_outputs":     d.ToolOutputs,
		"repeated_outputs": d.RepeatedOutcomes,
	}
}
