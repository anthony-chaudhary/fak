package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func cmdTokenDefaultsScorecard(argv []string) {
	p, c, asMarkdown, done := scorecardCmdSetup("fak token-defaults-scorecard", argv, collectTokenDefaultsScorecard)
	if done {
		return
	}
	if asMarkdown {
		fmt.Printf("# fak token-saving-defaults scorecard\n\n**token_defaults_debt: %v**; grade **%v**.\n", c["token_defaults_debt"], c["grade"])
		return
	}
	fmt.Printf("token-defaults-scorecard: %s (%s)\n  token_defaults_debt: %v   grade: %v\n", p["verdict"], p["finding"], c["token_defaults_debt"], c["grade"])
	if defects, _ := c["defects"].([]string); len(defects) > 0 {
		for _, d := range defects {
			fmt.Println("  - " + d)
		}
	}
}

func collectTokenDefaultsScorecard(root string) map[string]any {
	read := func(rel string) string {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return ""
		}
		return string(b)
	}
	serve := read("cmd/fak/serve.go")
	guard := read("cmd/fak/guard.go")
	gateway := read("internal/gateway/gateway.go")
	tui := read("cmd/fak/tui.go")
	defects := []string{}
	require := func(ok bool, msg string) {
		if !ok {
			defects = append(defects, msg)
		}
	}
	require(strings.Contains(gateway, "const DefaultCompactHistoryBudget = 48000"), "gateway.DefaultCompactHistoryBudget must stay default-on")
	require(strings.Contains(gateway, "DefaultElideResultBytes = DocumentedElideResultBytes"), "gateway.DefaultElideResultBytes must arm oversized-result elision on by default at the documented threshold")
	for _, entry := range []struct {
		name string
		src  string
	}{
		{"serve.go", serve},
		{"guard.go", guard},
	} {
		require(strings.Contains(entry.src, `fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget`), entry.name+" must wire compact-history-budget to gateway.DefaultCompactHistoryBudget")
		require(strings.Contains(entry.src, `fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes`), entry.name+" must wire elide-result-bytes to gateway.DefaultElideResultBytes")
		require(strings.Contains(entry.src, `fs.Int("ctx-view-budget", 0`), entry.name+" must keep ctx-view-budget dark at 0")
	}
	require(strings.Contains(serve, `fs.Bool("vdso", true`), "serve.go must default vDSO on")
	require(strings.Contains(guard, "VDSO:                  true") || strings.Contains(guard, "VDSO: true"), "guard.go must set VDSO true")
	require(strings.Contains(serve, "ToolFloorDenies:") && strings.Contains(guard, "ToolFloorDenies:"), "both front doors must wire ToolFloorDenies")
	// The per-turn debug-stats line is the observable cache/token-value debug layer. It is ON
	// by default on the flagship `fak guard` Claude-OAuth path (the cache + token-value economy
	// of every turn is visible with no flag; --debug-stats=false or --quiet silences it) and in
	// the console-agent launcher (fak console agent / fak c) overlay. `fak serve` keeps it off:
	// that daemon's observability is /metrics + /debug/vars + the access log, not a per-turn
	// stderr line. Lock both on-by-default front doors so the visible layer cannot silently regress.
	require(strings.Contains(guard, `fs.Bool("debug-stats", true`), "guard.go must default --debug-stats ON so the observable cache/token-value debug layer is visible by default on the Claude-OAuth path")
	require(strings.Contains(tui, `fs.Bool("debug-stats", true`), "tui.go must default --debug-stats to true in the console agent launcher (native per-turn token-usage overlay)")
	require(strings.Contains(tui, `"--debug-stats"`), "tui.go must wire --debug-stats into the guard command for the console launcher overlay")
	require(strings.Contains(tui, "gateway.DefaultCompactHistoryBudget") && strings.Contains(tui, "gateway.DefaultElideResultBytes"), "tui.go must pass the active token-saving guard defaults explicitly so they appear in dry-run output")

	debt := len(defects)
	score := 100
	grade := "A"
	ok, verdict, finding := true, "OK", "token_defaults_wired"
	reason := "zero token-defaults-debt; safe default token savers are wired and the console overlay is on"
	next := "rerun after changing serve/guard/gateway/tui token-saving defaults"
	if debt > 0 {
		ok, verdict, finding = false, "ACTION", "token_defaults_debt"
		score, grade = 70, "C"
		reason = fmt.Sprintf("%d token-defaults defect(s)", debt)
		next = "restore the default wiring named in corpus.defects"
	}
	return map[string]any{
		"schema":      "fak-token-defaults-scorecard/1",
		"ok":          ok,
		"verdict":     verdict,
		"finding":     finding,
		"reason":      reason,
		"next_action": next,
		"corpus": map[string]any{
			"token_defaults_debt": debt,
			"score":               score,
			"grade":               grade,
			"levers_total":        7,
			"stacked_on":          6,
			"defects":             defects,
			"lever_status": []map[string]any{
				{"key": "elideresult", "on": true, "witnessed": true},
				{"key": "ctxview", "on": false, "witnessed": true},
				{"key": "console-overlay", "on": true, "witnessed": true},
			},
		},
	}
}
