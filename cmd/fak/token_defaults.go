package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func cmdTokenDefaultsScorecard(argv []string) {
	fs := flag.NewFlagSet("fak token-defaults-scorecard", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit machine-readable scorecard JSON")
	asMarkdown := fs.Bool("markdown", false, "emit markdown")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	p := collectTokenDefaultsScorecard(repoRoot())
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		_ = enc.Encode(p)
		return
	}
	c := p["corpus"].(map[string]any)
	if *asMarkdown {
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
	defects := []string{}
	require := func(ok bool, msg string) {
		if !ok {
			defects = append(defects, msg)
		}
	}
	require(strings.Contains(gateway, "const DefaultCompactHistoryBudget = 48000"), "gateway.DefaultCompactHistoryBudget must stay default-on")
	require(strings.Contains(gateway, "DefaultElideResultBytes = 0"), "gateway.DefaultElideResultBytes must be the single dark default")
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

	debt := len(defects)
	score := 100
	grade := "A"
	ok, verdict, finding := true, "OK", "token_defaults_wired"
	reason := "zero token-defaults-debt; safe default token savers are wired and dark levers are pinned"
	next := "rerun after changing serve/guard/gateway token-saving defaults"
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
			"levers_total":        6,
			"stacked_on":          4,
			"defects":             defects,
			"lever_status": []map[string]any{
				{"key": "elideresult", "on": false, "witnessed": false},
				{"key": "ctxview", "on": false, "witnessed": true},
			},
		},
	}
}
