package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type tokenEffectivenessRow struct {
	Key          string   `json:"key"`
	Default      string   `json:"default"`
	Configured   string   `json:"configured"`
	Owner        string   `json:"owner"`
	Mechanism    string   `json:"runtime_mechanism"`
	EffectMetric string   `json:"effect_metric"`
	WitnessKind  string   `json:"witness_kind"`
	Witness      string   `json:"witness"`
	Paths        []string `json:"paths"`
	Control      string   `json:"control"`
	Observed     string   `json:"observed"`
	Scope        string   `json:"scope"`
}

type tokenEffectivenessReport struct {
	Schema string                  `json:"schema"`
	OK     bool                    `json:"ok"`
	Debt   int                     `json:"debt"`
	Rows   []tokenEffectivenessRow `json:"rows"`
	Note   string                  `json:"note"`
}

// tokenEffectivenessEvidence maps the source-derived defaults roster to the
// narrowest real effect witness and its control. It intentionally does not
// duplicate default state: collectTokenDefaultsScorecard remains authoritative.
var tokenEffectivenessEvidence = map[string]tokenEffectivenessRow{
	"provider_cache": {
		Owner: "internal/gateway", Mechanism: "provider prompt-cache prefix",
		EffectMetric: "cache-read tokens unlocked by stable fak breakpoint placement", WitnessKind: "exact-token A/B",
		Witness: "go test ./internal/gateway -run TestFakPlacementUnlocksProviderCacheSavings", Paths: []string{"internal/gateway/provider_cache_fak_placement_savings_test.go"},
		Control: "same turns with naive last-block placement", Scope: "synthetic provider-cache accounting; broader workload arms remain #6684 and #6690",
	},
	"toolfloor": {
		Owner: "internal/gateway", Mechanism: "unreachable tool-definition pruning",
		EffectMetric: "removed tool definitions and exact request-byte reduction while preserving the stable prefix", WitnessKind: "exact-byte A/B",
		Witness: "go test ./internal/gateway -run TestInboundToolsPrunesDeniedKeepsPrefix", Paths: []string{"internal/gateway/messages_inbound_tools_test.go"},
		Control: "nil/all-allowed tool-floor predicate", Scope: "synthetic Anthropic request fixture",
	},
	"mcptoolfilter": {
		Owner: "internal/gateway", Mechanism: "native MCP tools/list filtering",
		EffectMetric: "exact descriptor bytes removed with held-query recall and capability parity", WitnessKind: "exact-byte A/B",
		Witness: "go test ./internal/gateway -run TestNativeMCPFilterABProof", Paths: []string{"internal/gateway/mcp_filter_ab_test.go"},
		Control: "FAK_ABLATE_MCP_TOOL_FILTER=1", Scope: "native MCP descriptor registry and held intent queries",
	},
	"defercoldtools": {
		Owner: "internal/gateway", Mechanism: "outbound cold-tool deferral",
		EffectMetric: "resident tool-definition count with deterministic recovery; wire bytes may increase", WitnessKind: "resident-definition A/B",
		Witness: "go test ./internal/gateway -run 'TestDeferColdToolsABFiresOverRegistry|TestResidentToolDefsPartition'", Paths: []string{"internal/gateway/tooldefer_export_test.go"},
		Control: "--defer-cold-tools=false", Scope: "canonical registry fixture; this is not an exact wire-byte saving claim",
	},
	"vdso": {
		Owner: "internal/vdso", Mechanism: "repeated tool-call fast path",
		EffectMetric: "prompt tokens and engine calls avoided on one frozen trace", WitnessKind: "same-trace ablation",
		Witness: "fak ablate --sweep vdso --baseline all-off", Paths: []string{"internal/maturity/runtime-proofs.json", "internal/ablate"},
		Control: "all-off arm", Scope: "tau2 airline smoke fixture; run reports the current exact delta",
	},
	"compacthistory": {
		Owner: "internal/gateway", Mechanism: "history compaction",
		EffectMetric: "fak-authored token-equivalent shed net of metadata with provider-cache accounting separated", WitnessKind: "exact fixture proof",
		Witness: "go test ./internal/gateway -run TestFakCompactionShedNetSavingOnClaudeCodePath", Paths: []string{"internal/gateway/messages_compact_test.go"},
		Control: "--compact-history-budget 0", Scope: "Claude Code-shaped request fixture; live firing is in GET /debug/vars",
	},
	"elideresult": {
		Owner: "internal/gateway", Mechanism: "oversized result elision",
		EffectMetric: "exact request-byte reduction while retaining prefix and bounded head/tail evidence", WitnessKind: "exact-byte A/B",
		Witness: "go test ./internal/gateway -run TestMaybeElideOnShrinksKeepsPrefix", Paths: []string{"internal/gateway/messages_elide_test.go"},
		Control: "--elide-result-bytes 0", Scope: "oversized old tool_result fixture",
	},
	"elidestale": {
		Owner: "internal/gateway", Mechanism: "superseded-read elision",
		EffectMetric: "exact request-byte reduction plus verbatim restoration of elided reads", WitnessKind: "round-trip A/B",
		Witness: "go test ./internal/gateway -run TestMaybeElideStaleReadsRoundTrip", Paths: []string{"internal/gateway/messages_elide_stale_test.go"},
		Control: "--elide-stale-reads=false", Scope: "superseded Read/Edit request fixture; live saved tokens are in GET /debug/vars",
	},
	"ctxview": {
		Owner: "internal/agent", Mechanism: "budgeted ctxplan materialized view",
		EffectMetric: "rendered tokens under a fixed budget with verbatim demand-page recovery", WitnessKind: "bounded-view proof",
		Witness: "go test ./internal/agent -run TestCtxSeamRenderTurnPlansViewAndKeepsExactRecall", Paths: []string{"internal/agent/ctxplan_seam_test.go"},
		Control: "disabled seam identity path", Scope: "recorded session fixture; the default ablate smoke trace currently reports no-op, not savings",
	},
}

func buildTokenEffectivenessReport(scorecard map[string]any) tokenEffectivenessReport {
	corpus, _ := scorecard["corpus"].(map[string]any)
	status, _ := corpus["lever_status"].([]map[string]any)
	if status == nil {
		if raw, ok := corpus["lever_status"].([]any); ok {
			for _, item := range raw {
				if row, ok := item.(map[string]any); ok {
					status = append(status, row)
				}
			}
		}
	}
	report := tokenEffectivenessReport{Schema: "fak-token-effectiveness/1", OK: true, Note: "configured/default-on and live ready/not_observed are posture, not effectiveness; each row names a bounded captured witness and control"}
	root := repoRoot()
	for _, lever := range status {
		key, _ := lever["key"].(string)
		evidence, mapped := tokenEffectivenessEvidence[key]
		evidence.Key = key
		if on, _ := lever["on"].(bool); on {
			evidence.Default = "on"
			evidence.Configured = "default_on"
		} else {
			evidence.Default = "off"
			evidence.Configured = "default_off"
		}
		evidence.Observed = "captured"
		if !mapped || evidence.EffectMetric == "" || evidence.Witness == "" || evidence.Control == "" || len(evidence.Paths) == 0 {
			evidence.Observed = "missing"
		}
		for _, rel := range evidence.Paths {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); err != nil {
				evidence.Observed = "missing"
			}
		}
		if evidence.Observed == "missing" {
			report.Debt++
			report.OK = false
		}
		report.Rows = append(report.Rows, evidence)
	}
	return report
}

func writeTokenEffectivenessReport(stdout, stderr io.Writer, scorecard map[string]any, asJSON bool) int {
	report := buildTokenEffectivenessReport(scorecard)
	if asJSON {
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak token-defaults-scorecard --effectiveness: encode json: %v\n", err)
			return 1
		}
		return okExit(report.OK)
	}
	fmt.Fprintf(stdout, "token-saving effectiveness: %d methods, witness debt %d\n", len(report.Rows), report.Debt)
	fmt.Fprintln(stdout, "method          configured  observed  witness                 effect/control")
	for _, row := range report.Rows {
		fmt.Fprintf(stdout, "%-15s %-11s %-9s %-23s %s; control: %s\n", row.Key, row.Configured, row.Observed, row.WitnessKind, row.EffectMetric, row.Control)
	}
	fmt.Fprintln(stdout, "\n"+strings.TrimSpace(report.Note))
	return okExit(report.OK)
}
