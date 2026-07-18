package main

// footprint_audit.go — `fak footprint --audit` (#5050, epic #2063): the user-facing
// session-config token-bloat audit over agent.RequestFootprint. ctg (`claude-token-guard`,
// "ESLint for your Claude Code setup") is the prior art for the SURFACE; fak's substrate
// is strictly better than ctg's hardcoded weights, so the audit is a thin AGGREGATOR over
// measurement fak already owns, never a reimplementation:
//
//   - the system/tools/history/tail split + the FLOOR (= system + tools, the per-turn
//     tax) come from agent.RequestFootprint — the real bucketed estimator, not
//     ctg-style `WEIGHTS[Pn] * occurrences` constants;
//   - the cold-custom-tool count comes from the PRODUCTION #3232 defer transform
//     (gateway.DeferColdToolsAB), so "N tools could defer" is the same classification
//     the live lever would apply, never a second opinion;
//   - the config-waste check is the GH#746 exposure (#5049): a custom
//     ANTHROPIC_BASE_URL with ENABLE_TOOL_SEARCH unset makes Claude Code materialize
//     EVERY tool schema into context — the ~35.8k tool slice of the ~41k fresh-session
//     floor.
//
// Provenance is honest by construction: every number is labeled ESTIMATED
// (agent.FootprintProvenance, ~4 chars/token) — the audit says WHERE the bytes go and
// which lever cuts them; it never claims a provider-measured saving. Read-only: no
// auto-fix in v1 (ctg's `fix --auto` edits CLAUDE.md — out of scope; fak reports, the
// user acts).

import (
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// footprintAuditFinding is one verifiable config waste: a stable ID, the measured
// detail, and the exact fak lever that fixes it.
type footprintAuditFinding struct {
	ID     string `json:"id"`
	Detail string `json:"detail"`
	Lever  string `json:"lever"`
}

// footprintAudit is the aggregated report: the structural footprint of one request
// plus the config-waste findings. Findings is empty (not nil-vs-missing significant)
// when nothing fired — a clean audit is a valid result.
type footprintAudit struct {
	Source        string                  `json:"source"` // "representative" or the --req file path
	Footprint     agent.Footprint         `json:"footprint"`
	ColdToolCount int                     `json:"cold_tool_count"`
	DeferReason   string                  `json:"defer_reason,omitempty"` // transform stand-down reason when no cold tools counted
	Findings      []footprintAuditFinding `json:"findings"`
}

// Audit finding IDs — stable identifiers a script can key on.
const (
	findingColdTools       = "COLD_TOOLS_RESIDENT"
	findingToolSearchUnset = "TOOL_SEARCH_UNSET"
)

// deFoldAuditSystemReq mirrors gateway's unexported deFoldSystemReq: a decoded live
// request carries the system prompt BOTH in req.System and as a prepended RoleSystem
// message (DecodeAnthropicMessagesRequest folds it in), so a naive RequestFootprint
// counts it twice. Drop the leading folded duplicate; never mutate the original.
func deFoldAuditSystemReq(req *agent.AnthropicMessagesRequest) *agent.AnthropicMessagesRequest {
	if req == nil {
		return nil
	}
	if req.System != "" && len(req.Messages) > 0 &&
		req.Messages[0].Role == agent.RoleSystem && req.Messages[0].Content == req.System {
		cp := *req
		cp.Messages = req.Messages[1:]
		return &cp
	}
	return req
}

// buildFootprintAudit is the pure aggregator: raw is one Anthropic Messages request
// body; baseURL/toolSearch are the session-config signals (ANTHROPIC_BASE_URL /
// ENABLE_TOOL_SEARCH), injected so the table test can pin them.
func buildFootprintAudit(raw []byte, source, baseURL, toolSearch string) (footprintAudit, error) {
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		return footprintAudit{}, fmt.Errorf("undecodable Messages request: %w", err)
	}
	a := footprintAudit{
		Source:    source,
		Footprint: agent.RequestFootprint(deFoldAuditSystemReq(req)),
		Findings:  []footprintAuditFinding{},
	}

	// Cold-tool classification via the PRODUCTION #3232 transform — never a local
	// re-opinion of which tools are hot. Read-only: both arms are discarded.
	arms := gateway.DeferColdToolsAB(raw)
	if arms.Changed {
		a.ColdToolCount = arms.ColdCount
	} else {
		a.DeferReason = arms.Reason
	}
	if a.ColdToolCount > 0 {
		a.Findings = append(a.Findings, footprintAuditFinding{
			ID: findingColdTools,
			Detail: fmt.Sprintf("%d cold custom tool schema(s) resident on every turn — the deferral lever would keep only the hot core loaded",
				a.ColdToolCount),
			Lever: "fak serve --defer-cold-tools (#3232) — Anthropic faults a cold schema in on demand",
		})
	}

	// GH#746 exposure (#5049): a custom base-URL with ENABLE_TOOL_SEARCH unset makes
	// Claude Code materialize every tool schema into context. Only checkable config
	// facts fire this: base-URL active AND the env var absent.
	if baseURL != "" && toolSearch == "" {
		a.Findings = append(a.Findings, footprintAuditFinding{
			ID:     findingToolSearchUnset,
			Detail: "custom base-URL active (ANTHROPIC_BASE_URL) with ENABLE_TOOL_SEARCH unset — Claude Code materializes every tool schema into context (GH#746)",
			Lever:  "set ENABLE_TOOL_SEARCH=1 in the session env (#5049), or fak serve --defer-cold-tools (#3232)",
		})
	}
	return a, nil
}

// renderFootprintAudit is the human view: floor split first (the headline), then the
// top reducible tool schemas, then the findings with their levers.
func renderFootprintAudit(w io.Writer, a footprintAudit, top int) {
	fp := a.Footprint
	fmt.Fprintf(w, "footprint-audit (%s, ~4 chars/token) — source: %s\n", fp.Provenance, a.Source)
	fmt.Fprintf(w, "  floor (system+tools, the per-turn tax): %d est. tokens\n", fp.Floor.Tokens)
	fmt.Fprintf(w, "    system   %6d tok (%4.1f%%)\n", fp.System.Tokens, fp.System.Pct)
	fmt.Fprintf(w, "    tools    %6d tok (%4.1f%%, %d tool schemas)\n", fp.Tools.Tokens, fp.Tools.Pct, fp.ToolCount)
	fmt.Fprintf(w, "    history  %6d tok (%4.1f%%)\n", fp.History.Tokens, fp.History.Pct)
	fmt.Fprintf(w, "    tail     %6d tok (%4.1f%%)\n", fp.Tail.Tokens, fp.Tail.Pct)

	ranked := append([]agent.ToolFootprint(nil), fp.PerTool...)
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].Tokens > ranked[j].Tokens })
	if top <= 0 || top > len(ranked) {
		top = len(ranked)
	}
	if top > 0 {
		fmt.Fprintf(w, "  heaviest tool schemas (top %d):\n", top)
		for _, t := range ranked[:top] {
			fmt.Fprintf(w, "    %6d tok  %s\n", t.Tokens, t.Name)
		}
	}

	if len(a.Findings) == 0 {
		fmt.Fprintf(w, "findings: none — no verifiable session-config waste detected\n")
		if a.DeferReason != "" {
			fmt.Fprintf(w, "  (cold-tool check stood down: %s)\n", a.DeferReason)
		}
		return
	}
	fmt.Fprintf(w, "findings (%d):\n", len(a.Findings))
	for _, f := range a.Findings {
		fmt.Fprintf(w, "  [%s] %s\n", f.ID, f.Detail)
		fmt.Fprintf(w, "      lever: %s\n", f.Lever)
	}
}

// runFootprintAudit loads the request (a captured --req file, else the representative
// Claude-Code-shaped body over the real MCP registry), reads the two session-config
// signals from the live env, and prints the aggregated report.
func runFootprintAudit(out, errw io.Writer, reqFile string, top int, asJSON bool) int {
	raw := []byte(nil)
	source := "representative (Claude-Code-shaped body over the real fak MCP registry)"
	if reqFile != "" {
		b, err := os.ReadFile(reqFile)
		if err != nil {
			fmt.Fprintf(errw, "fak footprint --audit: %v\n", err)
			return 1
		}
		raw, source = b, reqFile
	} else {
		raw = gateway.CanonicalDeferABBody()
	}

	a, err := buildFootprintAudit(raw, source, os.Getenv("ANTHROPIC_BASE_URL"), os.Getenv("ENABLE_TOOL_SEARCH"))
	if err != nil {
		fmt.Fprintf(errw, "fak footprint --audit: %v\n", err)
		return 1
	}
	if asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":          "fak-footprint-audit/1",
			"provenance":      a.Footprint.Provenance,
			"source":          a.Source,
			"footprint":       a.Footprint,
			"cold_tool_count": a.ColdToolCount,
			"defer_reason":    a.DeferReason,
			"findings":        a.Findings,
		})
		return 0
	}
	renderFootprintAudit(out, a, top)
	return 0
}
