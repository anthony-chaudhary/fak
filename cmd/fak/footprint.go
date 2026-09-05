package main

// fak footprint — the systemic MCP tool-schema floor scorecard (issue #3230,
// epic #3229). Every fak_* tool fak's MCP server advertises adds its full JSON
// schema to `tools/list`, and that floor is paid on EVERY turn whether or not the
// tool is ever selected. This verb prices that always-sent floor offline and
// deterministically: it reads the gateway's real tool registry
// (gateway.MCPFloorToolDefs) and prices it through the same estimator the agent
// footprint primitive uses (agent.RequestFootprint, via internal/mcpfootprint), so
// the reported number can never drift from EstimateAnthropicTokens.
//
// It is the MEASUREMENT foundation of the baseline-floor epic: run it before and
// after a deferral change (#3231 cold-schema defer, #3232 the gateway lever) to
// witness the reduction. The committed baseline lives in
// docs/context-budget/mcp-tool-floor.md; the witness test is
// internal/mcpfootprint.TestRealFakMCPFloor.

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/mcpfootprint"
)

func runMCPFootprint(out, errw io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak footprint", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	top := fs.Int("top", 0, "show only the N heaviest tools (0 = all)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	ab := fs.Bool("ab", false, "run the #3532 cold-tool-deferral A/B token-delta scorecard")
	heldAcc := fs.Bool("held-accuracy", false, "run the #3533 cold-tool-deferral held-accuracy fault-in-recall eval")
	audit := fs.Bool("audit", false, "run the #5050 session-config token-bloat audit (floor split + verifiable config wastes + the fak lever for each)")
	reqFile := fs.String("req", "", "audit: path to a captured Anthropic Messages request JSON body (default: the representative Claude-Code-shaped body)")
	doc := fs.String("doc", "", "price a markdown doc's INSTRUCTION-PULLED floor per section (#5445, schema fak-doc-footprint/1)")
	workspace := fs.Bool("workspace", false, "audit whole-workspace turn-0 context floor and verify 3x reduction")
	flagArgs, _ := partitionArgs(argv, map[string]bool{"top": true, "req": true, "doc": true})
	if err := fs.Parse(flagArgs); err != nil {
		fmt.Fprintln(errw, err)
		footprintUsage(errw)
		return 2
	}

	if *workspace {
		return runFootprintWorkspace(out, errw, *asJSON)
	}
	if *doc != "" {
		return runFootprintDoc(out, errw, *doc, *top, *asJSON)
	}
	if *audit {
		return runFootprintAudit(out, errw, *reqFile, *top, *asJSON)
	}
	if *ab {
		return runFootprintAB(out, errw, *asJSON)
	}
	if *heldAcc {
		return runFootprintHeldAccuracy(out, errw, *asJSON)
	}

	fp := mcpfootprint.Price(gateway.MCPFloorToolDefs())
	ranked := mcpfootprint.PerToolSorted(fp)

	limit := *top
	if limit <= 0 || limit > len(ranked) {
		limit = len(ranked)
	}

	if *asJSON {
		perTool := make([]map[string]any, 0, limit)
		for _, t := range ranked[:limit] {
			perTool = append(perTool, map[string]any{
				"name":   t.Name,
				"bytes":  t.Bytes,
				"tokens": t.Tokens,
			})
		}
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":       "fak-mcp-footprint/1",
			"provenance":   fp.Provenance,
			"tool_count":   fp.ToolCount,
			"floor_bytes":  fp.Tools.Bytes,
			"floor_tokens": fp.Tools.Tokens,
			"shown":        limit,
			"per_tool":     perTool,
		})
		return 0
	}

	// Human view: for the full listing reuse the package renderer; for a --top N
	// slice print the same header then only the heaviest N (with a tail note).
	if limit == len(ranked) {
		fmt.Fprint(out, mcpfootprint.Render(fp))
		return 0
	}
	fmt.Fprintf(out, "mcp-footprint: %d tools · floor %d est. tokens (%d bytes, %s)\n",
		fp.ToolCount, fp.Tools.Tokens, fp.Tools.Bytes, fp.Provenance)
	for _, t := range ranked[:limit] {
		fmt.Fprintf(out, "  %6d tok  %7d B  %s\n", t.Tokens, t.Bytes, t.Name)
	}
	fmt.Fprintf(out, "  … and %d more tool(s) not shown (--top %d)\n", len(ranked)-limit, limit)
	return 0
}

func footprintUsage(w io.Writer) {
	fmt.Fprint(w, `usage: fak footprint [--top N] [--json]

Price fak's always-sent MCP tool-schema floor — the fixed per-turn token tax every
registered fak_* tool adds to tools/list, paid whether or not the tool is called.
Deterministic and offline (no model call): the number is the same estimator the
agent request footprint uses, so it never drifts from EstimateAnthropicTokens.

  --top N   show only the N heaviest tools (0 = all)
  --json    emit machine-readable JSON (schema fak-mcp-footprint/1)
  --ab      cold-tool-deferral A/B scorecard (#3532, schema fak-footprint-ab/1)
  --held-accuracy  cold-tool-deferral fault-in-recall eval (#3533, schema fak-footprint-held-accuracy/1)
  --audit   session-config token-bloat audit (#5050, schema fak-footprint-audit/1):
            the floor split (system/tools/history/tail) over one request plus the
            verifiable config wastes — cold custom tools that could defer (#3232)
            and a custom ANTHROPIC_BASE_URL with ENABLE_TOOL_SEARCH unset (GH#746,
            #5049) — each with the exact fak lever that fixes it. Read-only; every
            number is ESTIMATED (~4 chars/token), never a provider-measured saving.
  --req F   with --audit: audit a captured Anthropic Messages request body from file F
            instead of the representative Claude-Code-shaped body
  --workspace audit the whole-workspace turn-0 context floor (instructions, skills,
            MCP tools, and subagent thinking variants) and verify 3x conservation
  --doc P   price markdown doc P's INSTRUCTION-PULLED floor, per section (#5445,
            schema fak-doc-footprint/1). A doc a resident instruction tells the agent
            to read (CLAUDE.md -> AGENTS.md) is paid as a turn-1 Read, so it lands in
            neither /context nor the tool-schema floor above. Ranks sections
            heaviest-first so a paging-out lever has a cut line. Baseline:
            docs/context-budget/agents-md-floor.md

The measurement foundation of epic #3229: run before/after a deferral change
(#3231, #3232) to witness the reduction. Baseline: docs/context-budget/mcp-tool-floor.md.

--ab prices the #3232 deferral lever: the PROVIDER-resident tool-slice tokens ARMED
(cold defs deferred) vs ABLATED (all defs resident). The delta is ESTIMATED (house
tokenizer on the resident slice) — defer_loading GROWS request bytes, it does not
shrink them; the OBSERVED provider-side reduction is a live run (usage relay +
fak_gateway_tool_defer_* /metrics, #3233/#3536).
`)
}
