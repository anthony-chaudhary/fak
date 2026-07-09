package main

// footprint_ab.go — the #3532 cold-tool-deferral A/B token-delta scorecard, an `--ab`
// mode of `fak footprint` (epic #3229). It builds ONE deterministic Claude-Code-shaped
// Anthropic body (a hot core + the real MCP registry as the cold tail), runs it through
// the PRODUCTION #3232 transform via the gateway export, and prices the ARMED arm (cold
// defs deferred) against the ABLATED arm (identity) with the ONE house estimator.
//
// HONESTY: the reported delta is ESTIMATED — the house tokenizer on the PROVIDER-RESIDENT
// tool slice (non-deferred defs armed vs all defs ablated). defer_loading does NOT shrink
// request bytes; the armed body GROWS, which the scorecard states explicitly so nobody
// re-labels this a byte reduction. The truly OBSERVED provider-side reduction lives only
// in the usage relay + the fak_gateway_tool_defer_* /metrics counters on a live run
// (#3233/#3536), and the JSON carries a pointer field saying so.

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/mcpfootprint"
)

func runFootprintAB(out, errw io.Writer, asJSON bool) int {
	body := gateway.CanonicalDeferABBody()
	arms := gateway.DeferColdToolsAB(body)
	if !arms.Changed {
		// Fail-safe: never emit a fabricated zero-delta row when the lever stood down.
		fmt.Fprintf(errw, "fak footprint --ab: cold-tool deferral did not fire (%s); no A/B to report\n", arms.Reason)
		return 1
	}

	ablatedFp := mcpfootprint.Price(gateway.ResidentToolDefs(arms.Ablated))
	armedFp := mcpfootprint.Price(gateway.ResidentToolDefs(arms.Armed))
	ablatedTok := ablatedFp.Tools.Tokens
	armedTok := armedFp.Tools.Tokens
	deltaTok := ablatedTok - armedTok

	ablatedBytes := len(arms.Ablated)
	armedBytes := len(arms.Armed)
	byteGrowth := armedBytes - ablatedBytes
	cachePrefixStable := gateway.NonToolsByteIdentical(arms.Ablated, arms.Armed)

	const observedNote = "resident-token delta is ESTIMATED (house tokenizer on the provider-resident tool slice); " +
		"the OBSERVED provider-side reduction lives in the usage relay + fak_gateway_tool_defer_* /metrics on a live run (#3233/#3536)"

	if asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":                      "fak-footprint-ab/1",
			"provenance":                  ablatedFp.Provenance, // ESTIMATED
			"cold_deferred":               arms.ColdCount,
			"ablated_resident_tools":      ablatedFp.ToolCount,
			"armed_resident_tools":        armedFp.ToolCount,
			"ablated_resident_tokens":     ablatedTok,
			"armed_resident_tokens":       armedTok,
			"resident_token_delta":        deltaTok,
			"ablated_body_bytes":          ablatedBytes,
			"armed_body_bytes":            armedBytes,
			"request_byte_growth":         byteGrowth,
			"cache_prefix_byte_identical": cachePrefixStable,
			"observed_note":               observedNote,
		})
		return 0
	}

	fmt.Fprintf(out, "defer-ab (#3532, epic #3229) — cold-tool deferral A/B · %s\n", ablatedFp.Provenance)
	fmt.Fprintf(out, "  cold defs deferred (WITNESSED):       %d\n", arms.ColdCount)
	fmt.Fprintf(out, "  resident tool tokens  ablated:        %6d  (%d defs loaded every turn)\n", ablatedTok, ablatedFp.ToolCount)
	fmt.Fprintf(out, "  resident tool tokens  armed:          %6d  (%d defs: hot core + tool_search_tool)\n", armedTok, armedFp.ToolCount)
	fmt.Fprintf(out, "  provider-resident delta (ESTIMATED):  %6d est. tokens faulted-in on demand\n", deltaTok)
	fmt.Fprintf(out, "  request bytes  ablated → armed:       %d → %d  (+%d B — defer_loading keys + tool_search_tool; NOT a byte shrink)\n", ablatedBytes, armedBytes, byteGrowth)
	fmt.Fprintf(out, "  cache prefix byte-identical:          %v\n", cachePrefixStable)
	fmt.Fprintf(out, "  note: %s\n", observedNote)
	return 0
}
