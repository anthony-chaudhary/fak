package metrics

import "fmt"

// FloorFootprint is the plain-value slice of a request footprint the guard
// per-turn exit summary needs to render the always-sent floor (#3531, epic
// #3229). It mirrors the System/Tools partition of internal/agent's
// RequestFootprint (Floor = System + Tools) plus the deferrable cold-tool tail,
// carried as token counts so this metrics-lane renderer stays free of any
// upward gateway/agent import — an import the other way would red architest
// (ARCH_LAYER_VIOLATION). The gateway (which already observes the floor into
// fak_context_value, #3233) fills this in from its footprint before rendering.
type FloorFootprint struct {
	// SystemTokens is the always-sent system-prompt token cost.
	SystemTokens int
	// ToolsTokens is the always-sent tool-definition token cost. Together with
	// SystemTokens it is the floor: the tokens every turn pays before any history.
	ToolsTokens int
	// ColdToolTokens is the token cost of the cold (deferrable) tail of
	// ToolsTokens — the provider-side reduction cold-tool deferral drives when it
	// fires. A subset of ToolsTokens, so 0 <= ColdToolTokens <= ToolsTokens.
	ColdToolTokens int
}

// FloorDeferFragment renders the compact one-line floor + tool-defer fragment
// for the guard per-turn exit summary (#3531): "floor=<sys>+<tools> defer=<n>
// Δtools=<tokens>".
//
// deferCold is the WITNESSED count of cold tool definitions fak marked
// defer_loading this turn (0 when the lever is off — its default — or when every
// advertised tool was hot). Δtools is the OBSERVED provider-side token reduction
// the deferral drove: the cold-tool slice of the floor when deferral fired, and
// 0 when it did not (defer=0 => Δtools=0), so the fragment never claims a saving
// the lever did not take, even when the footprint still carries a cold tail.
//
// Pure: no I/O, no network — the same footprint + count always renders the same
// bytes, which is what makes the golden-line test deterministic. Negative inputs
// (never expected from a real footprint) are floored to 0 so the fragment can
// never render a nonsense "defer=-1".
func FloorDeferFragment(fp FloorFootprint, deferCold int) string {
	sys, tools := fp.SystemTokens, fp.ToolsTokens
	if sys < 0 {
		sys = 0
	}
	if tools < 0 {
		tools = 0
	}
	if deferCold < 0 {
		deferCold = 0
	}
	delta := 0
	if deferCold > 0 && fp.ColdToolTokens > 0 {
		delta = fp.ColdToolTokens
	}
	return fmt.Sprintf("floor=%d+%d defer=%d Δtools=%d", sys, tools, deferCold, delta)
}
