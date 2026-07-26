package policy

import "testing"

// The apply pipeline's scout phase drives the Playwright MCP browser through
// `mcp__playwright__browser_snapshot` (the accessibility-tree read). Before this
// grant landed, the embedded guard floor listed the `mcp__dos__*` and
// `mcp__fak__*` tools but NOT a single `mcp__playwright__browser_*` identity, so
// every browser call fell to DEFAULT_DENY. A scout that could not snapshot
// enumerated 0 form fields, which the apply classifier bucketed as
// `scout_zero_sections` structural-doom (6h cooldown + retirement) — a guard
// false-positive that walked healthy jobs toward permanent skip. This test locks
// the browser family into the floor so the grant can never be silently dropped.
//
// The floor grants the capability; the apply harness narrows per phase (its
// read-only phases deny the write/interactive browser tools in the SDK layer).
// That layering mirrors how the floor grants Bash/Write broadly while the harness
// restricts — so asserting the presence of the whole family here is correct.
func TestGuardFloorGrantsPlaywrightBrowserFamily(t *testing.T) {
	rt, err := ParseRuntime(realFloor(t))
	if err != nil {
		t.Fatalf("ParseRuntime(embedded floor): %v", err)
	}
	allow := rt.Adjudicator.Allow

	// The read tool whose DEFAULT_DENY broke every apply — the headline regression.
	if !allow["mcp__playwright__browser_snapshot"] {
		t.Fatalf("embedded guard floor does not grant mcp__playwright__browser_snapshot; " +
			"a scout under this floor cannot read the accessibility tree and every apply " +
			"regresses to scout_zero_sections structural-doom (the guard false-positive)")
	}

	// The rest of the family the apply pipeline drives across scout/fill/submit.
	// Read-only first (scout), then the write/interactive set (fill + submit).
	for _, tool := range []string{
		"mcp__playwright__browser_navigate",
		"mcp__playwright__browser_take_screenshot",
		"mcp__playwright__browser_tabs",
		"mcp__playwright__browser_click",
		"mcp__playwright__browser_type",
		"mcp__playwright__browser_fill_form",
		"mcp__playwright__browser_select_option",
		"mcp__playwright__browser_file_upload",
		"mcp__playwright__browser_press_key",
	} {
		if !allow[tool] {
			t.Errorf("embedded guard floor does not grant %s; the apply pipeline drives it "+
				"and it would fall to DEFAULT_DENY", tool)
		}
	}
}
