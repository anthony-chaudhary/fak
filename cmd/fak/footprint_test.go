package main

import (
	"bytes"
	"io"
	"testing"
)

// TestMCPFootprintVerbJSON witnesses the `fak footprint --json` contract: the
// verb prices the real MCP registry, the floor is a faithful partition of the
// per-tool bytes (floor_bytes == sum of shown per-tool bytes when nothing is
// truncated), and the per-tool list is largest-first. If the estimator or the
// registry export drifts, this fails — it is the CLI-level twin of
// internal/mcpfootprint.TestRealFakMCPFloor (#3230, epic #3229).
func TestMCPFootprintVerbJSON(t *testing.T) {
	var out bytes.Buffer
	if code := runMCPFootprint(&out, io.Discard, []string{"--json"}); code != 0 {
		t.Fatalf("runMCPFootprint exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())

	if m["schema"] != "fak-mcp-footprint/1" {
		t.Fatalf("schema=%v, want fak-mcp-footprint/1", m["schema"])
	}
	toolCount := m["tool_count"].(float64)
	if toolCount <= 0 {
		t.Fatalf("tool_count=%v, want >0 (registry not seen?)", toolCount)
	}
	floor := m["floor_bytes"].(float64)
	if floor <= 0 {
		t.Fatalf("floor_bytes=%v, want >0", floor)
	}
	if m["floor_tokens"].(float64) <= 0 {
		t.Fatalf("floor_tokens=%v, want >0", m["floor_tokens"])
	}

	perTool := m["per_tool"].([]any)
	if len(perTool) != int(toolCount) {
		t.Fatalf("per_tool len=%d, want tool_count=%d (default --top 0 shows all)", len(perTool), int(toolCount))
	}

	// Faithful partition: with all tools shown, the per-tool bytes sum to the floor.
	sum := 0.0
	prev := -1.0
	for i, e := range perTool {
		row := e.(map[string]any)
		b := row["bytes"].(float64)
		sum += b
		if prev >= 0 && b > prev {
			t.Fatalf("per_tool not largest-first at %d: %v > %v", i, b, prev)
		}
		prev = b
	}
	if sum != floor {
		t.Fatalf("floor_bytes %v != sum of per-tool bytes %v (not a faithful partition)", floor, sum)
	}
}

// TestMCPFootprintVerbTop witnesses --top N truncates the per-tool list to the N
// heaviest without changing the reported floor totals (the floor is the whole
// registry, not just the shown slice).
func TestMCPFootprintVerbTop(t *testing.T) {
	var full bytes.Buffer
	if code := runMCPFootprint(&full, io.Discard, []string{"--json"}); code != 0 {
		t.Fatalf("full exit=%d", code)
	}
	fm := decodeJSON(t, full.Bytes())
	total := int(fm["tool_count"].(float64))
	if total < 3 {
		t.Skipf("registry has only %d tools; --top test needs >=3", total)
	}

	var out bytes.Buffer
	if code := runMCPFootprint(&out, io.Discard, []string{"--top", "3", "--json"}); code != 0 {
		t.Fatalf("runMCPFootprint --top 3 exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())
	if got := m["per_tool"].([]any); len(got) != 3 {
		t.Fatalf("per_tool len=%d, want 3 (--top 3)", len(got))
	}
	// Totals are the whole floor regardless of truncation.
	if m["floor_bytes"] != fm["floor_bytes"] || m["tool_count"] != fm["tool_count"] {
		t.Fatalf("--top changed the floor totals: %v/%v vs %v/%v",
			m["floor_bytes"], m["tool_count"], fm["floor_bytes"], fm["tool_count"])
	}
}
