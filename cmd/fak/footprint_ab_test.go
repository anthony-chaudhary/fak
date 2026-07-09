package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// footprint_ab_test.go pins the #3532 A/B scorecard's load-bearing claims: the resident
// tool-slice token delta is positive (armed < ablated), the request bytes GROW (honesty:
// not a byte shrink), the cache prefix is byte-stable, the provenance is ESTIMATED, and
// the cold count equals the whole MCP registry.

func TestFootprintABJSONContract(t *testing.T) {
	var out, errw bytes.Buffer
	if code := runFootprintAB(&out, &errw, true); code != 0 {
		t.Fatalf("runFootprintAB exit=%d, stderr=%s", code, errw.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("--ab --json is not valid JSON: %v\n%s", err, out.String())
	}

	num := func(k string) float64 {
		v, ok := got[k].(float64)
		if !ok {
			t.Fatalf("field %q missing or non-numeric in %v", k, got)
		}
		return v
	}

	if got["schema"] != "fak-footprint-ab/1" {
		t.Fatalf("schema=%v, want fak-footprint-ab/1", got["schema"])
	}
	if got["provenance"] != "ESTIMATED" {
		t.Fatalf("provenance=%v, want ESTIMATED (the number is our tokenizer, not a provider counter)", got["provenance"])
	}

	ablated := num("ablated_resident_tokens")
	armed := num("armed_resident_tokens")
	delta := num("resident_token_delta")
	if armed >= ablated {
		t.Fatalf("armed resident tokens %v must be < ablated %v (deferral removes the cold slice from context)", armed, ablated)
	}
	if delta <= 0 {
		t.Fatalf("resident_token_delta=%v, want > 0", delta)
	}
	if delta != ablated-armed {
		t.Fatalf("delta %v != ablated-armed %v (algebra must close)", delta, ablated-armed)
	}

	// Honesty: the ARMED request is LARGER in bytes — defer_loading is not a byte shrink.
	if num("request_byte_growth") <= 0 {
		t.Fatalf("request_byte_growth=%v, want > 0 (armed body grows: defer_loading keys + search tool)", got["request_byte_growth"])
	}
	if num("armed_body_bytes") <= num("ablated_body_bytes") {
		t.Fatalf("armed body must be larger than ablated in bytes")
	}

	// Cache-safety invariant recomputed independently from the gateway export.
	if got["cache_prefix_byte_identical"] != true {
		t.Fatalf("cache_prefix_byte_identical=%v, want true", got["cache_prefix_byte_identical"])
	}

	if want := float64(len(gateway.MCPFloorToolDefs())); num("cold_deferred") != want {
		t.Fatalf("cold_deferred=%v, want %v (every MCP def is cold)", got["cold_deferred"], want)
	}
	if num("armed_resident_tools") != 3 {
		t.Fatalf("armed_resident_tools=%v, want 3 (Read, Bash, tool_search_tool)", got["armed_resident_tools"])
	}
}

func TestFootprintABHumanRow(t *testing.T) {
	var out, errw bytes.Buffer
	if code := runFootprintAB(&out, &errw, false); code != 0 {
		t.Fatalf("runFootprintAB exit=%d", code)
	}
	s := out.String()
	for _, want := range []string{"defer-ab", "ESTIMATED", "cold defs deferred", "NOT a byte shrink"} {
		if !strings.Contains(s, want) {
			t.Fatalf("human row missing %q:\n%s", want, s)
		}
	}
}

func TestFootprintABViaVerb(t *testing.T) {
	// the --ab flag must route through the existing `fak footprint` verb.
	var out, errw bytes.Buffer
	if code := runMCPFootprint(&out, &errw, []string{"--ab", "--json"}); code != 0 {
		t.Fatalf("fak footprint --ab --json exit=%d, stderr=%s", code, errw.String())
	}
	if !strings.Contains(out.String(), "fak-footprint-ab/1") {
		t.Fatalf("verb did not route to the A/B scorecard:\n%s", out.String())
	}
}
