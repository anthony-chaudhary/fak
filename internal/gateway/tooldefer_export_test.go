package gateway

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// tooldefer_export_test.go witnesses the #3532 A/B export contract: the canonical body
// decodes and carries the whole registry, the REAL transform fires over it, the ablated
// arm is the verbatim input, the resident-def partition is exact, and everything outside
// tools[] is byte-identical across the arms (the cache-safety invariant).

func TestCanonicalDeferABBodyDecodesAndCarriesRegistry(t *testing.T) {
	body := CanonicalDeferABBody()
	req, err := agent.DecodeAnthropicMessagesRequest(body)
	if err != nil {
		t.Fatalf("canonical body does not decode: %v", err)
	}
	wantTools := len(MCPFloorToolDefs()) + 2 // + Read + Bash hot core
	if len(req.Tools) != wantTools {
		t.Fatalf("canonical body tools=%d, want %d (registry + 2 hot)", len(req.Tools), wantTools)
	}
	// last element must carry the cache_control anchor the transform re-anchors from.
	elems := toolsOf(t, body)
	last := elems[len(elems)-1]
	if _, ok := last["cache_control"]; !ok {
		t.Fatalf("canonical body's last tool lacks cache_control anchor")
	}
}

func TestDeferColdToolsABFiresOverRegistry(t *testing.T) {
	body := CanonicalDeferABBody()
	arms := DeferColdToolsAB(body)
	if !arms.Changed {
		t.Fatalf("expected the transform to fire, got identity (reason=%q)", arms.Reason)
	}
	if want := len(MCPFloorToolDefs()); arms.ColdCount != want {
		t.Fatalf("ColdCount=%d, want %d (every MCP def is cold — none in the hot core)", arms.ColdCount, want)
	}
	if !bytes.Equal(arms.Ablated, body) {
		t.Fatalf("ablated arm must be the verbatim input body")
	}
	if bytes.Equal(arms.Armed, arms.Ablated) {
		t.Fatalf("armed arm must differ from ablated (defer_loading + injected search tool)")
	}
	// the armed body GROWS — defer_loading keys + the search tool are net-added bytes.
	if len(arms.Armed) <= len(arms.Ablated) {
		t.Fatalf("armed bytes %d must exceed ablated %d (defer_loading is not a byte shrink)", len(arms.Armed), len(arms.Ablated))
	}
}

func TestResidentToolDefsPartition(t *testing.T) {
	arms := DeferColdToolsAB(CanonicalDeferABBody())

	ablated := ResidentToolDefs(arms.Ablated)
	if want := len(MCPFloorToolDefs()) + 2; len(ablated) != want {
		t.Fatalf("ablated resident defs=%d, want %d (all defs resident)", len(ablated), want)
	}

	armed := ResidentToolDefs(arms.Armed)
	got := map[string]bool{}
	for _, d := range armed {
		got[d.Function.Name] = true
	}
	want := map[string]bool{"Read": true, "Bash": true, toolSearchToolName: true}
	if len(got) != len(want) {
		t.Fatalf("armed resident defs=%v, want exactly %v (hot core + tool_search_tool)", got, want)
	}
	for name := range want {
		if !got[name] {
			t.Fatalf("armed resident missing %q; got %v", name, got)
		}
	}
}

func TestNonToolsByteIdenticalHoldsAcrossArms(t *testing.T) {
	arms := DeferColdToolsAB(CanonicalDeferABBody())
	if !NonToolsByteIdentical(arms.Ablated, arms.Armed) {
		t.Fatalf("non-tools body bytes must be identical across arms (cache-prefix invariant)")
	}
	// negative control: perturb a non-tools byte and the invariant must break.
	perturbed := bytes.Replace(arms.Ablated, []byte(`"SYS-PROMPT"`), []byte(`"SYS-PROMPTX"`), 1)
	if NonToolsByteIdentical(perturbed, arms.Armed) {
		t.Fatalf("perturbing the system prompt must break non-tools byte-identity")
	}
}

func TestDeferColdToolsABDeterministic(t *testing.T) {
	body := CanonicalDeferABBody()
	a := DeferColdToolsAB(body)
	b := DeferColdToolsAB(body)
	if !bytes.Equal(a.Armed, b.Armed) {
		t.Fatalf("armed arm must be deterministic turn-over-turn (cache stability)")
	}
}

func TestDeferColdToolsABStandDownIsIdentity(t *testing.T) {
	// an already-deferred body (carries a tool_search_tool) must stand down: identity
	// with a named reason, never a double-apply.
	arms := DeferColdToolsAB(DeferColdToolsAB(CanonicalDeferABBody()).Armed)
	if arms.Changed {
		t.Fatalf("re-arming an already-deferred body must stand down, got Changed")
	}
	if arms.Reason == "" {
		t.Fatalf("stand-down must carry a fail-safe reason")
	}
	if !bytes.Equal(arms.Armed, arms.Ablated) {
		t.Fatalf("stand-down must be identity (armed == ablated)")
	}
}
