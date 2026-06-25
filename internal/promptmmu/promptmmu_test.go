package promptmmu

import (
	"encoding/json"
	"testing"
)

// tool builds one tools[] element. withCC marks it with a cache_control breakpoint.
func tool(name string, withCC bool) map[string]any {
	t := map[string]any{
		"name":         name,
		"description":  "does " + name,
		"input_schema": map[string]any{"type": "object"},
	}
	if withCC {
		t["cache_control"] = map[string]any{"type": "ephemeral"}
	}
	return t
}

// body builds a minimal but well-formed Anthropic /v1/messages body with the
// given tools and an optional cache_control on the system block.
func body(tb testing.TB, tools []map[string]any, sysCC bool) []byte {
	tb.Helper()
	var sys any = "you are a test agent"
	if sysCC {
		sys = []any{map[string]any{"type": "text", "text": "you are a test agent", "cache_control": map[string]any{"type": "ephemeral"}}}
	}
	obj := map[string]any{
		"model":      "claude-test",
		"max_tokens": 64,
		"system":     sys,
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
		"tools": toAnySlice(tools),
	}
	b, err := json.Marshal(obj)
	if err != nil {
		tb.Fatalf("marshal body: %v", err)
	}
	return b
}

func toAnySlice(ts []map[string]any) []any {
	out := make([]any, len(ts))
	for i, t := range ts {
		out[i] = t
	}
	return out
}

// okDecode is the test stand-in for agent.DecodeAnthropicMessagesRequest: a body
// that unmarshals to a JSON object is "valid enough" for the spine's re-check.
func okDecode(b []byte) error {
	var m map[string]json.RawMessage
	return json.Unmarshal(b, &m)
}

func drop(names ...string) ToolPlan {
	m := map[string]bool{}
	for _, n := range names {
		m[n] = true
	}
	return ToolPlan{Drop: m}
}

func toolNamesIn(tb testing.TB, raw []byte) []string {
	tb.Helper()
	var obj struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		tb.Fatalf("decode tools: %v", err)
	}
	out := make([]string, len(obj.Tools))
	for i, t := range obj.Tools {
		out[i] = t.Name
	}
	return out
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// TestPrune_DropsPostBreakpointTool_PrefixByteIdentical is the happy path AND the
// cache-prefix invariant in one: a tool strictly AFTER the breakpoint is dropped,
// and the protected prefix bytes are byte-identical to the input.
func TestPrune_DropsPostBreakpointTool_PrefixByteIdentical(t *testing.T) {
	// breakpoint on tool[0]; tool[2] is droppable (strictly after).
	raw := body(t, []map[string]any{tool("alpha", true), tool("beta", false), tool("gamma", false)}, false)
	res := CompactInboundTools(raw, drop("gamma"), okDecode)

	if !res.Changed {
		t.Fatalf("expected a prune, got identity (reason=%q)", res.SkipReason)
	}
	if len(res.Pruned) != 1 || res.Pruned[0] != "gamma" {
		t.Fatalf("Pruned = %v, want [gamma]", res.Pruned)
	}
	names := toolNamesIn(t, res.Body)
	if contains(names, "gamma") {
		t.Fatalf("gamma should be gone, tools = %v", names)
	}
	if !contains(names, "alpha") || !contains(names, "beta") {
		t.Fatalf("alpha+beta must survive, tools = %v", names)
	}
	if err := okDecode(res.Body); err != nil {
		t.Fatalf("result must re-decode: %v", err)
	}
	// The protected prefix (through tool[0]'s breakpoint, which precedes beta)
	// must be byte-identical to the original up to the first un-protected region.
	cut := indexOf(raw, []byte(`"beta"`))
	if cut <= 0 {
		t.Fatalf("fixture missing beta marker")
	}
	if string(raw[:cut]) != string(res.Body[:cut]) {
		t.Fatalf("protected prefix bytes diverged before the dropped region")
	}
}

// TestPrune_RefusesPreBreakpointTool_Identity proves the spine NEVER moves the
// breakpoint: a drop targeting a tool at/before the last breakpoint is refused.
func TestPrune_RefusesPreBreakpointTool_Identity(t *testing.T) {
	// breakpoint on the LAST tool (Claude Code's real shape) — nothing strictly after it.
	raw := body(t, []map[string]any{tool("alpha", false), tool("beta", false), tool("gamma", true)}, false)
	res := CompactInboundTools(raw, drop("alpha", "beta"), okDecode)

	if res.Changed {
		t.Fatalf("expected identity (pre-breakpoint drop must be refused), pruned=%v", res.Pruned)
	}
	if res.SkipReason != SkipNothingAfter {
		t.Fatalf("SkipReason = %q, want %q", res.SkipReason, SkipNothingAfter)
	}
	if &res.Body[0] != &raw[0] {
		t.Fatalf("identity must return the SAME backing slice")
	}
}

// TestPrune_NoBreakpoint_Identity proves the "can't anchor the boundary ⇒ don't
// touch it" floor.
func TestPrune_NoBreakpoint_Identity(t *testing.T) {
	raw := body(t, []map[string]any{tool("alpha", false), tool("beta", false)}, false)
	res := CompactInboundTools(raw, drop("beta"), okDecode)
	if res.Changed {
		t.Fatalf("no breakpoint anywhere must be identity, pruned=%v", res.Pruned)
	}
	if res.SkipReason != SkipNoBreakpoint {
		t.Fatalf("SkipReason = %q, want %q", res.SkipReason, SkipNoBreakpoint)
	}
}

// TestPrune_DegenerateInputs_Identity proves the fail-safe guards, all identity,
// all with named reasons.
func TestPrune_DegenerateInputs_Identity(t *testing.T) {
	valid := body(t, []map[string]any{tool("alpha", true), tool("beta", false)}, false)
	cases := []struct {
		name   string
		raw    []byte
		plan   ToolPlan
		reason string
	}{
		{"empty-plan", valid, ToolPlan{}, SkipEmptyPlan},
		{"empty-input", []byte{}, drop("beta"), SkipEmptyInput},
		{"not-json", []byte("not json at all"), drop("beta"), SkipNotJSONObject},
		{"no-tools", []byte(`{"model":"m","messages":[]}`), drop("beta"), SkipNoTools},
		{"empty-tools", []byte(`{"model":"m","tools":[],"system":"s"}`), drop("beta"), SkipNoTools},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res := CompactInboundTools(c.raw, c.plan, okDecode)
			if res.Changed {
				t.Fatalf("%s: expected identity, got change", c.name)
			}
			if res.SkipReason != c.reason {
				t.Fatalf("%s: SkipReason = %q, want %q", c.name, res.SkipReason, c.reason)
			}
		})
	}
}

// TestPrune_SystemBreakpointOnly_DropsTrailingTool proves the pfxEnd<0 &&
// sysHasCC branch: when only `system` is cached, the whole tools[] head is the
// protected array open and any plan-dropped tool is prunable, with system bytes
// preserved verbatim.
func TestPrune_SystemBreakpointOnly_DropsTrailingTool(t *testing.T) {
	raw := body(t, []map[string]any{tool("alpha", false), tool("beta", false)}, true /*sysCC*/)
	res := CompactInboundTools(raw, drop("beta"), okDecode)
	if !res.Changed {
		t.Fatalf("system-anchored prune expected, got identity (reason=%q)", res.SkipReason)
	}
	if len(res.Pruned) != 1 || res.Pruned[0] != "beta" {
		t.Fatalf("Pruned = %v, want [beta]", res.Pruned)
	}
	names := toolNamesIn(t, res.Body)
	if contains(names, "beta") || !contains(names, "alpha") {
		t.Fatalf("want [alpha] only, got %v", names)
	}
}

// TestPrune_SpliceProvenByDecode proves the final re-decode guard is load-bearing:
// a decode callback that always errors forces fail-safe identity even on an
// otherwise-prunable body.
func TestPrune_SpliceProvenByDecode(t *testing.T) {
	raw := body(t, []map[string]any{tool("alpha", true), tool("beta", false)}, false)
	failDecode := func([]byte) error { return errAlways }
	res := CompactInboundTools(raw, drop("beta"), failDecode)
	if res.Changed {
		t.Fatalf("a failing decode must force identity, pruned=%v", res.Pruned)
	}
	if res.SkipReason != SkipSpliceUnproven {
		t.Fatalf("SkipReason = %q, want %q", res.SkipReason, SkipSpliceUnproven)
	}
	if &res.Body[0] != &raw[0] {
		t.Fatalf("identity must return the SAME backing slice")
	}
}

var errAlways = constErr("forced decode failure")

type constErr string

func (e constErr) Error() string { return string(e) }

func indexOf(hay, needle []byte) int {
	for i := 0; i+len(needle) <= len(hay); i++ {
		if string(hay[i:i+len(needle)]) == string(needle) {
			return i
		}
	}
	return -1
}
