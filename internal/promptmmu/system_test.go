package promptmmu

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"
)

func systemBlock(name, block, text string, withCC bool) map[string]any {
	b := map[string]any{
		"type":  "text",
		"name":  name,
		"block": block,
		"text":  text,
	}
	if withCC {
		b["cache_control"] = map[string]any{"type": "ephemeral"}
	}
	return b
}

func systemBody(tb testing.TB, blocks []map[string]any) []byte {
	tb.Helper()
	raw, err := json.Marshal(map[string]any{
		"model":      "claude-test",
		"max_tokens": 64,
		"system":     toAnySlice(blocks),
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
	})
	if err != nil {
		tb.Fatalf("marshal body: %v", err)
	}
	return raw
}

func systemNamesIn(tb testing.TB, raw []byte) []string {
	tb.Helper()
	var obj struct {
		System []struct {
			Name string `json:"name"`
		} `json:"system"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		tb.Fatalf("decode system: %v", err)
	}
	out := make([]string, len(obj.System))
	for i, b := range obj.System {
		out[i] = b.Name
	}
	return out
}

// TestCompactInboundSystem_DropsNamedPostBreakpointBlock proves the #758 rung-6
// splice: a named skills block strictly after the system cache_control breakpoint
// can be dropped while the cached system prefix stays byte-identical.
func TestCompactInboundSystem_DropsNamedPostBreakpointBlock(t *testing.T) {
	raw := systemBody(t, []map[string]any{
		systemBlock("core", BlockSystem, "resident spine", false),
		systemBlock("policy", BlockSystem, "resident policy", true),
		systemBlock("current_skill", BlockSkills, "fresh skill", false),
		systemBlock("old_skill", BlockSkills, "stale skill", false),
		systemBlock("stale_memory", BlockMemory, "over budget", false),
	})
	_, prefixEnd, _, ok := ArraySplicePoints(raw, "system")
	if !ok {
		t.Fatal("fixture must carry a system cache_control breakpoint")
	}

	res := CompactInboundSystem(raw, BlockPlan{
		Block: BlockSkills,
		Drop:  map[string]bool{"old_skill": true, "stale_memory": true},
	}, okDecode)
	if !res.Changed {
		t.Fatalf("expected a system prune, got identity (%q)", res.SkipReason)
	}
	if len(res.Pruned) != 1 || res.Pruned[0] != "old_skill" {
		t.Fatalf("Pruned = %v, want [old_skill]", res.Pruned)
	}
	if prefixEnd > len(res.Body) || !bytes.Equal(raw[:prefixEnd], res.Body[:prefixEnd]) {
		t.Fatalf("system cache prefix bytes changed")
	}
	names := systemNamesIn(t, res.Body)
	if contains(names, "old_skill") {
		t.Fatalf("old_skill should be gone, system names = %v", names)
	}
	for _, want := range []string{"core", "policy", "current_skill", "stale_memory"} {
		if !contains(names, want) {
			t.Fatalf("%s should survive, system names = %v", want, names)
		}
	}
}

// TestCompactInboundSystem_RefusesPreBreakpointDrop proves the system pruner does
// not move the cache boundary: a named resident block at/before the last system
// breakpoint is not dropped.
func TestCompactInboundSystem_RefusesPreBreakpointDrop(t *testing.T) {
	raw := systemBody(t, []map[string]any{
		systemBlock("core", BlockSystem, "resident spine", false),
		systemBlock("policy", BlockSystem, "resident policy", true),
		systemBlock("overlay", BlockSystem, "tail", false),
	})
	res := CompactInboundSystem(raw, BlockPlan{Block: BlockSystem, Drop: map[string]bool{"core": true, "policy": true}}, okDecode)
	if res.Changed {
		t.Fatalf("pre-breakpoint system blocks must not be pruned, pruned=%v", res.Pruned)
	}
	if res.SkipReason != SkipNothingAfter {
		t.Fatalf("SkipReason = %q, want %q", res.SkipReason, SkipNothingAfter)
	}
	if &res.Body[0] != &raw[0] {
		t.Fatalf("identity must return the SAME backing slice")
	}
}

func TestCompactInboundSystem_NoBreakpointIsIdentity(t *testing.T) {
	raw := systemBody(t, []map[string]any{
		systemBlock("core", BlockSystem, "resident spine", false),
		systemBlock("old_skill", BlockSkills, "stale skill", false),
	})
	res := CompactInboundSystem(raw, BlockPlan{Block: BlockSkills, Drop: map[string]bool{"old_skill": true}}, okDecode)
	if res.Changed {
		t.Fatalf("no system breakpoint must be identity, pruned=%v", res.Pruned)
	}
	if res.SkipReason != SkipNoBreakpoint {
		t.Fatalf("SkipReason = %q, want %q", res.SkipReason, SkipNoBreakpoint)
	}
}

func TestCompactInboundSystem_BareStringIsIdentity(t *testing.T) {
	raw := []byte(`{"model":"m","system":"plain system","messages":[]}`)
	res := CompactInboundSystem(raw, BlockPlan{Block: BlockSystem, Drop: map[string]bool{"plain": true}}, okDecode)
	if res.Changed {
		t.Fatalf("bare-string system prompt must be identity, pruned=%v", res.Pruned)
	}
	if res.SkipReason != SkipNoSystem {
		t.Fatalf("SkipReason = %q, want %q", res.SkipReason, SkipNoSystem)
	}
}

func TestCompactInboundSystem_ReDecodeFailureIsSafe(t *testing.T) {
	raw := systemBody(t, []map[string]any{
		systemBlock("policy", BlockSystem, "resident policy", true),
		systemBlock("old_skill", BlockSkills, "stale skill", false),
	})
	res := CompactInboundSystem(raw, BlockPlan{Block: BlockSkills, Drop: map[string]bool{"old_skill": true}},
		func([]byte) error { return errors.New("decode failed") })
	if res.Changed {
		t.Fatalf("a failing decode must force identity, pruned=%v", res.Pruned)
	}
	if res.SkipReason != SkipSpliceUnproven {
		t.Fatalf("SkipReason = %q, want %q", res.SkipReason, SkipSpliceUnproven)
	}
}
