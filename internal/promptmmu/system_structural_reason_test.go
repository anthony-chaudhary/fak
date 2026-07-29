package promptmmu

import "testing"

// system_structural_reason_test.go — the #5446 half-A witness. CompactInboundSystem used to
// answer "there was no system[] array, or it was a bare string" for BOTH a system[] whose
// elements could not be read and one that was simply empty. The benign bucket is expected to
// be large and to grow, so a reader fault filed there is invisible: it reads as a quiet drop
// in prune rate rather than as an error. Its documented twin CompactInboundTools has split
// exactly this pair since it was written; the system side had not.
//
// The structural arm inside CompactInboundSystem itself is close to unreachable by
// construction (decodeCurateInput already validated the whole document before
// decodeArrayElements runs), the same posture #5442 recorded for ArrayUndecodable. What
// these tests pin is therefore the two things that ARE load-bearing: the benign shapes stay
// benign, and the new reason is not inert — it changes the mask-vs-remove verdict, and it is
// classified in the one place the closed set is partitioned. The REACHABLE structural
// failure on this path lives in the gateway reader one layer up and is witnessed there.

// TestSystemBenignShapesStayBenign: every legitimate wire shape of `system` that yields no
// prune must keep the benign reason and must NOT be classified structural. `system: null` is
// the specific trap #5442 named — json.Unmarshal of JSON null into a slice SUCCEEDS, so a
// re-probe that reaches for a decoder instead of testing the value's shape mislabels it.
func TestSystemBenignShapesStayBenign(t *testing.T) {
	plan := BlockPlan{Block: BlockSkills, Drop: map[string]bool{"old_skill": true}}
	for name, raw := range map[string][]byte{
		"absent":      []byte(`{"model":"m","messages":[]}`),
		"bare-string": []byte(`{"model":"m","system":"just a string system"}`),
		"json-null":   []byte(`{"model":"m","system":null}`),
		"empty-array": []byte(`{"model":"m","system":[]}`),
	} {
		res := CompactInboundSystem(raw, plan, okDecode)
		if res.Changed {
			t.Fatalf("%s: expected fail-safe identity, got a splice", name)
		}
		if res.SkipReason != SkipNoSystem {
			t.Errorf("%s: SkipReason = %q, want %q", name, res.SkipReason, SkipNoSystem)
		}
		if SkipReasonIsStructural(res.SkipReason) {
			t.Errorf("%s: a legitimate wire shape must never be classified structural", name)
		}
	}
}

// TestUndecodableSystemIsADistinctStructuralReason: the new reason must be its own member of
// the closed set — not an alias of the benign one — and must land on the structural side of
// the single partition, next to its tools twin.
func TestUndecodableSystemIsADistinctStructuralReason(t *testing.T) {
	if SkipUndecodableSystem == SkipNoSystem {
		t.Fatalf("SkipUndecodableSystem must not alias SkipNoSystem; both = %q", SkipNoSystem)
	}
	if !SkipReasonIsStructural(SkipUndecodableSystem) {
		t.Errorf("SkipReasonIsStructural(%q) = false, want true — an unreadable system[] is a fak fault",
			SkipUndecodableSystem)
	}
	if !SkipReasonIsStructural(SkipUndecodableTools) {
		t.Errorf("SkipReasonIsStructural(%q) = false, want true (the tools twin)", SkipUndecodableTools)
	}
	// The other side of the partition: the ordinary idle reasons must stay off it, or the
	// structural signal is worthless (everything would be an alarm).
	for _, benign := range []string{SkipEmptyInput, SkipEmptyPlan, SkipNoTools, SkipNoSystem, SkipNoBreakpoint, SkipNothingAfter} {
		if SkipReasonIsStructural(benign) {
			t.Errorf("SkipReasonIsStructural(%q) = true, want false — it is an ordinary idle reason", benign)
		}
	}
	if SkipReasonIsStructural(ArrayUndecodable) {
		t.Error("the Skip* partition must not answer for the deliberately disjoint Array* vocabulary")
	}
}

// TestUndecodableSystemMasksInsteadOfUnchanged is the pin that half A is NOT inert. The
// strategy explainer sends every "not safe to cut" reason to the defensive ToolSchemaMask;
// with no system-side structural reason in that arm, an unreadable system[] fell through to
// ToolSchemaUnchanged — the "nothing needed doing" verdict, which is the opposite of what the
// identical failure earns on the tools side.
func TestUndecodableSystemMasksInsteadOfUnchanged(t *testing.T) {
	got := ExplainToolSchemaStrategy(PruneResult{SkipReason: SkipUndecodableSystem})
	if got.Strategy != ToolSchemaMask {
		t.Fatalf("Strategy for %q = %q, want %q — an unreadable system[] means NOT SAFE TO CUT",
			SkipUndecodableSystem, got.Strategy, ToolSchemaMask)
	}
	twin := ExplainToolSchemaStrategy(PruneResult{SkipReason: SkipUndecodableTools})
	if got.Strategy != twin.Strategy {
		t.Errorf("the system and tools read failures earned different strategies (%q vs %q); they are the same event",
			got.Strategy, twin.Strategy)
	}
	if got.SkipReason != SkipUndecodableSystem {
		t.Errorf("SkipReason = %q, want %q carried through verbatim", got.SkipReason, SkipUndecodableSystem)
	}
	benign := ExplainToolSchemaStrategy(PruneResult{SkipReason: SkipNoSystem})
	if benign.Strategy != ToolSchemaUnchanged {
		t.Errorf("Strategy for %q = %q, want %q — an absent system[] needs no defensive posture",
			SkipNoSystem, benign.Strategy, ToolSchemaUnchanged)
	}
	if got.Strategy == benign.Strategy {
		t.Fatalf("an unreadable system[] and an absent one collapsed into one strategy %q", got.Strategy)
	}
}
