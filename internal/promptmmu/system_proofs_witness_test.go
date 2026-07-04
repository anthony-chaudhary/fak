package promptmmu

import (
	"bytes"
	"testing"
)

// system_proofs_witness_test is the SAFETY witness for promptmmu RUNG 6 (#758, epic
// #751): the invariant-witness discipline #759 established for the tool-def compactor
// (proofs_witness_test.go), extended to CompactInboundSystem — the system[]-block
// generalization that is now wired live into the gateway passthrough hot path
// (internal/gateway/messages.go maybeCompactInboundSystem). The epic calls the system
// block "the most cache-sensitive and behavior-critical surface", so the rung that
// prunes it MOST needs the same proof over a SPREAD of real-shaped bodies, not just a
// single happy-path fixture (system_test.go). Same three invariants as the tool path:
//
//   inv 3 - NAMED: every block removed appears in PruneResult.Pruned, and nothing not in
//           Pruned vanishes (no silent disappearance).
//   inv 4 - REVERSIBLE: re-running with an empty Drop reproduces the input bit-for-bit.
//   inv 5 - KERNEL-VIEW BYTE-UNCHANGED: the cached system prefix (through the last
//           system cache_control) is byte-identical after any prune; a prune only
//           deletes whole post-breakpoint blocks and never reorders or renames a
//           survivor, and it is scoped to the plan's own Block.

// systemWitnessBodies returns a spread of well-formed bodies exercising the real
// system-prune regimes: a breakpoint mid-list with a droppable block after it, multiple
// droppable blocks in one target block, a memory-block drop, no breakpoint at all
// (identity), a named-but-pre-breakpoint drop (refused), a plan naming a name that is
// absent, and a plan whose Block does not match the block the name lives in (block
// scoping — a name in a DIFFERENT block is never dropped).
func systemWitnessBodies(tb testing.TB) []struct {
	name string
	raw  []byte
	plan BlockPlan
} {
	tb.Helper()
	return []struct {
		name string
		raw  []byte
		plan BlockPlan
	}{
		{
			name: "breakpoint-mid-system-drop-skill-after",
			raw: systemBody(tb, []map[string]any{
				systemBlock("core", BlockSystem, "resident spine", false),
				systemBlock("policy", BlockSystem, "resident policy", true),
				systemBlock("current_skill", BlockSkills, "fresh skill", false),
				systemBlock("old_skill", BlockSkills, "stale skill", false),
			}),
			plan: BlockPlan{Block: BlockSkills, Drop: map[string]bool{"old_skill": true}},
		},
		{
			name: "drop-multiple-in-block-after-keeps-middle",
			raw: systemBody(tb, []map[string]any{
				systemBlock("policy", BlockSystem, "resident policy", true),
				systemBlock("skill_a", BlockSkills, "stale a", false),
				systemBlock("keep_b", BlockSkills, "fresh b", false),
				systemBlock("skill_c", BlockSkills, "stale c", false),
			}),
			plan: BlockPlan{Block: BlockSkills, Drop: map[string]bool{"skill_a": true, "skill_c": true}},
		},
		{
			name: "memory-block-drop-after",
			raw: systemBody(tb, []map[string]any{
				systemBlock("policy", BlockSystem, "resident policy", true),
				systemBlock("keep_skill", BlockSkills, "fresh skill", false),
				systemBlock("stale_memory", BlockMemory, "over budget", false),
			}),
			plan: BlockPlan{Block: BlockMemory, Drop: map[string]bool{"stale_memory": true}},
		},
		{
			name: "no-breakpoint-anywhere",
			raw: systemBody(tb, []map[string]any{
				systemBlock("core", BlockSystem, "resident spine", false),
				systemBlock("old_skill", BlockSkills, "stale skill", false),
			}),
			plan: BlockPlan{Block: BlockSkills, Drop: map[string]bool{"old_skill": true}},
		},
		{
			name: "pre-breakpoint-drop-refused",
			raw: systemBody(tb, []map[string]any{
				systemBlock("core", BlockSystem, "resident spine", false),
				systemBlock("policy", BlockSystem, "resident policy", true),
				systemBlock("overlay", BlockSystem, "tail", false),
			}),
			plan: BlockPlan{Block: BlockSystem, Drop: map[string]bool{"core": true, "policy": true}},
		},
		{
			name: "drop-name-not-present",
			raw: systemBody(tb, []map[string]any{
				systemBlock("policy", BlockSystem, "resident policy", true),
				systemBlock("old_skill", BlockSkills, "stale skill", false),
			}),
			plan: BlockPlan{Block: BlockSkills, Drop: map[string]bool{"nonexistent": true}},
		},
		{
			name: "block-scoped-name-in-other-block-not-dropped",
			raw: systemBody(tb, []map[string]any{
				systemBlock("policy", BlockSystem, "resident policy", true),
				systemBlock("old_skill", BlockSkills, "stale skill", false),
			}),
			// old_skill lives in the skills block; a memory-scoped plan must not touch it.
			plan: BlockPlan{Block: BlockMemory, Drop: map[string]bool{"old_skill": true}},
		},
	}
}

// TestSystemWitness_PrunedBlocksAreNamed (invariant 3): every block the system compactor
// removes is reported in Pruned, and no block absent from Pruned ever vanishes; the drop
// is always LEGIBLE, never silent.
func TestSystemWitness_PrunedBlocksAreNamed(t *testing.T) {
	for _, tc := range systemWitnessBodies(t) {
		t.Run(tc.name, func(t *testing.T) {
			before := systemNamesIn(t, tc.raw)
			res := CompactInboundSystem(tc.raw, tc.plan, okDecode)
			after := systemNamesIn(t, res.Body)

			// Every Pruned name was present in the input and is absent from the output.
			for _, p := range res.Pruned {
				if !contains(before, p) {
					t.Errorf("Pruned names %q which was never in the input system %v", p, before)
				}
				if contains(after, p) {
					t.Errorf("Pruned names %q but it is still present in the output %v", p, after)
				}
			}
			// Nothing vanishes silently: any input block gone from the output must be in Pruned.
			for _, b := range before {
				if !contains(after, b) && !contains(res.Pruned, b) {
					t.Errorf("block %q vanished without being named in Pruned (silent drop): before=%v after=%v pruned=%v",
						b, before, after, res.Pruned)
				}
			}
			// Pruned non-empty iff a change happened, and an identity always names a reason.
			if (len(res.Pruned) > 0) != res.Changed {
				t.Errorf("Pruned/Changed disagree: pruned=%v changed=%v", res.Pruned, res.Changed)
			}
			if !res.Changed && res.SkipReason == "" {
				t.Errorf("identity result must name a closed-set SkipReason, got empty")
			}
		})
	}
}

// TestSystemWitness_EmptyPlanIsReversibleIdentity (invariant 4): an empty Drop reproduces
// the input bit-for-bit: the same backing slice, a named SkipEmptyPlan, no change.
func TestSystemWitness_EmptyPlanIsReversibleIdentity(t *testing.T) {
	for _, tc := range systemWitnessBodies(t) {
		t.Run(tc.name, func(t *testing.T) {
			res := CompactInboundSystem(tc.raw, BlockPlan{Block: tc.plan.Block}, okDecode)
			if res.Changed {
				t.Errorf("empty plan changed the body (must be identity), pruned=%v", res.Pruned)
			}
			if !bytes.Equal(res.Body, tc.raw) {
				t.Errorf("empty plan is not byte-identical to the input")
			}
			// Identity returns the SAME backing array (the contract a caller relies on).
			if len(res.Body) > 0 && len(tc.raw) > 0 && &res.Body[0] != &tc.raw[0] {
				t.Errorf("empty-plan identity should return the same backing slice")
			}
			if res.SkipReason != SkipEmptyPlan {
				t.Errorf("SkipReason = %q, want %q", res.SkipReason, SkipEmptyPlan)
			}
		})
	}
}

// TestSystemWitness_KernelViewByteUnchanged (invariant 5): whatever the compactor does,
// the cached system prefix is byte-identical and every surviving block's name is
// preserved in order; only whole post-breakpoint blocks of the plan's own Block may be
// deleted. A prune can never alter the bytes upstream already cached, nor reorder/rename
// a survivor.
func TestSystemWitness_KernelViewByteUnchanged(t *testing.T) {
	for _, tc := range systemWitnessBodies(t) {
		t.Run(tc.name, func(t *testing.T) {
			res := CompactInboundSystem(tc.raw, tc.plan, okDecode)

			// The result must always re-decode (the request stays valid).
			if err := okDecode(res.Body); err != nil {
				t.Fatalf("result must re-decode as a valid request: %v", err)
			}

			before := systemNamesIn(t, tc.raw)
			after := systemNamesIn(t, res.Body)

			// Survivors keep their original RELATIVE order and exact names; a prune may
			// only DELETE elements, never reorder or rename them.
			si := 0
			for _, b := range before {
				if si < len(after) && after[si] == b {
					si++
				}
			}
			if si != len(after) {
				t.Errorf("output system order/names are not a subsequence of the input: before=%v after=%v", before, after)
			}

			if !res.Changed {
				// Identity: byte-for-byte unchanged, the strongest form of inv 5.
				if !bytes.Equal(res.Body, tc.raw) {
					t.Errorf("identity result is not byte-identical to the input")
				}
				return
			}

			// On a real prune the compactor only DELETES whole post-breakpoint blocks: the
			// cached prefix the provider is warm for is anchored on the last system
			// cache_control, so the input and output must share a common byte prefix that
			// runs at least past that breakpoint — no cached byte moved.
			if _, prefixEnd, _, ok := ArraySplicePoints(tc.raw, "system"); ok {
				if prefixEnd > len(res.Body) || !bytes.Equal(tc.raw[:prefixEnd], res.Body[:prefixEnd]) {
					t.Errorf("cached system prefix (through byte %d) changed under a prune", prefixEnd)
				}
				lcp := commonPrefixLen(tc.raw, res.Body)
				if lcp < prefixEnd {
					t.Errorf("common prefix (%d) does not reach the cached breakpoint end (%d): a cached byte moved under a prune",
						lcp, prefixEnd)
				}
			}
			// The output is strictly shorter (a prune only deletes) and differs from the input.
			if len(res.Body) >= len(tc.raw) {
				t.Errorf("a prune must SHRINK the body: in=%d out=%d", len(tc.raw), len(res.Body))
			}
		})
	}
}
