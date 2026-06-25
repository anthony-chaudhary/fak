package promptmmu

import (
	"bytes"
	"testing"
)

// proofs_witness_test is the SAFETY witness for promptmmu rung 7 (#759, epic #751):
// it proves invariants 3/4/5 of the inbound tool-def compactor hold over a SPREAD of
// real-shaped request bodies, not just the single happy-path fixture, so a future
// rung that wires the prune live (rungs 3-6) cannot regress the trust boundary silently.
//
//   inv 3 - NAMED: every tool removed appears in PruneResult.Pruned, and nothing not in
//           Pruned vanishes (no silent disappearance).
//   inv 4 - REVERSIBLE: re-running with an empty plan reproduces the input bit-for-bit.
//   inv 5 - KERNEL-VIEW BYTE-UNCHANGED: the protected prefix the kernel adjudicates is
//           byte-identical after any prune; only the advertised post-breakpoint surface
//           narrows, and every surviving tool's name is preserved.

// witnessBodies returns a spread of well-formed bodies that exercise the real prune
// regimes: a breakpoint mid-list with droppable tools after it, a system-only
// breakpoint (whole tools[] protected -> identity), no breakpoint at all (identity),
// and a larger list with a late breakpoint.
func witnessBodies(tb testing.TB) []struct {
	name string
	raw  []byte
	plan ToolPlan
} {
	tb.Helper()
	return []struct {
		name string
		raw  []byte
		plan ToolPlan
	}{
		{
			name: "breakpoint-mid-list-drop-after",
			raw:  body(tb, []map[string]any{tool("alpha", true), tool("beta", false), tool("gamma", false)}, false),
			plan: drop("gamma"),
		},
		{
			name: "breakpoint-mid-list-drop-multiple-after",
			raw:  body(tb, []map[string]any{tool("a", false), tool("b", true), tool("c", false), tool("d", false), tool("e", false)}, false),
			plan: drop("c", "e"),
		},
		{
			name: "system-only-breakpoint-protects-all-tools",
			raw:  body(tb, []map[string]any{tool("alpha", false), tool("beta", false)}, true),
			plan: drop("beta"),
		},
		{
			name: "no-breakpoint-anywhere",
			raw:  body(tb, []map[string]any{tool("alpha", false), tool("beta", false)}, false),
			plan: drop("alpha", "beta"),
		},
		{
			name: "drop-names-not-present",
			raw:  body(tb, []map[string]any{tool("alpha", true), tool("beta", false)}, false),
			plan: drop("nonexistent"),
		},
	}
}

// commonPrefixLen returns the length of the longest shared byte prefix of a and b.
func commonPrefixLen(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// TestWitness_PrunedToolsAreNamed (invariant 3): every tool the compactor removes is
// reported in Pruned, and no tool absent from Pruned ever vanishes; the drop is always
// LEGIBLE, never silent.
func TestWitness_PrunedToolsAreNamed(t *testing.T) {
	for _, tc := range witnessBodies(t) {
		t.Run(tc.name, func(t *testing.T) {
			before := toolNamesIn(t, tc.raw)
			res := CompactInboundTools(tc.raw, tc.plan, okDecode)
			after := toolNamesIn(t, res.Body)

			// Every Pruned name was present in the input and is absent from the output.
			for _, p := range res.Pruned {
				if !contains(before, p) {
					t.Errorf("Pruned names %q which was never in the input tools %v", p, before)
				}
				if contains(after, p) {
					t.Errorf("Pruned names %q but it is still advertised in the output %v", p, after)
				}
			}
			// Nothing vanishes silently: any input tool gone from the output must be in Pruned.
			for _, b := range before {
				if !contains(after, b) && !contains(res.Pruned, b) {
					t.Errorf("tool %q vanished without being named in Pruned (silent drop): before=%v after=%v pruned=%v",
						b, before, after, res.Pruned)
				}
			}
			// Pruned non-empty iff a change happened.
			if (len(res.Pruned) > 0) != res.Changed {
				t.Errorf("Pruned/Changed disagree: pruned=%v changed=%v", res.Pruned, res.Changed)
			}
		})
	}
}

// TestWitness_EmptyPlanIsReversibleIdentity (invariant 4): an empty plan reproduces the
// input bit-for-bit: the same backing slice, a named SkipEmptyPlan, no change.
func TestWitness_EmptyPlanIsReversibleIdentity(t *testing.T) {
	for _, tc := range witnessBodies(t) {
		t.Run(tc.name, func(t *testing.T) {
			res := CompactInboundTools(tc.raw, ToolPlan{}, okDecode)
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

// TestWitness_KernelViewByteUnchanged (invariant 5): whatever the compactor does, the
// protected prefix the kernel adjudicates is byte-identical and every surviving tool's
// name is preserved; only the advertised post-breakpoint surface may narrow. A prune
// can never alter the bytes the kernel already trusted, nor silently rename a survivor.
func TestWitness_KernelViewByteUnchanged(t *testing.T) {
	for _, tc := range witnessBodies(t) {
		t.Run(tc.name, func(t *testing.T) {
			res := CompactInboundTools(tc.raw, tc.plan, okDecode)

			// The result must always re-decode (the kernel view stays a valid request).
			if err := okDecode(res.Body); err != nil {
				t.Fatalf("result must re-decode as a valid request: %v", err)
			}

			before := toolNamesIn(t, tc.raw)
			after := toolNamesIn(t, res.Body)

			// Survivors keep their original RELATIVE order and exact names; a prune may
			// only DELETE elements, never reorder or rename them.
			si := 0
			for _, b := range before {
				if si < len(after) && after[si] == b {
					si++
				}
			}
			if si != len(after) {
				t.Errorf("output tool order/names are not a subsequence of the input: before=%v after=%v", before, after)
			}

			if !res.Changed {
				// Identity: byte-for-byte unchanged, the strongest form of inv 5.
				if !bytes.Equal(res.Body, tc.raw) {
					t.Errorf("identity result is not byte-identical to the input")
				}
				return
			}

			// On a real prune the compactor only DELETES whole post-breakpoint tool
			// elements: the input and output must share a common byte prefix that runs
			// at least through the last breakpoint tool (the kernel-trusted region the
			// breakpoint anchors cannot shift), and that prefix must reach at least up to
			// where the first dropped tool's element begins. We verify the kernel-view
			// stability structurally: the common prefix extends past the last cache_control
			// breakpoint in the input, so no trusted byte moved.
			lcp := commonPrefixLen(tc.raw, res.Body)
			lastBreakpoint := bytes.LastIndex(tc.raw, []byte(`"cache_control"`))
			if lastBreakpoint >= 0 && lcp <= lastBreakpoint {
				t.Errorf("common prefix (%d) does not reach past the last breakpoint (%d): a trusted byte moved under a prune",
					lcp, lastBreakpoint)
			}
			// The output is strictly shorter (a prune only deletes) and is not equal to
			// the input (Changed was true).
			if len(res.Body) >= len(tc.raw) {
				t.Errorf("a prune must SHRINK the body: in=%d out=%d", len(tc.raw), len(res.Body))
			}
		})
	}
}
