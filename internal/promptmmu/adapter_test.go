package promptmmu

import (
	"sort"
	"testing"
)

// fakeFloor builds a fake DeniesUnconditionally predicate from an explicit set of
// UNCONDITIONALLY-denied names. It stands in for policy.Manifest.DeniesToolUnconditionally
// (#773, a cross-lane package) so this tier-1 test never imports internal/policy.
// Crucially, an arg-conditional or allowed tool is NOT in the set ⇒ the predicate
// returns false for it, exactly mirroring the real predicate's contract.
func fakeFloor(unconditionallyDenied ...string) DeniesUnconditionally {
	deny := map[string]bool{}
	for _, n := range unconditionallyDenied {
		deny[n] = true
	}
	return func(tool string) bool { return deny[tool] }
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func eqStrs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestToolPlanFor_BlanketBlockedToolIsDropped is the core #752 rule: a tool the
// floor denies UNCONDITIONALLY is added to Drop, so its definition is prunable.
func TestToolPlanFor_BlanketBlockedToolIsDropped(t *testing.T) {
	advertised := []string{"read_file", "delete_repo", "write_file"}
	// delete_repo is a blanket block; the other two are allowed.
	plan := ToolPlanFor(advertised, fakeFloor("delete_repo"))
	if got := sortedKeys(plan.Drop); !eqStrs(got, []string{"delete_repo"}) {
		t.Fatalf("Drop = %v, want [delete_repo]", got)
	}
}

// TestToolPlanFor_ArgConditionalToolIsNotDropped is the TRAP the issue names: a tool
// denied only for SOME args (allowed otherwise) must NOT be pruned — the fake floor
// reports false for it, so it never enters Drop.
func TestToolPlanFor_ArgConditionalToolIsNotDropped(t *testing.T) {
	advertised := []string{"bash", "read_file"}
	// bash is arg-conditional (denied for `rm -rf` but allowed for `ls`), so the
	// real predicate reports false — modeled by leaving it out of the deny set.
	plan := ToolPlanFor(advertised, fakeFloor( /* nothing unconditionally denied */ ))
	if plan.Drop["bash"] {
		t.Fatalf("arg-conditional tool bash must NOT be in Drop (would remove a real capability)")
	}
	if len(plan.Drop) != 0 {
		t.Fatalf("Drop = %v, want empty", sortedKeys(plan.Drop))
	}
}

// TestToolPlanFor_AllowedAndAbsentToolsAreNotDropped covers the remaining #752 cases:
// an affirmatively-allowed tool and a tool absent from the manifest (default-allow
// posture) both stay advertised.
func TestToolPlanFor_AllowedAndAbsentToolsAreNotDropped(t *testing.T) {
	advertised := []string{"read_file" /* allowed */, "novel_tool" /* absent */, "rm_rf" /* blocked */}
	plan := ToolPlanFor(advertised, fakeFloor("rm_rf"))
	if plan.Drop["read_file"] {
		t.Errorf("an allowed tool must not be dropped")
	}
	if plan.Drop["novel_tool"] {
		t.Errorf("a tool absent under default-allow must not be dropped")
	}
	if !plan.Drop["rm_rf"] {
		t.Errorf("the blanket-blocked tool must be dropped")
	}
}

// TestToolPlanFor_NilPredicateFailsClosed: with no floor predicate, nothing is
// provably unconditionally denied, so the plan is empty (advertise everything).
func TestToolPlanFor_NilPredicateFailsClosed(t *testing.T) {
	plan := ToolPlanFor([]string{"a", "b", "c"}, nil)
	if len(plan.Drop) != 0 {
		t.Fatalf("nil predicate must yield an empty Drop (fail-closed), got %v", sortedKeys(plan.Drop))
	}
}

// TestToolPlanFor_EmptyNamesIgnored: a blank advertised name never enters Drop even
// if the (degenerate) predicate would match it.
func TestToolPlanFor_EmptyNamesIgnored(t *testing.T) {
	denyAll := func(string) bool { return true }
	plan := ToolPlanFor([]string{"", "real"}, denyAll)
	if plan.Drop[""] {
		t.Errorf("empty name must never enter Drop")
	}
	if !plan.Drop["real"] {
		t.Errorf("the real name should be dropped under deny-all")
	}
}

// TestToolPlanFor_PlanFeedsSpine end-to-end: the plan ToolPlanFor builds drives the
// spine to actually prune the unconditionally-denied tool's definition, proving the
// adapter output is a valid spine input.
func TestToolPlanFor_PlanFeedsSpine(t *testing.T) {
	// breakpoint on tool[0]; "denied" sits strictly after it (prunable).
	raw := body(t, []map[string]any{tool("read_file", true), tool("keep", false), tool("denied", false)}, false)
	plan := ToolPlanFor([]string{"read_file", "keep", "denied"}, fakeFloor("denied"))
	res := CompactInboundTools(raw, plan, okDecode)
	if !res.Changed {
		t.Fatalf("expected the adapter's plan to drive a prune, got identity (%q)", res.SkipReason)
	}
	if len(res.Pruned) != 1 || res.Pruned[0] != "denied" {
		t.Fatalf("Pruned = %v, want [denied]", res.Pruned)
	}
	if contains(toolNamesIn(t, res.Body), "denied") {
		t.Fatalf("denied tool definition should be gone")
	}
}

// TestToolPlanForRequest_SelfDropIsAdvertisedOnlyAndFloorMonotonic (#757): the
// request-shaped planner input unions self-drop with the policy floor, filters
// unknown names to the advertised surface, and cannot remove a kernel-denied drop.
func TestToolPlanForRequest_SelfDropIsAdvertisedOnlyAndFloorMonotonic(t *testing.T) {
	req := ToolPlanRequest{
		Advertised: []string{"read_file", "write_file", "rm_rf"},
		SelfDrop:   []string{"write_file", "not_advertised", ""},
	}
	plan := ToolPlanForRequest(req, fakeFloor("rm_rf"))
	if got := sortedKeys(plan.Drop); !eqStrs(got, []string{"rm_rf", "write_file"}) {
		t.Fatalf("Drop = %v, want [rm_rf write_file]", got)
	}
	if plan.Drop["not_advertised"] {
		t.Fatalf("self-drop must be scoped to the advertised surface")
	}
}

// TestToolPlanForRequest_DrivesSelfDropSpinePrune proves the request-level helper
// is a valid input to the unchanged spine: a purely self-imposed, advertised tool
// drop removes that post-breakpoint definition.
func TestToolPlanForRequest_DrivesSelfDropSpinePrune(t *testing.T) {
	raw := body(t, []map[string]any{tool("read_file", true), tool("write_file", false)}, false)
	plan := ToolPlanForRequest(ToolPlanRequest{
		Advertised: []string{"read_file", "write_file"},
		SelfDrop:   []string{"write_file"},
	}, fakeFloor())
	res := CompactInboundTools(raw, plan, okDecode)
	if !res.Changed || len(res.Pruned) != 1 || res.Pruned[0] != "write_file" {
		t.Fatalf("self-drop request should prune write_file, got changed=%v pruned=%v reason=%q", res.Changed, res.Pruned, res.SkipReason)
	}
}

// TestWithSelfDrop_UnionsAndDoesNotMutate (#757): self-withheld tools augment the
// policy Drop set as a union, and the input plan is never mutated.
func TestWithSelfDrop_UnionsAndDoesNotMutate(t *testing.T) {
	base := ToolPlanFor([]string{"rm_rf", "write_file"}, fakeFloor("rm_rf"))
	// agent in a read-only phase withholds its own write tool.
	aug := WithSelfDrop(base, []string{"write_file", ""})

	if got := sortedKeys(aug.Drop); !eqStrs(got, []string{"rm_rf", "write_file"}) {
		t.Fatalf("augmented Drop = %v, want [rm_rf write_file]", got)
	}
	if aug.Drop[""] {
		t.Errorf("empty self-withheld name must be ignored")
	}
	// The base plan must be untouched (no aliasing).
	if base.Drop["write_file"] {
		t.Errorf("WithSelfDrop mutated the input plan's Drop set")
	}
}

// TestWithSelfDrop_NilBaseDrop is robust to a base plan with a nil Drop map.
func TestWithSelfDrop_NilBaseDrop(t *testing.T) {
	aug := WithSelfDrop(ToolPlan{}, []string{"write_file"})
	if !aug.Drop["write_file"] {
		t.Fatalf("self-drop should apply even with a nil base Drop, got %v", sortedKeys(aug.Drop))
	}
}

// TestWithSelfDrop_DrivesSpinePrune end-to-end: a purely self-imposed drop (no policy
// denial) still prunes a post-breakpoint tool via the unchanged spine.
func TestWithSelfDrop_DrivesSpinePrune(t *testing.T) {
	raw := body(t, []map[string]any{tool("read_file", true), tool("write_file", false)}, false)
	plan := WithSelfDrop(ToolPlanFor([]string{"read_file", "write_file"}, fakeFloor()), []string{"write_file"})
	res := CompactInboundTools(raw, plan, okDecode)
	if !res.Changed || len(res.Pruned) != 1 || res.Pruned[0] != "write_file" {
		t.Fatalf("self-drop should prune write_file, got changed=%v pruned=%v reason=%q", res.Changed, res.Pruned, res.SkipReason)
	}
}

// TestBlockPlanFor_GeneralizesByName (#758): the generalized block planner selects
// named elements of any block with the same Drop-by-name shape.
func TestBlockPlanFor_GeneralizesByName(t *testing.T) {
	stale := map[string]bool{"old_skill": true}
	plan := BlockPlanFor(BlockSkills, []string{"current_skill", "old_skill", ""}, func(n string) bool { return stale[n] })
	if plan.Block != BlockSkills {
		t.Fatalf("Block = %q, want %q", plan.Block, BlockSkills)
	}
	if got := sortedKeys(plan.Drop); !eqStrs(got, []string{"old_skill"}) {
		t.Fatalf("Drop = %v, want [old_skill]", got)
	}
}

// TestBlockPlanFor_NilPredicateEmpty: no predicate ⇒ inject/advertise everything.
func TestBlockPlanFor_NilPredicateEmpty(t *testing.T) {
	plan := BlockPlanFor(BlockMemory, []string{"mem_a", "mem_b"}, nil)
	if len(plan.Drop) != 0 {
		t.Fatalf("nil drop predicate must yield an empty plan, got %v", sortedKeys(plan.Drop))
	}
	if plan.Block != BlockMemory {
		t.Fatalf("Block label should still be set, got %q", plan.Block)
	}
}
