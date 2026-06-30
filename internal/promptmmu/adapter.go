package promptmmu

// adapter.go projects the kernel's policy decision onto a ToolPlan — the bridge
// (epic #751 rung 2, #752) that makes the spine DO something — plus the two pure
// plan-augmentation rungs that ride on the same Drop set: agent self-control over
// its own advertised surface (#757) and the generalization to other inbound blocks
// (#758). Everything here is pure mapping over names; the spine still owns the
// cache-safe splice, the policy floor still owns the trust boundary.

// DeniesUnconditionally is the definition-level policy predicate the adapter needs:
// "is this tool denied for EVERY possible argument value?" — a blanket tool block,
// never an arg-conditional rule that allows some args. It is supplied as a CLOSURE
// so promptmmu stays a tier-1, stdlib-only package and never imports internal/policy
// (the predicate's home). The live wiring passes
// policy.Manifest.DeniesToolUnconditionally (#773); a test passes a fake.
//
// CONTRACT (the trap this exists to avoid): it MUST report false for an
// arg-conditional tool (allowed for some args) and false for a tool the floor
// allows or that is absent under a default-allow posture. Only an UNCONDITIONAL
// deny may return true. Any uncertainty ⇒ false (fail-closed: a missed prune costs
// a few tokens, a wrong prune removes a real capability — epic invariant 5).
type DeniesUnconditionally func(tool string) bool

// ToolPlanRequest is the planner-side input for building a prompt-MMU tool plan.
// Advertised names the MCP/tool definitions currently present in the model-visible
// surface; SelfDrop names the subset the agent asks to withhold from ITS OWN surface
// this turn (#757). The request is negative-only: it can add names to Drop, never
// remove a kernel-denied name from Drop or widen the advertised set.
type ToolPlanRequest struct {
	// Advertised is the set of tool names currently advertised in tools[].
	Advertised []string
	// SelfDrop is the agent's self-imposed tool-name drop list for this surface.
	SelfDrop []string
}

// ToolPlanFor builds the ToolPlan whose Drop set is exactly the advertised tools
// the policy floor denies UNCONDITIONALLY (denies for all args), so the spine may
// drop their DEFINITIONS with zero behavioral change. It is the #752 adapter: pure
// mapping, no new policy.
//
//   - advertised is the set of tool NAMES currently advertised in the request's
//     tools[] (the caller extracts them from agent.ToolDef — that decode lives
//     above this tier, so the adapter takes names, never the agent type).
//   - denies answers the definition-level "denied for all args?" question. A nil
//     denies ⇒ an empty Drop (nothing is provably unconditionally denied, so
//     advertise everything — the fail-closed default).
//
// A name is added to Drop IFF denies(name) is true. An arg-conditional or allowed
// tool is left advertised. Duplicate or empty names are ignored. The returned plan
// is safe to hand straight to CompactInboundTools, which independently refuses any
// drop that would move the cache boundary.
func ToolPlanFor(advertised []string, denies DeniesUnconditionally) ToolPlan {
	plan := ToolPlan{Drop: map[string]bool{}}
	if denies == nil {
		return plan
	}
	for _, name := range advertised {
		if name == "" {
			continue
		}
		if denies(name) {
			plan.Drop[name] = true
		}
	}
	return plan
}

// ToolPlanForRequest builds the complete planner-side ToolPlan for one advertised
// surface: kernel-floor unconditional denials UNION the agent's self-imposed drops
// (#757). SelfDrop entries are scoped to Advertised, so a request can only shrink
// the tool definitions it is actually offering this turn; unknown and empty names
// are ignored. The result is still only a plan — CompactInboundTools owns the
// cache-safe splice and may refuse a cache-unsafe drop.
func ToolPlanForRequest(req ToolPlanRequest, denies DeniesUnconditionally) ToolPlan {
	plan := ToolPlanFor(req.Advertised, denies)
	if len(req.SelfDrop) == 0 {
		return plan
	}
	advertised := make(map[string]bool, len(req.Advertised))
	for _, name := range req.Advertised {
		if name != "" {
			advertised[name] = true
		}
	}
	self := make([]string, 0, len(req.SelfDrop))
	for _, name := range req.SelfDrop {
		if advertised[name] {
			self = append(self, name)
		}
	}
	return WithSelfDrop(plan, self)
}

// WithSelfDrop returns a copy of plan augmented with the agent's OWN self-imposed
// drops (epic #751 rung 5, #757): tools the agent chooses to withhold from its own
// advertised surface — e.g. an agent in a read-only phase withholding its write
// tools. This is pure plan augmentation on top of the kernel's denied set; the
// spine is unchanged and still refuses any cache-unsafe drop.
//
// SAFETY: self-drops are always SAFE in the same sense the policy drops are — a
// withheld tool the agent later names is still adjudicated by the unchanged kernel
// floor, so withholding the advertisement can only shrink the tool-def tokens,
// never widen the reachable action set (epic invariant 5). The union is taken: a
// name already in the policy Drop stays dropped; empty names are ignored. The input
// plan is never mutated.
func WithSelfDrop(plan ToolPlan, selfWithheld []string) ToolPlan {
	out := ToolPlan{Drop: make(map[string]bool, len(plan.Drop)+len(selfWithheld))}
	for name := range plan.Drop {
		out.Drop[name] = true
	}
	for _, name := range selfWithheld {
		if name != "" {
			out.Drop[name] = true
		}
	}
	return out
}

// BlockPlan is the #758 generalization of ToolPlan to the OTHER inbound blocks the
// prompt-MMU will eventually curate: stale skills to demote, injected-memory
// segments to budget out, system segments to prune. It carries the SAME Drop-by-name
// shape so each sibling leaf reuses the spine's prefix-proof discipline — every drop
// lands strictly past that block's own cache_control breakpoint, named and reversible.
//
// This is the in-lane scaffold for rung 6; the per-block splice (which breakpoint
// anchors which block) and the gateway wiring are the deferred halves. Today it lets
// a caller name what it WOULD demote from any block in one uniform structure.
type BlockPlan struct {
	// Block names the inbound block this plan curates (a closed label, below).
	Block string
	// Drop is the set of element NAMES to remove from that block, strictly past the
	// block's own cache_control breakpoint. Membership only; order is irrelevant.
	Drop map[string]bool
}

// Closed set of curatable inbound blocks (#758). tools[] is the shipped spine
// (CompactInboundTools); the rest name the rung-6 siblings so a plan can be built
// and logged before the per-block splice lands.
const (
	BlockTools  = "tools"  // tool DEFINITIONS — the shipped spine
	BlockSkills = "skills" // harness-injected skill text — demote stale skills
	BlockMemory = "memory" // injected memory segments — budget the window
	BlockSystem = "system" // system-prompt sub-blocks — prune segments (most conservative)
)

// BlockPlanFor builds a BlockPlan naming the elements of one block a caller would
// drop. drop is the predicate "withhold this named element?"; names selecting it
// land in Drop. A nil drop or empty names ⇒ an empty plan (advertise/inject all).
// Like ToolPlanFor it is pure mapping — the per-block cache-safe splice is the
// deferred rung-6 spine.
func BlockPlanFor(block string, elements []string, drop func(name string) bool) BlockPlan {
	plan := BlockPlan{Block: block, Drop: map[string]bool{}}
	if drop == nil {
		return plan
	}
	for _, name := range elements {
		if name != "" && drop(name) {
			plan.Drop[name] = true
		}
	}
	return plan
}
