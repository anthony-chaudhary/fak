// submit_admitter.go — compute-region contention pricing at the Kernel.Submit seam
// (#3269). modelroute writes abi.ToolCall.Engine BEFORE Kernel.Submit
// (route-before-adjudicate); SubmitAdmitter is the adjudication-fold step that
// consults live COMPUTE leases at that seam, so two ensemble members targeting
// the same over-subscribed device serialize the same way two agents on the same
// file tree do: the second refuses with COLLISION_RISK instead of being
// silently co-scheduled. It is a decision only — no scheduler, no preemption,
// no queueing; the caller re-submits once the holder releases.
package computeadmit

import (
	"context"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// ReasonComputeCollision is the registered abi.ReasonCode a Submit-seam compute
// collision cites, carrying the SAME stable name (COLLISION_RISK) the dispatch
// loop and the dos arbiter refuse with — one vocabulary on every surface.
// Registered out-of-tree range: egressfloor holds 1024–1025, toolproc
// 1040–1044, taskgraph 1050–1052, ctxmmu 1060, gateway 1070; this leaf takes
// 1080.
const ReasonComputeCollision abi.ReasonCode = 1080

// ReasonComputeCollisionName is the stable name registered for
// ReasonComputeCollision.
const ReasonComputeCollisionName = "COLLISION_RISK"

// By is the adjudicator identity a SubmitAdmitter verdict carries (forensics).
const By = "computeadmit"

// The OPEN abi.ToolCall.Meta keys the routing layer writes alongside Engine,
// BEFORE Kernel.Submit, to declare the call's compute claim — the compute twin
// of a dispatch candidate declaring its file tree. A call carrying none of
// these (and no bound engine) is claim-less and the gate defers untouched.
const (
	MetaComputeClass  = "compute_class"
	MetaComputeRange  = "compute_range"
	MetaComputeMode   = "compute_mode"
	MetaComputeHolder = "compute_holder"
)

var registerReasonOnce sync.Once

// SubmitAdmitter is an abi.Adjudicator that prices compute-region contention inside
// the Kernel.Submit adjudication fold. It resolves each call's compute claim
// (Meta keys first, then the bound Engine→claim route map), asks the shared
// Decide against its live-lease table, DENIES a contending or out-of-taxonomy
// claim with the closed vocabulary, and records an admitted holder's claim as
// live until Release. Claim-less calls always defer (no opinion), which keeps
// every existing chain's verdicts byte-identical — the additive guarantee.
type SubmitAdmitter struct {
	mu      sync.Mutex
	tax     Taxonomy
	engines map[string]dispatchorder.ComputeClaim
	live    map[string]Lease
}

// NewSubmitAdmitter builds a gate over the given compute taxonomy (zero Taxonomy =
// contention pricing only) and registers the COLLISION_RISK reason name in the
// abi closed vocabulary (idempotent).
func NewSubmitAdmitter(tax Taxonomy) *SubmitAdmitter {
	registerReasonOnce.Do(func() {
		abi.RegisterReason(ReasonComputeCollision, ReasonComputeCollisionName)
	})
	return &SubmitAdmitter{
		tax:     tax,
		engines: map[string]dispatchorder.ComputeClaim{},
		live:    map[string]Lease{},
	}
}

// BindRoute records the compute region an engine route occupies, so a call
// whose Engine names that route (written by modelroute pre-Submit) carries the
// region's claim without any Meta plumbing.
func (g *SubmitAdmitter) BindRoute(engine string, claim dispatchorder.ComputeClaim) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.engines[strings.TrimSpace(engine)] = claim
}

// Release frees the live compute lease a holder acquired at admission — the
// caller's completion (Reap) or abandonment hook. Unknown holders are a no-op.
func (g *SubmitAdmitter) Release(holder string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	delete(g.live, holder)
}

// Live snapshots the gate's live compute leases, sorted by ID (observability).
func (g *SubmitAdmitter) Live() []Lease {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.snapshotLocked()
}

func (g *SubmitAdmitter) snapshotLocked() []Lease {
	out := make([]Lease, 0, len(g.live))
	for _, l := range g.live {
		out = append(out, l)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Caps advertises nothing; the gate is a pure decision rung.
func (g *SubmitAdmitter) Caps() []abi.Capability { return nil }

// Adjudicate prices the call's compute claim against the live lease table via
// the shared Decide. No claim → defer (no opinion). A refusal maps the closed
// string vocabulary onto the registered abi.ReasonCode (COLLISION_RISK →
// ReasonComputeCollision, POLICY_BLOCK → abi.ReasonPolicyBlock) and carries the
// rung, detail, and conflicting holder in Meta as evidence. An admitted claim
// with a known holder is recorded live until Release; admission itself defers —
// an admission gate is not an allow-source, so the rest of the chain still
// decides the call.
func (g *SubmitAdmitter) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: By}
	}
	claim, ok := g.claimFor(c)
	if !ok {
		return abi.Verdict{Kind: abi.VerdictDefer, By: By}
	}
	holder := holderOf(c)
	g.mu.Lock()
	defer g.mu.Unlock()
	dec := Decide(Request{Actor: holder, Claim: claim, SelfID: holder}, g.snapshotLocked(), g.tax)
	if !dec.Admit {
		meta := map[string]string{
			"reason": dec.Reason,
			"rung":   dec.Rung,
			"detail": dec.Detail,
			"region": dispatchorder.ComputeRegionLabel(claim),
		}
		if dec.Conflict != nil {
			meta["conflict"] = dec.Conflict.ID
		}
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: reasonCode(dec.Reason), By: By, Meta: meta}
	}
	if holder != "" {
		g.live[holder] = Lease{ID: holder, Holder: holder, Claim: claim}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: By}
}

// claimFor resolves the call's compute claim: explicit Meta keys first (the
// routing layer's declaration), then the bound Engine route map. ok=false means
// claim-less — the gate must not touch the call.
func (g *SubmitAdmitter) claimFor(c *abi.ToolCall) (dispatchorder.ComputeClaim, bool) {
	if class := strings.TrimSpace(c.Meta[MetaComputeClass]); class != "" {
		return dispatchorder.ComputeClaim{
			Class: class,
			Range: strings.TrimSpace(c.Meta[MetaComputeRange]),
			Mode:  strings.TrimSpace(c.Meta[MetaComputeMode]),
		}, true
	}
	if eng := strings.TrimSpace(c.Engine); eng != "" {
		g.mu.Lock()
		claim, ok := g.engines[eng]
		g.mu.Unlock()
		if ok {
			return claim, true
		}
	}
	return dispatchorder.ComputeClaim{}, false
}

// holderOf names the lease holder for a call: the explicit compute_holder Meta
// key, else the trace id (an ensemble member's identity). An empty holder is
// priced against the live set but never recorded — an anonymous claim cannot
// hold a seat it could never release.
func holderOf(c *abi.ToolCall) string {
	if h := strings.TrimSpace(c.Meta[MetaComputeHolder]); h != "" {
		return h
	}
	return strings.TrimSpace(c.TraceID)
}

// reasonCode maps the closed string vocabulary onto the abi registered codes.
func reasonCode(reason string) abi.ReasonCode {
	if reason == ReasonPolicyBlock {
		return abi.ReasonPolicyBlock
	}
	return ReasonComputeCollision
}
