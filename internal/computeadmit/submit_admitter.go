// submit_admitter.go — compute-region contention pricing at the Kernel.Submit seam
// (#3269). modelroute writes abi.ToolCall.Engine BEFORE Kernel.Submit
// (route-before-adjudicate); SubmitAdmitter is the adjudication-fold step that
// consults live COMPUTE leases at that seam, so two ensemble members targeting
// the same over-subscribed device serialize the same way two agents on the same
// file tree do: the second refuses with COLLISION_RISK instead of being
// silently co-scheduled. It is a decision only — no scheduler, no preemption,
// no queueing; the caller re-submits once the holder releases.
//
// SubmitAdmitter answers BOTH kernel registry seams, which is what makes the
// serialization a serialization instead of a one-way wedge:
//
//   - abi.Adjudicator — the Kernel.Submit fold: price the call's compute claim
//     against the live lease table; refuse a contender with COLLISION_RISK.
//   - abi.ResultAdmitter — the Kernel.Reap result-admission fold: RELEASE the
//     region the completing call's holder took at Submit, so the member that was
//     serialized behind it admits on its next Submit.
//
// Without the second seam nothing in a shipped process ever frees a claimed
// region: the first admitted member would hold its device forever and every
// co-targeted peer would refuse for the life of the process. Both seams are
// driver-blind registry seams, so the kernel never imports this leaf (it is the
// "driver-blind integrator"; see internal/architest TestKernelImportsOnlyAbi) —
// a wiring layer installs ONE gate at both:
//
//	gate := computeadmit.NewSubmitAdmitter(tax)
//	abi.RegisterAdjudicator(rank, gate)
//	abi.RegisterResultAdmitter(rank, gate)
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

// Admit is the Kernel.Reap half of the Submit-seam serialization: when an
// admitted call completes, the kernel folds the ResultAdmitter chain over its
// result, and THAT is the driver-blind moment this gate frees the compute region
// the call's holder took at Submit. The member serialized behind it therefore
// admits on its next Submit with no scheduler, no queue, and no test-only
// Release poke — the same "the holder finished, the seat is free" discipline two
// agents on one file tree get when a lease is dropped.
//
// It releases ONLY when the completing call carries a compute claim AND the live
// lease recorded under that holder is for the SAME region: a claim-less call
// that happens to share a trace id with a compute holder must never hand that
// holder's device away underneath it.
//
// The verdict is always Allow-with-no-opinion (fold rank 0, the same rank
// admitResult seeds its default-admit best with), so installing this gate cannot
// change ANY existing chain's result-admission verdict — the additive guarantee
// on the result side, matching the Defer-on-no-claim guarantee on the call side.
func (g *SubmitAdmitter) Admit(_ context.Context, c *abi.ToolCall, _ *abi.Result) abi.Verdict {
	noOpinion := abi.Verdict{Kind: abi.VerdictAllow, By: By}
	if c == nil {
		return noOpinion
	}
	claim, ok := g.claimFor(c)
	if !ok {
		return noOpinion
	}
	holder := holderOf(c)
	if holder == "" {
		return noOpinion
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if l, held := g.live[holder]; held && l.Claim == claim {
		delete(g.live, holder)
	}
	return noOpinion
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
