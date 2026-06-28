package main

// kvmmu_slot_bridge.go — issue #915, child C of the "one machine" epic (#912): close the wire
// the gateway and the session scheduler each built only HALF of, so "a session slot freed" BE
// "a KV block freed" for a waiting sequence.
//
// internal/session.Scheduler emits a structured "a slot freed" SlotEvent at a turn boundary
// (CauseDraining/CauseStopped for a terminal transition, off WatchTransitions). internal/gateway
// .Server.ReclaimKVOnSlotFreed consumes a wire-neutral SlotFreed and, with the in-kernel KVMMU
// flag on, drives the REAL KV reclaim (kvmmu.Context.EvictColdest -> model.KVCache.Evict). The
// two never referenced each other: the gateway deliberately does NOT import internal/session (it
// consumes wire-neutral projections, the same SessionState/SessionEvent discipline), so on the
// served path nothing actually freed KV when a slot freed — the edge was armed but inert. This
// host bridge is the missing wire: it projects the scheduler's SlotEvent onto the gateway's
// SlotFreed and routes it to the reclaim edge.
//
// SCOPE (the issue's fences): this only ROUTES the existing event to the existing reclaim — no
// new eviction policy, and no flag handling of its own (the gateway owns FAK_INKERNEL_KVMMU;
// off, ReclaimKVOnSlotFreed is a byte-identical no-op, so the bridge is advisory-degrading too).
// The live serve loop attaches a real scheduler and injects a residency-backed reclaimer
// separately (the #912/#916 step); this projection is the wire both of those route through.

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// slotEventToSlotFreed projects a scheduler SlotEvent onto the gateway's wire-neutral SlotFreed.
// Cause crosses as the lowercase SlotCause token (SlotCause.String()) — exactly the vocabulary
// the gateway's terminalSlotCause keys the KV-free edge on (both sourced from internal/lifecycle)
// — so a rename on either side surfaces as a missed terminal token in the witness, never a
// silent drift. TraceID names whose residency freed; Rev is the freeing write's table revision,
// so a host can order/de-duplicate against a /v1/fak/changes cursor.
func slotEventToSlotFreed(ev session.SlotEvent) gateway.SlotFreed {
	return gateway.SlotFreed{
		TraceID: ev.TraceID,
		Cause:   ev.Cause.String(),
		Rev:     ev.Rev,
	}
}

// wireSlotFreedKVReclaim registers the gateway's KV-reclaim edge as the scheduler's slot-freed
// observer (#915): every SlotEvent the scheduler emits at a turn boundary is projected and handed
// to srv.ReclaimKVOnSlotFreed, so a terminal drain/stop frees that session's real KV for a
// waiting sequence. It COMPOSES — pass, when non-nil, runs first for every event, so a host that
// also wants the raw SlotEvent (its own slot accounting) keeps it through pass even though the
// scheduler holds exactly ONE OnSlotFreed callback. A nil scheduler or server is a no-op,
// matching the rest of the host's defensive wiring.
func wireSlotFreedKVReclaim(sched *session.Scheduler, srv *gateway.Server, pass func(session.SlotEvent)) {
	if sched == nil || srv == nil {
		return
	}
	sched.OnSlotFreed(func(ev session.SlotEvent) {
		if pass != nil {
			pass(ev)
		}
		srv.ReclaimKVOnSlotFreed(slotEventToSlotFreed(ev))
	})
}

// inkernelKVMMUEnabledHost mirrors internal/gateway.inkernelKVMMUEnabled (and internal/agent's
// own gate) so the host's serve-path slot-freed attach engages on EXACTLY the truthy set the
// gateway's KV-reclaim edge does — flag-off is then byte-for-byte the pre-#1095 served path
// (no scheduler attached, the two table seams kept exactly as serve.go installs them today).
func inkernelKVMMUEnabledHost() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_KVMMU"))) {
	case "on", "1", "true", "yes":
		return true
	}
	return false
}

// servePathResidencyReclaimer builds the residency-backed KVResidencyReclaimer the slot-freed
// edge would drive on the live serve loop (#1095) — the implementation that maps a draining /
// stopped session's TraceID onto the KV positions it held and frees them via the proven
// kvmmu.Context.EvictColdest -> model.KVCache.Evict reclaim.
//
// IT RETURNS nil TODAY — deliberately, not as a stub to fill in casually. The reclaimer's
// contract is per-TRACE: ReclaimResidency(trace) must free the residency THAT session held, and
// the issue's own risk note is load-bearing ("a mis-keyed TraceID frees the wrong session" — the
// reclaim re-RoPEs survivors). But the in-kernel serve path holds NO trace-addressable residency:
// internal/agent.InKernelPlanner builds a FRESH model.Session every turn (p.m.NewSession(), often
// defer s.Close()) and its cross-turn reuse is the PREFIX-keyed radix tree (radixkv, keyed on
// token-prefix hashes), never a trace->kvmmu.Context map. By the time a drain/stop SlotEvent
// fires at the next boundary, the residency that session "held" is already gone or shared by
// prefix, so there is nothing a trace-keyed reclaimer can safely evict. Constructing one over the
// ephemeral session would either no-op (free nothing) or, worse, evict a prefix another live
// session reuses — the exact wrong-session free the risk note forbids.
//
// The missing construction is a PERSISTENT trace-keyed residency the planner surfaces (the same
// kvmmu.Segment{From,Len,KV} ledger the sibling pressure executor waits on — #1074 / #987). Once
// the planner exposes a trace->residency handle, this returns a reclaimer closing over it; the
// call site below (attachServeSlotFreedReclaim) already routes a real drain/stop to it. Returning
// nil keeps the edge ARMED-BUT-INERT on the serve path (ReclaimKVOnSlotFreed returns (0,false)
// when no reclaimer is wired) rather than faking a free.
func servePathResidencyReclaimer() gateway.KVResidencyReclaimer {
	return nil
}

// attachServeSlotFreedReclaim is the serve-path caller #1095 adds: it gives wireSlotFreedKVReclaim
// + SetKVResidencyReclaimer their FIRST non-test caller. With FAK_INKERNEL_KVMMU on, it builds a
// session.Scheduler over the LIVE serve table, registers the gateway's KV-reclaim edge as the
// scheduler's slot-freed observer, and installs the residency-backed reclaimer — so a real
// drain/stop transition on a served session routes, at the next boundary, to the gateway edge.
//
// COMPOSITION (the load-bearing correctness point). Table.WatchTransitions / WatchBudget each hold
// exactly ONE observer slot (last write wins), and serve.go already installs the notifier's
// transition observer + the combined budget observer there. So this MUST take over both seams via
// the scheduler's AttachOptions pass-throughs rather than calling Watch* again (which would clobber
// the notifier). The caller therefore hands the resolved observers in and lets this own the install
// when the flag is on; when the flag is off it returns false and the caller installs them directly,
// byte-for-byte as before.
//
// It returns whether it took over the seams (so the caller knows whether to install the observers
// itself). A nil table or server, or the flag off, returns false and changes nothing.
func attachServeSlotFreedReclaim(tbl *session.Table, srv *gateway.Server, warnFraction float64, budgetObs session.BudgetObserver, transObs session.TransitionObserver) bool {
	if tbl == nil || srv == nil || !inkernelKVMMUEnabledHost() {
		return false
	}
	sched := session.NewScheduler(session.StrictPriority)
	// Compose: the scheduler's Attach installs fan-out handlers that run the host's pass-through
	// observers FIRST, then its own slot-freed accounting — so the notifier still sees every
	// transition/budget event through transObs/budgetObs, and the scheduler's SlotEvent fires too.
	sched.Attach(tbl, session.AttachOptions{
		WarnFraction: warnFraction,
		Budget:       budgetObs,
		Transitions:  transObs,
	})
	// The gateway's KV-reclaim edge becomes the scheduler's slot-freed observer; a terminal
	// drain/stop projects to SlotFreed and drives srv.ReclaimKVOnSlotFreed. pass is nil — the host
	// has no separate slot-accounting consumer beyond the transition pass-through above.
	wireSlotFreedKVReclaim(sched, srv, nil)
	// Install the residency-backed reclaimer. nil today (see servePathResidencyReclaimer): the
	// edge is armed and reachable on the serve path, but inert until the planner surfaces a
	// trace-keyed residency to evict. SetKVResidencyReclaimer(nil) is an explicit no-op clear.
	srv.SetKVResidencyReclaimer(servePathResidencyReclaimer())
	return true
}
