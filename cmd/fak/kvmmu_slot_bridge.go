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
