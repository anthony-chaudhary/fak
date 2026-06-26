package gateway

// kvmmu_slot_reclaim.go — issue #915, child C of the "one machine" epic (#912): make the
// drain/stop ↔ evict correspondence row LOAD-BEARING. A boundary-taken drain/stop already
// emits a structured "a slot freed" event (internal/session Scheduler.SlotEvent with
// CauseDraining/CauseStopped), but on the served path nothing frees REAL KV when it fires —
// the event is consumed only by the in-memory session scheduler. This wires that event to an
// actual KV free (the existing kvmmu.Context.EvictColdest / model.KVCache.Evict reclaim), so
// "a session slot freed" IS "a KV block freed" for a waiting sequence.
//
// POSTURE (the epic's §5 fences, inherited):
//   - No new eviction policy. This child only ROUTES the existing event to the existing
//     reclaim; the reclaimer is injected by the host and backed by the live residency.
//   - Gated behind the in-kernel KVMMU flag (FAK_INKERNEL_KVMMU, default off). Off, the path
//     is byte-identical to today — advisory-degrading: the scheduler still consumes the
//     SlotEvent for its own accounting; only the KV-free EDGE is gated here.
//   - Boundary discipline: eviction is taken at the next turn boundary (the SlotEvent already
//     fires there), never mid-decode.
//   - Only the two TERMINAL causes free KV. A budget-exhausted or PAUSED slot is a HOLD, not
//     a free — the session may resume and reuse its warm KV (warm-swap on Paused→Running is
//     the sibling child #916), so reclaiming on a pause would throw away reusable state.
//
// The gateway never imports internal/session (it consumes wire-neutral projections, the same
// discipline as SessionState/SessionEvent), so the SlotEvent crosses as the SlotFreed struct:
// the host maps SlotEvent.{TraceID, Cause.String(), Rev} onto it and calls ReclaimKVOnSlotFreed.

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/lifecycle"
)

// SlotFreed is the gateway's wire-neutral projection of internal/session.SlotEvent — the "a
// session slot freed" signal the scheduler emits at a turn boundary. Cause is the lowercase
// SlotCause token ("budget-exhausted"|"paused"|"draining"|"stopped"); the gateway keys the
// KV-free edge on the TERMINAL tokens, sourced from internal/lifecycle so a vocabulary rename
// can never silently drift this off the served session's own RunState (the same sourcing
// session_admit.go's admission gate uses). Rev is the table revision at the freeing write, so
// a host can order/de-duplicate against a /v1/fak/changes cursor.
type SlotFreed struct {
	TraceID string `json:"trace_id"`
	Cause   string `json:"cause"`
	Rev     uint64 `json:"rev,omitempty"`
}

// KVResidencyReclaimer is the seam the gateway drives on a terminal slot-freed event to free
// the KV residency a draining/stopped session held. The host injects an implementation backed
// by the live served residency (a kvmmu.Context whose EvictColdest / model.KVCache.Evict is
// the proven reclaim — re-RoPE + renumber); a server with none wired leaves the edge armed but
// inert. It exists so the gateway routes the event WITHOUT importing the model/kvmmu reclaim
// concretely — the same injection discipline as the session control hooks.
type KVResidencyReclaimer interface {
	// ReclaimResidency frees the KV positions the named session held and returns the count
	// freed (0 if the session held no residency). It is called once per terminal transition,
	// at the boundary the SlotEvent already fires on — never mid-decode.
	ReclaimResidency(trace string) (freedPositions int)
}

// SetKVResidencyReclaimer installs the host's KV-residency reclaimer (#915). Pass nil to
// clear. Settable after New so the host can build it once the in-kernel model/residency is
// loaded (mirroring SetModelLoadProfile). A nil receiver is a no-op.
func (s *Server) SetKVResidencyReclaimer(r KVResidencyReclaimer) {
	if s == nil {
		return
	}
	s.kvReclaimMu.Lock()
	s.kvReclaimer = r
	s.kvReclaimMu.Unlock()
}

// ReclaimKVOnSlotFreed is the gateway's consumer of a Scheduler SlotEvent: it makes "a
// session slot freed" BE "a KV block freed" for a waiting sequence (#915). The host wires it
// to Scheduler.OnSlotFreed; on a TERMINAL cause (draining/stopped) it drives the injected
// reclaimer's real KV free.
//
// Returns (freed, fired): fired reports whether the KV-free edge engaged — flag on AND a
// terminal cause AND a reclaimer wired — and freed is the positions released. Every non-firing
// path returns (0, false): flag off (the default, byte-identical to pre-#915), a non-terminal
// HOLD cause (budget-exhausted/paused), or the edge armed but no reclaimer wired yet.
func (s *Server) ReclaimKVOnSlotFreed(ev SlotFreed) (freed int, fired bool) {
	if s == nil {
		return 0, false
	}
	// Flag-off default: a no-op. The scheduler still consumes the SlotEvent for its own
	// accounting upstream; only the KV-free edge is gated here (advisory-degrading, #912 §5).
	if !inkernelKVMMUEnabled() {
		return 0, false
	}
	// Only drain/stop free KV; a HOLD (paused/budget-exhausted) keeps the warm residency.
	if !terminalSlotCause(ev.Cause) {
		return 0, false
	}
	s.kvReclaimMu.RLock()
	r := s.kvReclaimer
	s.kvReclaimMu.RUnlock()
	if r == nil {
		return 0, false
	}
	freed = r.ReclaimResidency(ev.TraceID)
	s.logf("gateway: KV residency reclaimed on slot-freed trace=%s cause=%s rev=%d freed=%dpos",
		ev.TraceID, ev.Cause, ev.Rev, freed)
	return freed, true
}

// terminalSlotCause reports whether a SlotCause token is one that ENDS a session (draining or
// stopped) and therefore frees its KV for a waiting sequence. The two non-terminal causes a
// SlotEvent can carry — budget-exhausted and paused — are holds, not frees. Tokens are sourced
// from internal/lifecycle (not re-spelled) so this stays bound to the shared #913 vocabulary.
func terminalSlotCause(cause string) bool {
	return cause == lifecycle.TokenDraining || cause == lifecycle.TokenStopped
}

// inkernelKVMMUEnabled mirrors internal/agent's FAK_INKERNEL_KVMMU gate so the gateway's
// slot-freed KV-reclaim edge engages on EXACTLY the truthy set the in-kernel span bridge does
// — and so flag-off is byte-for-byte the pre-#915 served path.
func inkernelKVMMUEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_KVMMU"))) {
	case "on", "1", "true", "yes":
		return true
	}
	return false
}
