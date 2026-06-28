package ctxmmu

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---------------------------------------------------------------------------
// Capability-body page-out (C3 — issue #1106).
//
// A faulted capability body (a SKILL.md, an MCP tool schema, an A2A method
// spec) is resident in context until the loader evicts it under pressure. C3's
// eviction must round-trip through the SAME witness gate quarantine does: a
// paged-out body pages back in ONLY after a witness Clear(). This file exposes a
// thin capability-body face over the machinery the quarantine path already runs
// (pageOut -> CAS, the held ledger, the CAS pin, the FIFO bound, Clear, PageIn)
// so the residency tracker drives the real pager instead of a parallel one.
//
// The body bytes are BENIGN (a skill body is not poison) — so this is NOT a
// quarantine: it does not screen, does not stub a result payload, and does not
// move the pollution counters. It is a page-out + gated re-admission only. The
// id keyspace is "c<n>" (capability), disjoint from quarantine's "q<n>" and the
// digest stub's "d<n>", so the three never collide in the shared held map.
// ---------------------------------------------------------------------------

// capPaged counts capability-body page-outs over the gate's lifetime; it mints
// the "c<n>" held id, disjoint from the quarantine "q<n>" / digest "d<n>" spaces.
func (m *MMU) capPagedNext() string {
	return fmt.Sprintf("c%d", atomic.AddInt64(&m.capPaged, 1))
}

// PageOutBody pages a faulted capability body out to the shared CAS and records
// a held handle, returning the held id the loader keys the eviction on. The
// bytes leave context (the body is no longer resident); a re-fault must page
// them back in through the witness gate. The handle is CAS-pinned so the bounded
// blob store cannot reclaim it before a gated PageInBody resolves it — the same
// soundness pin quarantine takes. The FIFO held bound still applies, so the
// oldest capability/quarantine handles age out together under one cap.
//
// A nil/empty body or a CAS backend that refuses the page-out returns ok=false
// with no ledger mutation, so the caller keeps the body resident rather than
// dropping bytes it cannot recover (fail-closed — never an unwitnessed drop).
func (m *MMU) PageOutBody(ctx context.Context, body []byte) (id string, ok bool) {
	if len(body) == 0 {
		return "", false
	}
	handle := m.pageOut(ctx, body)
	if handle.Digest == "" {
		// No CAS backend resolved a durable handle: refuse rather than mint a
		// held id that can never page back in (would be a silent body loss).
		return "", false
	}
	id = m.capPagedNext()
	m.mu.Lock()
	m.held[id] = handle
	m.order = append(m.order, id)
	abi.PinResolved(handle) // pin under m.mu, exactly as quarantineResult does
	m.evictExcessLocked()
	// evictExcessLocked may have just aged this very id out if maxHeld is tiny;
	// report failure so the caller does not believe a non-resolvable id is held.
	_, stillHeld := m.held[id]
	m.mu.Unlock()
	if !stillHeld {
		return "", false
	}
	return id, true
}

// PageInBody restores a paged-out capability body for id, but ONLY after a
// witness Clear(id) — the identical gate quarantine's PageIn enforces. This is
// the "re-fault is a FAULT" round-trip: an evicted body pages back in through
// the witness, never silently. An uncleared or unknown/aged-out id is refused.
func (m *MMU) PageInBody(ctx context.Context, id string) ([]byte, error) {
	// PageIn already enforces held + cleared + a resolvable handle; reuse it so
	// the capability body and a quarantine share one gated re-admission path.
	return m.PageIn(ctx, id)
}

// IsHeld reports whether id is currently in the held ledger (paged out, not yet
// aged out by the FIFO bound). The residency tracker uses it to tell an evicted
// capability whose body is still recoverable (held) from one whose handle aged
// out (a re-fault must re-resolve from the source, not the CAS). Read-only.
func (m *MMU) IsHeld(id string) bool {
	m.mu.Lock()
	_, ok := m.held[id]
	m.mu.Unlock()
	return ok
}

// CapPaged reports the lifetime count of capability bodies paged out via
// PageOutBody — the C3 peer of Quarantine/Digested. It lets the residency
// tracker's snapshot reconcile its evict count against the gate's own counter
// (the witness reconciliation C6's LoaderJournal performs at the audit tier).
func (m *MMU) CapPaged() int64 { return atomic.LoadInt64(&m.capPaged) }
