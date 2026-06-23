package ctxresidency

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// State is the residency class of one context span — the read-side view of the
// coherence state kvmmu's write side (Append / AdmitResult / Quarantine / evict)
// already maintains. It is the promotion of ctxmmu's observability counters
// (Held / HeldLen / Evicted / PollutionRate) to a first-class, per-span read.
type State string

const (
	// StateResident: the span's K/V is in the kernel-owned cache AND evicting it
	// would invalidate one or more live cachemeta dependents (a derived
	// attention_index that parents this span's K/V). It is held in residence by
	// those references; EvictBlastRadius.DependentEntries > 0.
	StateResident State = "resident"
	// StateEvictable: the span's K/V is in the cache but no live cachemeta entry
	// parents it — a clean eviction candidate. EvictBlastRadius is its own token
	// count; DependentEntries == 0.
	StateEvictable State = "evictable"
	// StateHeld: the span was quarantined — its K/V span evicted from the cache
	// (kvmmu) and its bytes sealed out of context (ctxmmu). Tokens == 0.
	StateHeld State = "held"
)

// BlastRadius is what evicting a resident/evictable span would cost: the K/V
// positions removed from the cache and the live cachemeta entries (e.g. derived
// attention_index views) that would be invalidated with it. It is the read-only
// projection of the SAME invalidation walk kvmmu.evict performs on a real
// eviction (AttentionIndexReferences + the provider/remote self-reference) — so
// the blast radius a query reports is exactly the one an Evict would incur.
type BlastRadius struct {
	Tokens           int // K/V positions an eviction would drop from the cache.
	DependentEntries int // live cachemeta entries an eviction would invalidate.
}

// Span is one row of the context-residency snapshot. It carries the residency
// fields a >100k-token operator/agent query needs: which span, how big in K/V,
// where it lives (tier), its state, and what evicting it would cost.
//
// Two fields of the issue's target shape ({reason, bytes}) are layer-specific
// and live at the ctxmmu BYTE tier, not the kvmmu SPAN tier: a held span's
// quarantine reason and its byte length are stamped on the byte-quarantine
// ledger keyed by the ctxmmu quarantine id (q<n>), which the kvmmu span ledger
// (keyed by tool-call id) does not carry. Exposing them per-span would require
// enriching the kvmmu Segment; the byte-level totals are reported on the
// Snapshot (ByteHeld / ByteCleared) instead, and reconcile with ctxmmu.
type Span struct {
	ID               string                  // the kvmmu segment id (the tool-call id).
	Tool             string                  // the tool that produced the span.
	Tokens           int                     // K/V positions occupied; 0 once held.
	Tier             cachemeta.ResidencyTier // DRAM while resident, Disk once paged out.
	State            State
	EvictBlastRadius BlastRadius // meaningful for resident/evictable; zero for held.
}

// Snapshot is a consistent point-in-time read over the context's span residency,
// composed from the kvmmu span ledger (the KV-level coherence state) and the
// ctxmmu byte-quarantine ledger (the text-level quarantine/clearance counts).
//
// The per-span rows come from kvmmu — the authoritative ledger Admit/Evict
// maintains. The byte-level summaries come from ctxmmu. The two describe the
// SAME quarantined spans at two enforcement layers (one decision, two media:
// ctxmmu bars the bytes from the text context, kvmmu bars the K/V from the
// attention state); the witness tests assert their counts reconcile.
type Snapshot struct {
	Spans []Span

	// KV-level accounting. ResidentTokens reconciles with kvmmu.Context.CacheLen;
	// HeldSpans reconciles with kvmmu.Context.Evicted. A witness test asserts
	// both so the query can never miscount vs the kernel's own ledger.
	ResidentTokens int
	HeldSpans      int

	// Byte-level accounting (ctxmmu, the text-tier enforcement). The ctxmmu
	// ledger keys byte quarantines by quarantine id (q<n>), which the kvmmu span
	// ledger does not carry, so clearance is reported as a reconciled COUNT
	// rather than fabricated per span.
	ByteHeld      int     // ctxmmu held byte-quarantines (MMU.HeldLen).
	ByteCleared   int     // ctxmmu witness-cleared, page-in-eligible (len MMU.Cleared).
	PollutionRate float64 // ctxmmu quarantined/total.
}

// Query returns a consistent residency snapshot over the kvmmu span ledger,
// composed with the ctxmmu byte-quarantine counts. It is a pure READ: it touches
// no gate state and mutates nothing, so it cannot launder a poisoned span back
// into context — re-admission still re-screens through ctxmmu.PageIn's
// witness-clear requirement, which this query never satisfies on a caller's
// behalf. A nil mmu yields a KV-only view (byte fields left zero); a nil ctx
// yields an empty snapshot.
func Query(c *kvmmu.Context, mmu *ctxmmu.MMU) Snapshot {
	if c == nil {
		return Snapshot{}
	}
	live := c.Entries()
	out := Snapshot{Spans: make([]Span, 0, len(c.Segments()))}
	for _, s := range c.Segments() {
		row := Span{ID: s.ID, Tool: s.Tool, Tokens: s.Len, Tier: tierOf(s)}
		if s.Held {
			row.State = StateHeld
			out.HeldSpans++
			out.Spans = append(out.Spans, row)
			continue
		}
		out.ResidentTokens += s.Len
		deps := countDependents(live, s.KV)
		row.EvictBlastRadius = BlastRadius{Tokens: s.Len, DependentEntries: deps}
		if deps > 0 {
			row.State = StateResident
		} else {
			row.State = StateEvictable
		}
		out.Spans = append(out.Spans, row)
	}
	if mmu != nil {
		out.ByteHeld = mmu.HeldLen()
		out.ByteCleared = len(mmu.Cleared())
		_, _, out.PollutionRate = mmu.PollutionRate()
	}
	return out
}

// tierOf returns the residency tier of a span. A resident span's K/V lives in
// the kernel-owned cache (TierDRAM, matching cachemeta.FromKVPrefix); a held
// span's bytes were paged out of context to the content-addressed store
// (TierDisk, matching cachemeta's residencyOfRef over a paged Ref). No model
// call, no fabrication — the tier is a structural property of where the bytes
// physically live at each state.
func tierOf(s *kvmmu.Segment) cachemeta.ResidencyTier {
	if s.Held || s.Len == 0 {
		return cachemeta.TierDisk
	}
	return cachemeta.TierDRAM
}

// countDependents is the read-only projection of kvmmu.evict's reference
// invalidation: how many live cachemeta entries parent this span's K/V and would
// be invalidated by its eviction. It mirrors kvmmu.externalEntryReferencesKV
// (the provider/remote self-reference) alongside cachemeta.AttentionIndexReferences
// (the attention_index parent reference) so the count equals what an Evict drops.
func countDependents(live []cachemeta.Entry, kv cachemeta.EntryID) int {
	if !kv.Valid() {
		return 0
	}
	n := 0
	for _, e := range live {
		if cachemeta.AttentionIndexReferences(e, kv) || externalRefMatches(e, kv) {
			n++
		}
	}
	return n
}

// externalRefMatches mirrors kvmmu.externalEntryReferencesKV: an entry whose own
// identity IS the evicted K/V and that lives on a remote/provider tier is a
// self-reference that eviction must invalidate.
func externalRefMatches(e cachemeta.Entry, kv cachemeta.EntryID) bool {
	if e.ID != kv {
		return false
	}
	return e.Residency.Tier == cachemeta.TierProvider || e.Residency.Tier == cachemeta.TierRemote
}
