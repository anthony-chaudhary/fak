package ctxplan

// maintain.go — incremental index MAINTENANCE: the operations that keep a single
// long-lived Index consistent as a session runs, so the live loop maintains ONE index
// (O(tokens) per turn via Add) instead of rebuilding it from the store every turn. The
// rebuild-per-turn alternative re-scans all N spans each turn — Θ(N²) cumulative — which is
// exactly the cost the candidate index exists to flatten (index.go). The flatten is real
// only if the index is MAINTAINED, never rebuilt; this file is that maintenance surface, and
// maintain_test.go proves an incrementally-maintained index is STRUCTURALLY IDENTICAL to a
// fresh BuildIndex over the same final span set (issue #558).
//
// # Why the surface is so small (a content-immutability CONTRACT, defended where it can be)
//
// A recorded span's content is meant to be immutable: it is content-addressed (Digest), so
// its bytes, its extractive descriptor, its token set, and its durability class are fixed
// once recorded. This is a CALLER CONTRACT, not a guarantee the Span type enforces (its
// fields are exported) — callers MUST treat a recorded span's content fields (Role,
// Descriptor, Digest, Bytes, Durability, Attrs) as immutable. The index defends the one part
// it can: Add clones the reference-type field (Attrs) so a caller cannot mutate the index's
// stored metadata through a shared map (index.go), and Spans returns cloned copies. Given the
// contract, the ONLY mutation a recorded span undergoes through this API is its
// TRUST/SUPPRESSION flag — the trust gate Seals a quarantined result, context control
// Tombstones a suppressed span. So the complete maintenance surface is Add (a new turn's
// span) plus SetSealed / SetTombstoned (a flag flip on an existing span). A flag flip touches
// neither the posting lists nor the durable set (both derive only from the immutable content
// + durability), so the index stays structurally identical to a rebuild — that is the
// structural reason the equivalence witness holds, under the unique-id addressing contract
// (Add's doc): the mutators address a span by its id, so ids must be unique.
//
// A span that is Sealed/Tombstoned is deliberately NOT removed from the posting lists: Probe
// still surfaces it, and Optimize then elides it sealed/tombstoned (the planner's audit
// records it, so the partition stays honest), exactly as a full-scan plan would. Suppression
// is enforced at SCORING (Benefit == 0) and at the page-in trust gate, never by erasing the
// span from the index — an index that silently dropped a sealed span would hide it from the
// audit it must appear in.

// SetTombstoned marks the span addressed by id as suppressed by context control, mutating
// only its Tombstoned flag. It returns whether id named a span in the index. The index
// addresses by id (Add's unique-id contract): a duplicate id resolves to the MOST-RECENTLY-
// added span, so ids must be unique for the mutation to be unambiguous (every shipped store
// guarantees it). Idempotent: a second call is a no-op. The posting lists and durable set are
// untouched (suppression does not change the span's content or durability), so the index
// stays equivalent to one built fresh from the post-mutation span set — the property the live
// loop relies on to maintain a single index across turns instead of rebuilding it.
func (ix *Index) SetTombstoned(id string) bool {
	i, ok := ix.byID[id]
	if !ok {
		return false
	}
	ix.spans[i].Tombstoned = true
	return true
}

// SetSealed marks the span addressed by id as quarantined by the trust gate, mutating only
// its Sealed flag. It returns whether id named a span in the index. Like SetTombstoned it
// addresses by id (unique-id contract) and leaves the posting lists and durable set
// unchanged, so a sealed span is still PROBED (so the plan can record it elided-sealed for
// audit) but scores 0 in Benefit and is never selected — the poison-never-resident invariant,
// enforced at scoring + the page-in gate, never by erasing the span from the index.
func (ix *Index) SetSealed(id string) bool {
	i, ok := ix.byID[id]
	if !ok {
		return false
	}
	ix.spans[i].Sealed = true
	return true
}

// Spans returns a defensive copy of the index's span table in append (step) order — the SAFE
// metadata image the index maintains. Each returned Span is a value copy with its Attrs map
// cloned, so a caller cannot mutate the index's internal table (or its scoring inputs)
// through the result. It is the accessor a persistence layer serializes (the index is pure
// metadata: this table plus the inverted token map + durable set, all rederivable from it via
// BuildIndex) and the accessor a store-level audit reconciles a plan against (storeaudit.go).
func (ix *Index) Spans() []Span {
	out := make([]Span, len(ix.spans))
	copy(out, ix.spans)
	for i := range out {
		out[i].Attrs = cloneAttrs(out[i].Attrs)
	}
	return out
}

// cloneAttrs returns an independent copy of a span's open Attrs bag (the one reference-type
// field on Span), or nil for a nil/empty map. It is what lets Add and Spans own their
// metadata so a caller's later mutation of its own map can never reach into the index — the
// defense that makes the content-immutability contract structurally true for Attrs, the only
// field the value-copy of a Span would otherwise alias.
func cloneAttrs(a map[string]string) map[string]string {
	if len(a) == 0 {
		return nil
	}
	c := make(map[string]string, len(a))
	for k, v := range a {
		c[k] = v
	}
	return c
}
