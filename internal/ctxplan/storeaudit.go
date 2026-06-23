package ctxplan

import "sort"

// storeaudit.go — STORE-LEVEL faithfulness: the honesty witness the candidate index opened
// the need for. Audit (faithful.go) proves a plan's Selected/Elided sets partition the
// PROBED candidate set. With the index (index.go) the planner scores a bounded probe, not
// the whole store, so spans outside the probe are PRUNED — absent from the plan's accounting
// entirely. That is honest by design (a pruned span stays in the lossless store, demand-
// pageable, still gated), but Audit alone cannot SEE it: the pruned set never reaches the
// plan. StoreAudit closes that gap with a checkable, store-scoped statement — resident ∪
// elided ∪ pruned partitions the WHOLE store, and every pruned span carries a recovery
// handle — so "index pruning is a forecast miss, never a lost fact" is a witness, not a
// comment (issue #565).
//
// It is the index analogue of Audit: where Audit asks "did the PLAN drop a candidate?",
// StoreAudit asks "did PRUNING drop a store span with no way back?". A faithful store view
// destroys nothing — every span the resident view leaves out, whether the planner elided it
// (saw it, chose cold) or the index pruned it (never scored it this turn), is recoverable.
//
// # The recovery handle is the id, so the store's ids must be unique
//
// A pruned span pages back in by its ID (the key Store.Materialize / recall.Resolve look it
// up by). If two distinct store spans shared an id that handle would be AMBIGUOUS — paging
// "the pruned span" back could return the wrong bytes — so a store with duplicate ids cannot
// honestly be certified recoverable. StoreAudit DETECTS duplicate store ids and refuses to
// certify (DuplicateStoreIDs, Partition=false), the store-scope counterpart of the index's
// unique-id addressing contract (index.go). Every shipped store assigns unique ids
// ("span:<n>", "page:<step>"), so this never fires in practice; it is the fail-closed guard
// for a malformed store, not a normal path.

// StoreWitness is the store-level faithfulness audit of an index-bounded plan against the
// full store it was planned over. A view is store-faithful iff (a) the resident, elided, and
// pruned sets PARTITION every store span (each span in exactly one), (b) every pruned span
// carries a recovery handle (its id — the store's page-back-in key — so it is cold, not
// destroyed), (c) the store's ids are unique (so that handle is unambiguous), and (d) the
// plan itself is faithful (Audit), so the elided half is recoverable too.
type StoreWitness struct {
	StoreSpans        int      `json:"store_spans"`                   // |S| — every span in the lossless store
	Resident          int      `json:"resident"`                      // spans the plan kept in the O(1) view (Selected)
	Elided            int      `json:"elided"`                        // spans the plan SAW and kept cold (Elided — probed, not selected)
	Pruned            int      `json:"pruned"`                        // spans the INDEX never scored this turn (in neither plan set)
	PrunedTokens      int      `json:"pruned_tokens"`                 // resident-token cost of the pruned set (cold, recoverable)
	Recoverable       int      `json:"recoverable"`                   // pruned spans WITH a recovery handle (a non-blank id) — must equal Pruned to be Faithful
	Unrecoverable     []string `json:"unrecoverable,omitempty"`       // pruned spans with NO handle (a blank-id store span — destroyed/unaddressable)
	Foreign           []string `json:"foreign,omitempty"`             // plan ids not present in the store (a plan over a different/stale store) — a set
	DuplicateStoreIDs []string `json:"duplicate_store_ids,omitempty"` // ids carried by >1 store span (an ambiguous recovery handle) — refused
	Partition         bool     `json:"partition"`                     // resident+elided+pruned == StoreSpans, no foreign plan id, no duplicate/blank store id
	Faithful          bool     `json:"faithful"`                      // Partition AND every pruned span recoverable AND the plan itself Faithful
	PlanFaithful      bool     `json:"plan_faithful"`                 // Audit(plan).Faithful — the probed-set partition (folded into Faithful)
}

// StoreAudit computes the store-level faithfulness witness of an index-bounded plan against
// the full set of store spans it was planned over. It classifies every store span as
// resident (in the plan's Selected set), elided (in the plan's Elided set — the planner saw
// and chose to keep it cold), or pruned (in neither — the index never scored it this turn),
// and checks the three sets partition the store with every pruned span recoverable. It reads
// only ids and the plan's own accounting plus the SAFE store metadata (no bytes are paged
// in), so it is a cheap, deterministic check a caller runs before trusting that an index-
// bounded view left the lossless store intact.
//
// It fails closed on two malformed-store conditions, since both make the id recovery handle
// unusable: a DUPLICATE store id (the handle is ambiguous — which physical span pages back?)
// and a BLANK store id (unaddressable). It also fails on a FOREIGN plan id — a resident/elided
// id that names no store span — which means the plan was built over a different or stale
// store than the one audited, so the witness cannot reconcile it.
func StoreAudit(spans []Span, p Plan) StoreWitness {
	w := StoreWitness{StoreSpans: len(spans)}

	selected := make(map[string]bool, len(p.Selected))
	for _, s := range p.Selected {
		if s.ID != "" {
			selected[s.ID] = true
		}
	}
	elided := make(map[string]bool, len(p.Elided))
	for _, e := range p.Elided {
		if e.ID != "" {
			elided[e.ID] = true
		}
	}

	// Store id census: which ids exist, and which collide. A colliding id is an ambiguous
	// recovery handle, so the store cannot be certified (fail-closed).
	idCount := make(map[string]int, len(spans))
	for _, s := range spans {
		if s.ID != "" {
			idCount[s.ID]++
		}
	}
	for id, n := range idCount {
		if n > 1 {
			w.DuplicateStoreIDs = append(w.DuplicateStoreIDs, id)
		}
	}
	sort.Strings(w.DuplicateStoreIDs)

	// Classify every store span into exactly one of resident / elided / pruned. (On a store
	// with duplicate ids these id-keyed counts are not per-span trustworthy, but the
	// DuplicateStoreIDs guard above already fails the partition, so a duplicate-id store is
	// never certified regardless of the counts.)
	wellFormed := true
	for _, s := range spans {
		if s.ID == "" {
			// A store span with no id cannot be addressed, classified, or paged back in —
			// a malformed store. It breaks the partition (recovery handle absent).
			wellFormed = false
			w.Unrecoverable = append(w.Unrecoverable, "(blank-id store span)")
			continue
		}
		switch {
		case selected[s.ID]:
			w.Resident++
		case elided[s.ID]:
			w.Elided++
		default:
			// Pruned: the index never scored this span this turn. It is NOT destroyed — it
			// stays in the lossless store, recoverable by its handle (its id, the key the
			// store pages it back in by) and still guarded by the trust gate on any page-in.
			// A blank-id span was already counted Unrecoverable above (the loop `continue`d on
			// it), so a span reaching here is addressable by definition — its id is the
			// recovery handle (the store usually also carries a Digest, a secondary handle the
			// witness does not separately require). That guaranteed recoverability IS the
			// honesty claim ("pruning is a forecast miss, never a lost fact"), checked here.
			w.Pruned++
			w.PrunedTokens += TokenCost(s)
			w.Recoverable++
		}
	}

	// A plan id (resident or elided) that names no store span is FOREIGN — the plan was built
	// over a different or stale store, so the witness cannot reconcile it. Collected as a SET.
	foreign := map[string]bool{}
	for id := range selected {
		if idCount[id] == 0 {
			foreign[id] = true
		}
	}
	for id := range elided {
		if idCount[id] == 0 {
			foreign[id] = true
		}
	}
	for id := range foreign {
		w.Foreign = append(w.Foreign, id)
	}
	sort.Strings(w.Foreign)

	// The three sets partition the store iff their counts sum to |S| (every span classified
	// exactly once — disjointness is structural: a store span is matched against selected
	// first, then elided, then pruned, so it lands in exactly one bucket) AND no plan id is
	// foreign AND the store is well-formed: no blank id and no duplicate id (both make the
	// recovery handle unusable).
	w.Partition = wellFormed && len(w.Foreign) == 0 && len(w.DuplicateStoreIDs) == 0 &&
		(w.Resident+w.Elided+w.Pruned == w.StoreSpans)

	pa := Audit(p)
	w.PlanFaithful = pa.Faithful
	// Store-faithful iff the store partitions cleanly, every pruned span is recoverable
	// (Recoverable == Pruned — load-bearing, not decorative), AND the plan's own probed-set
	// partition is faithful (so the elided half is recoverable too). A compaction-style plan
	// (elided spans stripped of handles, CompactionView) makes PlanFaithful false and fails
	// the store audit too — the same contrast Audit draws, lifted to store scope.
	w.Faithful = w.Partition && len(w.Unrecoverable) == 0 && w.Recoverable == w.Pruned && w.PlanFaithful
	return w
}
