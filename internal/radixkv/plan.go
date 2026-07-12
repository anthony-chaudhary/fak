package radixkv

// plan.go — ratio-capped bounded eviction with a confirmed-delete ledger (#3387):
// evictToBudget is a CLIFF — one call drains EVERY over-budget victim in a single pass,
// so a large overage becomes a single long stall on whichever request triggered it. This
// file adds the bounded alternative alongside (evictToBudget is untouched), technique-
// inspired (clean-room) by LMCache's eviction_manager ratio-per-pass drain.
//
// Mechanism — SAME victims, smaller bites, receipted deletes:
//
//   - PlanBoundedEviction(ratio) is one TICK of the drain. It selects victims through the
//     SAME t.victimLeaf() selector evictToBudget uses, on the same evolving tree, so victim
//     identity and order are identical to the cliff by construction — it just STOPS after
//     at most cap = max(1, ceil(ratio*N)) victims (N = resident leaves at plan time, the
//     selector's candidate set), or sooner if budget is reached. A K-victim overage drains
//     over several ticks instead of one stall, and every side effect the cliff performs per
//     victim (thrash note, eviction counters, recordEvictChoice via the selector) happens
//     here too, so the two planes are indistinguishable to the observability rungs.
//
//   - The CONFIRMED-DELETE LEDGER is the two-phase receipt: each staged victim is recorded
//     {tokens, plen} in FIFO order BEFORE its detach, and stays PENDING until the caller
//     calls ConfirmEvictions — the reconcile step that settles the confirmed counters and
//     clears the entries, each exactly once. A partial tick (planned but not confirmed)
//     leaves consistent accounting: the detach already happened (Stats.Tokens is truthful),
//     the pending entries carry the record, and the next tick resumes staging AFTER them —
//     a staged victim is out of the tree, so it can never be re-picked (no double-evict),
//     and it is never dropped from the ledger except by its own confirm (no lost victim).
//
// BOUND (the documented contract, guarded by TestBoundedEvictionLedgerBounded): the ledger
// holds at most evictionLedgerCap pending records. A caller that stages and never confirms
// back-pressures the plane — staging capacity hits zero and PlanBoundedEviction stages
// nothing further (the tree stays over budget rather than the ledger growing without
// bound) — it never silently discards a pending receipt.
//
// Deterministic and wall-clock-free: no time.Now(); ticks are caller-driven calls under the
// tree's existing caller-serialized discipline, and victim order inherits the selectors'
// determinism (strict lastUsed / cost minima).

import "math"

// evictionLedgerCap is the hard entry bound on the confirmed-delete ledger: at most this
// many staged-unconfirmed victims may be pending. Staging capacity is exhausted, never the
// bound; PlanBoundedEviction stops staging when the ledger is full.
const evictionLedgerCap = 256

// evictionRecord is one staged victim awaiting confirmation: how many edge tokens its
// detach freed and the full-prefix length it cached — the accounting a confirm settles.
type evictionRecord struct {
	tokens int // victim edge tokens freed by the detach (the budget metric)
	plen   int // victim full-prefix length at staging time
}

// EvictionPlan is the summary of one PlanBoundedEviction tick.
type EvictionPlan struct {
	OverBudget   int // tokens above budget when the tick was planned (0 = at/under budget)
	Candidates   int // resident leaves N the ratio was applied to
	Cap          int // this tick's victim cap = max(1, ceil(ratio*Candidates))
	Staged       int // victims staged (and detached) this tick
	StagedTokens int // Σ victim edge tokens over this tick's staged victims
	Pending      int // ledger entries awaiting ConfirmEvictions after this tick
	Remaining    int // tokens still above budget after this tick (>0 = more ticks needed)
}

// PlanBoundedEviction runs ONE ratio-capped eviction tick: it stages at most
// max(1, ceil(ratio*N)) victims (N = resident leaves) into the confirmed-delete ledger,
// detaching each from the tree exactly as evictToBudget would — same selector, same order,
// same per-victim side effects — and stops early once the cached-token count is within
// budget or the ledger is full. It returns the tick's summary; the caller settles the
// staged receipts with ConfirmEvictions.
//
// ratio is clamped to (0, 1]: a ratio > 1 behaves as 1 (one full-candidate-set tick), and
// a non-positive or NaN ratio stages nothing (an explicitly disabled tick, still reporting
// OverBudget/Pending). An at-budget, under-budget, or keep-all tree returns an empty plan
// with zero victims staged. Victims the selector refuses (leased leaves) bound the tick
// exactly as they bound the cliff: staging stops and Remaining reports the shortfall.
func (t *Tree) PlanBoundedEviction(ratio float64) EvictionPlan {
	plan := EvictionPlan{Pending: len(t.evictLedger)}
	budget, evicts := t.resolveBudget()
	if !evicts || t.tokens <= budget {
		return plan // at/under budget (or keep-all): zero victims, empty plan
	}
	plan.OverBudget = t.tokens - budget
	plan.Remaining = plan.OverBudget
	if math.IsNaN(ratio) || ratio <= 0 {
		return plan // disabled tick: report the pressure, stage nothing
	}
	if ratio > 1 {
		ratio = 1
	}
	plan.Candidates = t.leafCount()
	plan.Cap = int(math.Ceil(ratio * float64(plan.Candidates)))
	if plan.Cap < 1 {
		plan.Cap = 1
	}
	capacity := evictionLedgerCap - len(t.evictLedger)
	if capacity > plan.Cap {
		capacity = plan.Cap
	}
	for t.tokens > budget && plan.Staged < capacity {
		v := t.victimLeaf()
		if v == nil {
			break // everything in budget-excess is leased; same bail as the cliff
		}
		t.noteEviction(v) // thrash detector (#3393): a bounded budget eviction can thrash too
		t.evictLedger = append(t.evictLedger, evictionRecord{tokens: len(v.key), plen: v.plen})
		t.removeLeaf(v)
		t.evictions++
		if t.policy == EvictionCostAware {
			t.costEvictions++
		}
		t.boundedEvictions++
		plan.Staged++
		plan.StagedTokens += len(v.key)
	}
	plan.Pending = len(t.evictLedger)
	plan.Remaining = 0
	if t.tokens > budget {
		plan.Remaining = t.tokens - budget
	}
	return plan
}

// ConfirmEvictions settles the confirmed-delete ledger: every pending staged victim is
// confirmed exactly once — the confirmed counters advance by its record and the entry is
// cleared — and the number of victims confirmed by THIS call is returned. Confirming with
// nothing pending is a no-op returning 0, so re-running the reconcile can never
// double-count. Frees the ledger's staging capacity for the next tick.
func (t *Tree) ConfirmEvictions() int {
	n := len(t.evictLedger)
	for _, rec := range t.evictLedger {
		t.ledgerConfirmed++
		t.ledgerConfirmedTokens += rec.tokens
	}
	t.evictLedger = t.evictLedger[:0] // capacity retained; bounded by evictionLedgerCap
	return n
}

// leafCount reports the resident eviction-candidate count: non-root leaves, leased or not —
// the same candidate set the victim selectors enumerate. It is the N a tick's ratio cap is
// taken over.
func (t *Tree) leafCount() int {
	count := 0
	var stack []*node
	t.forEachRoot(func(r *node) { // candidate leaves span every namespace root
		for _, c := range r.children {
			stack = append(stack, c)
		}
	})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if len(n.children) == 0 {
			count++
			continue
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	return count
}
