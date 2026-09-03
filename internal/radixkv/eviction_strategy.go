package radixkv

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// eviction_strategy.go opens the budget-pressure victim rule from a closed 2-value enum
// into SGLang's shape (@7090a49 python/sglang/srt/mem_cache/evict_policy.py:10-65): a
// strategy is ONE method, Priority(node) -> victimKey, and the eviction loop is a single
// argmin over that key — the tuple ordering does the tiering, so there is no bespoke loop
// per policy. A string-keyed factory registry (get_eviction_strategy in SGLang) resolves a
// policy by name and FAILS CLOSED on an unknown one, listing the supported set. Adding the
// next policy is a ~5-line registered strategy here, not an enum edit fanning out to two
// switches and a per-policy counter. The legacy EvictionPolicy enum (radixkv.go) stays as
// the two-value back-compat knob and routes THROUGH this registry, so both paths share one
// victim seam.

// victimKey is a total-order eviction priority: among all evictable leaves the tree evicts
// the one with the LEAST key first — fak's form of SGLang's
// heappush((strategy.get_priority(node), node)) min-heap, where the tuple ordering does the
// tiering. The fields compare lexicographically, exactly like SGLang's comparator tuples:
//
//   - seg is the coarse protection segment: every leaf in a lower segment evicts before any
//     leaf in a higher one, regardless of the finer keys. This is the lever SLRU pulls — a
//     probationary leaf (seg 0) always precedes a protected leaf (seg 1), so a burst of
//     one-off cold inserts cannot flush a frequently-hit prefix (scan resistance).
//   - cost is a within-segment score (lower evicts first). Cost-aware eviction puts its
//     value-of-keeping score here; recency-only strategies leave it 0.
//   - age is the final LRU tie-break: the oldest lastUsed evicts first, so any strategy that
//     leaves seg and cost uniform degenerates to pure LRU victim choice.
type victimKey struct {
	seg  int
	cost float64
	age  uint64
}

// less reports whether k should be evicted before o (k is the "smaller", lower-priority
// key). Lexicographic over (seg, cost, age); equal keys are neither less, so the eviction
// argmin keeps the first-seen leaf on an exact tie — the same stability the pure-LRU and
// cost-aware loops had.
func (k victimKey) less(o victimKey) bool {
	if k.seg != o.seg {
		return k.seg < o.seg
	}
	if k.cost != o.cost {
		return k.cost < o.cost
	}
	return k.age < o.age
}

// VictimStrategy is the open eviction seam: a policy is one Priority method returning the
// total-order victimKey the tree evicts the minimum of. It is SGLang's get_priority(node)
// contract. The method takes the unexported *node deliberately — a strategy reads a leaf's
// resident telemetry (hits, recency, span cost) exactly as the two seed strategies do, so
// new policies register in-package alongside them.
type VictimStrategy interface {
	// Name is the strategy's registry key — also what Stats.EvictionPolicy reports.
	Name() string
	// Priority scores one evictable leaf; the tree evicts the leaf with the least key.
	Priority(n *node) victimKey
}

// lruStrategy is pure recency: evict the least-recently-used leaf. Seg and cost stay 0, so
// the argmin reduces to the oldest lastUsed — byte-identical to the historical lruLeaf.
type lruStrategy struct{}

func (lruStrategy) Name() string               { return "lru" }
func (lruStrategy) Priority(n *node) victimKey { return victimKey{age: n.lastUsed} }

// costAwareStrategy evicts the cheapest-to-lose leaf per compute.KVEvictionCost (#2239):
// recompute-cost × reuse-probability ÷ bytes, LRU tie-break. It puts that score in cost and
// leaves seg 0, so the argmin reproduces PickEvictionVictim's ranking (lowest cost, oldest
// lastUsed on a tie) exactly — the value-aware generalization of LRU, unchanged.
type costAwareStrategy struct{}

func (costAwareStrategy) Name() string { return "cost-aware" }
func (costAwareStrategy) Priority(n *node) victimKey {
	return victimKey{cost: compute.KVEvictionCost(n.kvSpanStats()), age: n.lastUsed}
}

// slruProtectThreshold is SLRU's promotion bar: a leaf hit at least this many times (SGLang
// uses hit_count >= 2) is PROTECTED and evicts only after every probationary leaf is gone.
const slruProtectThreshold = 2

// slruStrategy is the scan-resistant 2-segment policy borrowed from SGLang's SLRUStrategy
// (comparator (is_protected, last_access_time)). A leaf with hits < protectThreshold is
// probationary (seg 0) and always evicts before any protected leaf (seg 1) REGARDLESS of
// recency, so a burst of one-off cold inserts cannot flush a hot prefix — the protected
// segment survives while an older low-frequency prefix in probation is reclaimed. Within a
// segment the oldest lastUsed goes first (LRU).
type slruStrategy struct{ protectThreshold int }

func (slruStrategy) Name() string { return "slru" }
func (s slruStrategy) Priority(n *node) victimKey {
	seg := 0
	if n.hits >= s.protectThreshold {
		seg = 1 // protected: strictly after every probationary leaf
	}
	return victimKey{seg: seg, age: n.lastUsed}
}

// TreePreparer is an optional interface a VictimStrategy can implement to inspect
// the tree and precompute page/chunk occupancy before victim selection.
type TreePreparer interface {
	PrepareTree(t *Tree)
}

// lowestScoreFirstStrategy is the value-aware LowestScoreFirst / cost-aware strategy
// providing explicit backward compatibility with Issue #10721 requirements.
type lowestScoreFirstStrategy struct{ costAwareStrategy }

func (lowestScoreFirstStrategy) Name() string { return "lowest-score-first" }

// victimStrategies is the string-keyed factory registry (SGLang's get_eviction_strategy
// table). Seed entries reproduce the two legacy enum behaviors byte-for-byte; slru is the
// first policy the open seam adds. RegisterVictimStrategy extends it in-package.
var victimStrategies = map[string]func() VictimStrategy{
	"lru":                func() VictimStrategy { return lruStrategy{} },
	"cost-aware":         func() VictimStrategy { return costAwareStrategy{} },
	"lowest-score-first": func() VictimStrategy { return lowestScoreFirstStrategy{costAwareStrategy{}} },
	"slru":               func() VictimStrategy { return slruStrategy{protectThreshold: slruProtectThreshold} },
	"page-aware":         func() VictimStrategy { return NewPageAwareStrategy(nil) },
	"page-aligned":       func() VictimStrategy { return NewPageAwareStrategy(nil) },
	"chunk-aware":        func() VictimStrategy { return NewPageAwareStrategy(nil) },
}

// RegisterVictimStrategy adds (or replaces) a named eviction strategy. Call it from an init
// so the name is resolvable before any tree selects it. This is the extension point the
// closed enum lacked: a new policy is a factory registration, not an enum-plus-two-switches
// edit.
func RegisterVictimStrategy(name string, factory func() VictimStrategy) {
	victimStrategies[name] = factory
}

// victimStrategyByName resolves a registered strategy, FAILING CLOSED on an unknown name
// with an error that lists the supported set (SGLang's get_eviction_strategy raise). A typo
// can never silently widen behavior into an unintended policy.
func victimStrategyByName(name string) (VictimStrategy, error) {
	if factory, ok := victimStrategies[name]; ok {
		return factory(), nil
	}
	return nil, fmt.Errorf("radixkv: unknown eviction strategy %q (supported: %s)", name, supportedStrategyNames())
}

// supportedStrategyNames lists the registered strategy keys, sorted for a deterministic
// error message and operator listing.
func supportedStrategyNames() string {
	names := make([]string, 0, len(victimStrategies))
	for name := range victimStrategies {
		names = append(names, name)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// evictionStrategy is the tree's current victim rule, defaulting an unset tree (a bare
// New()) to pure LRU so the zero value keeps its historical behavior.
func (t *Tree) evictionStrategy() VictimStrategy {
	if t.strategy == nil {
		return lruStrategy{}
	}
	return t.strategy
}

// SetEvictionStrategy selects the budget-pressure victim rule by REGISTERED NAME for future
// evictions only — the open-seam, operator-facing sibling of SetEvictionPolicy. It fails
// CLOSED: an unregistered name returns an error listing the supported set and leaves the
// current strategy unchanged, so an operator typo can never silently pick the wrong policy.
// The legacy cost-aware counter (Stats.CostEvictions) still tracks the "cost-aware" strategy.
func (t *Tree) SetEvictionStrategy(name string) error {
	s, err := victimStrategyByName(name)
	if err != nil {
		return err
	}
	t.strategy = s
	if tb, ok := s.(interface{ bindTree(*Tree) }); ok {
		tb.bindTree(t)
	}
	if t.pageTracker != nil {
		if ts, ok := s.(interface{ SetTracker(PageOccupancyTracker) }); ok {
			ts.SetTracker(t.pageTracker)
		}
	}
	switch s.Name() {
	case "cost-aware", "lowest-score-first":
		t.policy = EvictionCostAware
		t.SetAdmissionEnabled(true)
	case "page-aware", "page-aligned", "chunk-aware":
		t.policy = EvictionPageAware
		t.SetAdmissionEnabled(false)
	default:
		t.policy = EvictionLRU
		t.SetAdmissionEnabled(false)
	}
	return nil
}

// NewWithEvictionStrategy builds a tree whose victim rule is the registered strategy `name`
// — the string-knob constructor sibling of NewWithEvictionPolicy. It fails closed on an
// unknown name (the tree is not created), mirroring SetEvictionStrategy.
func NewWithEvictionStrategy(maxTokens int, name string) (*Tree, error) {
	t := New(maxTokens)
	if err := t.SetEvictionStrategy(name); err != nil {
		return nil, err
	}
	return t, nil
}
