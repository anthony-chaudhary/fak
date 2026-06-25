package vcachechain

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// chain.go is the prefix DAG — the recall substrate (§12 net-new #1). A ChainNode
// is one vBlock; its PARENT is the prefix it extends, and the path root→node is the
// chain vCache replays to recall this unit. The DAG carries the recall plan (parent
// chain per vBlock), not just identity: cachemeta's KVManifest binds what a span IS;
// PrefixDAG binds how a span is RECONSTRUCTED from its ancestors.
//
// All functions here are pure and deterministic. The caller injects everything
// (token counts, block counts, the warm depth M1 derived from
// cachemeta.FirstDivergeTokenOffset); nothing in this file issues a network call,
// reads a clock, or stores a payload.

// ChainNode is one vBlock in the prefix DAG.
type ChainNode struct {
	// ID is the node's identity (the vBlock's manifest key, scope by the full
	// identity tuple per Law B2). Unique within a PrefixDAG.
	ID string
	// ParentID is the node's parent in the prefix chain; "" denotes the root/anchor.
	// A chain is replayed root→node so the provider serves every ancestor from cache
	// and freshly prefills only this node's tail.
	ParentID string
	// Tokens is the span length THIS node adds beyond its parent (the fresh prefill
	// a cold recall of this node pays). The prefix length a recall replays is the
	// sum of every ANCESTOR's Tokens (PrefixDAG.PrefixTokens).
	Tokens int64
	// Blocks is the number of provider content blocks this node contributes. The
	// 20-block lookback (§8 + Rule C3) sums blocks along the chain to decide where
	// intermediate breakpoints must go; a chain adding >20 blocks between breakpoints
	// silently misses.
	Blocks int
	// Secret is the node's Law-D4 content class. A chain is warmable only if EVERY
	// node on it is warmable (vcachegov.Warmable): a secret ancestor makes the whole
	// chain no-cache, because rebuilding it would replay the secret byte-for-byte
	// through the provider prefix cache.
	Secret vcachegov.SecretClassification
}

// PrefixDAG is the parent-chain-per-vBlock recall substrate.
type PrefixDAG struct {
	Nodes []ChainNode
}

// ErrCycle / ErrMissingParent / ErrDuplicateID / ErrMultiRoot are the validation
// failures a malformed DAG raises. A recall plan is only defined over a valid DAG:
// exactly one root, every ParentID resolvable, acyclic.
var (
	ErrCycle         = dagErr("prefix DAG has a cycle")
	ErrMissingParent = dagErr("a node's ParentID resolves to no node")
	ErrDuplicateID   = dagErr("two nodes share an ID")
	ErrMultiRoot     = dagErr("prefix DAG has more than one root (anchor)")
	ErrEmpty         = dagErr("prefix DAG is empty")
	ErrMissingNode   = dagErr("target node id is not in the DAG")
)

type dagErr string

func (e dagErr) Error() string { return string(e) }

// Validate checks the DAG is a single-rooted arborescence: non-empty, unique IDs,
// every ParentID resolvable (except the one root's ""), exactly one root, acyclic.
// A recall plan is only meaningful over a valid DAG; the live loop builds this from
// the cachemeta manifest's parent pointers and validates once on admission.
func (d PrefixDAG) Validate() error {
	if len(d.Nodes) == 0 {
		return ErrEmpty
	}
	byID := make(map[string]ChainNode, len(d.Nodes))
	roots := 0
	for _, n := range d.Nodes {
		if _, dup := byID[n.ID]; dup {
			return ErrDuplicateID
		}
		byID[n.ID] = n
		if n.ParentID == "" {
			roots++
		}
	}
	if roots != 1 {
		return ErrMultiRoot
	}
	for _, n := range d.Nodes {
		if n.ParentID == "" {
			continue
		}
		if _, ok := byID[n.ParentID]; !ok {
			return ErrMissingParent
		}
	}
	// Cycle detection: walk each node's parent chain to the root, bounding the walk
	// by the node count. A chain longer than the node count must contain a cycle.
	for _, n := range d.Nodes {
		cur := n.ParentID
		for steps := 0; steps < len(d.Nodes); steps++ {
			if cur == "" {
				break
			}
			parent, ok := byID[cur]
			if !ok {
				return ErrMissingParent
			}
			cur = parent.ParentID
		}
		if cur != "" {
			return ErrCycle
		}
	}
	return nil
}

// node is the internal lookup helper; callers should Validate first.
func (d PrefixDAG) node(id string) (ChainNode, bool) {
	for _, n := range d.Nodes {
		if n.ID == id {
			return n, true
		}
	}
	return ChainNode{}, false
}

// ChainTo reconstructs the ordered root→node path (the recall chain). The first
// element is the anchor; the last is the target. Prefix replay walks this path in
// order so the provider serves every ancestor from cache.
func (d PrefixDAG) ChainTo(nodeID string) ([]ChainNode, error) {
	if _, ok := d.node(nodeID); !ok {
		return nil, ErrMissingNode
	}
	// Walk parent pointers, then reverse to root→node order.
	var path []ChainNode
	cur, ok := d.node(nodeID)
	if !ok {
		return nil, ErrMissingNode
	}
	seen := map[string]bool{}
	for {
		if seen[cur.ID] {
			return nil, ErrCycle
		}
		seen[cur.ID] = true
		path = append(path, cur)
		if cur.ParentID == "" {
			break
		}
		parent, ok := d.node(cur.ParentID)
		if !ok {
			return nil, ErrMissingParent
		}
		cur = parent
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path, nil
}

// PrefixTokens is the cacheable prefix length for a node — the sum of every
// ANCESTOR's Tokens (the bytes a rebuild replays from cache). The node's own Tokens
// are the fresh prefill a cold recall pays (the unit being recalled).
func (d PrefixDAG) PrefixTokens(nodeID string) (int64, error) {
	chain, err := d.ChainTo(nodeID)
	if err != nil {
		return 0, err
	}
	if len(chain) == 0 {
		return 0, nil
	}
	// All nodes except the last (the target) are the replayed prefix.
	var prefix int64
	for _, n := range chain[:len(chain)-1] {
		prefix += n.Tokens
	}
	return prefix, nil
}

// BlocksTo is the cumulative content-block count root→node (inclusive). It feeds the
// 20-block lookback: PlaceBreakpoints places an intermediate breakpoint every ~15
// cumulative blocks so no span between breakpoints exceeds the ≤20-block walk-back.
func (d PrefixDAG) BlocksTo(nodeID string) (int, error) {
	chain, err := d.ChainTo(nodeID)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, n := range chain {
		total += n.Blocks
	}
	return total, nil
}

// ReplayLevel is one fan boundary of the send-one-then-fan replay (§8 + Rule C2).
// Within a level, Lead is sent FIRST; its first streamed content delta proves its
// cache write is readable, which releases Fan — the siblings that then read the
// cache Lead just wrote. Releasing Fan before Lead's write is observable is the
// ordering race that strands dependents on a cold prefix (§8).
type ReplayLevel struct {
	Lead ChainNode   // sent first; its first streamed token releases Fan
	Fan  []ChainNode // siblings released only after Lead's write is readable
}

// ReplayPlan is the topological replay schedule for a set of targets.
type ReplayPlan struct {
	Levels      []ReplayLevel // send-one-then-fan, parent depth before child depth
	Breakpoints []int         // cumulative-block indices needing an intermediate breakpoint (Rule C3)
}

// TopologicalReplay schedules the recall of one or more targets as a sequence of
// send-one-then-fan fan levels (§8 + Rule C2). It collects the union of every
// target's chain, drops the warm prefix (every node at depth < WarmDepth — already
// cached), and groups the remaining cold nodes by depth. Depth 0 of the cold set is
// the first cold ancestor; its first node is the level's Lead and its siblings are
// the Fan. Each deeper depth is a new fan level released only after the prior level
// writes are readable. A single-target linear chain yields one node per level (Fan
// empty); a multi-target fan-out groups siblings into one level's Fan.
//
// Breakpoints are placed over the FULL chain's cumulative block count (Rule C3),
// not just the cold tail: the 20-block walk-back limits how far back a breakpoint
// can reach, so placement is a property of the whole replayed span.
func (d PrefixDAG) TopologicalReplay(targets []string, warmDepth int) (ReplayPlan, error) {
	cold, err := d.coldSubgraph(targets, warmDepth)
	if err != nil {
		return ReplayPlan{}, err
	}
	// Group cold nodes by depth (rootward-first). Depth is the node's index in its
	// own root→node chain, so grouping by depth puts a parent and its children in
	// consecutive, correctly-ordered fan levels.
	byDepth := map[int][]ChainNode{}
	maxDepth := -1
	for depth, nodes := range cold {
		byDepth[depth] = nodes
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	plan := ReplayPlan{}
	for depth := 0; depth <= maxDepth; depth++ {
		nodes := byDepth[depth]
		if len(nodes) == 0 {
			continue
		}
		sort.Slice(nodes, func(i, j int) bool { return nodes[i].ID < nodes[j].ID })
		lvl := ReplayLevel{Lead: nodes[0]}
		if len(nodes) > 1 {
			lvl.Fan = append([]ChainNode{}, nodes[1:]...)
		}
		plan.Levels = append(plan.Levels, lvl)
	}
	// Breakpoints over the deepest target's full cumulative block count.
	for _, t := range targets {
		if b, err := d.BlocksTo(t); err == nil {
			plan.Breakpoints = MergeBreakpoints(plan.Breakpoints, PlaceBreakpoints(b))
		}
	}
	return plan, nil
}

// coldSubgraph collects, for a set of targets, every COLD node on any target's chain
// (depth >= WarmDepth), keyed by chain-depth. A node already warm (depth < WarmDepth)
// is served from cache and needs no replay.
func (d PrefixDAG) coldSubgraph(targets []string, warmDepth int) (map[int][]ChainNode, error) {
	cold := map[int][]ChainNode{}
	seen := map[string]bool{}
	for _, t := range targets {
		chain, err := d.ChainTo(t)
		if err != nil {
			return nil, err
		}
		for depth, n := range chain {
			if depth < warmDepth {
				continue
			}
			if seen[n.ID] {
				continue
			}
			seen[n.ID] = true
			cold[depth] = append(cold[depth], n)
		}
	}
	return cold, nil
}

// BreakpointBlock is the ~15-block spacing Rule C3 prescribes: an intermediate
// breakpoint every ~15 content blocks stays inside Anthropic's ≤20-block look-back
// with margin. The value is exported so a test can pin it and a caller can override
// (e.g. a future provider with a different walk-back).
const BreakpointBlock = 15

// BreakpointCap is the per-request cap (Rule C3): at most 4 intermediate
// breakpoints per request. A chain needing >4 anchors (>~60 blocks) cannot place
// them all — the design-note fix is aggregation (collapse tiny units into one
// ≥-min parent), not deeper chaining.
const BreakpointCap = 4

// PlaceBreakpoints returns the cumulative-block indices at which an intermediate
// breakpoint must be placed so no span between breakpoints exceeds the ≤20-block
// look-back (§8 + Rule C3). It drops one every BreakpointBlock blocks, capped at
// BreakpointCap, and only while the index stays below totalBlocks. A span shorter
// than BreakpointBlock needs none.
//
// For 50 blocks: [15, 30, 45] (the 60th would exceed the span). For 100 blocks:
// [15, 30, 45, 60] (capped at 4 — the 75th/90th cannot be placed, which is exactly
// the aggregation signal). For 10 blocks: [] (the span fits one look-back).
func PlaceBreakpoints(totalBlocks int) []int {
	if totalBlocks <= BreakpointBlock {
		return nil
	}
	var out []int
	for b := BreakpointBlock; b < totalBlocks && len(out) < BreakpointCap; b += BreakpointBlock {
		out = append(out, b)
	}
	return out
}

// MergeBreakpoints merges two sorted-unique breakpoint lists into one sorted-unique
// list. It is exported so a multi-target replay can combine each target's placement
// without duplicating an index.
func MergeBreakpoints(a, b []int) []int {
	seen := map[int]bool{}
	var out []int
	for _, x := range a {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	for _, x := range b {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	sort.Ints(out)
	return out
}

// String renders a ReplayLevel for diagnostics (not on any hot path).
func (l ReplayLevel) String() string {
	ids := []string{l.Lead.ID}
	for _, n := range l.Fan {
		ids = append(ids, n.ID)
	}
	return fmt.Sprintf("{lead=%s fan=%v}", l.Lead.ID, ids[1:])
}
