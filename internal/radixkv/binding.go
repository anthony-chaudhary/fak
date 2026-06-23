package radixkv

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// binding.go enforces the materialization binding at the KV REUSE POINT (issue
// #432). The bare radix Tree is keyed PURELY by token ids: walk/Lookup match a
// prefix on token identity alone and hand back the node's kernel-owned KVCache for
// reuse. That KV was computed under one model / tokenizer / serializer / RoPE
// position regime, and is GARBAGE under another — yet nothing in Lookup stops a
// model-B request whose tokens happen to share a prefix from being served model-A's
// span. The tree is single-model only by caller CONVENTION (the package doc's "one
// tree per workload"), which is exactly the silent cross-model reuse #432 warns
// against: "after a source-linked semantic view is selected, it may be materialized
// as a local KV prefix/span only if model/runtime constraints hold."
//
// The constraint already ships as the proven cachemeta gate (#432:
// MaterializationKey.Complete / .Matches). This file binds that gate onto the radix
// reuse path so the convention becomes MECHANICAL: a BoundTree carries the
// materialization key its spans were computed under, and its Lookup consults the key
// BEFORE touching the tree. A request keyed to a divergent model/tokenizer/
// serializer/position/policy/admitter regime fails CLOSED — no node, no lease, no
// split — so a cross-binding KV span is never served as a hit. It does NOT
// reimplement the tree or the gate; it is a pure consumer of both, the same posture
// radixkv takes toward the verified model.KVCache primitives.

// BoundTree is a radix prefix cache PINNED to one runtime materialization binding:
// the model / tokenizer / serializer / RoPE-position / policy / admitter identity
// every KV span it holds was computed under. It wraps the proven Tree (so eviction,
// leasing, splitting, and the bit-exact Clone/Evict reuse are unchanged) and adds the
// #432 fail-closed reuse guard the bare Tree lacks: reuse is admitted only when the
// requesting key matches the binding on every axis.
type BoundTree struct {
	*Tree
	key cachemeta.MaterializationKey
}

// NewBound builds a prefix cache pinned to key. maxTokens is the LRU budget in
// cached tokens (0 = unbounded), exactly as New. Every span inserted into the
// returned tree is, by the caller's contract, materialized under key — and Lookup
// enforces that no request under a divergent binding may reuse one.
func NewBound(maxTokens int, key cachemeta.MaterializationKey) *BoundTree {
	return &BoundTree{Tree: New(maxTokens), key: key}
}

// Key returns the materialization binding this tree's spans were computed under.
func (b *BoundTree) Key() cachemeta.MaterializationKey { return b.key }

// Reusable reports whether a request keyed by want may reuse this tree's spans,
// returning the FIRST divergent axis as a typed reason (ReasonNone on a match). An
// incomplete binding on EITHER side cannot prove a match — a missing axis is an
// unprovable identity — so it fails closed with ReasonModelMismatch rather than
// admit a possibly-mismatched span. This is the cachemeta key gate, no more.
func (b *BoundTree) Reusable(want cachemeta.MaterializationKey) (bool, cachemeta.LookupReason) {
	if !b.key.Complete() || !want.Complete() {
		return false, cachemeta.ReasonModelMismatch
	}
	return b.key.Matches(want)
}

// Lookup is the binding-guarded reuse entry point. It consults the materialization
// binding BEFORE touching the tree: on a mismatch (cross-model / cross-tokenizer /
// cross-serializer / cross-position / cross-policy / cross-admitter, or any
// incomplete key) it fails CLOSED — returning (nil, 0, false) with NO lease taken
// and NO edge split performed — so a request that merely shares a token prefix can
// never be served a KV span computed under a different binding. On a match it
// delegates to the proven Tree.Lookup, which takes the lease the caller MUST release
// (Insert hands it to the leaf; Done releases it). ok reports whether the binding
// admitted the reuse; when false the returned node is nil and the tree is untouched.
func (b *BoundTree) Lookup(want cachemeta.MaterializationKey, tokens []int) (boundary *node, matched int, ok bool) {
	if admit, _ := b.Reusable(want); !admit {
		return nil, 0, false
	}
	n, m := b.Tree.Lookup(tokens)
	return n, m, true
}
