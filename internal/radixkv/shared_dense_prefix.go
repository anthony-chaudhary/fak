// SPDX-License-Identifier: Apache-2.0
// Copyright 2023-2024 SGLang Team
// Ownership adaptation of SGLang at 181380682b50f4606d1a09df3cbbbfc21de931e3,
// described in THIRD_PARTY_NOTICES.md ("SGLang").
package radixkv

import "github.com/anthony-chaudhary/fak/internal/model"

// EnableSharedDensePrefixes selects immutable page-sharing for the existing
// clone-owning insertion API. Configure before the first payload is inserted.
// Live-pointer InsertWithLogits remains a distinct, unchanged API.
func (t *Tree) EnableSharedDensePrefixes() bool {
	if t == nil || t.tokens != 0 {
		return false
	}
	t.sharedDensePrefixes = true
	return true
}

func (t *Tree) SharedDensePrefixesEnabled() bool { return t != nil && t.sharedDensePrefixes }

// HasKV is a presence probe: it never allocates a materialized Session cache.
func (n *node) HasKV() bool { return n != nil && (n.kv != nil || n.densePrefix != nil) }

func (n *node) releaseDensePrefix() {
	if n != nil && n.densePrefix != nil {
		n.densePrefix.Free()
		n.densePrefix = nil
	}
}

// hasDenseSubtree protects immutable ownership when a caller retains a stale
// Lookup boundary across another insertion. Refuse replacement instead of
// detaching pages from the tree's retained-byte census or invalidating a lease.
func hasDenseSubtree(n *node) bool {
	if n == nil {
		return false
	}
	if n.densePrefix != nil {
		return true
	}
	for _, child := range n.children {
		if hasDenseSubtree(child) {
			return true
		}
	}
	return false
}

func releaseDenseSubtree(n *node) {
	if n == nil {
		return
	}
	n.releaseDensePrefix()
	for _, child := range n.children {
		releaseDenseSubtree(child)
	}
}

// insertSharedDenseClone is called only by the explicitly clone-owning API.
// handled=false preserves the historical clone for unsupported layouts.
func (t *Tree) insertSharedDenseClone(boundary *node, suffix []int, kv *model.KVCache, logits []float32) (*node, bool) {
	if boundary == nil {
		return nil, true
	}
	if boundary.plen < 0 || len(suffix) > int(^uint(0)>>1)-boundary.plen {
		return boundary, true
	}
	if kv == nil || kv.Len() != boundary.plen+len(suffix) {
		return nil, false
	}
	if len(suffix) == 0 && boundary.HasKV() {
		return t.InsertWithLogits(boundary, suffix, nil, logits), true
	}
	// A stale boundary must not overwrite a previously admitted child/lease.
	// Ordinary Lookup splits a divergent edge before this insertion point.
	if len(suffix) > 0 && boundary.children[suffix[0]] != nil {
		return boundary, true
	}
	plan, ok := model.PlanDensePrefixCapture(kv, boundary.densePrefix)
	if !ok {
		return nil, false
	}
	if !t.makeCPUCacheRoom(cacheByteSum(plan.AdditionalBytes(), cpuLogitsBytes(len(logits))), 0, boundary) {
		return boundary, true
	}
	// Keep the existing value/frequency, token-budget and lease transition.
	// Do not allocate or reserve pool metadata until that admission succeeds:
	// a rejected value gate must not retain unpublished allocator backing.
	// The entire validation/admission/capture sequence holds the tree lock;
	// the admitted leaf protects its prefix throughout token-budget eviction.
	leaf, admitted := t.insertWithLogits(boundary, suffix, nil, nil, nil)
	if !admitted || leaf == nil {
		return leaf, true
	}
	owned := plan.Capture()
	leaf.releaseDensePrefix()
	leaf.kv = nil
	leaf.densePrefix = owned
	leaf.logits = copyCPULogits(logits)
	return leaf, true
}
