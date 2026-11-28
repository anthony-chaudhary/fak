package model

import (
	"strconv"
	"sync"
)

// dsa_indexer_share.go — layer-group-scoped DSA light-indexer selection sharing.
//
//
// In a GLM-5.2 DSA block several layers alias the SAME light indexer: they share
// one indexer projection and therefore one top-k page/key selection. Recomputing
// that selection per aliasing layer is pure duplicated work on the sparse-attention
// prefill path. The upstream fix computes the selection once per layer group and
// broadcasts it to every dependent layer.
//
// DSA selector seam (dsaTopKIndices / dsaIndexDigest / cloneIndexDecision) rather
// than a parallel type. It is metadata/control-flow only: it decides WHICH keys a
// group attends to, not the attention math. It is host-buildable (no CUDA tag).

// dsaIndexerGroupCache hoists the DSA light-indexer top-k selection to layer-group
// scope so aliasing layers in a group share one computation per decode step.
//
// Keying and invalidation:
//   - The cache is keyed by layer-group id. Each entry records the decode step it
//     was computed for.
//   - A Select for a (group, step) that matches the stored entry returns the cached
//     selection WITHOUT recomputing — this is the broadcast to aliasing layers.
//   - A new step for the same group, or a group id never seen, is a miss and
//     recomputes. Advancing a group to a new step replaces its entry, so the cache
//     holds at most one selection per group (bounded memory).
//   - A layer outside the group presents a different group id, so it never shares
//     another group's selection; it computes its own.
//
// All layers in a group MUST present the same indexer inputs (scores, positions,
// topK) for a given step — that is what "aliasing the same light indexer" means.
// The cache trusts the group key and does not re-derive the inputs.
//
// The cache is safe for concurrent use: aliasing layers may resolve their selection
// from different goroutines, and the compute-once guarantee holds under contention.
type dsaIndexerGroupCache struct {
	mu       sync.Mutex
	groups   map[string]dsaIndexerGroupEntry
	computes int
}

// dsaIndexerGroupEntry is one group's cached selection for one decode step.
type dsaIndexerGroupEntry struct {
	step      int
	selection [][]int
	digest    string
}

// newDSAIndexerGroupCache returns an empty group-scoped selection cache.
func newDSAIndexerGroupCache() *dsaIndexerGroupCache {
	return &dsaIndexerGroupCache{groups: make(map[string]dsaIndexerGroupEntry)}
}

// Select returns the group's top-k indexer selection for the given step, computing
// it at most once per (group, step). Every aliasing layer in the group receives an
// identical (DeepEqual) selection. The returned slice is a clone, so a caller
// mutating it cannot corrupt the cached entry or a sibling layer's view.
//
// It returns false (and caches nothing) when the group id is empty or the selector
// rejects the inputs, so a malformed group fails closed rather than broadcasting a
// partial selection.
func (c *dsaIndexerGroupCache) Select(group string, step int, scores [][]float64, queryPositions, keyPositions []int, topK int) ([][]int, bool) {
	if c == nil || group == "" {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.groups[group]; ok && e.step == step {
		return cloneIndexDecision(e.selection), true
	}
	selection, ok := dsaTopKIndices(scores, queryPositions, keyPositions, topK)
	if !ok {
		return nil, false
	}
	c.groups[group] = dsaIndexerGroupEntry{
		step:      step,
		selection: cloneIndexDecision(selection),
		digest:    dsaIndexDigest(selection),
	}
	c.computes++
	return cloneIndexDecision(selection), true
}

// Computes reports how many times the cache actually ran the top-k selection.
// Aliasing layers in one group at one step add exactly one; a new step or a new
// group adds another.
func (c *dsaIndexerGroupCache) Computes() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.computes
}

// Digest returns the cached selection digest for a (group, step), or "" if the
// group has no entry for that step. It lets a caller witness that two aliasing
// layers resolved to the same selection without re-running the selector.
func (c *dsaIndexerGroupCache) Digest(group string, step int) string {
	if c == nil {
		return ""
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.groups[group]; ok && e.step == step {
		return e.digest
	}
	return ""
}

// Invalidate drops a group's cached selection, forcing the next Select for that
// group to recompute. Used when a group's indexer inputs change mid-step (e.g. a
// re-prefill) so a stale broadcast cannot leak to aliasing layers.
func (c *dsaIndexerGroupCache) Invalidate(group string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.groups, group)
}

// dsaIndexerGroupForLayer derives a layer's indexer group from the arch
// descriptor's indexer plan: a "full" layer heads its own group and each
// "shared" layer joins the nearest preceding full layer's group. Kinds are read
// through the descriptor's own normalization.
//
// The whole plan is judged, not just the asked layer: one shared layer with no
// full source (a dense layer in between clears the source) or one unknown kind
// refuses every layer of the model, since a plan that is wrong somewhere gives
// no reason to trust the rest of it. A "dense" layer publishes no sparse
// selection and is refused on its own. Without a layer count there is no model
// to be inside of, so that refuses too.
func dsaIndexerGroupForLayer(cfg Config, layer int) (string, bool) {
	if layer < 0 || layer >= cfg.NumLayers {
		return "", false
	}
	source, group := -1, -1
	for l := 0; l < cfg.NumLayers; l++ {
		switch glmDsaIndexerKind(cfg, l) {
		case "full":
			source = l
		case "dense":
			source = -1
		case "shared":
			if source < 0 {
				return "", false
			}
		default:
			return "", false
		}
		if l == layer {
			group = source
		}
	}
	if group < 0 {
		return "", false
	}
	return "indexer-layer-" + strconv.Itoa(group), true
}

// SelectForLayer is Select with the group taken from the arch descriptor
// instead of from the caller, so the descriptor's refusals reach the cache: a
// layer with no legal indexer source gets no selection and caches nothing.
func (c *dsaIndexerGroupCache) SelectForLayer(cfg Config, layer, step int, scores [][]float64, queryPositions, keyPositions []int, topK int) ([][]int, bool) {
	group, ok := dsaIndexerGroupForLayer(cfg, layer)
	if !ok {
		return nil, false
	}
	return c.Select(group, step, scores, queryPositions, keyPositions, topK)
}
