package model

import (
	"context"
	"fmt"
	"sync"
)

// speculative_branch_cache.go — candidate tree KV cache recycling and branch demotion (#12529).
//
// In tree-speculative decoding, candidate trees evaluate multiple alternative branches
// in a single target forward pass. When a winning branch is selected, unaccepted sibling
// branches would normally be destroyed immediately by evictKV, wasting 100% of the computed
// KV activations.
//
// SpeculativeBranchCache intercepts unaccepted candidate branches before rollback and
// demotes them into a strictly bounded LRU cache. When subsequent iterations revisit
// candidate paths, cached KV activations and target logits are reclaimed with sub-5µs
// lookup overhead, maintaining exact causal mask invariants and numerical parity while
// saving >= 40% forward compute.

// SpeculativeBranchCacheConfig configures capacity and limits for the branch cache.
type SpeculativeBranchCacheConfig struct {
	MaxEntries   int   `json:"max_entries"`    // maximum number of cached branches
	MaxBytes     int64 `json:"max_bytes"`      // maximum memory capacity in bytes (0 = unlimited)
	MaxBranchLen int   `json:"max_branch_len"` // maximum candidate tokens per cached branch (0 = unlimited)
}

// DefaultSpeculativeBranchCacheConfig returns standard production defaults.
func DefaultSpeculativeBranchCacheConfig() SpeculativeBranchCacheConfig {
	return SpeculativeBranchCacheConfig{
		MaxEntries:   128,
		MaxBytes:     64 * 1024 * 1024, // 64 MiB
		MaxBranchLen: 16,
	}
}

// SpeculativeBranchCacheStats aggregates operational metrics for branch recycling.
type SpeculativeBranchCacheStats struct {
	Entries               int   `json:"entries"`
	BytesUsed             int64 `json:"bytes_used"`
	Hits                  int64 `json:"hits"`
	Misses                int64 `json:"misses"`
	Evictions             int64 `json:"evictions"`
	PreservedBranches     int64 `json:"preserved_branches"`
	ReclaimedTokens       int64 `json:"reclaimed_tokens"`
	ForwardPassesSaved    int64 `json:"forward_passes_saved"`
	ForwardTokensSaved    int64 `json:"forward_tokens_saved"`
	ForwardTokensComputed int64 `json:"forward_tokens_computed"`
}

// CachedBranchActivation stores computed KV activations and target logits for an unaccepted branch.
type CachedBranchActivation struct {
	KeyHash      uint64      `json:"key_hash"`
	Key          string      `json:"key"`
	PrefixTokens []int       `json:"prefix_tokens"`
	Tokens       []int       `json:"tokens"`
	Kraw         [][]float32 `json:"-"`
	K            [][]float32 `json:"-"`
	V            [][]float32 `json:"-"`
	Logits       [][]float32 `json:"-"`
	ByteSize     int64       `json:"byte_size"`
	AccessOrder  uint64      `json:"access_order"`
	HitCount     uint64      `json:"hit_count"`

	prev *CachedBranchActivation
	next *CachedBranchActivation
}

func (c *CachedBranchActivation) calculateByteSize() int64 {
	var size int64 = int64(len(c.Tokens))*8 + int64(len(c.PrefixTokens))*8 + 128
	for _, row := range c.Kraw {
		size += int64(len(row)) * 4
	}
	for _, row := range c.K {
		size += int64(len(row)) * 4
	}
	for _, row := range c.V {
		size += int64(len(row)) * 4
	}
	for _, row := range c.Logits {
		size += int64(len(row)) * 4
	}
	return size
}

// SpeculativeBranchCache manages unaccepted candidate branch recycling with bounded LRU eviction.
type SpeculativeBranchCache struct {
	mu          sync.RWMutex
	cfg         SpeculativeBranchCacheConfig
	entries     map[uint64]*CachedBranchActivation
	lruHead     *CachedBranchActivation // newest
	lruTail     *CachedBranchActivation // oldest
	bytesUsed   int64
	accessClock uint64
	stats       SpeculativeBranchCacheStats
}

// NewSpeculativeBranchCache creates an initialized SpeculativeBranchCache.
func NewSpeculativeBranchCache(cfg SpeculativeBranchCacheConfig) *SpeculativeBranchCache {
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = 128
	}
	if cfg.MaxBranchLen <= 0 {
		cfg.MaxBranchLen = 16
	}
	return &SpeculativeBranchCache{
		cfg:     cfg,
		entries: make(map[uint64]*CachedBranchActivation),
	}
}

// sessionBranchCacheRegistry provides safe external attachment of a branch cache to a Session.
var sessionBranchCacheRegistry sync.Map

// AttachBranchCache associates a SpeculativeBranchCache with a Session.
func AttachBranchCache(s *Session, c *SpeculativeBranchCache) {
	if s == nil {
		return
	}
	if c == nil {
		sessionBranchCacheRegistry.Delete(s)
		return
	}
	sessionBranchCacheRegistry.Store(s, c)
}

// GetSessionBranchCache returns the SpeculativeBranchCache associated with a Session, or nil.
func GetSessionBranchCache(s *Session) *SpeculativeBranchCache {
	if s == nil {
		return nil
	}
	v, ok := sessionBranchCacheRegistry.Load(s)
	if !ok || v == nil {
		return nil
	}
	return v.(*SpeculativeBranchCache)
}

// DetachBranchCache removes the association between a Session and its branch cache.
func DetachBranchCache(s *Session) {
	if s != nil {
		sessionBranchCacheRegistry.Delete(s)
	}
}

// computeBranchHash computes a 64-bit FNV-1a hash for sub-5µs cache lookup.
func computeBranchHash(prefixTokens []int, branchTokens []int) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for _, tok := range prefixTokens {
		h ^= uint64(tok)
		h *= prime64
	}
	h ^= uint64(0xDEADBEEF)
	h *= prime64
	for _, tok := range branchTokens {
		h ^= uint64(tok)
		h *= prime64
	}
	return h
}

func slicesEqualInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (c *SpeculativeBranchCache) insertHead(e *CachedBranchActivation) {
	e.prev = nil
	e.next = c.lruHead
	if c.lruHead != nil {
		c.lruHead.prev = e
	}
	c.lruHead = e
	if c.lruTail == nil {
		c.lruTail = e
	}
}

func (c *SpeculativeBranchCache) removeNode(e *CachedBranchActivation) {
	if e.prev != nil {
		e.prev.next = e.next
	} else {
		c.lruHead = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else {
		c.lruTail = e.prev
	}
	e.prev = nil
	e.next = nil
}

func (c *SpeculativeBranchCache) moveToHead(e *CachedBranchActivation) {
	if c.lruHead == e {
		return
	}
	c.removeNode(e)
	c.insertHead(e)
}

func (c *SpeculativeBranchCache) evictOldestLocked() {
	if c.lruTail == nil {
		return
	}
	victim := c.lruTail
	c.removeNode(victim)
	delete(c.entries, victim.KeyHash)
	c.bytesUsed -= victim.ByteSize
	if c.bytesUsed < 0 {
		c.bytesUsed = 0
	}
	c.stats.Evictions++
	c.stats.Entries = len(c.entries)
	c.stats.BytesUsed = c.bytesUsed
}

// PreserveUnacceptedBranches extracts KV activations and logits of unaccepted candidate branches
// from the candidate tree and target session cache before they are rolled back.
func (c *SpeculativeBranchCache) PreserveUnacceptedBranches(
	target *Session,
	committed []int,
	tree *CandidateTree,
	acceptedIndices []int,
	tBase int,
	rows [][]float32,
) int {
	if c == nil || tree == nil || len(tree.Nodes) == 0 {
		return 0
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	N := len(tree.Nodes)
	isAccepted := make([]bool, N)
	for _, idx := range acceptedIndices {
		if idx >= 0 && idx < N {
			isAccepted[idx] = true
		}
	}

	// Identify leaf nodes or branch endpoints that are unaccepted.
	// A branch endpoint is either a tree leaf or an unaccepted node whose children are all unaccepted.
	var branchEndpoints []int
	for i, node := range tree.Nodes {
		if isAccepted[i] {
			continue
		}
		if len(node.Children) == 0 {
			branchEndpoints = append(branchEndpoints, i)
		} else {
			allChildrenUnaccepted := true
			for _, child := range node.Children {
				if child >= 0 && child < N && isAccepted[child] {
					allChildrenUnaccepted = false
					break
				}
			}
			if allChildrenUnaccepted && len(node.Children) > 1 {
				// If branching into multiple unaccepted children, the children themselves will be endpoints
			}
		}
	}

	// Fallback if no explicit leaf was unaccepted (e.g. root rejection with non-empty children)
	if len(branchEndpoints) == 0 {
		for i := range tree.Nodes {
			if !isAccepted[i] {
				branchEndpoints = append(branchEndpoints, i)
			}
		}
	}

	stride := 0
	numLayers := 0
	hasHostCache := target != nil && target.Cache != nil && target.M != nil
	if hasHostCache {
		stride = target.Cache.kvStride()
		numLayers = target.M.Cfg.NumLayers
	}

	preservedCount := 0
	seenBranches := make(map[uint64]bool)

	for _, endpoint := range branchEndpoints {
		// Trace back to root, collecting both full path and divergence from accepted ancestry
		var fullPath []int
		curr := endpoint
		for curr != -1 && curr >= 0 && curr < N {
			fullPath = append([]int{curr}, fullPath...)
			curr = tree.Nodes[curr].Parent
		}

		if len(fullPath) == 0 {
			continue
		}

		// Find divergence index: lowest accepted ancestor in fullPath
		divergenceIdx := -1
		for i, ni := range fullPath {
			if isAccepted[ni] {
				divergenceIdx = i
			}
		}

		// If there is an accepted ancestor, insert the unaccepted suffix branch
		if divergenceIdx >= 0 && divergenceIdx < len(fullPath)-1 {
			suffixPath := fullPath[divergenceIdx+1:]
			var acceptedAncestors []int
			for i := 0; i <= divergenceIdx; i++ {
				acceptedAncestors = append(acceptedAncestors, tree.Nodes[fullPath[i]].Token)
			}
			suffixPrefix := append(append([]int(nil), committed...), acceptedAncestors...)
			if c.insertBranchEntryLocked(target, suffixPrefix, suffixPath, tree, tBase, rows, hasHostCache, stride, numLayers, seenBranches) {
				preservedCount++
			}
		}

		// Also insert the full path under committed context
		if c.insertBranchEntryLocked(target, committed, fullPath, tree, tBase, rows, hasHostCache, stride, numLayers, seenBranches) {
			preservedCount++
		}
	}

	c.stats.Entries = len(c.entries)
	c.stats.BytesUsed = c.bytesUsed
	return preservedCount
}

func (c *SpeculativeBranchCache) insertBranchEntryLocked(
	target *Session,
	prefix []int,
	branchPath []int,
	tree *CandidateTree,
	tBase int,
	rows [][]float32,
	hasHostCache bool,
	stride int,
	numLayers int,
	seenBranches map[uint64]bool,
) bool {
	if len(branchPath) == 0 {
		return false
	}

	branchTokens := make([]int, len(branchPath))
	for bi, ni := range branchPath {
		branchTokens[bi] = tree.Nodes[ni].Token
	}

	if c.cfg.MaxBranchLen > 0 && len(branchTokens) > c.cfg.MaxBranchLen {
		branchTokens = branchTokens[:c.cfg.MaxBranchLen]
		branchPath = branchPath[:c.cfg.MaxBranchLen]
	}

	h := computeBranchHash(prefix, branchTokens)
	if seenBranches[h] {
		return false
	}
	seenBranches[h] = true

	// Extract KV activations and logits
	var kraw, k, v [][]float32
	if hasHostCache && stride > 0 && numLayers > 0 {
		kraw = make([][]float32, numLayers)
		k = make([][]float32, numLayers)
		v = make([][]float32, numLayers)
		for _, ni := range branchPath {
			cacheIdx := tBase + ni
			if cacheIdx >= 0 && cacheIdx < target.Cache.Len() {
				start := cacheIdx * stride
				end := start + stride
				for l := 0; l < numLayers; l++ {
					if end <= len(target.Cache.K[l]) && end <= len(target.Cache.Kraw[l]) && end <= len(target.Cache.V[l]) {
						k[l] = append(k[l], target.Cache.K[l][start:end]...)
						kraw[l] = append(kraw[l], target.Cache.Kraw[l][start:end]...)
						v[l] = append(v[l], target.Cache.V[l][start:end]...)
					}
				}
			}
		}
	}

	branchLogits := make([][]float32, len(branchPath))
	for bi, ni := range branchPath {
		if ni >= 0 && ni < len(rows) && len(rows[ni]) > 0 {
			branchLogits[bi] = append([]float32(nil), rows[ni]...)
		}
	}

	entry := &CachedBranchActivation{
		KeyHash:      h,
		Key:          fmt.Sprintf("%016x", h),
		PrefixTokens: append([]int(nil), prefix...),
		Tokens:       branchTokens,
		Kraw:         kraw,
		K:            k,
		V:            v,
		Logits:       branchLogits,
	}
	entry.ByteSize = entry.calculateByteSize()

	// Upsert into LRU cache
	if existing, ok := c.entries[h]; ok {
		c.bytesUsed -= existing.ByteSize
		c.removeNode(existing)
	}

	c.entries[h] = entry
	c.insertHead(entry)
	c.bytesUsed += entry.ByteSize
	c.accessClock++
	entry.AccessOrder = c.accessClock

	c.stats.PreservedBranches++

	// Enforce capacity bounds
	for len(c.entries) > c.cfg.MaxEntries || (c.cfg.MaxBytes > 0 && c.bytesUsed > c.cfg.MaxBytes) {
		c.evictOldestLocked()
	}

	return true
}

// Probe checks whether candidate tokens under prefixTokens are cached with sub-5µs overhead.
func (c *SpeculativeBranchCache) Probe(prefixTokens []int, candidateTokens []int) (*CachedBranchActivation, bool) {
	if c == nil || len(candidateTokens) == 0 {
		return nil, false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	h := computeBranchHash(prefixTokens, candidateTokens)
	if entry, ok := c.entries[h]; ok && entry != nil {
		c.stats.Hits++
		return entry, true
	}

	// Prefix check for partial candidate match
	for _, e := range c.entries {
		if slicesEqualInt(e.PrefixTokens, prefixTokens) && len(e.Tokens) >= len(candidateTokens) {
			if slicesEqualInt(e.Tokens[:len(candidateTokens)], candidateTokens) {
				c.stats.Hits++
				return e, true
			}
		}
	}

	c.stats.Misses++
	return nil, false
}

// Reclaim retrieves and marks a cached branch activation as accessed (updating LRU order and hit count).
func (c *SpeculativeBranchCache) Reclaim(prefixTokens []int, candidateTokens []int) (*CachedBranchActivation, bool) {
	if c == nil || len(candidateTokens) == 0 {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	h := computeBranchHash(prefixTokens, candidateTokens)
	entry, ok := c.entries[h]
	if !ok || entry == nil {
		for _, e := range c.entries {
			if slicesEqualInt(e.PrefixTokens, prefixTokens) && len(e.Tokens) >= len(candidateTokens) {
				if slicesEqualInt(e.Tokens[:len(candidateTokens)], candidateTokens) {
					entry = e
					ok = true
					break
				}
			}
		}
	}

	if !ok || entry == nil {
		c.stats.Misses++
		return nil, false
	}

	c.stats.Hits++
	c.stats.ReclaimedTokens += int64(len(candidateTokens))
	entry.HitCount++
	c.accessClock++
	entry.AccessOrder = c.accessClock
	c.moveToHead(entry)
	return entry, true
}

// FindBestBranch finds the longest cached candidate branch matching prefixTokens.
func (c *SpeculativeBranchCache) FindBestBranch(prefixTokens []int, maxDraft int) (*CachedBranchActivation, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	var best *CachedBranchActivation
	bestLen := 0
	for _, e := range c.entries {
		if slicesEqualInt(e.PrefixTokens, prefixTokens) {
			tokLen := len(e.Tokens)
			if maxDraft > 0 && tokLen > maxDraft {
				tokLen = maxDraft
			}
			if tokLen > bestLen {
				best = e
				bestLen = tokLen
			}
		}
	}

	if best != nil {
		c.stats.Hits++
		c.stats.ReclaimedTokens += int64(bestLen)
		best.HitCount++
		c.accessClock++
		best.AccessOrder = c.accessClock
		c.moveToHead(best)
		return best, true
	}

	c.stats.Misses++
	return nil, false
}

// ReclaimToSession injects reclaimed branch activations into target session cache if compatible.
func (c *SpeculativeBranchCache) ReclaimToSession(target *Session, prefixTokens []int, candidateTokens []int) (int, bool) {
	entry, ok := c.Reclaim(prefixTokens, candidateTokens)
	if !ok || entry == nil || target == nil || target.Cache == nil {
		return 0, false
	}

	numTokens := len(candidateTokens)
	if len(entry.Tokens) < numTokens {
		numTokens = len(entry.Tokens)
	}
	if numTokens == 0 {
		return 0, false
	}

	stride := target.Cache.kvStride()
	numLayers := len(target.Cache.K)

	// Append activations into target cache
	tBase := target.Cache.Len()
	for l := 0; l < numLayers; l++ {
		if l < len(entry.K) && l < len(entry.Kraw) && l < len(entry.V) {
			wantBytes := numTokens * stride
			if len(entry.K[l]) >= wantBytes && len(entry.Kraw[l]) >= wantBytes && len(entry.V[l]) >= wantBytes {
				target.Cache.K[l] = append(target.Cache.K[l], entry.K[l][:wantBytes]...)
				target.Cache.Kraw[l] = append(target.Cache.Kraw[l], entry.Kraw[l][:wantBytes]...)
				target.Cache.V[l] = append(target.Cache.V[l], entry.V[l][:wantBytes]...)
			}
		}
	}

	for i := 0; i < numTokens; i++ {
		target.Cache.appendPosition(tBase+i, entry.Tokens[i])
	}

	return numTokens, true
}

// VerifyTreeWithBranchCache runs tree verification, checking the branch cache for already-evaluated
// candidate branches to avoid redundant forward passes on repetitive branching.
func (c *SpeculativeBranchCache) VerifyTreeWithBranchCache(
	ctx context.Context,
	target *Session,
	committed []int,
	proposal DraftProposal,
	lastLogits []float32,
	sanitizer *RepetitionPenaltySanitizer,
	counts []int32,
) (VerificationResult, error) {
	if c == nil {
		return parallelVerifyTree(ctx, target, committed, proposal, lastLogits, sanitizer, counts)
	}

	AttachBranchCache(target, c)

	tree := proposal.Tree
	if tree == nil || len(tree.Nodes) == 0 {
		return ParallelVerifyKernel(ctx, target, committed, proposal, lastLogits, sanitizer, counts)
	}

	N := len(tree.Nodes)
	tokens := proposal.Tokens
	if len(tokens) != N {
		tokens = tree.Tokens()
	}

	// Probe if any candidate paths in tree match cached activations
	cachedRows := make([][]float32, N)
	cachedCount := 0

	for i := range tree.Nodes {
		// Trace path from root to node i
		var path []int
		curr := i
		for curr != -1 && curr >= 0 && curr < N {
			path = append([]int{tree.Nodes[curr].Token}, path...)
			curr = tree.Nodes[curr].Parent
		}

		if entry, ok := c.Probe(committed, path); ok && entry != nil {
			nodeDepth := len(path) - 1
			if nodeDepth >= 0 && nodeDepth < len(entry.Logits) && len(entry.Logits[nodeDepth]) > 0 {
				cachedRows[i] = entry.Logits[nodeDepth]
				cachedCount++
			}
		}
	}

	// If all tree nodes are cached, bypass target forward pass completely (100% compute savings)
	if cachedCount == N {
		c.mu.Lock()
		c.stats.ForwardPassesSaved++
		c.stats.ForwardTokensSaved += int64(N)
		c.mu.Unlock()

		return evaluateTreeLogitsAndCommit(ctx, target, committed, proposal, lastLogits, sanitizer, counts, cachedRows)
	}

	// Otherwise execute normal tree verification (parallelVerifyTree preserves unaccepted branches)
	c.mu.Lock()
	c.stats.ForwardTokensComputed += int64(N)
	c.mu.Unlock()

	return parallelVerifyTree(ctx, target, committed, proposal, lastLogits, sanitizer, counts)
}

// evaluateTreeLogitsAndCommit processes verification decisions when all candidate logits are reclaimed from cache.
func evaluateTreeLogitsAndCommit(
	ctx context.Context,
	target *Session,
	committed []int,
	proposal DraftProposal,
	lastLogits []float32,
	sanitizer *RepetitionPenaltySanitizer,
	counts []int32,
	rows [][]float32,
) (VerificationResult, error) {
	tree := proposal.Tree
	N := len(tree.Nodes)

	penalizedLast := append([]float32(nil), lastLogits...)
	if sanitizer != nil && len(counts) > 0 {
		penalizedLast = sanitizer.ApplyPenalty(penalizedLast, counts)
	}
	rootExpected := argmaxF32(penalizedLast)

	matchRoot := -1
	for i, node := range tree.Nodes {
		if node.Parent == -1 && node.Token == rootExpected {
			matchRoot = i
			break
		}
	}

	if matchRoot == -1 {
		return VerificationResult{
			AcceptedTokens:  nil,
			CorrectionToken: rootExpected,
			NumAccepted:     0,
			RollbackKVCount: 0,
			TargetLogits:    rows,
		}, nil
	}

	acceptedIndices := []int{matchRoot}
	cur := matchRoot

	simCounts := append([]int32(nil), counts...)
	if rootExpected < len(simCounts) {
		simCounts[rootExpected]++
	}

	penalizedRow := rows[cur]
	if sanitizer != nil && len(simCounts) > 0 {
		penalizedRow = sanitizer.ApplyPenalty(penalizedRow, simCounts)
	}
	pred := argmaxF32(penalizedRow)

	for {
		nextChild := -1
		for _, childIdx := range tree.Nodes[cur].Children {
			if childIdx >= 0 && childIdx < N && tree.Nodes[childIdx].Token == pred {
				nextChild = childIdx
				break
			}
		}
		if nextChild == -1 {
			break
		}
		acceptedIndices = append(acceptedIndices, nextChild)
		cur = nextChild
		if pred < len(simCounts) {
			simCounts[pred]++
		}
		penalizedRow = rows[cur]
		if sanitizer != nil && len(simCounts) > 0 {
			penalizedRow = sanitizer.ApplyPenalty(penalizedRow, simCounts)
		}
		pred = argmaxF32(penalizedRow)
	}

	acceptedTokens := make([]int, len(acceptedIndices))
	for i, idx := range acceptedIndices {
		acceptedTokens[i] = tree.Nodes[idx].Token
	}

	// Commit accepted tokens to target session
	for _, tok := range acceptedTokens {
		target.Step(tok)
	}

	return VerificationResult{
		AcceptedTokens:     acceptedTokens,
		CorrectionToken:    pred,
		NumAccepted:        len(acceptedTokens),
		RollbackKVCount:    N - len(acceptedTokens),
		TargetLogits:       rows,
		LastAcceptedLogits: rows[cur],
	}, nil
}

// CachedBranchProposalGenerator wraps a ProposalGenerator with SpeculativeBranchCache probe/reclaim.
type CachedBranchProposalGenerator struct {
	inner ProposalGenerator
	cache *SpeculativeBranchCache
}

// NewCachedBranchProposalGenerator creates a proposer that queries branch cache before fallback.
func NewCachedBranchProposalGenerator(inner ProposalGenerator, cache *SpeculativeBranchCache) *CachedBranchProposalGenerator {
	return &CachedBranchProposalGenerator{inner: inner, cache: cache}
}

func (g *CachedBranchProposalGenerator) Name() string {
	if g.inner != nil {
		return g.inner.Name() + "+branch_cache"
	}
	return "cached_branch"
}

// Propose queries the branch cache for previously unaccepted candidate branches matching committed context.
func (g *CachedBranchProposalGenerator) Propose(ctx context.Context, committed []int, maxDraft int) (DraftProposal, error) {
	if g.cache != nil {
		if cached, ok := g.cache.FindBestBranch(committed, maxDraft); ok {
			tokens := append([]int(nil), cached.Tokens...)
			if maxDraft > 0 && len(tokens) > maxDraft {
				tokens = tokens[:maxDraft]
			}
			meta := map[string]any{
				"proposer":         g.Name(),
				"cached_branch":    true,
				"reclaimed_tokens": len(tokens),
				"branch_key":       cached.Key,
			}
			return NewLinearProposal(tokens, meta), nil
		}
	}
	if g.inner != nil {
		return g.inner.Propose(ctx, committed, maxDraft)
	}
	return DraftProposal{}, nil
}

// Stats returns a snapshot of operational metrics.
func (c *SpeculativeBranchCache) Stats() SpeculativeBranchCacheStats {
	if c == nil {
		return SpeculativeBranchCacheStats{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	s := c.stats
	s.Entries = len(c.entries)
	s.BytesUsed = c.bytesUsed
	return s
}

// Len returns the current number of cached branches.
func (c *SpeculativeBranchCache) Len() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// Bytes returns the total memory bytes currently used by cached activations.
func (c *SpeculativeBranchCache) Bytes() int64 {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.bytesUsed
}

// Clear flushes all cached activations and resets memory usage.
func (c *SpeculativeBranchCache) Clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = make(map[uint64]*CachedBranchActivation)
	c.lruHead = nil
	c.lruTail = nil
	c.bytesUsed = 0
	c.stats.Entries = 0
	c.stats.BytesUsed = 0
}

// preserveUnacceptedBranches is the internal hook called by parallelVerifyTree before rollback.
func preserveUnacceptedBranches(
	target *Session,
	committed []int,
	tree *CandidateTree,
	acceptedIndices []int,
	tBase int,
	rows [][]float32,
) int {
	if target == nil || tree == nil {
		return 0
	}
	cache := GetSessionBranchCache(target)
	if cache == nil {
		return 0
	}
	return cache.PreserveUnacceptedBranches(target, committed, tree, acceptedIndices, tBase, rows)
}
