package model

// KVCache is the first-class, kernel-OWNED attention state. This is the object the
// whole fusion is about: because it lives in the Go kernel's address space (not
// behind an HTTP boundary in a vLLM process), the context-MMU can EVICT a poisoned
// token span from it (so the model physically cannot attend to that span), and the
// vDSO can reuse a cached prefix — real operations on real K/V tensors, not
// metaphors. Each layer holds K and V as flat row-major [pos * (NumKVHeads*HeadDim)]
// slices; pos[] records each entry's absolute RoPE position so eviction can compact
// the cache and relabel subsequent positions to equal a run that never saw the span.
type KVCache struct {
	cfg    Config
	K      [][]float32 // [layer] -> flat, NumKVHeads*HeadDim per cached position (post-RoPE)
	Kraw   [][]float32 // [layer] -> the SAME entries pre-RoPE, so a span can be repositioned
	V      [][]float32
	pos    []int // absolute position of each cached entry (shared across layers)
	linear *linearAttnCache
	glm    *glmDsaKVCache
	msa    *minimaxKVCache // MiniMax-M3 lightning-indexer key cache (sparse layers only)
}

// NewKVCache allocates an empty cache for a model. Kraw (pre-RoPE K) is kept so that
// eviction can re-derive a survivor's post-RoPE K at its new position in a SINGLE
// rotation (bit-exact to a fresh prefill), rather than composing two rotations
// (which is mathematically equal but drifts ~1e-6 — enough to flip a greedy token).
func NewKVCache(cfg Config) *KVCache {
	return &KVCache{
		cfg:    cfg,
		K:      make([][]float32, cfg.NumLayers),
		Kraw:   make([][]float32, cfg.NumLayers),
		V:      make([][]float32, cfg.NumLayers),
		linear: newLinearAttnCache(cfg),
		glm:    newGLMDsaKVCache(cfg),
		msa:    newMinimaxKVCache(cfg),
	}
}

// Len is the number of cached positions.
func (c *KVCache) Len() int { return len(c.pos) }

// kvStride is the per-position width of one layer's K (and V) row.
func (c *KVCache) kvStride() int { return c.cfg.NumKVHeads * c.cfg.HeadDim }

// RecurrentEvictUnsupportedError is the typed verdict KVCache.Evict / TryEvict / CanEvict
// return for a hybrid Gated-DeltaNet (Qwen3.5/3.6) cache. The linear-attention layers hold
// an ACCUMULATED recurrent state (and a short-conv window), not a per-token K/V row, so a
// middle span has no rows to compact and no checkpoint/journal to replay the deletion from.
// Quarantining a span out of such a state is therefore a formally UNSUPPORTED operation
// under the current cache. The verdict fails CLOSED — the cache is left byte-for-byte
// unchanged — rather than silently dropping the softmax-KV rows of the full-attention
// layers while leaving the poison fused into the recurrence (a quarantine that did not
// quarantine). A caller that must quarantine a hybrid session rebuilds from a clean prefix
// (re-prefill the survivor span) instead of evicting in place.
type RecurrentEvictUnsupportedError struct {
	// Layers are the linear-attention (recurrent) layer indices whose state blocks eviction.
	Layers []int
}

// Error reports that KVCache.Evict cannot quarantine a hybrid Gated-DeltaNet cache, naming the recurrent layers and advising a rebuild from a clean prefix.
func (e *RecurrentEvictUnsupportedError) Error() string {
	return "model: KVCache.Evict does not support Gated-DeltaNet recurrent state " +
		"(hybrid linear-attention layers " + itoaSlice(e.Layers) + " hold an accumulated " +
		"recurrence with no per-token journal); rebuild from a clean prefix instead"
}

// CanEvict reports whether this cache supports span eviction. It returns nil for an
// ordinary softmax-KV cache (and for the GLM-MoE-DSA cache, which compacts its DSA state)
// and a typed *RecurrentEvictUnsupportedError for a hybrid Gated-DeltaNet cache. It is the
// witnessable verdict callers consult BEFORE Evict so a recurrent model surfaces a typed
// limitation at the boundary instead of crashing inside the package.
func (c *KVCache) CanEvict() error {
	if c.linear != nil {
		return &RecurrentEvictUnsupportedError{Layers: c.linear.recurrentLayers()}
	}
	return nil
}

// Evict removes a contiguous span [from, from+n) of cached positions from EVERY
// layer and compacts the survivors so the cache is byte-for-byte what it would be
// if the span had NEVER been seen. This is the KV-level quarantine primitive.
//
// The subtlety the adversarial review caught: a survivor that came AFTER the evicted
// span had its K rotated by RoPE at its ORIGINAL absolute position; after compaction
// it sits at a lower position, so its cached K must be RE-ROTATED by the position
// delta. RoPE is linear in position (angle = pos·inv_freq), so re-rotating a K vector
// from old position p to new position p' is exactly applying RoPE of (p'-p). Without
// this, end-span eviction passes by luck but MIDDLE-span eviction is silently wrong.
// (V is not rotated, so it never needs fixing.) Returns positions removed.
//
// Evict is the convenience form for the softmax-KV / GLM-DSA paths whose eviction is
// always supported. For a hybrid Gated-DeltaNet cache it panics the typed
// *RecurrentEvictUnsupportedError (a loud boundary for an unchecked caller); a caller that
// can encounter a hybrid cache uses TryEvict (or CanEvict) to handle the verdict instead.
func (c *KVCache) Evict(from, n int) int {
	removed, err := c.TryEvict(from, n)
	if err != nil {
		panic(err)
	}
	return removed
}

// TryEvict is Evict with the unsupported verdict surfaced as a typed error rather than
// panicked. It returns (0, *RecurrentEvictUnsupportedError) for a hybrid recurrent cache,
// leaving the cache unchanged (fail-closed), and (removed, nil) otherwise.
func (c *KVCache) TryEvict(from, n int) (int, error) {
	if err := c.CanEvict(); err != nil {
		return 0, err
	}
	return c.evictSupported(from, n), nil
}

// evictSupported performs the actual span compaction for caches whose eviction is
// supported (ordinary softmax-KV and GLM-MoE-DSA). CanEvict has already cleared the
// hybrid-recurrent boundary by the time this runs.
func (c *KVCache) evictSupported(from, n int) int {
	if from < 0 || n <= 0 || from >= len(c.pos) {
		return 0
	}
	end := from + n
	if end > len(c.pos) {
		end = len(c.pos)
	}
	if c.cfg.isGLMMoeDsa() {
		return c.evictGLMDsa(from, end)
	}
	w := c.kvStride()
	hd, nKV := c.cfg.HeadDim, c.cfg.NumKVHeads
	for l := 0; l < c.cfg.NumLayers; l++ {
		c.K[l] = append(c.K[l][:from*w], c.K[l][end*w:]...)
		c.Kraw[l] = append(c.Kraw[l][:from*w], c.Kraw[l][end*w:]...)
		c.V[l] = append(c.V[l][:from*w], c.V[l][end*w:]...)
	}
	// MiniMax-M3 MSA keeps its main K/V in the rows just compacted; the per-layer
	// lightning-indexer key cache is compacted in lock-step and re-RoPEd below.
	if c.msa != nil {
		c.msa.evict(c.cfg, from, end)
	}
	// c.pos still holds each survivor's ORIGINAL absolute position; its new position
	// is its new index i. Where they differ, re-derive K[i] from the PRE-RoPE Kraw[i]
	// in one rotation at position i — bit-exact to a prefill that never saw the span.
	c.pos = append(c.pos[:from], c.pos[end:]...)
	for i := range c.pos {
		if c.pos[i] != i {
			for l := 0; l < c.cfg.NumLayers; l++ {
				if c.cfg.Alibi {
					copy(c.K[l][i*w:(i+1)*w], c.Kraw[l][i*w:(i+1)*w])
					continue
				}
				cos, sin := ropeRowForLayer(c.cfg, l, i)
				for h := 0; h < nKV; h++ {
					dst := c.K[l][i*w+h*hd : i*w+(h+1)*hd]
					copy(dst, c.Kraw[l][i*w+h*hd:i*w+(h+1)*hd]) // raw (pre-RoPE)
					applyRopeRow(dst, cos, sin)                 // single rotation at new pos
				}
			}
			c.msa.rerotateSurvivor(c.cfg, i) // nil-safe: re-RoPE the survivor's index key
		}
		c.pos[i] = i
	}
	return end - from
}

func (c *KVCache) evictGLMDsa(from, end int) int {
	if c.glm != nil {
		c.glm.evict(c.cfg, from, end)
	}
	c.pos = append(c.pos[:from], c.pos[end:]...)
	for i := range c.pos {
		if c.pos[i] != i && c.glm != nil {
			c.glm.rerotateSurvivor(c.cfg, i)
		}
		c.pos[i] = i
	}
	return end - from
}

// Clone deep-copies the cache so a computed prefix can be REUSED across sessions —
// the vDSO payoff: a shared system-prompt / tool-result prefix's K/V is computed once
// and spliced into the next session, skipping its prefill entirely. Because the copy
// is exact, the reusing session is bit-identical to one that prefilled the prefix.
func (c *KVCache) Clone() *KVCache {
	return c.CloneWithReserve(0)
}

// CloneWithReserve is Clone plus spare per-layer capacity for extra future positions.
// Fleet serving knows the planned decode/result tail length; reserving it here avoids
// re-copying the already-cloned prefix on the first append and on later growth steps.
func (c *KVCache) CloneWithReserve(extraPositions int) *KVCache {
	if extraPositions < 0 {
		extraPositions = 0
	}
	extraFloats := extraPositions * c.kvStride()
	n := &KVCache{
		cfg:    c.cfg,
		K:      make([][]float32, len(c.K)),
		Kraw:   make([][]float32, len(c.Kraw)),
		V:      make([][]float32, len(c.V)),
		pos:    cloneIntsWithReserve(c.pos, extraPositions),
		linear: c.linear.clone(),
	}
	if c.glm != nil {
		n.glm = c.glm.cloneWithReserve(c.cfg, extraPositions)
	}
	if c.msa != nil {
		n.msa = c.msa.cloneWithReserve(c.cfg, extraPositions)
	}
	for l := range c.K {
		n.K[l] = cloneFloat32WithReserve(c.K[l], extraFloats)
		n.Kraw[l] = cloneFloat32WithReserve(c.Kraw[l], extraFloats)
		n.V[l] = cloneFloat32WithReserve(c.V[l], extraFloats)
	}
	return n
}

// Reserve grows this cache's spare capacity for extra future positions while preserving
// its exact current contents. It does not change Len().
func (c *KVCache) Reserve(extraPositions int) {
	if extraPositions <= 0 {
		return
	}
	extraFloats := extraPositions * c.kvStride()
	c.pos = reserveInts(c.pos, extraPositions)
	for l := range c.K {
		c.K[l] = reserveFloat32(c.K[l], extraFloats)
		c.Kraw[l] = reserveFloat32(c.Kraw[l], extraFloats)
		c.V[l] = reserveFloat32(c.V[l], extraFloats)
	}
	if c.glm != nil {
		c.glm.reserve(c.cfg, extraPositions)
	}
	if c.msa != nil {
		c.msa.reserve(c.cfg, extraPositions)
	}
}

func cloneFloat32WithReserve(src []float32, extra int) []float32 {
	dst := make([]float32, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func cloneIntsWithReserve(src []int, extra int) []int {
	dst := make([]int, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func reserveFloat32(src []float32, extra int) []float32 {
	if cap(src) >= len(src)+extra {
		return src
	}
	dst := make([]float32, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func reserveInts(src []int, extra int) []int {
	if cap(src) >= len(src)+extra {
		return src
	}
	dst := make([]int, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func cloneFloat64WithReserve(src []float64, extra int) []float64 {
	dst := make([]float64, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func reserveFloat64(src []float64, extra int) []float64 {
	if cap(src) >= len(src)+extra {
		return src
	}
	dst := make([]float64, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}
