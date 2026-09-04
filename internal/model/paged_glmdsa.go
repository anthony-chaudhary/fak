package model

import "fmt"

// paged_glmdsa.go carries the GLM-MoE-DSA exact-span eviction cache onto paged rows.
// GLM-DSA does not use the dense GQA K/V geometry that PagedKVPool assumes: attention K
// and V have different strides, and the learned-indexer cache is stored as widened f32
// values in []float64. This bridge keeps that layout honest by paging each row plane with
// its own fixed-size block table, then applying the same single-rotation-from-raw
// reposition rule as glmDsaKVCache.rerotateSurvivor.

// PagedGLMDsaByteBudget is the maximum allowed byte allocation for paged GLM-DSA caches.
var PagedGLMDsaByteBudget int64 = 64 << 30

// PagedGLMDsaBudgetError is returned when paged GLM-DSA cache construction
// violates dimension constraints, encounters multiplication overflow, or exceeds
// the configured allocation budget.
type PagedGLMDsaBudgetError struct {
	Field     string
	Requested int64
	Limit     int64
	Bytes     int64
	Reason    string
}

func (e *PagedGLMDsaBudgetError) Error() string {
	return fmt.Sprintf("model: paged GLM-DSA cache rejected (%s): field=%s requested=%d limit=%d bytes=%d",
		e.Reason, e.Field, e.Requested, e.Limit, e.Bytes)
}

type pagedRows[T any] struct {
	blockTokens int
	stride      int
	blocks      [][]T
	table       []int
	free        []int
	nTokens     int
}

func newPagedRows[T any](blockTokens, stride int) (*pagedRows[T], error) {
	if blockTokens <= 0 {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "blockTokens",
			Requested: int64(blockTokens),
			Limit:     1,
			Reason:    "non-positive block tokens",
		}
	}
	if stride <= 0 {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "stride",
			Requested: int64(stride),
			Limit:     1,
			Reason:    "non-positive stride",
		}
	}
	if _, ok := checkedMulInt(blockTokens, stride); !ok {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "blockWidth",
			Requested: int64(blockTokens) * int64(stride),
			Reason:    "multiplication overflow",
		}
	}
	return &pagedRows[T]{blockTokens: blockTokens, stride: stride}, nil
}

func (r *pagedRows[T]) blockWidth() int {
	w, ok := checkedMulInt(r.blockTokens, r.stride)
	if !ok {
		panic(fmt.Sprintf("model: pagedRows blockWidth overflow (%d * %d)", r.blockTokens, r.stride))
	}
	return w
}

func (r *pagedRows[T]) alloc() int {
	if n := len(r.free); n > 0 {
		id := r.free[n-1]
		r.free = r.free[:n-1]
		clear(r.blocks[id])
		return id
	}
	id := len(r.blocks)
	r.blocks = append(r.blocks, make([]T, r.blockWidth()))
	return id
}

func (r *pagedRows[T]) append(row []T) {
	li := r.nTokens / r.blockTokens
	off := r.nTokens % r.blockTokens
	if li == len(r.table) {
		r.table = append(r.table, r.alloc())
	}
	blk := r.blocks[r.table[li]]
	dst := off * r.stride
	copy(blk[dst:dst+r.stride], row)
	r.nTokens++
}

func (r *pagedRows[T]) gather() []T {
	total, ok := checkedMulInt(r.nTokens, r.stride)
	if !ok {
		panic(fmt.Sprintf("model: pagedRows gather overflow (%d * %d)", r.nTokens, r.stride))
	}
	out := make([]T, total)
	for pos := 0; pos < r.nTokens; pos++ {
		blk := r.blocks[r.table[pos/r.blockTokens]]
		off := pos % r.blockTokens
		copy(out[pos*r.stride:(pos+1)*r.stride], blk[off*r.stride:(off+1)*r.stride])
	}
	return out
}

func (r *pagedRows[T]) freeAll() {
	for _, id := range r.table {
		r.free = append(r.free, id)
	}
	r.table = nil
	r.nTokens = 0
}

type pagedGLMDsaKVCache struct {
	cfg         Config
	blockTokens int
	nTokens     int
	K           []*pagedRows[float32]
	Kraw        []*pagedRows[float32]
	V           []*pagedRows[float32]
	IndexK      []*pagedRows[float64]
	IndexKraw   []*pagedRows[float64]
}

func newPagedGLMDsaKVCache(cfg Config, blockTokens int) (*pagedGLMDsaKVCache, error) {
	if cfg.NumLayers <= 0 {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "NumLayers",
			Requested: int64(cfg.NumLayers),
			Limit:     1,
			Reason:    "non-positive layers",
		}
	}
	if blockTokens <= 0 {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "blockTokens",
			Requested: int64(blockTokens),
			Limit:     1,
			Reason:    "non-positive block tokens",
		}
	}
	kStride := glmDsaAttentionKStride(cfg)
	if kStride <= 0 {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "kStride",
			Requested: int64(kStride),
			Limit:     1,
			Reason:    "non-positive stride",
		}
	}
	vStride := glmDsaAttentionVStride(cfg)
	if vStride <= 0 {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "vStride",
			Requested: int64(vStride),
			Limit:     1,
			Reason:    "non-positive stride",
		}
	}
	idxStride := cfg.IndexHeadDim
	if idxStride <= 0 {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "IndexHeadDim",
			Requested: int64(idxStride),
			Limit:     1,
			Reason:    "non-positive stride",
		}
	}

	kBlockCells, ok1 := checkedMulInt(blockTokens, kStride)
	vBlockCells, ok2 := checkedMulInt(blockTokens, vStride)
	idxBlockCells, ok3 := checkedMulInt(blockTokens, idxStride)
	if !ok1 || !ok2 || !ok3 {
		return nil, &PagedGLMDsaBudgetError{
			Field:  "blockWidth",
			Reason: "multiplication overflow",
		}
	}

	// 2*K (K, Kraw f32), 1*V (f32), 2*Index (IndexK, IndexKraw f64)
	kBytes := int64(kBlockCells) * 4
	vBytes := int64(vBlockCells) * 4
	idxBytes := int64(idxBlockCells) * 8
	blockBytesPerLayer := 2*kBytes + vBytes + 2*idxBytes
	totalInitialBytes, ok4 := checkedMul(blockBytesPerLayer, int64(cfg.NumLayers))
	if !ok4 || totalInitialBytes > PagedGLMDsaByteBudget {
		return nil, &PagedGLMDsaBudgetError{
			Field:     "totalInitialBytes",
			Requested: totalInitialBytes,
			Limit:     PagedGLMDsaByteBudget,
			Bytes:     totalInitialBytes,
			Reason:    "allocation budget exceeded",
		}
	}

	p := &pagedGLMDsaKVCache{
		cfg:         cfg,
		blockTokens: blockTokens,
		K:           make([]*pagedRows[float32], cfg.NumLayers),
		Kraw:        make([]*pagedRows[float32], cfg.NumLayers),
		V:           make([]*pagedRows[float32], cfg.NumLayers),
		IndexK:      make([]*pagedRows[float64], cfg.NumLayers),
		IndexKraw:   make([]*pagedRows[float64], cfg.NumLayers),
	}
	for l := 0; l < cfg.NumLayers; l++ {
		var err error
		if p.K[l], err = newPagedRows[float32](blockTokens, kStride); err != nil {
			return nil, err
		}
		if p.Kraw[l], err = newPagedRows[float32](blockTokens, kStride); err != nil {
			return nil, err
		}
		if p.V[l], err = newPagedRows[float32](blockTokens, vStride); err != nil {
			return nil, err
		}
		if p.IndexK[l], err = newPagedRows[float64](blockTokens, idxStride); err != nil {
			return nil, err
		}
		if p.IndexKraw[l], err = newPagedRows[float64](blockTokens, idxStride); err != nil {
			return nil, err
		}
	}
	return p, nil
}

// GLMDsaKVCacheToPaged snapshots a GLM-MoE-DSA KVCache into fixed-size paged row
// blocks. It is the GLM-DSA counterpart of KVCacheToPaged: dense attention K/Kraw/V
// and the learned-indexer K/Kraw planes are copied byte-for-byte into paged storage so
// exact-span Evict can be proven without changing the live GLM decode loop.
func GLMDsaKVCacheToPaged(c *KVCache, blockTokens int) (*pagedGLMDsaKVCache, error) {
	if c == nil {
		return nil, fmt.Errorf("model: cannot snapshot nil GLM-DSA KVCache")
	}
	if !c.cfg.isGLMMoeDsa() || c.glm == nil {
		return nil, fmt.Errorf("model: GLMDsaKVCacheToPaged requires a GLM-MoE-DSA cache")
	}
	p, err := newPagedGLMDsaKVCache(c.cfg, blockTokens)
	if err != nil {
		return nil, err
	}
	n := c.Len()
	kStride := glmDsaAttentionKStride(c.cfg)
	vStride := glmDsaAttentionVStride(c.cfg)
	idxStride := c.cfg.IndexHeadDim

	if n > 0 {
		tokenBytesPerLayer := int64(2*kStride+vStride)*4 + int64(2*idxStride)*8
		totalTokenBytes, ok := checkedMul(tokenBytesPerLayer*int64(n), int64(c.cfg.NumLayers))
		if !ok || totalTokenBytes > PagedGLMDsaByteBudget {
			return nil, &PagedGLMDsaBudgetError{
				Field:     "totalTokenBytes",
				Requested: totalTokenBytes,
				Limit:     PagedGLMDsaByteBudget,
				Bytes:     totalTokenBytes,
				Reason:    "allocation budget exceeded",
			}
		}
	}
	for l := 0; l < c.cfg.NumLayers; l++ {
		if len(c.glm.K[l]) != n*kStride || len(c.glm.Kraw[l]) != n*kStride || len(c.glm.V[l]) != n*vStride {
			return nil, fmt.Errorf("model: GLM-DSA KV layer %d has inconsistent attention rows", l)
		}
		hasIndex := len(c.glm.IndexK[l]) != 0 || len(c.glm.IndexKraw[l]) != 0
		if hasIndex && (len(c.glm.IndexK[l]) != n*idxStride || len(c.glm.IndexKraw[l]) != n*idxStride) {
			return nil, fmt.Errorf("model: GLM-DSA KV layer %d has inconsistent index rows", l)
		}
		for pos := 0; pos < n; pos++ {
			p.K[l].append(c.glm.K[l][pos*kStride : (pos+1)*kStride])
			p.Kraw[l].append(c.glm.Kraw[l][pos*kStride : (pos+1)*kStride])
			p.V[l].append(c.glm.V[l][pos*vStride : (pos+1)*vStride])
			if hasIndex {
				p.IndexK[l].append(c.glm.IndexK[l][pos*idxStride : (pos+1)*idxStride])
				p.IndexKraw[l].append(c.glm.IndexKraw[l][pos*idxStride : (pos+1)*idxStride])
			}
		}
	}
	p.nTokens = n
	return p, nil
}

func (p *pagedGLMDsaKVCache) Len() int {
	if p == nil {
		return 0
	}
	return p.nTokens
}

// Evict removes [from, from+n) from paged GLM-DSA cache rows and re-rotates shifted
// survivors from Kraw / IndexKraw at their new logical positions. It returns the number
// of removed positions.
func (p *pagedGLMDsaKVCache) Evict(from, n int) int {
	if p == nil || from < 0 || n <= 0 || from >= p.nTokens {
		return 0
	}
	end := from + n
	if end > p.nTokens {
		end = p.nTokens
	}
	survivors := make([]int, 0, p.nTokens-(end-from))
	for pos := 0; pos < from; pos++ {
		survivors = append(survivors, pos)
	}
	for pos := end; pos < p.nTokens; pos++ {
		survivors = append(survivors, pos)
	}

	kStride := glmDsaAttentionKStride(p.cfg)
	vStride := glmDsaAttentionVStride(p.cfg)
	idxStride := p.cfg.IndexHeadDim
	for l := 0; l < p.cfg.NumLayers; l++ {
		K := p.K[l].gather()
		Kraw := p.Kraw[l].gather()
		V := p.V[l].gather()
		IndexK := p.IndexK[l].gather()
		IndexKraw := p.IndexKraw[l].gather()
		hasIndex := len(IndexK) != 0 || len(IndexKraw) != 0

		p.K[l].freeAll()
		p.Kraw[l].freeAll()
		p.V[l].freeAll()
		p.IndexK[l].freeAll()
		p.IndexKraw[l].freeAll()

		for ni, op := range survivors {
			rawK := append([]float32(nil), Kraw[op*kStride:(op+1)*kStride]...)
			p.Kraw[l].append(rawK)
			p.V[l].append(V[op*vStride : (op+1)*vStride])
			if op == ni {
				p.K[l].append(K[op*kStride : (op+1)*kStride])
			} else {
				p.K[l].append(glmDsaReropeAttentionKFromRaw(p.cfg, l, ni, rawK))
			}
			if hasIndex {
				rawIdx := append([]float64(nil), IndexKraw[op*idxStride:(op+1)*idxStride]...)
				p.IndexKraw[l].append(rawIdx)
				if op == ni {
					p.IndexK[l].append(IndexK[op*idxStride : (op+1)*idxStride])
				} else {
					p.IndexK[l].append(glmDsaReropeIndexKFromRaw(p.cfg, l, ni, rawIdx))
				}
			}
		}
	}
	p.nTokens = len(survivors)
	return end - from
}

func glmDsaReropeAttentionKFromRaw(cfg Config, layer, pos int, kraw []float32) []float32 {
	out := make([]float32, len(kraw))
	qkNope, qkRope := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim
	qkHead := qkNope + qkRope
	cos, sin := ropeRowForLayer(cfg, layer, pos)
	for h := 0; h < cfg.NumHeads; h++ {
		src := kraw[h*qkHead : (h+1)*qkHead]
		dst := out[h*qkHead : (h+1)*qkHead]
		copy(dst[:qkNope], src[:qkNope])
		copy(dst[qkNope:], glmDsaApplyInterleavedRoPE(src[qkNope:], cos, sin))
	}
	return out
}

func glmDsaReropeIndexKFromRaw(cfg Config, layer, pos int, kraw []float64) []float64 {
	out := float64To32(kraw)
	cos, sin := ropeRowForLayer(cfg, layer, pos)
	glmDsaApplyIndexerRoPE(out[:cfg.QKRopeHeadDim], cos, sin)
	return float32To64(out)
}

// ToKVCache materializes the paged GLM-DSA rows back into a live KVCache with logical
// positions relabeled to 0..Len-1, matching the contiguous Evict contract.
func (p *pagedGLMDsaKVCache) ToKVCache() *KVCache {
	c := NewKVCache(p.cfg)
	for l := 0; l < p.cfg.NumLayers; l++ {
		c.glm.K[l] = p.K[l].gather()
		c.glm.Kraw[l] = p.Kraw[l].gather()
		c.glm.V[l] = p.V[l].gather()
		c.glm.IndexK[l] = p.IndexK[l].gather()
		c.glm.IndexKraw[l] = p.IndexKraw[l].gather()
	}
	c.pos = make([]int, p.nTokens)
	for i := range c.pos {
		c.pos[i] = i
	}
	return c
}
