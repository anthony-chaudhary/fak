package model

// v41LayerCacheGet returns a retained f32 block for name, or nil on a miss.
//
// A HIT refreshes the key's recency (it is moved to the most-recently-used end
// of layerExpertFIFO) so v41LayerCachePut's eviction becomes LRU rather than
// FIFO. This is the #13294 frontier lever: the prefill's routed-expert reads are
// interleaved across the token dimension, so the key re-read last pass is the
// one FIFO had evicted first. Re-touching on the read that just used it keeps
// the hot set resident, which is exactly the "raise the bounded resident hit
// fraction" the first-token throughput seam names. The map lookup and promotion
// are observation-only w.r.t. arithmetic: the RETURNED block is the same bytes.
func (s *v41ProjScratch) v41LayerCacheGet(name string) ([]float32, bool) {
	if s == nil || s.layerExperts == nil {
		return nil, false
	}
	w, ok := s.layerExperts[name]
	if !ok {
		return nil, false
	}
	s.v41LayerCacheTouch(name)
	return w, true
}

// v41LayerCacheTouch moves name to the most-recently-used end of the eviction
// queue. It is O(len) over the layer's resident keys (bounded by the cache
// budget / block size, i.e. small) and is called on a cache hit and on the
// insertion path, so the queue always reflects recency, never bare insertion
// order. A name absent from the queue is a no-op (defensive: the map and queue
// are only ever mutated together under the sequential MoE loop).
func (s *v41ProjScratch) v41LayerCacheTouch(name string) {
	if s == nil || len(s.layerExpertFIFO) == 0 {
		return
	}
	// The common case is a touch of a key already at the MRU end (the
	// immediately preceding insert); skip the rebuild then.
	if s.layerExpertFIFO[len(s.layerExpertFIFO)-1] == name {
		return
	}
	for i, v := range s.layerExpertFIFO {
		if v == name {
			s.layerExpertFIFO = append(s.layerExpertFIFO[:i], s.layerExpertFIFO[i+1:]...)
			s.layerExpertFIFO = append(s.layerExpertFIFO, name)
			return
		}
	}
}

// v41LayerCachePut retains name's f32 block, evicting the LEAST-RECENTLY-USED
// entry until the declared byte budget admits it (a hit refreshes recency
// through v41LayerCacheGet, so a re-read key is never evicted ahead of a colder
// one). A block larger than the whole budget is not cached (the caller still
// holds its own copy).
func (s *v41ProjScratch) v41LayerCachePut(name string, w []float32) {
	if s == nil || s.expertLayerCacheBytes <= 0 || len(w) == 0 {
		return
	}
	sz := int64(len(w)) * 4
	if sz > s.expertLayerCacheBytes {
		return
	}
	if s.layerExperts == nil {
		s.layerExperts = map[string][]float32{}
	}
	if _, dup := s.layerExperts[name]; dup {
		// Already resident: refresh recency, do not double-count bytes.
		s.v41LayerCacheTouch(name)
		return
	}
	for s.layerExpertBytes+sz > s.expertLayerCacheBytes && len(s.layerExpertFIFO) > 0 {
		old := s.layerExpertFIFO[0]
		s.layerExpertFIFO = s.layerExpertFIFO[1:]
		if prev, ok := s.layerExperts[old]; ok {
			s.layerExpertBytes -= int64(len(prev)) * 4
			delete(s.layerExperts, old)
		}
	}
	s.layerExperts[name] = w
	s.layerExpertFIFO = append(s.layerExpertFIFO, name)
	s.layerExpertBytes += sz
}

// v41LayerCacheReset drops every retained block. Called at the top of a layer so
// the cache's working set is exactly one layer's routed experts.
func (s *v41ProjScratch) v41LayerCacheReset() {
	if s == nil {
		return
	}
	s.layerExperts = nil
	s.layerExpertFIFO = nil
	s.layerExpertBytes = 0
}

// v41SizeExpertLayerBudget sizes the layer-scoped expert budget once per
// scratch from the model's budget (or the test override).
func (s *v41ProjScratch) v41SizeExpertLayerBudget(m *Model) {
	if s.expertLayerCacheBytes != 0 {
		return
	}
	if v41TestLayerCacheBudgetOverride > 0 {
		s.expertLayerCacheBytes = v41TestLayerCacheBudgetOverride
	} else {
		s.expertLayerCacheBytes = m.v41LayerExpertCacheBudget()
	}
}
