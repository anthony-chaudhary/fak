package model

// expert_pagecache_advise.go - the access-hint seam over a mapped expert range (#1302). It is the
// model-side analogue of ggufload's ChooseMmapAdvice / MmapAdvice shim (#5284): the pure choice
// vocabulary lives in expert_pagecache.go as PageAdvice, and this file translates one expert's
// byte range into the OS hint the mapped region should carry.
//
// WHY ADVISE AT ALL. A cold routed-expert fault is a scattered read with no sequential locality,
// so the honest hint is MADV_RANDOM (no eager read-ahead); a workload that walks the slab in
// expert order instead wants MADV_SEQUENTIAL. Neither hint is ever required for correctness - a
// failed madvise only forfeits an IO optimization - so adviseRange swallows every error and
// returns only whether a hint was issued, for a bytes/token witness, never for control flow.
//
// It advises ONLY when a mapping is actually present: on the ReadAt path there is no mapped region
// to hint, so it declines with false exactly as willneedExpertSlice does (expert_readahead.go).
// The page-align start-down rule matches madviseWillneed: madvise requires a page-aligned start,
// so off is rounded DOWN and advising a few leading bytes is harmless. The INVALIDATING
// assumption mirrors the source's: enablePageCache only accepted a mapping exactly the size of the
// source, so the [start,end) range here is in-bounds by construction; a mapping that no longer
// matches the source is a caller bug this file does not paper over.

// adviseRange issues the given access hint over expert e's byte range inside the named fused
// tensor's mapping, and returns whether the hint actually fired. It is a pure performance hint
// layered on the page-cache read path: it fires only when this source has an enabled mapping that
// covers the expert, maps PageAdvice to the OS madvise constant through madviseAdvice, and
// reports false - never an error - when there is no mapping, the expert is unknown, or the
// underlying madvise refused. It never reads a byte and never changes the ledger, so a caller may
// call it freely to steer the kernel without affecting what a read returns.
func (s *ggufExpertSource) adviseRange(name string, e int, advice PageAdvice) bool {
	if s == nil {
		return false
	}
	data, ok := s.mappedData()
	if !ok {
		return false
	}
	entry, ok := s.tensors[name]
	if !ok || e < 0 || e >= entry.desc.Experts {
		return false
	}
	start := entry.desc.Offset + int64(e)*entry.stride
	end := start + entry.stride
	if start < 0 || end < start || end > int64(len(data)) {
		return false
	}
	return madviseAdvice(data, int(start), int(entry.stride), advice)
}

// mappedData returns this source's mapped region when the page cache is enabled, or (nil,false)
// when it is off. It is the shared gate for the advice seam and any future mapped-range consumer.
func (s *ggufExpertSource) mappedData() ([]byte, bool) {
	if s == nil || s.page == nil {
		return nil, false
	}
	st := s.page
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.mode != PageCacheMapped || st.data == nil {
		return nil, false
	}
	return st.data, true
}
