//go:build cuda && cgo

package compute

import "testing"

// Synthetic host owners exercise the production observer without CUDA copies,
// allocations or frees. Nonparallel: uploadCache is a process-global owner.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAUploadHostRetentionCountsIndexedLogicalViews(t *testing.T) {
	f32 := &hostBuf{f32: make([]float32, 8)}
	raw := &hostBuf{i8: make([]int8, 144)}
	q8 := &cudaQ8UploadSource{HostBuffer: &hostBuf{i8: make([]int8, 256)}, scales: make([]float32, 8)}
	rawOwner := &cudaBuf{hostKeep: raw}
	cache := map[ucKey]Tensor{
		{1, F32, RowMajor}:  {buf: &cudaBuf{hostKeep: f32}},
		{1, F16, RowMajor}:  {buf: &cudaBuf{hostKeep: f32}}, // same backing counts once per entry
		{2, Q4_K, RowMajor}: {buf: rawOwner},
		{3, Q8_0, RowMajor}: {buf: &cudaBuf{hostKeep: q8}},
		{4, F32, RowMajor}:  {buf: &cudaBuf{}}, // no strong host owner
		{5, F32, RowMajor}:  {buf: f32},        // not a CUDA cache owner
		{6, F32, RowMajor}:  {buf: (*cudaBuf)(nil)},
	}
	cudaMu.Lock()
	saved := uploadCache
	uploadCache = cache
	cudaMu.Unlock()
	defer func() {
		cudaMu.Lock()
		uploadCache = saved
		cudaMu.Unlock()
	}()

	cb := &cudaBackend{}
	entries, payload, scales := cb.CUDAUploadHostRetentionSnapshot()
	if entries != 4 || payload != 2*8*4+144+256 || scales != 8*4 {
		t.Fatalf("indexed logical views = (%d, %d, %d), want (4, 464, 32)", entries, payload, scales)
	}
	cudaMu.Lock()
	if len(uploadCache) != 7 || rawOwner.hostKeep != raw || rawOwner.ptr != nil {
		cudaMu.Unlock()
		t.Fatal("observation changed source or cache ownership")
	}
	delete(uploadCache, ucKey{2, Q4_K, RowMajor})
	cudaMu.Unlock()
	entries, payload, scales = cb.CUDAUploadHostRetentionSnapshot()
	if entries != 3 || payload != 320 || scales != 32 || rawOwner.hostKeep != raw {
		t.Fatalf("unindexed owner must remain intact but excluded: (%d, %d, %d)", entries, payload, scales)
	}
}
