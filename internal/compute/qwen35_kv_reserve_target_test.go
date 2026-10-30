package compute

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestQwen35KVReserveTargetChunksGrowableStores(t *testing.T) {
	const stride = 1024
	for _, tc := range []struct{ needed, stride, maxPos, want int }{
		{1, stride, 0, 256 * stride},
		{256 * stride, stride, 0, 256 * stride},
		{256*stride + 1, stride, 0, 512 * stride},
		{37 * stride, stride, 0, 256 * stride},
		{37 * stride, stride, 4096, 37 * stride}, // fixed graph store: exact
		{0, stride, 0, 0},
		{29, 0, 0, 29},
		{int(qwen35GDNMaxCInt) - 5, stride, 0, int(qwen35GDNMaxCInt) - 5}, // rounding would overflow C int
	} {
		if got := qwen35KVReserveTarget(tc.needed, tc.stride, tc.maxPos); got != tc.want {
			t.Fatalf("target(%d, %d, %d) = %d, want %d", tc.needed, tc.stride, tc.maxPos, got, tc.want)
		}
	}
	// Two requests of different prompt lengths in one chunk share a size class.
	if qwen35KVReserveTarget(100*stride, stride, 0) != qwen35KVReserveTarget(200*stride, stride, 0) {
		t.Fatal("same-chunk requests got distinct KV size classes")
	}
}
