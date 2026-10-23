package compute

import (
	"os"
	"strings"
	"testing"
)

// Pure arithmetic and source binding only; no device allocation or CUDA call.
// Runtime estimate only, not measured.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAKVAppendBounds(t *testing.T) {
	t.Parallel()
	maxByteFloats := int(^uint(0)>>1) / 4
	limit := min(cudaKVMaxCopyFloats, maxByteFloats)
	for _, tc := range []struct {
		name                    string
		length, capacity, count int
		want                    int
		refuse                  bool
	}{
		{"empty", 0, 0, 4, 4, false},
		{"spare", 4, 16, 4, 16, false},
		{"grow", 8, 8, 4, 20, false},
		{"exact-limit", limit - 1, limit, 1, limit, false},
		{"one-over", limit, limit, 1, 0, true},
		{"oversized-count", 0, 0, limit + 1, 0, true},
		{"negative-count", 0, 0, -1, 0, true},
		{"negative-length", -1, 0, 1, 0, true},
		{"short-capacity", 4, 3, 1, 0, true},
		{"byte-overflow", 0, maxByteFloats + 1, 1, 0, true},
		{"saturate-growth", limit/2 + 1, limit/2 + 1, 1, limit, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := cudaKVAppendCapacity(tc.length, tc.capacity, tc.count, "bounds-test")
			if got != tc.want || (err != nil) != tc.refuse {
				t.Fatalf("capacity/error = %d/%v, want %d/refuse=%v", got, err, tc.want, tc.refuse)
			}
			if tc.refuse {
				e, ok := err.(*CUDAOpError)
				if !ok || e.Op != "AppendKV" || e.Site != "bounds-test" || e.Class != MemoryKVCache {
					t.Fatalf("refusal lost typed operation/site/class: %v", err)
				}
			}
		})
	}
	// Pin the pure planner before the first native operation, and all three row
	// preflights before the first append. This remains runnable without CUDA tags.
	raw, err := os.ReadFile("cuda_kv.go")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	for _, pair := range [][2]string{
		{"ncap, err := cudaKVAppendCapacity(d.len, d.cap, nFloats, site)", "np := c.dallocKV(ncap*F32.Bytes(), site).ptr"},
		{"for _, row := range [3]dslice{k.Kraw[layer], k.K[layer], k.V[layer]}", "k.be.growAppend(&k.Kraw[layer]"},
		{"cudaKVAppendCapacity(row.len, row.cap, w, \"kv-append\")", "k.be.growAppend(&k.Kraw[layer]"},
		{"k.cfg.NumKVHeads > cudaKVMaxCopyFloats/k.cfg.HeadDim", "w := k.stride()"},
	} {
		check, operation := strings.Index(source, pair[0]), strings.Index(source, pair[1])
		if check < 0 || operation < 0 || check >= operation {
			t.Fatalf("missing or late bounds check %q before %q", pair[0], pair[1])
		}
	}
	for _, assertion := range []string{"var _ [4 - C.sizeof_int]byte", "var _ [C.sizeof_int - 4]byte"} {
		if !strings.Contains(source, assertion) {
			t.Fatalf("missing signed 32-bit C ABI width assertion %q", assertion)
		}
	}
}
