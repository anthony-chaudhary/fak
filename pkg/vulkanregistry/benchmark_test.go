package vulkanregistry

import "testing"

// BenchmarkCurrent times Current(), the read-and-hash path a receipt-admitting
// consumer runs before trusting a Vulkan build receipt: one lookup of the
// builder's current registry plus a SHA-256 over its sorted module stems. A
// jump here shows up directly as admission latency on every receipt check.
//
// fak-test:runtime fast est=1s lane=default
// fak-test:usefulness protects=vulkan-registry-current-benchmark
func BenchmarkCurrent(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := Current(); err != nil {
			b.Fatal(err)
		}
	}
}
