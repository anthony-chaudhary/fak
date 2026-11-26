package compute

import "testing"

const loadBenchBytes = 64 << 20

// nvfp4LoadScales is a valid scale stream shaped like a real checkpoint: every
// non-negative finite E4M3 byte, never the negative zero or NaN encodings.
func nvfp4LoadScales(n int) []byte {
	scales := make([]byte, n)
	for i := range scales {
		scales[i] = byte(i % 0x7F)
	}
	return scales
}

func BenchmarkNVFP4ValidateScales64MiB(b *testing.B) {
	scales := nvfp4LoadScales(loadBenchBytes)
	b.SetBytes(int64(len(scales)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := nvfp4ValidateScales(scales, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBytesAsI864MiB(b *testing.B) {
	payload := nvfp4LoadScales(loadBenchBytes)
	b.SetBytes(int64(len(payload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := bytesAsI8(payload); len(got) != len(payload) {
			b.Fatal(len(got))
		}
	}
}
