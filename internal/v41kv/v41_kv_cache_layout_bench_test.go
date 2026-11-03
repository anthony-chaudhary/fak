package v41kv

import (
	"testing"
)

// BenchmarkV41KVQuantizeDequantizeFP8 times one full host-side round trip of a
// deterministic 512-wide KV row through the published 528-byte fp8 record: tile
// scale selection plus e4m3 encode for quantization, and the inverse decode.
// Throughput is reported per record byte via SetBytes(BytesPerToken).
func BenchmarkV41KVQuantizeDequantizeFP8(b *testing.B) {
	l, ok := V41KVCacheLayoutFor(V41KVFP8)
	if !ok {
		b.Fatal("fp8 layout is not in the published set")
	}
	nope := v41KVTestRow(l.DNoPE)
	rope := v41KVTestRow(l.DRoPE)
	b.SetBytes(int64(l.BytesPerToken))
	for i := 0; i < b.N; i++ {
		record, err := l.V41KVQuantizeRow(nope, rope)
		if err != nil {
			b.Fatal(err)
		}
		row, err := l.V41KVDequantizeRow(record)
		if err != nil {
			b.Fatal(err)
		}
		if len(row.NoPE) != l.DNoPE || len(row.RoPE) != l.DRoPE {
			b.Fail()
		}
	}
}

// BenchmarkV41KVQuantizeDequantizeFP4 does the same round-trip measurement for
// the published 288-byte fp4 record: e2m1 nibble packing two-per-byte with the
// e4m3 amax/6 tile scales, and its unpacking. Throughput is reported per record
// byte via SetBytes(BytesPerToken).
func BenchmarkV41KVQuantizeDequantizeFP4(b *testing.B) {
	l, ok := V41KVCacheLayoutFor(V41KVFP4)
	if !ok {
		b.Fatal("fp4 layout is not in the published set")
	}
	nope := v41KVTestRow(l.DNoPE)
	rope := v41KVTestRow(l.DRoPE)
	b.SetBytes(int64(l.BytesPerToken))
	for i := 0; i < b.N; i++ {
		record, err := l.V41KVQuantizeRow(nope, rope)
		if err != nil {
			b.Fatal(err)
		}
		row, err := l.V41KVDequantizeRow(record)
		if err != nil {
			b.Fatal(err)
		}
		if len(row.NoPE) != l.DNoPE || len(row.RoPE) != l.DRoPE {
			b.Fail()
		}
	}
}
