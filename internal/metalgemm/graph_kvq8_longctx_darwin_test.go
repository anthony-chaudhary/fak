//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

// TestProjectionGraphQwenSplitDecodeCoversLongContext pins qg_split_policy: a rows==1
// decode past 32 splits x 2048 = 65536 tokens must still attend every row. The prefix
// keys are zero (uniform scores) and only the rows past 65536 carry a non-zero V, so
// an attention that stops at 65536 returns ~0 where the oracle returns the tail's
// softmax share. Both the F32 and the packed Q8 entries share the policy.
func TestProjectionGraphQwenSplitDecodeCoversLongContext(t *testing.T) {
	requireKVQ8(t)
	defer ResetQ4K()
	const input, nH, nKV, hd, rotary, base, tail = 256, 2, 1, 32, 16, 70000, 65536
	kvwidth := nKV * hd
	l := newQwenQ8Layer(t, nH, nKV, hd, rotary, input, 1298400)
	prefixK, prefixV := make([]float32, base*kvwidth), make([]float32, base*kvwidth)
	for i := tail * kvwidth; i < len(prefixV); i++ {
		prefixV[i] = 10
	}
	packed := packKVQ8(prefixK, prefixV)
	x := q4kTestVector(input, 1298401)
	cosv, sinv := l.rope(base, 1)
	for _, mode := range []qwenQ8Mode{qwenQ8ModeF32, qwenQ8ModeHost} {
		encoders := map[string]int{}
		// "1" takes the split-KV decode; "0" (the kill switch) the single-pass
		// online kernel. Both must attend every row.
		for _, split := range []string{"1", "0"} {
			t.Setenv("FAK_QWEN35_ATTN_SPLIT", split)
			out := l.encode(t, x, 1, base, qwenQ8Prefix{k: prefixK, v: prefixV, packed: packed}, mode)
			encoders[split] = l.receipt.Encoders
			allK := append(append([]float32(nil), prefixK...), out[6]...)
			allV := append(append([]float32(nil), prefixV...), out[7]...)
			if mode == qwenQ8ModeHost {
				all := appendKVQ8(packed, q8Slots(out))
				allK, allV = dequantKVQ8(all.KCodes, all.KScales), dequantKVQ8(all.VCodes, all.VScales)
			}
			want := qwenQ8AttentionCPU(out[0], out[1], l.qnorm, cosv, sinv, allK, allV, 1, base, nH, nKV, hd, rotary, l.scale, l.eps)
			for i := range want {
				if d := math.Abs(float64(out[4][i] - want[i])); math.IsNaN(float64(out[4][i])) || d > 8e-4+1e-4*math.Abs(float64(want[i])) {
					t.Fatalf("mode %d split=%s: decode attention[%d]=%g oracle %g past %d tokens (tail dropped?)", mode, split, i, out[4][i], want[i], tail)
				}
			}
		}
		// The split path encodes split + combine where the single pass encodes one.
		if encoders["1"] != encoders["0"]+1 {
			t.Fatalf("mode %d: split-KV decode did not run (encoders split=%d single=%d)", mode, encoders["1"], encoders["0"])
		}
	}
}

// BenchmarkQwen35GraphAttentionDecode20k measures one Qwen3.8-27B full-attention
// layer (24 query heads, 4 KV heads, head_dim 256) decoding one token over a
// 20480-token context through each KV entry: the F32 host prefix the live P1 route
// uses today, the packed Q8 host prefix, and the persistent F32 and packed stores.
// gpu-ms/op is the command buffer's GPU time; ns/op includes the host prefix copy.
func BenchmarkQwen35GraphAttentionDecode20k(b *testing.B) {
	requireKVQ8(b)
	defer ResetQ4K()
	const input, nH, nKV, hd, rotary, base = 256, 24, 4, 256, 64, 20479
	kvwidth, qwidth := nKV*hd, nH*hd

	l := newQwenQ8Layer(b, nH, nKV, hd, rotary, input, 1298500)
	prefixK, prefixV := q4kTestVector(base*kvwidth, 1298501), q4kTestVector(base*kvwidth, 1298502)
	packed := packKVQ8(prefixK, prefixV)
	dkv := NewDeviceKV(1, base+1, kvwidth)
	store := NewDeviceKVQ8(1, base+1, kvwidth)
	if dkv == nil || store == nil {
		b.Fatal("device KV allocation")
	}
	defer dkv.Close()
	defer store.Close()
	for side, src := range [][]float32{prefixK, prefixK, prefixV} {
		if err := dkv.Upload(side, src); err != nil {
			b.Fatal(err)
		}
	}
	if err := store.WriteRows(0, 0, packed); err != nil {
		b.Fatal(err)
	}
	x := q4kTestVector(input, 1298503)
	cosv, sinv := l.rope(base, 1)
	for _, bc := range []struct {
		name string
		mode int
	}{{"f32_host_prefix", 0}, {"q8_host_prefix", 1}, {"f32_device_kv", 2}, {"q8_device_kv", 3}} {
		b.Run(bc.name, func(b *testing.B) {
			var gpu float64
			for i := 0; i < b.N; i++ {
				g, err := BeginProjectionGraph(x, nil, nil, 1, input)
				if err != nil {
					b.Fatal(err)
				}
				qg, _ := g.EncodeQ4K(l.qgate)
				k, _ := g.EncodeQ4K(l.kw)
				v, _ := g.EncodeQ4K(l.vw)
				q, gate, err := g.SplitGatedQ(qg, qwidth, hd)
				if err != nil {
					b.Fatal(err)
				}
				var att Qwen35GraphAttentionResult
				switch bc.mode {
				case 0:
					att, err = g.FullAttention(q, k, v, gate, l.qnorm, l.knorm, cosv, sinv, prefixK, prefixV, base, nH, nKV, hd, rotary, l.scale, l.eps, true, true)
				case 1:
					var r Qwen35GraphAttentionQ8Result
					r, err = g.FullAttentionQ8(q, k, v, gate, l.qnorm, l.knorm, cosv, sinv, packed, base, nH, nKV, hd, rotary, l.scale, l.eps, true, true)
					att = r.Qwen35GraphAttentionResult
				case 2:
					att, err = g.FullAttentionDevice(q, k, v, gate, dkv, 0, l.qnorm, l.knorm, cosv, sinv, base, nH, nKV, hd, rotary, l.scale, l.eps, true, true)
				case 3:
					var r Qwen35GraphAttentionQ8Result
					r, err = g.FullAttentionDeviceQ8(q, k, v, gate, store, 0, l.qnorm, l.knorm, cosv, sinv, base, nH, nKV, hd, rotary, l.scale, l.eps, true, true)
					att = r.Qwen35GraphAttentionResult
				}
				if err != nil {
					b.Fatal(err)
				}
				_, receipt, err := g.FinishRead(att.Output)
				if err != nil {
					b.Fatal(err)
				}
				gpu += receipt.GPUMilliseconds
				g.Free()
			}
			b.ReportMetric(gpu/float64(b.N), "gpu-ms/op")
		})
	}
}
