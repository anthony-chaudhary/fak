//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

func TestQwen35FullAttentionDecodeBatchIndependentKVSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	for _, batch := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("B%d", batch), func(t *testing.T) {
			baseOwners, baseBuffers := qwen35AttentionBatchGraphLiveCounts()
			const hidden = 256
			const nH = 4
			const nKV = 2
			const hd = 64
			const rotary = 64
			kvw := nKV * hd
			weights := Qwen35FullAttentionWeights{Q: attentionBatchQ8(t, 2*hidden, hidden, 11), K: attentionBatchQ8(t, kvw, hidden, 13), V: attentionBatchQ4K(t, kvw, hidden, 17)}
			defer weights.Q.Release()
			defer weights.K.Release()
			defer weights.V.Release()
			lanes := make([]Qwen35FullAttentionLane, batch)
			input := make([]float32, batch*hidden)
			maxPos := 0
			for row := 0; row < batch; row++ {
				pos := row + 2
				if pos > maxPos {
					maxPos = pos
				}
				lanes[row] = Qwen35FullAttentionLane{Position: pos, PrefixK: attentionBatchValues(pos*kvw, row+3), PrefixV: attentionBatchValues(pos*kvw, row+7)}
				copy(input[row*hidden:], attentionBatchValues(hidden, row+11))
			}
			cosv, sinv := attentionBatchRope(maxPos+1, rotary)
			req := Qwen35FullAttentionBatchRequest{Input: input, Weights: weights, Lanes: lanes, QNorm: attentionBatchOnes(hd), KNorm: attentionBatchOnes(hd), Cos: cosv, Sin: sinv, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, RotaryDim: rotary, Scale: 1 / float32(math.Sqrt(hd)), QKNormEpsilon: 1e-6, QKNorm: true}
			got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
			if err != nil || !accepted {
				t.Fatalf("accepted=%v err=%v", accepted, err)
			}
			if receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 || receipt.ProjectionDispatches != 3 || receipt.InputUploads != 1 || receipt.IntermediateReadbacks != 0 || receipt.FinalReadbacks != 1 {
				t.Fatalf("receipt=%+v", receipt)
			}
			if owners, buffers := qwen35AttentionBatchGraphLiveCounts(); owners != baseOwners || buffers != baseBuffers {
				t.Fatalf("B=%d success retained graph owners/buffers=%d/%d, baseline=%d/%d", batch, owners, buffers, baseOwners, baseBuffers)
			}
			for row := 0; row < batch; row++ {
				one := req
				one.Input = append([]float32(nil), input[row*hidden:(row+1)*hidden]...)
				one.Lanes = []Qwen35FullAttentionLane{lanes[row]}
				want := attentionBatchSerialOracle(t, one)
				attentionBatchClose(t, "output", got.Output[row], want.Output[0])
				attentionBatchClose(t, "kraw", got.KRaw[row], want.KRaw[0])
				attentionBatchClose(t, "kpost", got.KPost[row], want.KPost[0])
				attentionBatchClose(t, "v", got.V[row], want.V[0])
			}
			changed := req
			changed.Lanes = append([]Qwen35FullAttentionLane(nil), lanes...)
			changed.Lanes[0].PrefixK = append([]float32(nil), lanes[0].PrefixK...)
			changed.Lanes[0].PrefixK[0] += .5
			alt, _, ok, err := RunQwen35FullAttentionDecodeBatch(changed)
			if err != nil || !ok {
				t.Fatal(err)
			}
			if owners, buffers := qwen35AttentionBatchGraphLiveCounts(); owners != baseOwners || buffers != baseBuffers {
				t.Fatalf("B=%d lane-isolation run retained graph owners/buffers=%d/%d, baseline=%d/%d", batch, owners, buffers, baseOwners, baseBuffers)
			}
			if attentionBatchSame(alt.Output[0], got.Output[0]) {
				t.Fatal("changed lane did not change")
			}
			for row := 1; row < batch; row++ {
				attentionBatchBits(t, alt.Output[row], got.Output[row])
			}
		})
	}
	t.Run("committed_failure_is_not_replayed", func(t *testing.T) {
		const hidden, nH, nKV, hd, rotary = 256, 4, 2, 64, 64
		kvw := nKV * hd
		weights := Qwen35FullAttentionWeights{Q: attentionBatchQ8(t, 2*hidden, hidden, 23), K: attentionBatchQ8(t, kvw, hidden, 29), V: attentionBatchQ4K(t, kvw, hidden, 31)}
		defer weights.Q.Release()
		defer weights.K.Release()
		defer weights.V.Release()
		cosv, sinv := attentionBatchRope(4, rotary)
		req := Qwen35FullAttentionBatchRequest{
			Input: attentionBatchValues(2*hidden, 37), Weights: weights,
			Lanes: []Qwen35FullAttentionLane{
				{Position: 2, PrefixK: attentionBatchValues(2*kvw, 41), PrefixV: attentionBatchValues(2*kvw, 43)},
				{Position: 3, PrefixK: attentionBatchValues(3*kvw, 47), PrefixV: attentionBatchValues(3*kvw, 53)},
			},
			QNorm: attentionBatchOnes(hd), KNorm: attentionBatchOnes(hd), Cos: cosv, Sin: sinv,
			NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, RotaryDim: rotary,
			Scale: 1 / float32(math.Sqrt(hd)), QKNormEpsilon: 1e-6, QKNorm: true,
			InjectPostSubmitFailureForTest: true,
		}
		baseOwners, baseBuffers := qwen35AttentionBatchGraphLiveCounts()
		for attempt := 0; attempt < 3; attempt++ {
			_, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
			var post *GraphPostSubmitError
			if !accepted || !errors.As(err, &post) || !receipt.Committed || receipt.Commits != 1 || receipt.CompletionWaits != 1 || receipt.FinalReadbacks != 0 {
				t.Fatalf("attempt=%d accepted=%v receipt=%+v err=%v", attempt, accepted, receipt, err)
			}
			if owners, buffers := qwen35AttentionBatchGraphLiveCounts(); owners != baseOwners || buffers != baseBuffers {
				t.Fatalf("attempt=%d failure retained graph owners/buffers=%d/%d, baseline=%d/%d", attempt, owners, buffers, baseOwners, baseBuffers)
			}
		}
	})
}

// The serial oracle deliberately uses row zero of the established P=32 graph
// operation. At t=0 that graph attends exactly prefix+1 tokens; placing the
// decode row at t=31 would silently add 31 zero-panel keys to the softmax and
// would not be an independent single-token decode oracle.
func attentionBatchSerialOracle(t *testing.T, req Qwen35FullAttentionBatchRequest) Qwen35FullAttentionBatchResult {
	t.Helper()
	l := req.Lanes[0]
	hidden := req.NumHeads * req.HeadDim
	kvw := req.NumKVHeads * req.HeadDim
	panel := make([]float32, 32*hidden)
	copy(panel, req.Input)
	g, err := BeginProjectionGraph(panel, nil, nil, 32, hidden)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	x, _ := g.Input(hidden)
	qx, _ := g.QuantizeQ8(x)
	qg, _ := g.EncodeQ8From(req.Weights.Q, qx)
	k, _ := g.EncodeQ8From(req.Weights.K, qx)
	v, _ := g.EncodeQ4KFrom(req.Weights.V, x)
	q, gate, err := g.SplitGatedQ(qg, hidden, req.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	// The established operation requires 32 rows, but row zero must carry the
	// lane's absolute rotary position for this decode comparison.
	cosv, sinv := attentionBatchRope(32, req.RotaryDim)
	halfRotary := req.RotaryDim / 2
	copy(cosv[:halfRotary], req.Cos[l.Position*halfRotary:(l.Position+1)*halfRotary])
	copy(sinv[:halfRotary], req.Sin[l.Position*halfRotary:(l.Position+1)*halfRotary])
	attn, err := g.FullAttention(q, k, v, gate, req.QNorm, req.KNorm, cosv, sinv, l.PrefixK, l.PrefixV, l.Position, req.NumHeads, req.NumKVHeads, req.HeadDim, req.RotaryDim, req.Scale, req.QKNormEpsilon, req.Gain1p, req.QKNorm)
	if err != nil {
		t.Fatal(err)
	}
	o, r, e := g.FinishRead(attn.Output, attn.KRaw, attn.KPost, attn.V)
	if e != nil || !r.Committed {
		t.Fatal(e)
	}
	return Qwen35FullAttentionBatchResult{Output: [][]float32{o[0][:hidden]}, KRaw: [][]float32{o[1][:kvw]}, KPost: [][]float32{o[2][:kvw]}, V: [][]float32{o[3][:kvw]}}
}
func attentionBatchQ8(t *testing.T, out, in, seed int) *Q8Weight {
	t.Helper()
	c := make([]int8, out*in)
	s := make([]float32, out*(in/32))
	for i := range c {
		c[i] = int8((i+seed)%15) - 7
	}
	for i := range s {
		s[i] = .004 + float32((i+seed)%7)*.001
	}
	w := UploadQ8(c, s, out, in)
	if w == nil {
		t.Fatal("Q8 upload")
	}
	return w
}
func attentionBatchQ4K(t *testing.T, out, in, seed int) *Q4KWeight {
	t.Helper()
	w := UploadQ4K(q4kTestRaw(out, in, uint64(seed)), out, in)
	if w == nil {
		t.Fatal("Q4K upload")
	}
	return w
}
func attentionBatchValues(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i+seed))) * .1
	}
	return x
}
func attentionBatchOnes(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = 1
	}
	return x
}
func attentionBatchRope(pos, rot int) ([]float32, []float32) {
	c := make([]float32, pos*rot/2)
	s := make([]float32, len(c))
	for p := 0; p < pos; p++ {
		for j := 0; j < rot/2; j++ {
			a := float64(p) * math.Pow(10000, -2*float64(j)/float64(rot))
			c[p*rot/2+j] = float32(math.Cos(a))
			s[p*rot/2+j] = float32(math.Sin(a))
		}
	}
	return c, s
}
func attentionBatchClose(t *testing.T, n string, g, w []float32) {
	t.Helper()
	if len(g) != len(w) {
		t.Fatalf("%s len", n)
	}
	var d, gg, ww, m float64
	for i := range g {
		a, b := float64(g[i]), float64(w[i])
		d += a * b
		gg += a * a
		ww += b * b
		if x := math.Abs(a - b); x > m {
			m = x
		}
	}
	if c := d / (math.Sqrt(gg*ww) + 1e-30); c < .999999 || m > .0001 {
		t.Fatalf("%s cosine=%g max=%g", n, c, m)
	}
}
func attentionBatchSame(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
func attentionBatchBits(t *testing.T, a, b []float32) {
	t.Helper()
	if !attentionBatchSame(a, b) {
		t.Fatal("cross-lane mutation")
	}
}
