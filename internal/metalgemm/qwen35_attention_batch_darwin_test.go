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
	t.Run("checked_product_boundaries", func(t *testing.T) {
		const batch = 2
		atLimit, ok := qwen35CheckedFullAttentionBatchSizes(batch, qwen35MaxCInt/batch, qwen35MaxCInt/(2*batch), qwen35MaxCInt/batch)
		if !ok || atLimit.Input != batch*(qwen35MaxCInt/batch) || atLimit.GatedQ != batch*2*(qwen35MaxCInt/(2*batch)) || atLimit.Attention != batch*(qwen35MaxCInt/(2*batch)) || atLimit.KV != batch*(qwen35MaxCInt/batch) {
			t.Fatalf("INT_MAX boundary sizes=%+v ok=%v", atLimit, ok)
		}
		if _, ok := qwen35CheckedFullAttentionBatchSizes(batch, qwen35MaxCInt/batch+1, 256, 128); ok {
			t.Fatal("B*modelWidth overflow admitted")
		}
		if _, ok := qwen35CheckedFullAttentionBatchSizes(batch, 256, qwen35MaxCInt/(2*batch)+1, 128); ok {
			t.Fatal("B*2*attentionWidth overflow admitted")
		}
		if _, ok := qwen35CheckedFullAttentionBatchSizes(batch, 256, 256, qwen35MaxCInt/batch+1); ok {
			t.Fatal("B*kvWidth overflow admitted")
		}
		if _, ok := qwen35CheckedCIntProduct(qwen35MaxCInt/2+1, 2); ok {
			t.Fatal("packed KV overflow admitted")
		}
	})
	t.Run("exact_split_width", func(t *testing.T) {
		const modelWidth, nH, nKV, hd, rotary = 5120, 24, 4, 256, 64
		const attentionWidth, kvWidth = nH * hd, nKV * hd
		baseQ8 := LiveQ8Weights()
		weights := Qwen35FullAttentionWeights{
			Q: attentionBatchQ8(t, 2*attentionWidth, modelWidth, 11),
			K: attentionBatchQ8(t, kvWidth, modelWidth, 13),
			V: attentionBatchQ4K(t, kvWidth, modelWidth, 17),
		}
		defer func() {
			weights.Q.Release()
			weights.K.Release()
			weights.V.Release()
			if got := LiveQ8Weights(); got != baseQ8 {
				t.Errorf("resident Q8 weights after cleanup=%d, baseline=%d", got, baseQ8)
			}
		}()

		for _, batch := range []int{2, 4, 8} {
			t.Run(fmt.Sprintf("B%d", batch), func(t *testing.T) {
				req := attentionBatchRequest(batch, modelWidth, nH, nKV, hd, rotary, weights)
				attentionBatchRunParityAndIsolation(t, req, modelWidth, attentionWidth, kvWidth)
			})
		}

		t.Run("invalid_dimensions_clean_decline", func(t *testing.T) {
			req := attentionBatchRequest(2, modelWidth, nH, nKV, hd, rotary, weights)
			badQ := &Q8Weight{In: modelWidth, Out: 2*attentionWidth - 1}
			badK := &Q8Weight{In: modelWidth + 32, Out: kvWidth}
			badV := &Q4KWeight{In: modelWidth + 256, Out: kvWidth}
			shortPrefix := req
			shortPrefix.Lanes = append([]Qwen35FullAttentionLane(nil), req.Lanes...)
			shortPrefix.Lanes[0].PrefixK = shortPrefix.Lanes[0].PrefixK[:len(shortPrefix.Lanes[0].PrefixK)-1]
			modelProductOverflow := req
			modelProductOverflow.Weights.Q = &Q8Weight{In: qwen35MaxCInt/2 + 1, Out: 2 * attentionWidth}
			attentionProductOverflow := req
			attentionProductOverflow.NumHeads = (qwen35MaxCInt/hd - 3) &^ 3
			for _, tc := range []struct {
				name string
				req  Qwen35FullAttentionBatchRequest
			}{
				{"input_model_width", func() Qwen35FullAttentionBatchRequest { x := req; x.Input = x.Input[:len(x.Input)-1]; return x }()},
				{"q_attention_width", func() Qwen35FullAttentionBatchRequest { x := req; x.Weights.Q = badQ; return x }()},
				{"k_model_width", func() Qwen35FullAttentionBatchRequest { x := req; x.Weights.K = badK; return x }()},
				{"v_model_width", func() Qwen35FullAttentionBatchRequest { x := req; x.Weights.V = badV; return x }()},
				{"prefix_kv_width", shortPrefix},
				{"batch_model_product_overflow", modelProductOverflow},
				{"batch_gated_q_product_overflow", attentionProductOverflow},
			} {
				t.Run(tc.name, func(t *testing.T) {
					baseOwners, baseBuffers := qwen35AttentionBatchGraphLiveCounts()
					beforeInput := append([]float32(nil), tc.req.Input...)
					_, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(tc.req)
					var decline *MixedQKVError
					if accepted || !errors.As(err, &decline) || decline.Stage != MixedQKVDeclined || !attentionBatchZeroReceipt(receipt) {
						t.Fatalf("accepted=%v receipt=%+v err=%v", accepted, receipt, err)
					}
					attentionBatchBits(t, tc.req.Input, beforeInput)
					if owners, buffers := qwen35AttentionBatchGraphLiveCounts(); owners != baseOwners || buffers != baseBuffers {
						t.Fatalf("decline retained graph owners/buffers=%d/%d, baseline=%d/%d", owners, buffers, baseOwners, baseBuffers)
					}
				})
			}
		})

		t.Run("committed_failure_is_not_replayed", func(t *testing.T) {
			req := attentionBatchRequest(2, modelWidth, nH, nKV, hd, rotary, weights)
			req.InjectPostSubmitFailureForTest = true
			baseOwners, baseBuffers := qwen35AttentionBatchGraphLiveCounts()
			for attempt := 0; attempt < 3; attempt++ {
				_, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
				var post *GraphPostSubmitError
				if !accepted || !errors.As(err, &post) || !receipt.Committed || receipt.Commits != 1 || receipt.CompletionWaits != 1 || receipt.ProjectionDispatches != 3 || receipt.FinalReadbacks != 0 {
					t.Fatalf("attempt=%d accepted=%v receipt=%+v err=%v", attempt, accepted, receipt, err)
				}
				if owners, buffers := qwen35AttentionBatchGraphLiveCounts(); owners != baseOwners || buffers != baseBuffers {
					t.Fatalf("attempt=%d failure retained graph owners/buffers=%d/%d, baseline=%d/%d", attempt, owners, buffers, baseOwners, baseBuffers)
				}
			}
		})
	})

	t.Run("equal_width_compatibility", func(t *testing.T) {
		const modelWidth, nH, nKV, hd, rotary = 256, 4, 2, 64, 64
		const attentionWidth, kvWidth = nH * hd, nKV * hd
		weights := Qwen35FullAttentionWeights{
			Q: attentionBatchQ8(t, 2*attentionWidth, modelWidth, 23),
			K: attentionBatchQ8(t, kvWidth, modelWidth, 29),
			V: attentionBatchQ4K(t, kvWidth, modelWidth, 31),
		}
		defer weights.Q.Release()
		defer weights.K.Release()
		defer weights.V.Release()
		attentionBatchRunParityAndIsolation(t, attentionBatchRequest(2, modelWidth, nH, nKV, hd, rotary, weights), modelWidth, attentionWidth, kvWidth)
	})
}

func attentionBatchRequest(batch, modelWidth, nH, nKV, hd, rotary int, weights Qwen35FullAttentionWeights) Qwen35FullAttentionBatchRequest {
	kvWidth := nKV * hd
	lanes := make([]Qwen35FullAttentionLane, batch)
	input := make([]float32, batch*modelWidth)
	maxPos := 0
	for row := 0; row < batch; row++ {
		pos := row + 2
		if pos > maxPos {
			maxPos = pos
		}
		lanes[row] = Qwen35FullAttentionLane{Position: pos, PrefixK: attentionBatchValues(pos*kvWidth, row+3), PrefixV: attentionBatchValues(pos*kvWidth, row+7)}
		copy(input[row*modelWidth:], attentionBatchValues(modelWidth, row+11))
	}
	cosv, sinv := attentionBatchRope(maxPos+1, rotary)
	return Qwen35FullAttentionBatchRequest{Input: input, Weights: weights, Lanes: lanes, QNorm: attentionBatchOnes(hd), KNorm: attentionBatchOnes(hd), Cos: cosv, Sin: sinv, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, RotaryDim: rotary, Scale: 1 / float32(math.Sqrt(float64(hd))), QKNormEpsilon: 1e-6, QKNorm: true}
}

func attentionBatchRunParityAndIsolation(t *testing.T, req Qwen35FullAttentionBatchRequest, modelWidth, attentionWidth, kvWidth int) {
	t.Helper()
	batch := len(req.Lanes)
	baseOwners, baseBuffers := qwen35AttentionBatchGraphLiveCounts()
	got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
	if err != nil || !accepted {
		t.Fatalf("accepted=%v err=%v", accepted, err)
	}
	if receipt.Batch != batch || receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 || receipt.ProjectionDispatches != 3 || receipt.AttentionDispatches != 4 || receipt.InputUploads != 1 || receipt.IntermediateReadbacks != 0 || receipt.FinalReadbacks != 1 || !receipt.Committed || !receipt.CompletedWait || len(receipt.AppendElements) != batch {
		t.Fatalf("receipt=%+v", receipt)
	}
	for row, n := range receipt.AppendElements {
		if n != kvWidth {
			t.Fatalf("row %d append elements=%d want %d", row, n, kvWidth)
		}
	}
	if owners, buffers := qwen35AttentionBatchGraphLiveCounts(); owners != baseOwners || buffers != baseBuffers {
		t.Fatalf("B=%d success retained graph owners/buffers=%d/%d, baseline=%d/%d", batch, owners, buffers, baseOwners, baseBuffers)
	}
	for row := 0; row < batch; row++ {
		if len(got.Output[row]) != attentionWidth || len(got.KRaw[row]) != kvWidth || len(got.KPost[row]) != kvWidth || len(got.V[row]) != kvWidth ||
			!attentionBatchFiniteNonzero(got.Output[row]) || !attentionBatchFiniteNonzero(got.KRaw[row]) || !attentionBatchFiniteNonzero(got.KPost[row]) || !attentionBatchFiniteNonzero(got.V[row]) {
			t.Fatalf("row %d malformed or non-finite result output=%d kraw=%d kpost=%d v=%d", row, len(got.Output[row]), len(got.KRaw[row]), len(got.KPost[row]), len(got.V[row]))
		}
		one := req
		one.Input = append([]float32(nil), req.Input[row*modelWidth:(row+1)*modelWidth]...)
		one.Lanes = []Qwen35FullAttentionLane{req.Lanes[row]}
		want := attentionBatchCPUOracle(t, one)
		attentionBatchClose(t, "output", got.Output[row], want.Output[0])
		attentionBatchClose(t, "kraw", got.KRaw[row], want.KRaw[0])
		attentionBatchClose(t, "kpost", got.KPost[row], want.KPost[0])
		attentionBatchClose(t, "v", got.V[row], want.V[0])
	}
	changed := req
	changed.Lanes = append([]Qwen35FullAttentionLane(nil), req.Lanes...)
	changed.Lanes[0].PrefixK = append([]float32(nil), req.Lanes[0].PrefixK...)
	changed.Lanes[0].PrefixK[0] += .5
	alt, altReceipt, ok, err := RunQwen35FullAttentionDecodeBatch(changed)
	if err != nil || !ok || altReceipt.Commits != 1 || altReceipt.CompletionWaits != 1 {
		t.Fatalf("lane-isolation accepted=%v receipt=%+v err=%v", ok, altReceipt, err)
	}
	if owners, buffers := qwen35AttentionBatchGraphLiveCounts(); owners != baseOwners || buffers != baseBuffers {
		t.Fatalf("B=%d lane-isolation run retained graph owners/buffers=%d/%d, baseline=%d/%d", batch, owners, buffers, baseOwners, baseBuffers)
	}
	if attentionBatchSame(alt.Output[0], got.Output[0]) {
		t.Fatal("changed lane did not change")
	}
	for row := 1; row < batch; row++ {
		attentionBatchBits(t, alt.Output[row], got.Output[row])
		attentionBatchBits(t, alt.KRaw[row], got.KRaw[row])
		attentionBatchBits(t, alt.KPost[row], got.KPost[row])
		attentionBatchBits(t, alt.V[row], got.V[row])
	}
}

func attentionBatchZeroReceipt(r Qwen35FullAttentionBatchReceipt) bool {
	return r.Batch == 0 && r.CommandBuffers == 0 && r.Commits == 0 && r.CompletionWaits == 0 && r.ProjectionDispatches == 0 && r.AttentionDispatches == 0 && r.InputUploads == 0 && r.FinalReadbacks == 0 && r.IntermediateReadbacks == 0 && len(r.AppendElements) == 0 && !r.Committed && !r.CompletedWait
}

func attentionBatchFiniteNonzero(x []float32) bool {
	nonzero := false
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
		nonzero = nonzero || v != 0
	}
	return nonzero
}

// attentionBatchCPUOracle obtains Q/K/V through the established P=32 projection
// graph, then performs Q/K normalization, RoPE, softmax attention, and output
// gating on the CPU. The projection graph is independent of the lane-attention
// kernels under test; the CPU portion pins their width/stride and prefix math.
func attentionBatchCPUOracle(t *testing.T, req Qwen35FullAttentionBatchRequest) Qwen35FullAttentionBatchResult {
	t.Helper()
	modelWidth := req.Weights.Q.In
	attentionWidth := req.NumHeads * req.HeadDim
	kvWidth := req.NumKVHeads * req.HeadDim
	panel := make([]float32, 32*modelWidth)
	copy(panel, req.Input)
	g, err := BeginProjectionGraph(panel, nil, nil, 32, modelWidth)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	x, _ := g.Input(modelWidth)
	qx, _ := g.QuantizeQ8(x)
	qg, _ := g.EncodeQ8From(req.Weights.Q, qx)
	k, _ := g.EncodeQ8From(req.Weights.K, qx)
	v, _ := g.EncodeQ4KFrom(req.Weights.V, x)
	q, gate, err := g.SplitGatedQ(qg, attentionWidth, req.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	o, r, e := g.FinishRead(q, gate, k, v)
	if e != nil || !r.Committed {
		t.Fatal(e)
	}
	qrow := append([]float32(nil), o[0][:attentionWidth]...)
	gaterow := o[1][:attentionWidth]
	kraw := attentionBatchNormalizeQK(o[2][:kvWidth], req.KNorm, req.NumKVHeads, req.HeadDim, req.QKNormEpsilon, req.Gain1p, req.QKNorm)
	qrow = attentionBatchNormalizeQK(qrow, req.QNorm, req.NumHeads, req.HeadDim, req.QKNormEpsilon, req.Gain1p, req.QKNorm)
	kpost := append([]float32(nil), kraw...)
	attentionBatchApplyRope(qrow, req.NumHeads, req.HeadDim, req.RotaryDim, req.Lanes[0].Position, req.Cos, req.Sin)
	attentionBatchApplyRope(kpost, req.NumKVHeads, req.HeadDim, req.RotaryDim, req.Lanes[0].Position, req.Cos, req.Sin)
	vrow := append([]float32(nil), o[3][:kvWidth]...)
	output := attentionBatchCPUAttention(qrow, gaterow, kpost, vrow, req)
	return Qwen35FullAttentionBatchResult{Output: [][]float32{output}, KRaw: [][]float32{kraw}, KPost: [][]float32{kpost}, V: [][]float32{vrow}}
}

func attentionBatchNormalizeQK(x, weight []float32, heads, hd int, eps float32, gain1p, enabled bool) []float32 {
	out := append([]float32(nil), x...)
	if !enabled {
		return out
	}
	if len(weight) == hd {
		for h := 0; h < heads; h++ {
			row := out[h*hd : (h+1)*hd]
			var sum float64
			for _, v := range row {
				sum += float64(v * v)
			}
			inv := float32(1 / math.Sqrt(sum/float64(hd)+float64(eps)))
			for d := range row {
				w := weight[d]
				if gain1p {
					w++
				}
				row[d] *= inv * w
			}
		}
		return out
	}
	var sum float64
	for _, v := range out {
		sum += float64(v * v)
	}
	inv := float32(1 / math.Sqrt(sum/float64(len(out))+float64(eps)))
	for i := range out {
		w := weight[i]
		if gain1p {
			w++
		}
		out[i] *= inv * w
	}
	return out
}

func attentionBatchApplyRope(x []float32, heads, hd, rotary, position int, cosv, sinv []float32) {
	half := rotary / 2
	cosrow := cosv[position*half : (position+1)*half]
	sinrow := sinv[position*half : (position+1)*half]
	for h := 0; h < heads; h++ {
		base := h * hd
		for j := 0; j < half; j++ {
			a, b := x[base+j], x[base+half+j]
			x[base+j] = a*cosrow[j] - b*sinrow[j]
			x[base+half+j] = a*sinrow[j] + b*cosrow[j]
		}
	}
}

func attentionBatchCPUAttention(q, gate, kcurrent, vcurrent []float32, req Qwen35FullAttentionBatchRequest) []float32 {
	lane := req.Lanes[0]
	nH, nKV, hd := req.NumHeads, req.NumKVHeads, req.HeadDim
	kvWidth := nKV * hd
	keys := append(append([]float32(nil), lane.PrefixK...), kcurrent...)
	values := append(append([]float32(nil), lane.PrefixV...), vcurrent...)
	tokens := lane.Position + 1
	out := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kh := h / (nH / nKV)
		scores := make([]float64, tokens)
		maxScore := math.Inf(-1)
		for token := 0; token < tokens; token++ {
			var dot float64
			for d := 0; d < hd; d++ {
				dot += float64(q[h*hd+d] * keys[token*kvWidth+kh*hd+d])
			}
			scores[token] = dot * float64(req.Scale)
			if scores[token] > maxScore {
				maxScore = scores[token]
			}
		}
		var denom float64
		for token := range scores {
			scores[token] = math.Exp(scores[token] - maxScore)
			denom += scores[token]
		}
		for d := 0; d < hd; d++ {
			var sum float64
			for token, score := range scores {
				sum += score * float64(values[token*kvWidth+kh*hd+d])
			}
			out[h*hd+d] = float32(sum/denom) / (1 + float32(math.Exp(-float64(gate[h*hd+d]))))
		}
	}
	return out
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
		if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
			t.Fatalf("%s non-finite at %d: got=%v want=%v", n, i, g[i], w[i])
		}
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
