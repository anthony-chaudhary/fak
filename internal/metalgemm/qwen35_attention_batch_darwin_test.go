//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"testing"
	"time"
)

func TestQwen35FullAttentionDecodeBatchIndependentKVSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	geometry := Qwen35FullAttentionGeometry{Hidden: 5120, NumHeads: 24, NumKVHeads: 4, HeadDim: 256, RotaryDim: 64}
	weights := qwen35AttentionBatchWeights(t, geometry)
	defer func() {
		weights.Q.Release()
		weights.K.Release()
		weights.V.Release()
	}()
	t.Logf("exact Qwen3.8 full-attention geometry: hidden=%d nH=%d nKV=%d hd=%d rotary=%d gqa=%d",
		geometry.Hidden, geometry.NumHeads, geometry.NumKVHeads, geometry.HeadDim, geometry.RotaryDim, geometry.NumHeads/geometry.NumKVHeads)
	qnorm, knorm := make([]float32, geometry.HeadDim), make([]float32, geometry.HeadDim)
	for i := range qnorm {
		qnorm[i] = float32((i%9)-4) * 0.015
		knorm[i] = float32((i%7)-3) * 0.02
	}
	requestFor := func(batch int) Qwen35FullAttentionDecodeBatchRequest {
		return Qwen35FullAttentionDecodeBatchRequest{
			Input: qwen35AttentionBatchInput(batch, geometry.Hidden), Weights: weights,
			Lanes: qwen35AttentionBatchLanes(batch, geometry), Geometry: geometry,
			QNorm: qnorm, KNorm: knorm, Scale: 1 / float32(math.Sqrt(float64(geometry.HeadDim))),
			QKNormEps: 1e-5, NormGain1p: true, QKNorm: true,
		}
	}

	var b4Rows []Qwen35FullAttentionDecodeRow
	const graphBufferDelta = 11 // xf + quantized q/d + three projections + five attention intermediates.
	serialRequest := requestFor(Qwen35FullAttentionDecodeBatchMax)
	serialRows := make([]Qwen35FullAttentionDecodeRow, Qwen35FullAttentionDecodeBatchMax)
	for row := range serialRows {
		serialRows[row] = qwen35AttentionSerialP32(t, serialRequest, row)
	}
	for _, batch := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("B%d serial parity and accounting", batch), func(t *testing.T) {
			req := requestFor(batch)
			baseOwners, baseBuffers := qwen35FullAttentionDecodeBatchNativeLiveCounts()
			peakOwners, peakBuffers := -1, -1
			req.afterGraphEncodeForTest = func() { peakOwners, peakBuffers = qwen35FullAttentionDecodeBatchNativeLiveCounts() }
			got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
			if err != nil || !accepted {
				t.Fatalf("B=%d accepted=%v err=%v", batch, accepted, err)
			}
			qwen35RequireAttentionBatchReceipt(t, receipt, req, geometry.NumKVHeads*geometry.HeadDim, 1)
			if peakOwners != baseOwners+1 || peakBuffers != baseBuffers+graphBufferDelta {
				t.Fatalf("B=%d native peak owners=%d buffers=%d, want %d/%d", batch, peakOwners, peakBuffers, baseOwners+1, baseBuffers+graphBufferDelta)
			}
			if owners, buffers := qwen35FullAttentionDecodeBatchNativeLiveCounts(); owners != baseOwners || buffers != baseBuffers {
				t.Fatalf("B=%d success retained native owners=%d buffers=%d, baseline=%d/%d", batch, owners, buffers, baseOwners, baseBuffers)
			}
			for row := range got {
				qwen35RequireAttentionBatchParity(t, fmt.Sprintf("B=%d row=%d output", batch, row), serialRows[row].Output, got[row].Output)
				qwen35RequireAttentionBatchParity(t, fmt.Sprintf("B=%d row=%d KRaw", batch, row), serialRows[row].KRaw, got[row].KRaw)
				qwen35RequireAttentionBatchParity(t, fmt.Sprintf("B=%d row=%d KPost", batch, row), serialRows[row].KPost, got[row].KPost)
				qwen35RequireAttentionBatchParity(t, fmt.Sprintf("B=%d row=%d V append", batch, row), serialRows[row].VAppend, got[row].VAppend)
			}
			if batch == 4 {
				b4Rows = qwen35CloneAttentionBatchRows(got)
			}
			if qwen35AttentionBatchMaxAbs(got[0].Output, got[len(got)-1].Output) == 0 {
				t.Fatalf("B=%d independent exact-geometry rows produced identical outputs", batch)
			}
		})
	}

	t.Run("absolute position above scratch limit uses supplied rotary row", func(t *testing.T) {
		req := requestFor(2)
		for row := range req.Lanes {
			req.Lanes[row].Position = 8192 + row*4097
			req.Lanes[row].Cos, req.Lanes[row].Sin = qwen35AttentionRotaryRow(req.Lanes[row].Position, geometry.RotaryDim)
		}
		got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
		if err != nil || !accepted || len(got) != 2 {
			t.Fatalf("large absolute position accepted=%v rows=%d receipt=%+v err=%v", accepted, len(got), receipt, err)
		}
		qwen35RequireAttentionBatchReceipt(t, receipt, req, geometry.NumKVHeads*geometry.HeadDim, 1)
		if receipt.Positions[0] <= qwen35FullAttentionScratchTokens || receipt.Positions[1] <= receipt.Positions[0] {
			t.Fatalf("large absolute positions were not preserved: %v", receipt.Positions)
		}
	})

	t.Run("changing one prefix changes only its lane", func(t *testing.T) {
		req := requestFor(4)
		const changed = 2
		req.Lanes[changed].PrefixK = append([]float32(nil), req.Lanes[changed].PrefixK...)
		req.Lanes[changed].PrefixV = append([]float32(nil), req.Lanes[changed].PrefixV...)
		req.Lanes[changed].PrefixK[3] += 0.75
		req.Lanes[changed].PrefixV[7] -= 0.5
		got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
		if err != nil || !accepted {
			t.Fatalf("lane isolation accepted=%v err=%v", accepted, err)
		}
		qwen35RequireAttentionBatchReceipt(t, receipt, req, geometry.NumKVHeads*geometry.HeadDim, 1)
		for row := range got {
			qwen35RequireAttentionBatchParity(t, fmt.Sprintf("lane isolation row=%d KRaw", row), b4Rows[row].KRaw, got[row].KRaw)
			qwen35RequireAttentionBatchParity(t, fmt.Sprintf("lane isolation row=%d KPost", row), b4Rows[row].KPost, got[row].KPost)
			qwen35RequireAttentionBatchParity(t, fmt.Sprintf("lane isolation row=%d V append", row), b4Rows[row].VAppend, got[row].VAppend)
			maxAbs := qwen35AttentionBatchMaxAbs(b4Rows[row].Output, got[row].Output)
			if row == changed && maxAbs <= 1e-7 {
				t.Fatalf("changed lane %d output maxAbs=%g, want nonzero", row, maxAbs)
			}
			if row != changed && maxAbs != 0 {
				t.Fatalf("changed lane %d mutated row %d output maxAbs=%g", changed, row, maxAbs)
			}
		}
	})

	t.Run("invalid lane declines before graph allocation", func(t *testing.T) {
		req := requestFor(2)
		req.Lanes[1].PrefixV = req.Lanes[1].PrefixV[:len(req.Lanes[1].PrefixV)-1]
		allocated := false
		req.afterGraphAllocationForTest = func() { allocated = true }
		baseOwners, baseBuffers := qwen35FullAttentionDecodeBatchNativeLiveCounts()
		got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
		var declined *Qwen35FullAttentionDecodeBatchDeclinedError
		if got != nil || accepted || !errors.As(err, &declined) || receipt.Committed || allocated {
			t.Fatalf("pre-submit got=%v accepted=%v receipt=%+v allocated=%v err=%T %v", got, accepted, receipt, allocated, err, err)
		}
		if owners, buffers := qwen35FullAttentionDecodeBatchNativeLiveCounts(); owners != baseOwners || buffers != baseBuffers {
			t.Fatalf("decline changed native owners=%d buffers=%d, baseline=%d/%d", owners, buffers, baseOwners, baseBuffers)
		}
	})

	t.Run("committed failure has no replay and exact cleanup", func(t *testing.T) {
		baseOwners, baseBuffers := qwen35FullAttentionDecodeBatchNativeLiveCounts()
		for attempt := 0; attempt < 3; attempt++ {
			req := requestFor(2)
			req.InjectPostSubmitFailureForTest = true
			peakOwners, peakBuffers := -1, -1
			req.afterGraphEncodeForTest = func() { peakOwners, peakBuffers = qwen35FullAttentionDecodeBatchNativeLiveCounts() }
			got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
			var post *GraphPostSubmitError
			if got != nil || !accepted || !errors.As(err, &post) || !receipt.Committed || !receipt.CompletedWait {
				t.Fatalf("attempt=%d got=%v accepted=%v receipt=%+v err=%T %v", attempt, got, accepted, receipt, err, err)
			}
			if receipt.ProjectionDispatches != 3 || receipt.ReplayAttempts != 0 || receipt.FinalReadbacks != 0 ||
				receipt.GraphOwners != 1 || receipt.GraphOwnerReleases != 1 || peakOwners != baseOwners+1 || peakBuffers != baseBuffers+graphBufferDelta {
				t.Fatalf("attempt=%d lifecycle receipt=%+v native peak=%d/%d baseline=%d/%d", attempt, receipt, peakOwners, peakBuffers, baseOwners, baseBuffers)
			}
			if owners, buffers := qwen35FullAttentionDecodeBatchNativeLiveCounts(); owners != baseOwners || buffers != baseBuffers {
				t.Fatalf("attempt=%d retained native owners=%d buffers=%d, baseline=%d/%d", attempt, owners, buffers, baseOwners, baseBuffers)
			}
		}
	})

	t.Run("Q4_K V owner stays live through the synchronous graph", func(t *testing.T) {
		req := requestFor(2)
		entered, resume := make(chan struct{}), make(chan struct{})
		req.afterGraphAllocationForTest = func() {
			close(entered)
			<-resume
		}
		type callResult struct {
			receipt  Qwen35FullAttentionDecodeBatchReceipt
			accepted bool
			err      error
		}
		calls := make(chan callResult, 1)
		go func() {
			_, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(req)
			calls <- callResult{receipt: receipt, accepted: accepted, err: err}
		}()
		<-entered
		released := make(chan struct{})
		go func() {
			weights.V.Release()
			close(released)
		}()
		select {
		case <-released:
			close(resume)
			t.Fatal("Q4_K V release crossed an in-flight graph")
		case <-time.After(50 * time.Millisecond):
		}
		close(resume)
		select {
		case result := <-calls:
			if result.err != nil || !result.accepted || result.receipt.GraphOwnerReleases != 1 {
				t.Fatalf("synchronous owner call accepted=%v receipt=%+v err=%v", result.accepted, result.receipt, result.err)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("synchronous owner call did not finish")
		}
		select {
		case <-released:
		case <-time.After(2 * time.Second):
			t.Fatal("Q4_K V release did not resume after graph completion")
		}
	})
}

func qwen35AttentionBatchWeights(t *testing.T, g Qwen35FullAttentionGeometry) Qwen35FullAttentionDecodeWeights {
	t.Helper()
	uploadQ8 := func(name string, out, phase int) *Q8Weight {
		codes := make([]int8, out*g.Hidden)
		scales := make([]float32, out*(g.Hidden/32))
		for i := range codes {
			codes[i] = int8((i*7+phase)%17 - 8)
		}
		for i := range scales {
			scales[i] = 0.006 + float32((i+phase)%7)*0.00075
		}
		w := UploadQ8(codes, scales, out, g.Hidden)
		if w == nil {
			t.Fatalf("upload %s Q8 [%d,%d]", name, out, g.Hidden)
		}
		return w
	}
	qwidth, kvwidth := g.NumHeads*g.HeadDim, g.NumKVHeads*g.HeadDim
	v := UploadQ4K(q4kTestRaw(kvwidth, g.Hidden, 951603), kvwidth, g.Hidden)
	if v == nil {
		t.Fatalf("upload V Q4_K [%d,%d]", kvwidth, g.Hidden)
	}
	return Qwen35FullAttentionDecodeWeights{Q: uploadQ8("Q", 2*qwidth, 3), K: uploadQ8("K", kvwidth, 5), V: v}
}

func qwen35AttentionBatchInput(batch, hidden int) []float32 {
	x := make([]float32, batch*hidden)
	for row := 0; row < batch; row++ {
		for i := 0; i < hidden; i++ {
			x[row*hidden+i] = float32(((row+2)*11+i*5)%37-18) / 64
		}
	}
	return x
}

func qwen35AttentionBatchLanes(batch int, g Qwen35FullAttentionGeometry) []Qwen35FullAttentionDecodeLane {
	kvwidth := g.NumKVHeads * g.HeadDim
	lanes := make([]Qwen35FullAttentionDecodeLane, batch)
	for row := range lanes {
		prefix := row + 1
		position := 17 + row*3
		cosv, sinv := qwen35AttentionRotaryRow(position, g.RotaryDim)
		lane := Qwen35FullAttentionDecodeLane{Position: position, PrefixK: make([]float32, prefix*kvwidth), PrefixV: make([]float32, prefix*kvwidth), Cos: cosv, Sin: sinv}
		for i := range lane.PrefixK {
			lane.PrefixK[i] = float32(((row+1)*13+i*7)%41-20) / 80
			lane.PrefixV[i] = float32(((row+3)*17+i*3)%43-21) / 72
		}
		lanes[row] = lane
	}
	return lanes
}

func qwen35AttentionRotaryRow(position, rotary int) ([]float32, []float32) {
	cosv, sinv := make([]float32, rotary/2), make([]float32, rotary/2)
	for j := range cosv {
		angle := float64(position) / math.Pow(10000, float64(2*j)/float64(rotary))
		cosv[j], sinv[j] = float32(math.Cos(angle)), float32(math.Sin(angle))
	}
	return cosv, sinv
}

func qwen35AttentionSerialP32(t *testing.T, req Qwen35FullAttentionDecodeBatchRequest, row int) Qwen35FullAttentionDecodeRow {
	t.Helper()
	g := req.Geometry
	qwidth, kvwidth := g.NumHeads*g.HeadDim, g.NumKVHeads*g.HeadDim
	panel := make([]float32, 32*g.Hidden)
	copy(panel, req.Input[row*g.Hidden:(row+1)*g.Hidden])
	for token := 1; token < 32; token++ {
		for i := 0; i < g.Hidden; i++ {
			panel[token*g.Hidden+i] = float32((token*3+i)%23-11) / 96
		}
	}
	weights := []*Q8Weight{req.Weights.Q, req.Weights.K}
	if !lockQ8Group(weights) {
		t.Fatal("serial Q/K handles unavailable")
	}
	defer unlockQ8Group(weights)
	q4kPinMu.Lock()
	defer q4kPinMu.Unlock()
	graph, err := BeginProjectionGraph(panel, nil, nil, 32, g.Hidden)
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Free()
	input, err := graph.Input(g.Hidden)
	if err != nil {
		t.Fatal(err)
	}
	quantized, err := graph.QuantizeQ8(input)
	if err != nil {
		t.Fatal(err)
	}
	qg, err := graph.EncodeQ8From(req.Weights.Q, quantized)
	if err != nil {
		t.Fatal(err)
	}
	kg, err := graph.EncodeQ8From(req.Weights.K, quantized)
	if err != nil {
		t.Fatal(err)
	}
	vg, err := graph.EncodeQ4K(req.Weights.V)
	if err != nil {
		t.Fatal(err)
	}
	q, gate, err := graph.SplitGatedQ(qg, qwidth, g.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	cosv, sinv := make([]float32, 32*(g.RotaryDim/2)), make([]float32, 32*(g.RotaryDim/2))
	copy(cosv, req.Lanes[row].Cos)
	copy(sinv, req.Lanes[row].Sin)
	for token := 1; token < 32; token++ {
		position := req.Lanes[row].Position + token
		for j := 0; j < g.RotaryDim/2; j++ {
			angle := float64(position) / math.Pow(10000, float64(2*j)/float64(g.RotaryDim))
			cosv[token*(g.RotaryDim/2)+j] = float32(math.Cos(angle))
			sinv[token*(g.RotaryDim/2)+j] = float32(math.Sin(angle))
		}
	}
	base := len(req.Lanes[row].PrefixK) / kvwidth
	attention, err := graph.FullAttention(q, kg, vg, gate, req.QNorm, req.KNorm, cosv, sinv,
		req.Lanes[row].PrefixK, req.Lanes[row].PrefixV, base, g.NumHeads, g.NumKVHeads, g.HeadDim, g.RotaryDim,
		req.Scale, req.QKNormEps, req.NormGain1p, req.QKNorm)
	if err != nil {
		t.Fatal(err)
	}
	packed, receipt, err := graph.FinishRead(attention.Output, attention.KRaw, attention.KPost, attention.V)
	if err != nil || !receipt.Committed || !receipt.CompletedWait || receipt.HostReadbacks != 1 {
		t.Fatalf("serial row=%d receipt=%+v err=%v", row, receipt, err)
	}
	return Qwen35FullAttentionDecodeRow{
		Output: append([]float32(nil), packed[0][:qwidth]...), KRaw: append([]float32(nil), packed[1][:kvwidth]...),
		KPost: append([]float32(nil), packed[2][:kvwidth]...), VAppend: append([]float32(nil), packed[3][:kvwidth]...),
	}
}

func qwen35RequireAttentionBatchParity(t *testing.T, label string, want, got []float32) {
	t.Helper()
	if len(want) == 0 || len(want) != len(got) {
		t.Fatalf("%s elements=%d, want %d", label, len(got), len(want))
	}
	var dot, nw, ng, maxAbs float64
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		if math.IsNaN(w) || math.IsInf(w, 0) || math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s non-finite element %d: want=%g got=%g", label, i, w, g)
		}
		dot, nw, ng = dot+w*g, nw+w*w, ng+g*g
		maxAbs = math.Max(maxAbs, math.Abs(w-g))
	}
	if nw <= 0 || ng <= 0 || math.IsNaN(nw) || math.IsInf(nw, 0) || math.IsNaN(ng) || math.IsInf(ng, 0) {
		t.Fatalf("%s vacuous or non-finite norms: want=%g got=%g", label, nw, ng)
	}
	cosine := dot / math.Sqrt(nw*ng)
	if math.IsNaN(cosine) || math.IsInf(cosine, 0) || math.IsNaN(maxAbs) || math.IsInf(maxAbs, 0) ||
		cosine < 0.999999 || maxAbs > 0.0001 {
		t.Fatalf("%s cosine=%.9f maxAbs=%g, want cosine>=0.999999 maxAbs<=0.0001", label, cosine, maxAbs)
	}
}

func qwen35RequireAttentionBatchReceipt(t *testing.T, got Qwen35FullAttentionDecodeBatchReceipt, req Qwen35FullAttentionDecodeBatchRequest, kvwidth, wantReadbacks int) {
	t.Helper()
	batch := len(req.Lanes)
	if got.Rows != batch || got.CommandBuffers != 1 || got.Commits != 1 || got.CompletionWaits != 1 ||
		got.ProjectionDispatches != 3 || got.Quantizers != 1 || got.AttentionDispatches != 2 || got.Encoders != 6 ||
		got.InputUploads != 1 || got.ConstantUploads != 7 || got.FinalReadbacks != wantReadbacks || got.IntermediateReadbacks != 0 || got.ReplayAttempts != 0 ||
		got.GraphOwners != 1 || got.GraphOwnerReleases != 1 || !got.Committed || !got.CompletedWait {
		t.Fatalf("batch receipt=%+v", got)
	}
	for row, lane := range req.Lanes {
		if got.PrefixTokens[row] != len(lane.PrefixK)/kvwidth || got.Positions[row] != lane.Position ||
			got.KAppendElements[row] != kvwidth || got.VAppendElements[row] != kvwidth {
			t.Fatalf("batch receipt row=%d got=%+v", row, got)
		}
	}
}

func qwen35CloneAttentionBatchRows(in []Qwen35FullAttentionDecodeRow) []Qwen35FullAttentionDecodeRow {
	out := make([]Qwen35FullAttentionDecodeRow, len(in))
	for row := range in {
		out[row] = Qwen35FullAttentionDecodeRow{Output: append([]float32(nil), in[row].Output...), KRaw: append([]float32(nil), in[row].KRaw...), KPost: append([]float32(nil), in[row].KPost...), VAppend: append([]float32(nil), in[row].VAppend...)}
	}
	return out
}

func qwen35AttentionBatchMaxAbs(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return math.Inf(1)
	}
	var maxAbs float64
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
			return math.Inf(1)
		}
		maxAbs = math.Max(maxAbs, math.Abs(float64(a[i]-b[i])))
	}
	return maxAbs
}
