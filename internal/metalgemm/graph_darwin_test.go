//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"testing"
	"time"
)

func TestProjectionGraphMixedQuantizedSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer ResetQ8()
	const P, in, out = 2, 256, 32
	xf := q4kTestVector(P*in, 9267)
	q4 := UploadQ4K(q4kTestRaw(out, in, 9267), out, in)
	if q4 == nil {
		t.Fatal("q4 upload")
	}
	// Q6 remains a valid zero matrix in this mixed packet; Q4/Q8 carry non-zero parity.
	q6 := UploadQ6K(make([]byte, out*(in/256)*210), out, in)
	if q6 == nil {
		t.Fatal("q6 upload")
	}
	q8codes := make([]int8, out*in)
	q8scales := make([]float32, out*(in/32))
	for i := range q8codes {
		q8codes[i] = int8(i%15 - 7)
	}
	for i := range q8scales {
		q8scales[i] = 0.02
	}
	q8 := UploadQ8(q8codes, q8scales, out, in)
	if q8 == nil {
		t.Fatal("q8 upload")
	}
	xq := make([]int8, P*in)
	xd := make([]float32, P*(in/32))
	for row := 0; row < P; row++ {
		for b := 0; b < in/32; b++ {
			xd[row*(in/32)+b] = 0.01
			for j := 0; j < 32; j++ {
				xq[row*in+b*32+j] = int8((row+b+j)%17 - 8)
			}
		}
	}

	want4 := make([]float32, P*out)
	q4.GEMM(xf, P, want4)
	want6 := make([]float32, P*out)
	q6.GEMM(xf, P, want6)
	want8 := make([]float32, P*out)
	q8.GEMM(xq, xd, P, want8)
	g, err := BeginProjectionGraph(xf, xq, xd, P, in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	r4, err := g.EncodeQ4K(q4)
	if err != nil {
		t.Fatal(err)
	}
	r6, err := g.EncodeQ6K(q6)
	if err != nil {
		t.Fatal(err)
	}
	r8, err := g.EncodeQ8(q8)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := g.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 3 || receipt.HostReadbacks != 0 {
		t.Fatalf("receipt=%+v", receipt)
	}
	wantUpload := uint64(len(xf))*4 + uint64(len(xq)) + uint64(len(xd))*4
	if receipt.HostUploadBytes != wantUpload || receipt.HostReadbackBytes != 0 {
		t.Fatalf("transfer bytes = upload %d readback %d, want %d/0", receipt.HostUploadBytes, receipt.HostReadbackBytes, wantUpload)
	}
	for i, pair := range []struct {
		r    *GraphResult
		want []float32
	}{{r4, want4}, {r6, want6}, {r8, want8}} {
		got, err := g.Read(pair.r)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(pair.want) {
			t.Fatalf("result %d len=%d", i, len(got))
		}
		for j := range got {
			d := math.Abs(float64(got[j] - pair.want[j]))
			if d > 1e-5 {
				t.Fatalf("result %d[%d] got=%g want=%g delta=%g", i, j, got[j], pair.want[j], d)
			}
		}
	}
}

func TestProjectionGraphDeviceResultChainingSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer ResetQ8()
	const P, hidden, value, intermediate, output = 32, 256, 512, 768, 256
	x := q4kTestVector(P*hidden, 9456)
	first := UploadQ4K(q4kTestRaw(value, hidden, 9456), value, hidden)
	if first == nil {
		t.Fatal("first Q4_K upload")
	}
	q8Weight := func(out, in, phase int) *Q8Weight {
		codes := make([]int8, out*in)
		scales := make([]float32, out*(in/32))
		for i := range codes {
			codes[i] = int8((i+phase)%15 - 7)
		}
		for i := range scales {
			scales[i] = 0.02
		}
		return UploadQ8(codes, scales, out, in)
	}
	second := q8Weight(intermediate, value, 3)
	third := q8Weight(output, intermediate, 9)
	if second == nil || third == nil {
		t.Fatal("width-changing Q8 uploads")
	}
	quantize := func(values []float32) ([]int8, []float32) {
		q, d := make([]int8, len(values)), make([]float32, len(values)/32)
		for block := range d {
			var amax float32
			for i := 0; i < 32; i++ {
				v := values[block*32+i]
				if v < 0 {
					v = -v
				}
				if v > amax {
					amax = v
				}
			}
			d[block] = amax / 127
			if d[block] == 0 {
				continue
			}
			for i := 0; i < 32; i++ {
				v := values[block*32+i] / d[block]
				if v < 0 {
					v = float32(math.Ceil(float64(v - .5)))
				} else {
					v = float32(math.Floor(float64(v + .5)))
				}
				q[block*32+i] = int8(v)
			}
		}
		return q, d
	}

	g, err := BeginProjectionGraph(x, nil, nil, P, hidden)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	firstResult, err := g.EncodeQ4K(first)
	if err != nil {
		t.Fatal(err)
	}
	quantized, err := g.QuantizeQ8(firstResult)
	if err != nil {
		t.Fatal(err)
	}
	secondResult, err := g.EncodeQ8From(second, quantized)
	if err != nil {
		t.Fatal(err)
	}
	quantizedAgain, err := g.QuantizeQ8(secondResult)
	if err != nil {
		t.Fatal(err)
	}
	thirdResult, err := g.EncodeQ8From(third, quantizedAgain)
	if err != nil {
		t.Fatal(err)
	}
	outputs, receipt, err := g.FinishRead(thirdResult)
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 || len(outputs[0]) != P*output {
		t.Fatalf("outputs shape=%d/%d, want 1/%d", len(outputs), len(outputs[0]), P*output)
	}
	if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 5 || receipt.HostReadbacks != 1 {
		t.Fatalf("receipt=%+v, want one command buffer, five encoders, one terminal wait/readback", receipt)
	}
	if wantUpload, wantReadback := uint64(len(x))*4, uint64(P*output)*4; receipt.HostUploadBytes != wantUpload || receipt.HostReadbackBytes != wantReadback {
		t.Fatalf("transfer bytes = upload %d readback %d, want %d/%d", receipt.HostUploadBytes, receipt.HostReadbackBytes, wantUpload, wantReadback)
	}
	firstHost := make([]float32, P*value)
	first.GEMM(x, P, firstHost)
	q, d := quantize(firstHost)
	secondHost := make([]float32, P*intermediate)
	second.GEMM(q, d, P, secondHost)
	q, d = quantize(secondHost)
	want := make([]float32, P*output)
	third.GEMM(q, d, P, want)
	cosine, maxRel := q4kTestCosineMaxRel(want, outputs[0])
	if cosine < 0.999999 || maxRel > 1e-5 {
		t.Fatalf("width-changing device chain cosine=%g maxRel=%g", cosine, maxRel)
	}
}

func qwenOrderedSplitCPU(src []float32, rows, qwidth, hd int) ([]float32, []float32) {
	q, gate := make([]float32, rows*qwidth), make([]float32, rows*qwidth)
	for row := 0; row < rows; row++ {
		for j := 0; j < qwidth; j++ {
			head, dim := j/hd, j%hd
			base := row*2*qwidth + head*2*hd
			q[row*qwidth+j] = src[base+dim]
			gate[row*qwidth+j] = src[base+hd+dim]
		}
	}
	return q, gate
}

func qwenOrderedNormalizeCPU(src, weight []float32, rows, heads, hd int, eps float32, gain1p bool) []float32 {
	out := make([]float32, len(src))
	for row := 0; row < rows; row++ {
		for head := 0; head < heads; head++ {
			base := (row*heads + head) * hd
			var sum float64
			for dim := 0; dim < hd; dim++ {
				v := float64(src[base+dim])
				sum += v * v
			}
			inv := float32(1 / math.Sqrt(sum/float64(hd)+float64(eps)))
			for dim := 0; dim < hd; dim++ {
				gain := weight[dim]
				if gain1p {
					gain++
				}
				out[base+dim] = src[base+dim] * inv * gain
			}
		}
	}
	return out
}

func qwenOrderedAttentionCPU(q, k, v, gate, qnorm, knorm, cosv, sinv, prefixK, prefixV []float32, rows, base, nH, nKV, hd, rotary int, scale, eps float32, gain1p bool) (out, kraw, kpost []float32) {
	qpost := qwenOrderedNormalizeCPU(q, qnorm, rows, nH, hd, eps, gain1p)
	kraw = qwenOrderedNormalizeCPU(k, knorm, rows, nKV, hd, eps, gain1p)
	kpost = append([]float32(nil), kraw...)
	half := rotary / 2
	for row := 0; row < rows; row++ {
		for head := 0; head < nH; head++ {
			off := (row*nH + head) * hd
			for dim := 0; dim < half; dim++ {
				x, y := qpost[off+dim], qpost[off+half+dim]
				c, s := cosv[row*half+dim], sinv[row*half+dim]
				qpost[off+dim], qpost[off+half+dim] = x*c-y*s, x*s+y*c
			}
		}
		for head := 0; head < nKV; head++ {
			off := (row*nKV + head) * hd
			for dim := 0; dim < half; dim++ {
				x, y := kpost[off+dim], kpost[off+half+dim]
				c, s := cosv[row*half+dim], sinv[row*half+dim]
				kpost[off+dim], kpost[off+half+dim] = x*c-y*s, x*s+y*c
			}
		}
	}
	allK, allV := append(append([]float32(nil), prefixK...), kpost...), append(append([]float32(nil), prefixV...), v...)
	out = make([]float32, rows*nH*hd)
	for row := 0; row < rows; row++ {
		for head := 0; head < nH; head++ {
			kvHead, upto := head/(nH/nKV), base+row+1
			scores := make([]float64, upto)
			maxScore := math.Inf(-1)
			for token := 0; token < upto; token++ {
				var dot float64
				for dim := 0; dim < hd; dim++ {
					dot += float64(qpost[(row*nH+head)*hd+dim] * allK[(token*nKV+kvHead)*hd+dim])
				}
				scores[token] = dot * float64(scale)
				maxScore = math.Max(maxScore, scores[token])
			}
			var denom float64
			for token := range scores {
				scores[token] = math.Exp(scores[token] - maxScore)
				denom += scores[token]
			}
			for dim := 0; dim < hd; dim++ {
				var sum float64
				for token := 0; token < upto; token++ {
					sum += scores[token] * float64(allV[(token*nKV+kvHead)*hd+dim])
				}
				index := (row*nH+head)*hd + dim
				out[index] = float32(sum/denom) / (1 + float32(math.Exp(-float64(gate[index]))))
			}
		}
	}
	return out, kraw, kpost
}

func qwenOrderedRMSNormCPU(input, weight []float32, rows, width int, eps float32) []float32 {
	out := make([]float32, len(input))
	for row := 0; row < rows; row++ {
		var sum float64
		for _, value := range input[row*width : (row+1)*width] {
			sum += float64(value * value)
		}
		inv := float32(1 / math.Sqrt(sum/float64(width)+float64(eps)))
		for dim := 0; dim < width; dim++ {
			out[row*width+dim] = input[row*width+dim] * inv * weight[dim]
		}
	}
	return out
}

func TestProjectionGraphQwenOrderedPanelAttentionAndFinalNorm(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const input, nH, nKV, hd, rotary, base = 256, 2, 1, 32, 16, 2
	const qwidth, kvwidth = nH * hd, nKV * hd
	qgateRaw := q4kTestRaw(2*qwidth, input, 1223811)
	kRaw := q4kTestRaw(kvwidth, input, 1223812)
	vRaw := q4kTestRaw(kvwidth, input, 1223813)
	qgateWeight := UploadQ4K(qgateRaw, 2*qwidth, input)
	kWeight := UploadQ4K(kRaw, kvwidth, input)
	vWeight := UploadQ4K(vRaw, kvwidth, input)
	if qgateWeight == nil || kWeight == nil || vWeight == nil {
		t.Fatal("ordered-panel Q4_K upload")
	}
	panelReference := func(raw []byte, out int, x []float32, rows int) []float32 {
		result := make([]float32, rows*out)
		for row := 0; row < rows; row++ {
			copy(result[row*out:], q4kVectorizedReference(raw, out, input, x[row*input:(row+1)*input]))
		}
		return result
	}
	assertClose := func(t *testing.T, name string, want, got []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s elements=%d, want %d", name, len(got), len(want))
		}
		var maxAbs float64
		for i := range want {
			maxAbs = math.Max(maxAbs, math.Abs(float64(got[i]-want[i])))
		}
		cosine, _ := q4kTestCosineMaxRel(want, got)
		if cosine < 0.99999 || maxAbs > 5e-4 {
			t.Fatalf("%s cosine=%g maxAbs=%g", name, cosine, maxAbs)
		}
	}

	for _, rows := range []int{2, 3, 4, 32} {
		t.Run(fmt.Sprintf("P%d", rows), func(t *testing.T) {
			x := q4kTestVector(rows*input, int64(1223820+rows))
			qgateHost := panelReference(qgateRaw, 2*qwidth, x, rows)
			kHost := panelReference(kRaw, kvwidth, x, rows)
			vHost := panelReference(vRaw, kvwidth, x, rows)
			qWant, gateWant := qwenOrderedSplitCPU(qgateHost, rows, qwidth, hd)
			qnorm, knorm := make([]float32, hd), make([]float32, hd)
			for i := 0; i < hd; i++ {
				qnorm[i], knorm[i] = float32(i%7-3)*0.01, float32(i%5-2)*0.015
			}
			cosv, sinv := make([]float32, rows*(rotary/2)), make([]float32, rows*(rotary/2))
			for row := 0; row < rows; row++ {
				for dim := 0; dim < rotary/2; dim++ {
					angle := float64((row+base+1)*(dim+1)) * 0.003
					cosv[row*(rotary/2)+dim], sinv[row*(rotary/2)+dim] = float32(math.Cos(angle)), float32(math.Sin(angle))
				}
			}
			prefixK, prefixV := make([]float32, base*kvwidth), make([]float32, base*kvwidth)
			for i := range prefixK {
				prefixK[i], prefixV[i] = float32(i%11-5)*0.02, float32(i%13-6)*0.018
			}
			const scale, qkEps, normEps = float32(0.1767767), float32(1e-6), float32(1e-5)
			attentionWant, krawWant, kpostWant := qwenOrderedAttentionCPU(qWant, kHost, vHost, gateWant, qnorm, knorm, cosv, sinv, prefixK, prefixV, rows, base, nH, nKV, hd, rotary, scale, qkEps, true)
			normWeight := make([]float32, qwidth)
			for i := range normWeight {
				normWeight[i] = 0.9 + float32(i%9)*0.025
			}
			normWant := qwenOrderedRMSNormCPU(attentionWant, normWeight, rows, qwidth, normEps)

			g, err := BeginProjectionGraph(x, nil, nil, rows, input)
			if err != nil {
				t.Fatal(err)
			}
			defer g.Free()
			qgate, err := g.EncodeQ4K(qgateWeight)
			if err != nil {
				t.Fatal(err)
			}
			k, err := g.EncodeQ4K(kWeight)
			if err != nil {
				t.Fatal(err)
			}
			v, err := g.EncodeQ4K(vWeight)
			if err != nil {
				t.Fatal(err)
			}
			q, gate, err := g.SplitGatedQ(qgate, qwidth, hd)
			if err != nil {
				t.Fatal(err)
			}
			attention, err := g.FullAttention(q, k, v, gate, qnorm, knorm, cosv, sinv, prefixK, prefixV, base, nH, nKV, hd, rotary, scale, qkEps, true, true)
			if err != nil {
				t.Fatal(err)
			}
			norm, err := g.RMSNorm(attention.Output, normWeight, normEps, false)
			if err != nil {
				t.Fatal(err)
			}
			last, err := g.LastRMSNorm(attention.Output, normWeight, normEps, false)
			if err != nil {
				t.Fatal(err)
			}
			for name, result := range map[string]*GraphResult{"q": q, "gate": gate, "attention": attention.Output, "kraw": attention.KRaw, "kpost": attention.KPost, "v": attention.V, "norm": norm} {
				if result.p != rows {
					t.Fatalf("%s result P=%d, want %d", name, result.p, rows)
				}
			}
			if last.p != 1 || last.out != qwidth {
				t.Fatalf("last norm shape P/out=%d/%d, want 1/%d", last.p, last.out, qwidth)
			}
			outputs, receipt, err := g.FinishRead(q, gate, attention.Output, attention.KRaw, attention.KPost, attention.V, norm, last)
			if err != nil {
				t.Fatal(err)
			}
			if !receipt.Committed || !receipt.CompletedWait || receipt.IntermediateWaits != 0 || receipt.IntermediateReadbacks != 0 || receipt.HostReadbacks != 1 {
				t.Fatalf("ordered-panel terminal receipt=%+v", receipt)
			}
			assertClose(t, "split q", qWant, outputs[0])
			assertClose(t, "split gate", gateWant, outputs[1])
			assertClose(t, "attention", attentionWant, outputs[2])
			assertClose(t, "K raw", krawWant, outputs[3])
			assertClose(t, "K post", kpostWant, outputs[4])
			assertClose(t, "V", vHost, outputs[5])
			assertClose(t, "all-row norm", normWant, outputs[6])
			assertClose(t, "last-row norm", normWant[(rows-1)*qwidth:], outputs[7])
		})
	}
}

func graphGDNPanel(g GDNGeometry, tokens int) GDNPanel {
	panel := GDNPanel{
		Tokens: tokens, Mixed: make([]float32, tokens*g.convDim()),
		Z: make([]float32, tokens*g.valueDim()), B: make([]float32, tokens*g.NumValueHeads),
		A: make([]float32, tokens*g.NumValueHeads), Conv1D: make([]float32, g.convDim()*g.ConvKernel),
		ALog: make([]float32, g.NumValueHeads), DTBias: make([]float32, g.NumValueHeads),
		Norm: make([]float32, g.ValueHeadDim), RMSNormEpsilon: 1e-5,
	}
	for i := range panel.Norm {
		panel.Norm[i] = 1
	}
	for i := range panel.ALog {
		panel.ALog[i] = -2
		panel.DTBias[i] = .1
	}
	return panel
}

func graphGDNPanelUploadBytes(panel GDNPanel) uint64 {
	return uint64(len(panel.Conv1D)+len(panel.ALog)+len(panel.DTBias)+len(panel.Norm)) * 4
}

func newProjectionGraphGDNLeaseFixture(t *testing.T) (*ProjectionGraph, *GDNState, *GraphResult, GDNGeometry) {
	t.Helper()
	const P, input = 32, 256
	geometry := GDNGeometry{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 32, ValueHeadDim: 32, ConvKernel: 2}
	g, err := BeginProjectionGraph(q4kTestVector(P*input, 945603), nil, nil, P, input)
	if err != nil {
		t.Fatal(err)
	}
	encode := func(width, seed int) *GraphResult {
		t.Helper()
		weight := UploadQ4K(q4kTestRaw(width, input, uint64(seed)), width, input)
		if weight == nil {
			t.Fatalf("GDN fixture Q4_K upload width=%d", width)
		}
		result, encodeErr := g.EncodeQ4K(weight)
		if encodeErr != nil {
			t.Fatalf("GDN fixture projection width=%d: %v", width, encodeErr)
		}
		return result
	}
	mixed := encode(geometry.convDim(), 945604)
	z := encode(geometry.valueDim(), 945605)
	b := encode(geometry.NumValueHeads, 945606)
	a := encode(geometry.NumValueHeads, 945607)
	state, err := NewGDNState(geometry)
	if err != nil {
		g.Free()
		t.Fatal(err)
	}
	panel := graphGDNPanel(geometry, P)
	core, err := g.GDN(state, mixed, z, b, a, panel)
	if err != nil {
		state.Close()
		g.Free()
		t.Fatal(err)
	}
	return g, state, core, geometry
}

func TestProjectionGraphGDNCheckpointRestoreSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const P, input = 4, 256
	geometry := GDNGeometry{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 32, ValueHeadDim: 32, ConvKernel: 2}
	baseline := GDNLiveBufferCount()
	live, err := NewGDNState(geometry)
	if err != nil {
		t.Fatal(err)
	}
	backup, err := NewGDNState(geometry)
	if err != nil {
		live.Close()
		t.Fatal(err)
	}
	defer live.Close()
	defer backup.Close()

	seed := func(phase float32) ([]float32, []float32) {
		conv := make([]float32, (geometry.ConvKernel-1)*geometry.convDim())
		recurrent := make([]float32, geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)
		for i := range conv {
			conv[i] = phase + float32(i%17-8)*0.003
		}
		for i := range recurrent {
			recurrent[i] = phase*0.5 + float32(i%23-11)*0.002
		}
		return conv, recurrent
	}
	liveConv, liveRecurrent := seed(0.25)
	backupConv, backupRecurrent := seed(-0.5)
	if err := live.Seed(liveConv, liveRecurrent); err != nil {
		t.Fatal(err)
	}
	if err := backup.Seed(backupConv, backupRecurrent); err != nil {
		t.Fatal(err)
	}
	liveHandles := [2]GDNStateHandle{}
	liveHandles[0], liveHandles[1] = live.Handles()
	backupHandles := [2]GDNStateHandle{}
	backupHandles[0], backupHandles[1] = backup.Handles()
	originalConv, originalRecurrent, err := live.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got := GDNLiveBufferCount(); got != baseline+4 {
		t.Fatalf("two checkpoint owners have %d live buffers, want %d", got, baseline+4)
	}

	x := q4kTestVector(P*input, 1223801)
	g, err := BeginProjectionGraph(x, nil, nil, P, input)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	if _, err := g.CheckpointGDN(live, nil); err == nil {
		t.Fatal("nil backup checkpoint accepted")
	}
	if _, err := g.CheckpointGDN(live, live); err == nil {
		t.Fatal("aliased checkpoint accepted")
	}
	checkpoint, err := g.CheckpointGDN(live, backup)
	if err != nil {
		t.Fatal(err)
	}
	if got := checkpoint.Receipt(); got != (GDNCheckpointReceipt{EncodedDeviceCopies: 2}) {
		t.Fatalf("encoded checkpoint receipt=%+v", got)
	}
	if err := checkpoint.Restore(); err == nil {
		t.Fatal("checkpoint restored before terminal completion")
	}
	encode := func(width, seed int) *GraphResult {
		t.Helper()
		weight := UploadQ4K(q4kTestRaw(width, input, uint64(seed)), width, input)
		if weight == nil {
			t.Fatalf("checkpoint Q4_K upload width=%d", width)
		}
		result, encodeErr := g.EncodeQ4K(weight)
		if encodeErr != nil {
			t.Fatalf("checkpoint projection width=%d: %v", width, encodeErr)
		}
		return result
	}
	mixed := encode(geometry.convDim(), 1223802)
	z := encode(geometry.valueDim(), 1223803)
	b := encode(geometry.NumValueHeads, 1223804)
	a := encode(geometry.NumValueHeads, 1223805)
	panel := graphGDNPanel(geometry, P)
	for i := range panel.Conv1D {
		panel.Conv1D[i] = 0.015 + float32(i%13)*0.001
	}
	core, err := g.GDN(live, mixed, z, b, a, panel)
	if err != nil {
		t.Fatal(err)
	}
	outputs, receipt, err := g.FinishRead(core)
	if err != nil {
		t.Fatal(err)
	}
	wantUpload := uint64(len(x))*4 + graphGDNPanelUploadBytes(panel)
	if len(outputs) != 1 || len(outputs[0]) != P*geometry.valueDim() || !receipt.Committed || !receipt.CompletedWait ||
		receipt.Encoders != 6 || receipt.IntermediateWaits != 0 || receipt.IntermediateReadbacks != 0 || receipt.HostReadbacks != 1 ||
		receipt.HostUploadBytes != wantUpload || receipt.HostReadbackBytes != uint64(P*geometry.valueDim())*4 {
		t.Fatalf("checkpoint graph output=%d receipt=%+v want upload=%d", len(outputs), receipt, wantUpload)
	}
	nonzero := false
	for _, value := range outputs[0] {
		nonzero = nonzero || value != 0
	}
	if !nonzero {
		t.Fatal("checkpoint graph produced only zero output")
	}

	equal := func(a, b []float32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	if got := checkpoint.Receipt(); got != (GDNCheckpointReceipt{EncodedDeviceCopies: 2, DeviceCopies: 2}) {
		t.Fatalf("pre-restore checkpoint receipt=%+v", got)
	}
	if err := checkpoint.Restore(); err != nil {
		t.Fatal(err)
	}
	restoredConv, restoredRecurrent, err := live.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	rollbackConv, rollbackRecurrent, err := backup.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if !equal(restoredConv, originalConv) || !equal(restoredRecurrent, originalRecurrent) {
		t.Fatal("restore did not recover exact original live state")
	}
	if equal(rollbackConv, originalConv) && equal(rollbackRecurrent, originalRecurrent) {
		t.Fatal("restore did not swap advanced state into backup")
	}
	if got := checkpoint.Receipt(); got != (GDNCheckpointReceipt{EncodedDeviceCopies: 2, DeviceCopies: 2, BufferSwaps: 2}) {
		t.Fatalf("restored checkpoint receipt=%+v", got)
	}
	if err := checkpoint.Restore(); err == nil {
		t.Fatal("checkpoint restored twice")
	}
	checkpoint.Close()
	checkpoint.Close()
	if c, r := live.Handles(); [2]GDNStateHandle{c, r} != liveHandles {
		t.Fatalf("live handles changed: got %v want %v", [2]GDNStateHandle{c, r}, liveHandles)
	}
	if c, r := backup.Handles(); [2]GDNStateHandle{c, r} != backupHandles {
		t.Fatalf("backup handles changed: got %v want %v", [2]GDNStateHandle{c, r}, backupHandles)
	}
	if got := GDNLiveBufferCount(); got != baseline+4 {
		t.Fatalf("checkpoint restore changed liveness: got %d want %d", got, baseline+4)
	}
	live.Close()
	backup.Close()
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("checkpoint owners leaked buffers=%d, baseline=%d", got, baseline)
	}
}

func newProjectionGraphGDNCheckpointOnlyFixture(t *testing.T) (*ProjectionGraph, *GDNState, *GDNState, *GDNGraphCheckpoint) {
	t.Helper()
	geometry := GDNGeometry{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 32, ValueHeadDim: 32, ConvKernel: 2}
	live, err := NewGDNState(geometry)
	if err != nil {
		t.Fatal(err)
	}
	backup, err := NewGDNState(geometry)
	if err != nil {
		live.Close()
		t.Fatal(err)
	}
	g, err := BeginProjectionGraph(make([]float32, 32), nil, nil, 1, 32)
	if err != nil {
		live.Close()
		backup.Close()
		t.Fatal(err)
	}
	checkpoint, err := g.CheckpointGDN(live, backup)
	if err != nil {
		g.Free()
		live.Close()
		backup.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		checkpoint.Close()
		g.Free()
		live.Close()
		backup.Close()
	})
	return g, live, backup, checkpoint
}

func TestProjectionGraphGDNCheckpointUnsubmittedFreeReceiptAndCloseLifecycle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	baseline := GDNLiveBufferCount()
	g, live, backup, checkpoint := newProjectionGraphGDNCheckpointOnlyFixture(t)
	checkpoint.Close()
	checkpoint.Close()
	if got := checkpoint.Receipt(); got != (GDNCheckpointReceipt{EncodedDeviceCopies: 2}) {
		t.Fatalf("closed unsubmitted checkpoint receipt=%+v", got)
	}

	resetDone := make(chan error, 1)
	go func() { resetDone <- backup.Reset() }()
	waitForGDNGraphWaiters(t, backup, 1)
	g.Free()
	select {
	case err := <-resetDone:
		if err != nil {
			t.Fatalf("backup reset after unsubmitted Free: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("pre-terminal Close released or stranded checkpoint lease")
	}
	if got := checkpoint.Receipt(); got != (GDNCheckpointReceipt{EncodedDeviceCopies: 2}) {
		t.Fatalf("freed unsubmitted checkpoint receipt=%+v", got)
	}
	if err := checkpoint.Restore(); err == nil {
		t.Fatal("unsubmitted freed checkpoint restored")
	}
	if err := live.Reset(); err != nil {
		t.Fatalf("live owner remained leased after unsubmitted Free: %v", err)
	}
	checkpoint.Close()
	live.Close()
	backup.Close()
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("unsubmitted checkpoint leaked buffers=%d, baseline=%d", got, baseline)
	}
}

func TestProjectionGraphGDNCheckpointRejectsMutatedBackupAfterClose(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	baseline := GDNLiveBufferCount()
	g, live, backup, checkpoint := newProjectionGraphGDNCheckpointOnlyFixture(t)
	receipt, err := g.Finish()
	if err != nil || !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 1 {
		t.Fatalf("checkpoint-only terminal receipt=%+v err=%v", receipt, err)
	}
	if got := checkpoint.Receipt(); got != (GDNCheckpointReceipt{EncodedDeviceCopies: 2, DeviceCopies: 2}) {
		t.Fatalf("completed checkpoint receipt=%+v", got)
	}

	resetDone := make(chan error, 1)
	go func() { resetDone <- backup.Reset() }()
	waitForGDNGraphWaiters(t, backup, 1)
	checkpoint.Close()
	select {
	case err := <-resetDone:
		if err != nil {
			t.Fatalf("backup reset after checkpoint close: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("checkpoint Close did not release completed owner pair")
	}
	if err := checkpoint.Restore(); err == nil {
		t.Fatal("closed checkpoint restored a subsequently mutated backup")
	}
	checkpoint.Close()
	live.Close()
	backup.Close()
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("closed checkpoint leaked buffers=%d, baseline=%d", got, baseline)
	}
}

func TestProjectionGraphGDNCheckpointStateIdentityLifecycleAndMutation(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	g, live, backup, checkpoint := newProjectionGraphGDNCheckpointOnlyFixture(t)
	if identity, err := checkpoint.StateIdentity(); err == nil || identity != "" {
		t.Fatalf("pre-completion checkpoint identity=%q err=%v", identity, err)
	}
	receipt, err := g.Finish()
	if err != nil || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("checkpoint-only terminal receipt=%+v err=%v", receipt, err)
	}
	identity, err := checkpoint.StateIdentity()
	if err != nil || len(identity) != 64 {
		t.Fatalf("completed checkpoint identity=%q err=%v", identity, err)
	}
	identityAgain, err := checkpoint.StateIdentity()
	if err != nil || identityAgain != identity {
		t.Fatalf("untouched checkpoint identity changed: first=%q again=%q err=%v", identity, identityAgain, err)
	}
	checkpoint.Close()
	if stale, err := checkpoint.StateIdentity(); err == nil || stale != "" {
		t.Fatalf("closed checkpoint identity=%q err=%v", stale, err)
	}

	geometry := live.geometry
	conv := make([]float32, (geometry.ConvKernel-1)*geometry.convDim())
	recurrent := make([]float32, geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)
	for i := range conv {
		conv[i] = float32(i+1) / 97
	}
	for i := range recurrent {
		recurrent[i] = float32(i+1) / 1025
	}
	if err := live.Seed(conv, recurrent); err != nil {
		t.Fatalf("seed same-geometry changed state: %v", err)
	}

	g2, err := BeginProjectionGraph(make([]float32, 32), nil, nil, 1, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer g2.Free()
	changedCheckpoint, err := g2.CheckpointGDN(live, backup)
	if err != nil {
		t.Fatal(err)
	}
	defer changedCheckpoint.Close()
	if changedIdentity, err := changedCheckpoint.StateIdentity(); err == nil || changedIdentity != "" {
		t.Fatalf("second pre-completion checkpoint identity=%q err=%v", changedIdentity, err)
	}
	receipt, err = g2.Finish()
	if err != nil || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("changed-state checkpoint receipt=%+v err=%v", receipt, err)
	}
	changedIdentity, err := changedCheckpoint.StateIdentity()
	if err != nil || len(changedIdentity) != 64 {
		t.Fatalf("changed-state checkpoint identity=%q err=%v", changedIdentity, err)
	}
	if changedIdentity == identity {
		t.Fatalf("same-geometry changed state reused checkpoint identity %q", identity)
	}
	if err := changedCheckpoint.Restore(); err != nil {
		t.Fatalf("restore changed-state checkpoint: %v", err)
	}
	if stale, err := changedCheckpoint.StateIdentity(); err == nil || stale != "" {
		t.Fatalf("restored checkpoint identity=%q err=%v", stale, err)
	}
}

func TestGDNStateMutationVersionCoversStandaloneAndGraphPaths(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	geometry := GDNGeometry{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 32, ValueHeadDim: 32, ConvKernel: 2}
	state, err := NewGDNState(geometry)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	version := state.version
	conv := make([]float32, (geometry.ConvKernel-1)*geometry.convDim())
	recurrent := make([]float32, geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)
	if err := state.Seed(conv, recurrent); err != nil {
		t.Fatal(err)
	}
	if state.version != version+1 {
		t.Fatalf("Seed version=%d want %d", state.version, version+1)
	}
	version = state.version
	if _, _, accepted, err := state.Run(graphGDNPanel(geometry, 1)); err != nil || !accepted {
		t.Fatalf("Run accepted=%v err=%v", accepted, err)
	}
	if state.version != version+1 {
		t.Fatalf("Run version=%d want %d", state.version, version+1)
	}
	version = state.version
	if err := state.Reset(); err != nil {
		t.Fatal(err)
	}
	if state.version != version+1 {
		t.Fatalf("Reset version=%d want %d", state.version, version+1)
	}

	g, graphState, core, _ := newProjectionGraphGDNLeaseFixture(t)
	defer g.Free()
	defer graphState.Close()
	version = graphState.version
	if _, receipt, err := g.FinishRead(core); err != nil || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("graph GDN receipt=%+v err=%v", receipt, err)
	}
	if graphState.version != version+1 {
		t.Fatalf("graph GDN version=%d want %d", graphState.version, version+1)
	}
}

func waitForGDNGraphWaiters(t *testing.T, state *GDNState, want int) {
	t.Helper()
	for i := 0; i < 100000; i++ {
		state.mu.Lock()
		got := state.graphWaiters
		state.mu.Unlock()
		if got == want {
			return
		}
		runtime.Gosched()
	}
	state.mu.Lock()
	got := state.graphWaiters
	state.mu.Unlock()
	t.Fatalf("GDN graph waiters=%d, want %d", got, want)
}

func TestProjectionGraphGDNLeaseSerializesOwnerLifecycle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	baseline := GDNLiveBufferCount()
	g, state, core, geometry := newProjectionGraphGDNLeaseFixture(t)

	type operationResult struct {
		name string
		err  error
	}
	results := make(chan operationResult, 4)
	seedConv := make([]float32, (geometry.ConvKernel-1)*geometry.convDim())
	seedRecurrent := make([]float32, geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)
	go func() { results <- operationResult{name: "reset", err: state.Reset()} }()
	go func() { results <- operationResult{name: "seed", err: state.Seed(seedConv, seedRecurrent)} }()
	go func() {
		_, _, _, err := state.Run(graphGDNPanel(geometry, 1))
		results <- operationResult{name: "run", err: err}
	}()
	go func() { state.Close(); results <- operationResult{name: "close"} }()
	waitForGDNGraphWaiters(t, state, 4)
	if got := GDNLiveBufferCount(); got != baseline+2 {
		t.Fatalf("in-flight graph owner buffers=%d, want %d", got, baseline+2)
	}
	outputs, receipt, err := g.FinishRead(core)
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 || len(outputs[0]) != 32*geometry.valueDim() || !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 5 || receipt.HostReadbacks != 1 {
		t.Fatalf("GDN graph terminal result=%d receipt=%+v", len(outputs), receipt)
	}
	if wantUpload, wantReadback := uint64(32*256)*4+graphGDNPanelUploadBytes(graphGDNPanel(geometry, 32)), uint64(32*geometry.valueDim())*4; receipt.HostUploadBytes != wantUpload || receipt.HostReadbackBytes != wantReadback {
		t.Fatalf("GDN transfer bytes = upload %d readback %d, want %d/%d", receipt.HostUploadBytes, receipt.HostReadbackBytes, wantUpload, wantReadback)
	}
	for i := 0; i < 4; i++ {
		select {
		case result := <-results:
			var declined *GDNDeclinedError
			if result.err != nil && !errors.As(result.err, &declined) {
				t.Fatalf("queued %s after graph completion: %v", result.name, result.err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("queued GDN owner operation deadlocked after terminal fence")
		}
	}
	g.Free()
	state.Close()
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("serialized graph owner leaked buffers=%d, baseline=%d", got, baseline)
	}
}

func TestProjectionGraphGDNLeaseReleasesOnFailureAndFree(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	baseline := GDNLiveBufferCount()

	t.Run("post-submit failure", func(t *testing.T) {
		g, state, _, geometry := newProjectionGraphGDNLeaseFixture(t)
		g.InjectPostSubmitFailureForTest()
		receipt, err := g.Finish()
		var post *GraphPostSubmitError
		if !errors.As(err, &post) || !receipt.Committed || !receipt.CompletedWait {
			t.Fatalf("injected graph failure receipt=%+v err=%T %v", receipt, err, err)
		}
		wantUpload := uint64(32*256)*4 + graphGDNPanelUploadBytes(graphGDNPanel(geometry, 32))
		if receipt.HostUploadBytes != wantUpload || receipt.HostReadbackBytes != 0 {
			t.Fatalf("post-submit transfer bytes = upload %d readback %d, want %d/0", receipt.HostUploadBytes, receipt.HostReadbackBytes, wantUpload)
		}
		state.Close()
		g.Free()
	})
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("failed graph leaked buffers=%d, baseline=%d", got, baseline)
	}

	t.Run("unsubmitted free", func(t *testing.T) {
		g, state, _, _ := newProjectionGraphGDNLeaseFixture(t)
		g.Free()
		g.Free()
		state.Close()
		state.Close()
	})
	if got := GDNLiveBufferCount(); got != baseline {
		t.Fatalf("freed graph leaked buffers=%d, baseline=%d", got, baseline)
	}
}
