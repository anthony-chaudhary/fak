//go:build cuda

package compute

import (
	"errors"
	"math"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

func qwen35SequenceGeometryFixture(layers int) Qwen35SequencePrefillRequest {
	req := Qwen35SequencePrefillRequest{
		Path: Qwen35SequencePrefillPath, TokenIDs: []int{1, 2}, StartPos: qwen38CausalAttentionPanelPrefix,
		Hidden: Qwen35DenseHidden, Intermediate: Qwen35DenseIntermediate,
		NumHeads: Qwen35DenseQueryHeads, NumKVHeads: Qwen35DenseKVHeads,
		HeadDim: Qwen35DenseHeadDim, RotaryDim: Qwen35DenseHeadDim / 4,
		NumKeyHeads: Qwen35DenseGDNGroups, NumValueHeads: Qwen35DenseGDNRank,
		KeyHeadDim: Qwen35DenseGDNState, ValueHeadDim: Qwen35DenseGDNState,
		ConvKernel: Qwen35DenseGDNConv, RMSNormEpsilon: 1e-6,
		Layers: make([]Qwen35SequenceLayer, layers), States: make([]Qwen35SequenceState, layers),
		RoPEThetaForLayer: make([]float64, layers),
	}
	for layer := range req.Layers {
		req.Layers[layer].Linear = (layer+1)%4 != 0
		req.RoPEThetaForLayer[layer] = 1e7
	}
	return req
}

func TestQwen35SequenceProductionGeometryContract(t *testing.T) {
	req := qwen35SequenceGeometryFixture(Qwen35DenseMainLayers)
	if err := validateQwen35SequenceGeometry(req); err != nil {
		t.Fatalf("exact dense Qwen3.8 geometry refused: %v", err)
	}
	for name, mutate := range map[string]func(*Qwen35SequencePrefillRequest){
		"metadata layer included": func(r *Qwen35SequencePrefillRequest) {
			r.Layers = append(r.Layers, Qwen35SequenceLayer{})
			r.States = append(r.States, Qwen35SequenceState{})
			r.RoPEThetaForLayer = append(r.RoPEThetaForLayer, 1e7)
		},
		"wrong full-attention cadence": func(r *Qwen35SequencePrefillRequest) { r.Layers[2].Linear = false },
		"wrong hidden size":            func(r *Qwen35SequencePrefillRequest) { r.Hidden-- },
		"wrong GDN rank":               func(r *Qwen35SequencePrefillRequest) { r.NumValueHeads-- },
	} {
		t.Run(name, func(t *testing.T) {
			bad := qwen35SequenceGeometryFixture(Qwen35DenseMainLayers)
			mutate(&bad)
			var contractErr *Qwen35SequenceError
			if err := validateQwen35SequenceGeometry(bad); err == nil || !errors.As(err, &contractErr) {
				t.Fatalf("malformed production geometry error = %v, want typed refusal", err)
			}
		})
	}
}

func TestQwen35SequenceBoundedFixtureGeometry(t *testing.T) {
	req := qwen35SequenceGeometryFixture(4)
	req.Hidden, req.Intermediate = 32, 64
	req.NumHeads, req.NumKVHeads, req.HeadDim, req.RotaryDim = 4, 2, 8, 8
	req.NumKeyHeads, req.NumValueHeads = 2, 4
	req.KeyHeadDim, req.ValueHeadDim, req.ConvKernel = 8, 8, 3
	var contractErr *Qwen35SequenceError
	if err := validateQwen35SequenceGeometry(req); err == nil || !errors.As(err, &contractErr) {
		t.Fatalf("non-Qwen3.8 prompt geometry error = %v, want typed source-spine refusal", err)
	}
}

func TestQwen35CausalAttentionPanelGeometryContract(t *testing.T) {
	if err := validateQwen35CausalAttentionPanelGeometry(
		qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix,
		qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads,
		qwen38CausalAttentionPanelHeadDim,
	); err != nil {
		t.Fatalf("exact Qwen3.8 source-spine geometry refused without a CUDA device: %v", err)
	}

	for _, geometry := range []struct {
		name                                    string
		tokens, prefix, heads, kvHeads, headDim int
	}{
		{"tokens-minus-one", qwen38CausalAttentionPanelTokens - 1, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim},
		{"tokens-plus-one", qwen38CausalAttentionPanelTokens + 1, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim},
		{"prefix-minus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix - 1, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim},
		{"prefix-plus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix + 1, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim},
		{"query-heads-minus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads - 1, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim},
		{"query-heads-plus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads + 1, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim},
		{"KV-heads-minus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads - 1, qwen38CausalAttentionPanelHeadDim},
		{"KV-heads-plus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads + 1, qwen38CausalAttentionPanelHeadDim},
		{"head-dimension-minus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim - 1},
		{"head-dimension-plus-one", qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix, qwen38CausalAttentionPanelHeads, qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim + 1},
	} {
		t.Run("refuse/"+geometry.name, func(t *testing.T) {
			var contractErr *Qwen35SequenceError
			if err := validateQwen35CausalAttentionPanelGeometry(geometry.tokens, geometry.prefix, geometry.heads, geometry.kvHeads, geometry.headDim); err == nil || !errors.As(err, &contractErr) {
				t.Fatalf("unsupported geometry error = %v, want typed refusal before launch", err)
			}
		})
	}
	for _, scale := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		var contractErr *Qwen35SequenceError
		if err := validateQwen35CausalAttentionPanelScale(scale); err == nil || !errors.As(err, &contractErr) {
			t.Fatalf("unsupported scale %v error = %v, want typed refusal before launch", scale, err)
		}
	}
	if err := validateQwen35CausalAttentionPanelScale(1 / float32(math.Sqrt(256))); err != nil {
		t.Fatalf("production attention scale refused: %v", err)
	}
}

func TestQwen35CausalAttentionPanelRefusesBeforeKVReservation(t *testing.T) {
	kv := &cudaKV{
		cfg:  KVConfig{NumKVHeads: 4, HeadDim: qwen35CausalAttentionPanelMaxHeadDim + 1},
		Kraw: []dslice{{len: 11, cap: 17}},
		K:    []dslice{{len: 13, cap: 19}},
		V:    []dslice{{len: 15, cap: 23}},
		pos:  []int{0, 1},
	}
	beforeKraw, beforeK, beforeV := kv.Kraw[0], kv.K[0], kv.V[0]
	beforePos := append([]int(nil), kv.pos...)

	var contractErr *Qwen35SequenceError
	if err := (&cudaBackend{}).qwen35SequenceReserveKVLocked(kv, 1, 29); err == nil || !errors.As(err, &contractErr) {
		t.Fatalf("unsupported reservation error = %v, want typed refusal", err)
	}
	if kv.Kraw[0] != beforeKraw || kv.K[0] != beforeK || kv.V[0] != beforeV {
		t.Fatalf("unsupported geometry mutated KV reservation: Kraw=%+v K=%+v V=%+v", kv.Kraw[0], kv.K[0], kv.V[0])
	}
	if len(kv.pos) != len(beforePos) {
		t.Fatalf("unsupported geometry mutated KV positions: got %v want %v", kv.pos, beforePos)
	}
	for i := range beforePos {
		if kv.pos[i] != beforePos[i] {
			t.Fatalf("unsupported geometry mutated KV position %d: got %d want %d", i, kv.pos[i], beforePos[i])
		}
	}
	if strconv.IntSize == 64 {
		oversized := &cudaKV{
			cfg:  KVConfig{NumKVHeads: 1, HeadDim: 1},
			Kraw: []dslice{{len: 7, cap: 11}},
			K:    []dslice{{len: 7, cap: 11}},
			V:    []dslice{{len: 7, cap: 11}},
			pos:  []int{0},
		}
		before := []dslice{oversized.Kraw[0], oversized.K[0], oversized.V[0]}
		if err := (&cudaBackend{}).qwen35SequenceReserveKVLocked(oversized, 1, int(qwen35GDNMaxCInt)+1); err == nil || !errors.As(err, &contractErr) {
			t.Fatalf("oversized valid-head reservation error = %v, want typed CUDA-int refusal", err)
		}
		if oversized.Kraw[0] != before[0] || oversized.K[0] != before[1] || oversized.V[0] != before[2] || len(oversized.pos) != 1 || oversized.pos[0] != 0 {
			t.Fatalf("oversized reservation mutated KV metadata: Kraw=%+v K=%+v V=%+v pos=%v", oversized.Kraw[0], oversized.K[0], oversized.V[0], oversized.pos)
		}
	}

	req := qwen35SequenceGeometryFixture(4)
	req.Hidden, req.Intermediate = 32, 64
	req.NumHeads, req.NumKVHeads = 4, 2
	req.HeadDim, req.RotaryDim = qwen35CausalAttentionPanelMaxHeadDim+1, 8
	req.NumKeyHeads, req.NumValueHeads = 2, 4
	req.KeyHeadDim, req.ValueHeadDim, req.ConvKernel = 8, 8, 3
	req.KV = kv
	conv := NewF32(Default(), []int{2}, []float32{3.25, -4.5})
	recurrent := NewF32(Default(), []int{2}, []float32{-6.75, 8})
	req.States[0] = Qwen35SequenceState{Conv: conv, Recurrent: recurrent}
	be := &cudaBackend{faultLatch: NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget)}
	if _, err := be.Qwen35SequencePrefill(req); err == nil || !errors.As(err, &contractErr) {
		t.Fatalf("unsupported full-prefill error = %v, want typed preflight refusal", err)
	}
	if kv.Kraw[0] != beforeKraw || kv.K[0] != beforeK || kv.V[0] != beforeV {
		t.Fatalf("unsupported full prefill mutated KV reservation: Kraw=%+v K=%+v V=%+v", kv.Kraw[0], kv.K[0], kv.V[0])
	}
	if got := Default().Read(conv); got[0] != 3.25 || got[1] != -4.5 {
		t.Fatalf("unsupported full prefill mutated convolution state: %v", got)
	}
	if got := Default().Read(recurrent); got[0] != -6.75 || got[1] != 8 {
		t.Fatalf("unsupported full prefill mutated recurrent state: %v", got)
	}
}

func TestQwen35CausalAttentionPanelRefusesBeforeOutputAllocation(t *testing.T) {
	sentinel := &cudaBuf{n: 17}
	be := &cudaBackend{transient: []*cudaBuf{sentinel}}
	kv := &cudaKV{cfg: KVConfig{
		NumKVHeads: Qwen35DenseKVHeads,
		HeadDim:    qwen35CausalAttentionPanelMaxHeadDim + 1,
	}}
	var contractErr *Qwen35SequenceError
	_, _, err := be.qwen35SequenceAttentionLocked(
		Tensor{}, kv, 0, 2, 1, Qwen35DenseQueryHeads,
		1/float32(math.Sqrt(qwen38CausalAttentionPanelHeadDim)),
	)
	if err == nil || !errors.As(err, &contractErr) {
		t.Fatalf("unsupported launcher geometry error = %v, want typed refusal", err)
	}
	if len(be.transient) != 1 || be.transient[0] != sentinel {
		t.Fatalf("unsupported launcher geometry allocated or mutated transient inventory: %#v", be.transient)
	}
}

func TestQwen35CausalAttentionPanelAppendPreflightsEveryKVSlice(t *testing.T) {
	const (
		tokens  = qwen38CausalAttentionPanelTokens
		prefix  = qwen38CausalAttentionPanelPrefix
		kvHeads = qwen38CausalAttentionPanelKVHeads
		headDim = qwen38CausalAttentionPanelHeadDim
		width   = kvHeads * headDim
	)
	be := &cudaBackend{}
	resident := func() Tensor {
		storage := make([]byte, tokens*width*F32.Bytes())
		return Tensor{
			Dtype: F32, Layout: RowMajor, Shape: []int{tokens, width},
			buf: &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: len(storage)},
			be:  be,
		}
	}
	kRawStorage := make([]byte, (prefix+tokens)*width*F32.Bytes())
	kStorage := make([]byte, (prefix+tokens)*width*F32.Bytes())
	vStorage := make([]byte, (prefix+tokens)*width*F32.Bytes())
	kv := &cudaKV{
		cfg:  KVConfig{NumKVHeads: kvHeads, HeadDim: headDim},
		Kraw: []dslice{{ptr: unsafe.Pointer(&kRawStorage[0]), len: prefix * width, cap: (prefix + tokens) * width}},
		K:    []dslice{{ptr: unsafe.Pointer(&kStorage[0]), len: prefix * width, cap: (prefix+tokens)*width - 1}},
		V:    []dslice{{ptr: unsafe.Pointer(&vStorage[0]), len: prefix * width, cap: (prefix + tokens) * width}},
	}
	beforeKraw, beforeK, beforeV := kv.Kraw[0], kv.K[0], kv.V[0]
	if err := be.qwen35SequenceAppendKVLocked(kv, 0, resident(), resident(), resident(), prefix, tokens); err == nil {
		t.Fatal("undersized middle KV slice accepted")
	}
	if kv.Kraw[0] != beforeKraw || kv.K[0] != beforeK || kv.V[0] != beforeV || len(kv.pos) != 0 {
		t.Fatalf("failed append mutated KV metadata: Kraw=%+v K=%+v V=%+v pos=%v", kv.Kraw[0], kv.K[0], kv.V[0], kv.pos)
	}
}

type qwen35PanelLCG uint64

func (r *qwen35PanelLCG) values(n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		*r = *r*6364136223846793005 + 1442695040888963407
		out[i] = (float32(uint32(*r>>32))/float32(uint64(1)<<32) - 0.5) * scale
	}
	return out
}

func qwen35CausalAttentionPanelCPUReference(q, k, v []float32, tokens, prefix, heads, kvHeads, headDim int, scale float32) []float32 {
	ref := Default()
	kv := ref.NewKV(KVConfig{NumLayers: 1, NumKVHeads: kvHeads, HeadDim: headDim})
	width := kvHeads * headDim
	group := heads / kvHeads
	out := make([]float32, 0, tokens*heads*headDim)
	for position := 0; position < prefix+tokens; position++ {
		key := NewF32(ref, []int{width}, k[position*width:(position+1)*width])
		value := NewF32(ref, []int{width}, v[position*width:(position+1)*width])
		kv.AppendKV(0, key, key, value, position)
		if position < prefix {
			continue
		}
		token := position - prefix
		query := NewF32(ref, []int{heads * headDim}, q[token*heads*headDim:(token+1)*heads*headDim])
		out = append(out, ref.Read(ref.Attention(query, kv, 0, true, group, scale))...)
	}
	return out
}

func TestCUDAQwen35CausalAttentionPanelMatchesCPUReference(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	for _, geometry := range []struct {
		name                                    string
		tokens, prefix, heads, kvHeads, headDim int
	}{
		{"qwen3.8-24x4-hd256", 2, 1, 24, 4, qwen38CausalAttentionPanelHeadDim},
	} {
		t.Run(geometry.name, func(t *testing.T) {
			const sentinel = float32(-12345.75)
			scale := float32(1 / math.Sqrt(float64(geometry.headDim)))
			rng := qwen35PanelLCG(0x9786c0da)
			q := rng.values(geometry.tokens*geometry.heads*geometry.headDim, 0.5)
			k := rng.values((geometry.prefix+geometry.tokens)*geometry.kvHeads*geometry.headDim, 0.5)
			v := rng.values((geometry.prefix+geometry.tokens)*geometry.kvHeads*geometry.headDim, 0.5)
			want := qwen35CausalAttentionPanelCPUReference(q, k, v, geometry.tokens, geometry.prefix, geometry.heads, geometry.kvHeads, geometry.headDim, scale)
			sentinels := make([]float32, len(want))
			for i := range sentinels {
				sentinels[i] = sentinel
			}
			upload := func(shape []int, data []float32, site string) Tensor {
				resident := be.UploadClass(NewF32(Default(), shape, data), F32, MemoryScratchpad, site)
				t.Cleanup(func() { be.Free(resident) })
				return resident
			}
			qDevice := upload([]int{geometry.tokens, geometry.heads * geometry.headDim}, q, "qwen35-panel-witness-query")
			kDevice := upload([]int{geometry.prefix + geometry.tokens, geometry.kvHeads * geometry.headDim}, k, "qwen35-panel-witness-key")
			vDevice := upload([]int{geometry.prefix + geometry.tokens, geometry.kvHeads * geometry.headDim}, v, "qwen35-panel-witness-value")
			outDevice := upload([]int{geometry.tokens, geometry.heads * geometry.headDim}, sentinels, "qwen35-panel-witness-sentinel-output")

			be.ResetHostXfer()
			be.ResetH2DXfer()
			if err := be.qwen35CausalAttentionPanelIntoForTest(qDevice, kDevice, vDevice, outDevice, geometry.tokens, geometry.prefix, geometry.heads, geometry.kvHeads, geometry.headDim, scale); err != nil {
				t.Fatalf("prompt-panel execution: %v", err)
			}
			if d2h, h2d := be.HostXferBytes(), be.H2DXferBytes(); d2h != 0 || h2d != 0 {
				t.Fatalf("resident prompt-panel kernel transferred host bytes before proof Read: D2H=%d H2D=%d", d2h, h2d)
			}
			got := be.Read(outDevice)
			for i, value := range got {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("output[%d] is non-finite: %v", i, value)
				}
				if value == sentinel {
					t.Fatalf("output[%d] retained caller sentinel: kernel returned without writing the panel", i)
				}
			}
			similarity := cosine(want, got)
			if math.IsNaN(similarity) || math.IsInf(similarity, 0) {
				t.Fatalf("panel vs cpuref cosine is non-finite: %v", similarity)
			}
			if similarity < cudaFlashAttnCosineMin {
				t.Fatalf("panel vs cpuref cosine %.8f < %.4f (max abs delta %.3e)", similarity, cudaFlashAttnCosineMin, maxAbsDelta(want, got))
			} else {
				t.Logf("prompt panel wrote every sentinel with zero activation transfer and matched cpuref: cosine=%.8f max_abs_delta=%.3e", similarity, maxAbsDelta(want, got))
			}
		})
	}
}

func TestCUDAPartialRoPEQKMatchesRotateHalfReference(t *testing.T) {
	be := cudaGDNBackend(t)
	const (
		pos = 7
		nQ  = 2
		nK  = 1
		hd  = 8
		rd  = 6
	)
	qHost := []float32{.1, .2, .3, .4, .5, .6, .7, .8, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8}
	kHost := []float32{.9, .8, .7, .6, .5, .4, .3, .2}
	q := uploadCUDAGDN(t, be, []int{nQ * hd}, qHost, MemoryActivation, "partial-rope-q-test")
	k := uploadCUDAGDN(t, be, []int{nK * hd}, kHost, MemoryActivation, "partial-rope-k-test")
	be.ResetHostXfer()
	be.ResetH2DXfer()
	qOut, kOut := be.PartialRoPEQK(q, k, pos, nQ, nK, hd, rd, 10000)
	t.Cleanup(func() { be.Free(qOut); be.Free(kOut) })
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes D2H", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes H2D", got)
	}

	ref := func(in []float32, heads int) []float32 {
		out := append([]float32(nil), in...)
		half := rd / 2
		for h := 0; h < heads; h++ {
			for j := 0; j < half; j++ {
				freq := math.Pow(10000, -2*float64(j)/rd)
				cs, sn := float32(math.Cos(pos*freq)), float32(math.Sin(pos*freq))
				a, b := in[h*hd+j], in[h*hd+j+half]
				out[h*hd+j], out[h*hd+j+half] = a*cs-b*sn, b*cs+a*sn
			}
		}
		return out
	}
	assertNear := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length %d != %d", label, len(got), len(want))
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 2e-5 {
				t.Fatalf("%s[%d]=%g want %g", label, i, got[i], want[i])
			}
		}
	}
	assertNear("q", be.Read(qOut), ref(qHost, nQ))
	assertNear("k", be.Read(kOut), ref(kHost, nK))
	assertNear("q input", be.Read(q), qHost)
	assertNear("k input", be.Read(k), kHost)
}

func TestCUDASigmoidMulInPlace(t *testing.T) {
	be := cudaGDNBackend(t)
	xHost := []float32{2, -3, 4, -5}
	gateHost := []float32{-2, 0, 2, 8}
	x := uploadCUDAGDN(t, be, []int{len(xHost)}, xHost, MemoryActivation, "sigmoid-x-test")
	gate := uploadCUDAGDN(t, be, []int{len(gateHost)}, gateHost, MemoryActivation, "sigmoid-gate-test")
	be.ResetHostXfer()
	be.ResetH2DXfer()
	be.SigmoidMulInPlace(x, gate)
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes D2H", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes H2D", got)
	}
	got := be.Read(x)
	for i := range got {
		want := xHost[i] / (1 + float32(math.Exp(float64(-gateHost[i]))))
		if math.Abs(float64(got[i]-want)) > 2e-6 {
			t.Fatalf("x[%d]=%g want %g", i, got[i], want)
		}
	}

	short := uploadCUDAGDN(t, be, []int{1}, []float32{1}, MemoryActivation, "sigmoid-short-test")
	defer func() {
		r := recover()
		var msg string
		switch v := r.(type) {
		case string:
			msg = v
		case error:
			msg = v.Error()
		}
		if r == nil || !strings.Contains(msg, "shape mismatch") {
			t.Fatalf("shape mismatch panic = %v", r)
		}
	}()
	be.SigmoidMulInPlace(x, short)
}

func TestCUDASplitQwen35QueryGate(t *testing.T) {
	be := cudaGDNBackend(t)
	qgHost := []float32{
		1, 2, 3, 4, 11, 12, 13, 14,
		5, 6, 7, 8, 15, 16, 17, 18,
	}
	qg := uploadCUDAGDN(t, be, []int{len(qgHost)}, qgHost, MemoryActivation, "qg-split-test")
	be.ResetHostXfer()
	be.ResetH2DXfer()
	q, gate := be.SplitQwen35QueryGate(qg, 2, 4)
	t.Cleanup(func() { be.Free(q); be.Free(gate) })
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes D2H", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("device operation copied %d bytes H2D", got)
	}
	assert := func(label string, got, want []float32) {
		t.Helper()
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d]=%g want %g", label, i, got[i], want[i])
			}
		}
	}
	assert("query", be.Read(q), []float32{1, 2, 3, 4, 5, 6, 7, 8})
	assert("gate", be.Read(gate), []float32{11, 12, 13, 14, 15, 16, 17, 18})
}
