//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// Executes the actual four-layer sequence with full Qwen attention geometry.
// Zero branch projections make raw residual bits analytically equal to the input
// rows. This is a synthetic sequence witness, not a checkpoint or MTP round.
// FAK_CUDA_RAW_HIDDEN_REQUIRED=1 turns missing hardware into failure.
// fak-test:runtime integration est=5s lane=optin
func TestCUDARawHiddenSequenceLifetime(t *testing.T) {
	backend, present := Lookup("cuda")
	be, ok := backend.(*cudaBackend)
	if !present || !ok || be == nil {
		if os.Getenv("FAK_CUDA_RAW_HIDDEN_REQUIRED") == "1" {
			t.Fatal("required exact CUDA backend is absent")
		}
		t.Skip("CUDA backend absent")
	}
	owner := be.ForwardExecutionOwner()
	owner.Lock()
	defer owner.Unlock()
	// These dimensions exercise the exact CUDA attention geometry with bounded
	// residual/FFN/state storage, not the 64-layer production checkpoint.
	const hidden, intermediate, vocab = 32, 64, 4
	const heads, kvheads, hd, rotary = 24, 4, 256, 64
	const nk, nv, khd, vhd, kernel = 1, 1, 8, 8, 3
	const convDim = 2*nk*khd + nv*vhd
	var weights []Tensor
	upload := func(shape []int, values []float32, class MemoryClass) Tensor {
		x := be.UploadClass(NewF32(Default(), shape, values), F32, class, "raw-hidden-sequence-witness")
		if class == MemoryWeights {
			weights = append(weights, x)
		}
		return x
	}
	defer func() {
		for _, w := range weights {
			be.Free(w)
		}
		be.Recycle()
	}()
	zeros := func(shape ...int) Tensor {
		n := 1
		for _, v := range shape {
			n *= v
		}
		return upload(shape, make([]float32, n), MemoryWeights)
	}
	norm := func(n int, scale float32) Tensor {
		v := make([]float32, n)
		for i := range v {
			v[i] = scale
		}
		return upload([]int{n}, v, MemoryWeights)
	}
	layers := make([]Qwen35SequenceLayer, 4)
	for i := range layers {
		l := &layers[i]
		l.Linear = i != 3
		l.InputNorm = norm(hidden, 1)
		l.PostNorm = norm(hidden, 1)
		l.Gate = zeros(intermediate, hidden)
		l.Up = zeros(intermediate, hidden)
		l.Down = zeros(hidden, intermediate)
		if l.Linear {
			l.GDNInQKV = zeros(convDim, hidden)
			l.GDNInZ = zeros(nv*vhd, hidden)
			l.GDNInB = zeros(nv, hidden)
			l.GDNInA = zeros(nv, hidden)
			l.GDNConv = zeros(convDim, kernel)
			l.GDNALog = zeros(nv)
			l.GDNDTBias = zeros(nv)
			l.GDNNorm = norm(vhd, 1)
			l.GDNOut = zeros(hidden, nv*vhd)
		} else {
			l.Q = zeros(2*heads*hd, hidden)
			l.K = zeros(kvheads*hd, hidden)
			l.V = zeros(kvheads*hd, hidden)
			l.O = zeros(hidden, heads*hd)
			l.QNorm = norm(hd, 1)
			l.KNorm = norm(hd, 1)
		}
	}
	outputNorm := norm(hidden, 2)
	outputData := make([]float32, vocab*hidden)
	for row := 0; row < vocab; row++ {
		outputData[row*hidden+row] = float32(row + 1)
	}
	output := upload([]int{vocab, hidden}, outputData, MemoryWeights)
	same := func(name string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s lengths %d/%d", name, len(a), len(b))
		}
		for i := range a {
			if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%s[%d]: %g != %g", name, i, a[i], b[i])
			}
		}
	}
	type observation struct{ last, logits, all []float32 }
	for _, tokens := range []int{1, 3, 8} {
		t.Run(fmt.Sprintf("P%d", tokens), func(t *testing.T) {
			values := make([]float32, tokens*hidden)
			for i := range values {
				values[i] = float32(i%23+1) / 16
			}
			embedding := upload([]int{tokens, hidden}, values, MemoryActivation)
			defer be.Free(embedding)
			ids := make([]int, tokens)
			for i := range ids {
				ids[i] = i % vocab
			}
			run := func(capture bool) observation {
				if capture {
					t.Setenv("FAK_CUDA_QWEN35_MTP", "1")
				} else {
					t.Setenv("FAK_CUDA_QWEN35_MTP", "0")
				}
				kv := be.newContiguousKV(KVConfig{NumLayers: 1, NumKVHeads: kvheads, HeadDim: hd, RopeTheta: 10000, Precision: KVPrecisionF32})
				states := make([]Qwen35SequenceState, 4)
				var stateRoots []Tensor
				for i := 0; i < 3; i++ {
					states[i].Conv = upload([]int{kernel - 1, convDim}, make([]float32, (kernel-1)*convDim), MemoryKVCache)
					states[i].Recurrent = upload([]int{nv, khd, vhd}, make([]float32, nv*khd*vhd), MemoryKVCache)
					stateRoots = append(stateRoots, states[i].Conv, states[i].Recurrent)
				}
				defer func() {
					be.Recycle()
					kv.Free()
					for _, s := range stateRoots {
						be.Free(s)
					}
				}()
				req := Qwen35SequencePrefillRequest{
					Path: Qwen35SequencePrefillPath, TokenIDs: ids, TokenEmbedding: embedding, TokenEmbeddingRows: true, TokenEmbeddingVocab: vocab,
					OutputNorm: outputNorm, Output: output, Layers: layers, States: states, KV: kv,
					Hidden: hidden, Intermediate: intermediate, NumHeads: heads, NumKVHeads: kvheads, HeadDim: hd, RotaryDim: rotary,
					NumKeyHeads: nk, NumValueHeads: nv, KeyHeadDim: khd, ValueHeadDim: vhd, ConvKernel: kernel, RMSNormEpsilon: 1e-6,
					RoPEThetaForLayer: []float64{10000, 10000, 10000, 10000}, NeedLogits: true, NeedAllLogits: true, CaptureRawHidden: capture,
				}
				if !capture {
					bad := req
					bad.CaptureRawHidden = true
					beforeTransient, beforeLive := be.cudaAllocationCountsForTest()
					if _, err := be.Qwen35SequencePrefill(bad); err == nil {
						t.Fatal("unselected raw capture admitted")
					}
					afterTransient, afterLive := be.cudaAllocationCountsForTest()
					if beforeTransient != afterTransient || beforeLive != afterLive || kv.Len() != 0 {
						t.Fatal("off refusal changed allocation/KV")
					}
				} else if be.Qwen35SequenceRawHiddenPath() != Qwen35SequenceRawHiddenPath {
					t.Fatal("selected raw marker missing")
				}
				h0, d0 := be.H2DXferBytes(), be.HostXferBytes()
				result, err := be.Qwen35SequencePrefill(req)
				if err != nil {
					t.Fatal(err)
				}
				if be.H2DXferBytes() != h0 || be.HostXferBytes() != d0 || result.Transfers.ActivationH2DBytes != 0 || result.Transfers.ActivationD2HBytes != 0 {
					t.Fatal("sequence activation transfer")
				}
				if result.Tokens != tokens || kv.Len() != tokens || !result.LastHidden.Ready() || !result.Logits.Ready() || !result.LogitsRows.Ready() {
					t.Fatal("incomplete sequence result")
				}
				if capture {
					raw := result.RawHiddenRows
					if !raw.Ready() || raw.Buf() == nil || len(raw.Shape) != 2 || raw.Shape[0] != tokens || raw.Shape[1] != hidden || raw.Buf() == result.LastHidden.Buf() {
						t.Fatal("raw identity/shape/readiness")
					}
					same("raw", be.Read(raw), values)
					// Ordinary additional operations must not retire request-owned raw.
					temporary := be.RMSNorm(result.LastHidden, outputNorm, 1e-6)
					_ = be.Read(temporary)
					same("retained raw", be.Read(raw), values)
					cudaMu.Lock()
					found := false
					for _, b := range be.transient {
						if b == raw.buf.(*cudaBuf) {
							found = true
						}
					}
					cudaMu.Unlock()
					if !found {
						t.Fatal("raw left the ordinary transient owner")
					}
				} else if result.RawHiddenRows.Buf() != nil {
					t.Fatal("unrequested raw result")
				}
				got := observation{be.Read(result.LastHidden), be.Read(result.Logits), be.Read(result.LogitsRows)}
				different := false
				for i, v := range got.last {
					if math.Float32bits(v) != math.Float32bits(values[(tokens-1)*hidden+i]) {
						different = true
					}
				}
				if !different {
					t.Fatal("fixture failed to distinguish raw from normalized hidden")
				}
				same("input unchanged", be.Read(embedding), values)
				if capture {
					rawBuf := result.RawHiddenRows.buf.(*cudaBuf)
					be.Recycle()
					cudaMu.Lock()
					retired := rawBuf.ptr == nil
					cudaMu.Unlock()
					if !retired {
						t.Fatal("raw survived explicit request retirement")
					}
				}
				return got
			}
			baseline := run(false)
			candidate := run(true)
			same("normalized last", candidate.last, baseline.last)
			same("last logits", candidate.logits, baseline.logits)
			same("all logits", candidate.all, baseline.all)
		})
	}
}
