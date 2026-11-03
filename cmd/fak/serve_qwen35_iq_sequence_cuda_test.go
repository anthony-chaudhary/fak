//go:build cuda && cgo

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

var qwen35IQ2SequenceRequired = flag.Bool("qwen35-iq2-sequence-required", false,
	"require real CUDA, typed IQ2 sequence selection, and existing CUDA timestamp profile; missing evidence fails instead of skipping")

type serveIQSequenceDevice interface {
	compute.Backend
	HostXferBytes() uint64
	ResetHostXfer()
	H2DXferBytes() uint64
	ResetH2DXfer()
	Qwen35GDNOperationCount() uint64
	ResetQwen35GDNOperationCount()
	Recycle()
}

// This is a complete, nonzero 4-layer GGUF fixture, not a loader-only stub or a
// direct matmul benchmark. The ordinary serve loader decides retention and the
// ordinary model Session owns sequence selection, KV/GDN state, and next decode.
// The reduced shape is NOT a full-model Qwen3.8-27B performance qualification.
// fak-test:runtime slow est=30s lane=optin
func TestServeCUDAQwen35RetainedIQSequence(t *testing.T) {
	selected := compute.CUDAQwen35IQ2SequenceQualificationEnabled()
	if *qwen35IQ2SequenceRequired && !selected {
		t.Fatal("required typed --qwen35-iq2-sequence-qualification was not built")
	}
	backend, ok := compute.Lookup("cuda")
	if !ok || backend == nil || backend.Name() != "cuda" {
		if *qwen35IQ2SequenceRequired {
			t.Fatal("required real CUDA backend is absent")
		}
		t.Skip("CUDA unavailable; acceptance must set -qwen35-iq2-sequence-required")
	}
	be, ok := backend.(serveIQSequenceDevice)
	if !ok {
		t.Fatalf("CUDA lacks existing operation/transfer witness surface: %T", backend)
	}
	t.Setenv("FAK_W3_MLP", "0") // existing loader policy: do not replace the fixture's quant format
	for _, kind := range []ggufload.TensorType{ggufload.TensorIQ2_XXS, ggufload.TensorIQ2_XS, ggufload.TensorF32, ggufload.TensorQ8_0, ggufload.TensorIQ4_XS} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			m, raw := serveIQSequenceFixture(t, be, kind)
			for name, want := range raw {
				got, resident := m.KQuantRaw(name)
				if !resident || !bytes.Equal(got, want) || m.HasQ8(name) || m.HasQ4K(name) {
					t.Fatalf("ordinary serve loader widened or changed retained %s bytes (%s)", kind, name)
				}
			}
			isIQ2 := kind == ggufload.TensorIQ2_XXS || kind == ggufload.TensorIQ2_XS
			for _, tokens := range []int{1, 2, 4, 128} {
				t.Run(fmt.Sprint(tokens), func(t *testing.T) {
					prompt := make([]int, tokens)
					for i := range prompt {
						prompt[i] = (3 + i*7) % 32
					}
					cpu := m.NewSession()
					defer cpu.Close()
					want := append([]float32(nil), cpu.Prefill(prompt)...)
					wantNext := append([]float32(nil), cpu.Step(17)...)
					device, err := m.NewBackendSessionChecked(be)
					if err != nil {
						t.Fatal(err)
					}
					defer device.Close()
					defer be.Recycle()
					device.SetExecutionPolicy(fakmodel.ExecutionPolicyDeviceOnly)
					be.ResetHostXfer()
					be.ResetH2DXfer()
					be.ResetQwen35GDNOperationCount()
					before := compute.KernelProfileSnapshot()
					var got []float32
					var failure any
					func() { defer func() { failure = recover() }(); got = device.Prefill(prompt) }()
					if isIQ2 && !selected && tokens > 1 {
						cause, _ := failure.(error)
						var seq *compute.Qwen35SequenceError
						if !errors.As(cause, &seq) || !strings.Contains(cause.Error(), "qwen35-iq2-sequence-qualification") || !device.BackendSessionClosed() {
							t.Fatalf("default-off request did not fail closed with typed refusal: %v", failure)
						}
						if len(got) != 0 || device.HostFallbackObserved() || be.Qwen35GDNOperationCount() != 0 || be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
							t.Fatal("default-off refusal executed, transferred, or fell back")
						}
						return
					}
					if failure != nil {
						t.Fatalf("ordinary Prefill failed: %v", failure)
					}
					if tokens > 1 {
						if err := device.RequireQwen35SequencePrefillNativePerformance(); err != nil {
							t.Fatal(err)
						}
						if be.Qwen35GDNOperationCount() != uint64(3*tokens) {
							t.Fatal("ordinary caller missed the complete GDN sequence")
						}
					}
					serveIQSequenceParity(t, "prefill", want, got)
					if isIQ2 && tokens > 1 {
						delta := compute.KernelProfileDelta(before, compute.KernelProfileSnapshot())
						if delta == nil {
							if *qwen35IQ2SequenceRequired {
								t.Fatal("required FAK_CUDA_TIMESTAMP_PROFILE=1 native kernel identity is missing")
							}
							t.Log("kernel identity UNKNOWN (timestamp profile disabled); not acceptance evidence")
						} else {
							if delta.Backend != "cuda" || delta.Truncated != 0 || delta.UntimedCompute != 0 {
								t.Fatalf("incomplete kernel trace: %+v", delta)
							}
							kernel := "k_iq2xxs_gemv"
							if kind == ggufload.TensorIQ2_XS {
								kernel = "k_iq2xs_gemv"
							}
							var calls, panels uint64
							for _, row := range delta.Rows {
								if row.Kernel == kernel {
									calls += row.Calls
									if row.GY == uint32(tokens) {
										panels += row.Calls
									}
								}
							}
							// 12 FFN + attention V/O + one last-row output head.
							if calls != 15 || panels != 14 {
								t.Fatalf("retained IQ projection route missing/widened: %s calls=%d full_panels=%d", kernel, calls, panels)
							}
						}
					}
					serveIQSequenceParity(t, "decode-after-prefill", wantNext, device.Step(17))
					if device.HostFallbackObserved() || device.BackendSessionClosed() {
						t.Fatal("device-only path fell back or retired")
					}
					route, _ := device.Qwen35SequencePrefillRouteStatus()
					t.Logf("format=%s P=%d route=%+v retained_weights=%d next_decode=true host_fallback=false", kind, tokens, route, len(raw))
				})
			}
		})
	}
}

func serveIQSequenceParity(t *testing.T, label string, reference, got []float32) {
	t.Helper()
	if len(reference) == 0 || len(reference) != len(got) {
		t.Fatalf("%s output extent %d vs %d", label, len(got), len(reference))
	}
	var dot, rr, gg, maxAbs, maxRef float64
	ri, gi := 0, 0
	for i, r32 := range reference {
		r, g := float64(r32), float64(got[i])
		if math.IsNaN(r) || math.IsInf(r, 0) || math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s non-finite row %d", label, i)
		}
		dot += r * g
		rr += r * r
		gg += g * g
		maxAbs = math.Max(maxAbs, math.Abs(r-g))
		maxRef = math.Max(maxRef, math.Abs(r))
		if r32 > reference[ri] {
			ri = i
		}
		if got[i] > got[gi] {
			gi = i
		}
	}
	cosine := dot / math.Sqrt(rr*gg)
	if rr == 0 || gg == 0 || math.IsNaN(cosine) || cosine < compute.Qwen35SequenceParityCosineMin || maxAbs > 0.05+0.05*maxRef || ri != gi {
		t.Fatalf("%s parity cosine=%g max_abs=%g max_ref=%g argmax=%d/%d", label, cosine, maxAbs, maxRef, gi, ri)
	}
	t.Logf("%s cosine=%.9f max_abs=%.6g argmax=%d", label, cosine, maxAbs, gi)
}

func serveIQSequenceFixture(t *testing.T, be compute.Backend, kind ggufload.TensorType) (*fakmodel.Model, map[string][]byte) {
	t.Helper()
	const h, vocab, convDim = 256, 32, 512
	// Keep the production attention panel contract even in this reduced
	// four-layer fixture; smaller head geometry is not admitted by CUDA.
	const qHeads, kvHeads, headDim = 24, 4, 256
	meta := map[string]ggufload.Value{
		"general.architecture":                    {Type: ggufload.TypeString, Value: "qwen35"},
		"qwen35.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
	}
	for key, value := range map[string]uint64{
		"context_length": 256, "embedding_length": h, "block_count": 4, "feed_forward_length": h,
		"attention.head_count": qHeads, "attention.head_count_kv": kvHeads, "attention.key_length": headDim, "attention.value_length": headDim,
		"rope.dimension_count": 64, "full_attention_interval": 4, "nextn_predict_layers": 0,
		"ssm.conv_kernel": 4, "ssm.state_size": 64, "ssm.group_count": 2, "ssm.time_step_rank": 4, "ssm.inner_size": 256,
	} {
		meta["qwen35."+key] = ggufload.Value{Type: ggufload.TypeUint64, Value: value}
	}
	text := make([]ggufload.Value, vocab)
	for i := range text {
		text[i] = ggufload.Value{Type: ggufload.TypeString, Value: fmt.Sprint(i)}
	}
	meta["tokenizer.ggml.tokens"] = ggufload.Value{Type: ggufload.TypeArray, Value: text}
	file := &ggufload.File{Metadata: meta}
	var payload bytes.Buffer
	retained := make(map[string][]byte)
	rng := rand.New(rand.NewSource(12983))
	add := func(name, canonical string, typ ggufload.TensorType, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		var raw []byte
		if typ == ggufload.TensorF32 {
			raw = make([]byte, n*4)
			for i := 0; i < n; i++ {
				v := float32(rng.NormFloat64() * 0.015)
				switch {
				case strings.HasSuffix(name, "norm.weight"):
					v += 1
				case strings.HasSuffix(name, ".ssm_a"):
					v = -float32(math.Exp(float64(i) * 0.25))
				case name == "token_embd.weight":
					v *= 12
				}
				binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
			}
		} else {
			block, values := 0, 256
			switch typ {
			case ggufload.TensorIQ2_XXS:
				block = 66
			case ggufload.TensorIQ2_XS:
				block = 74
			case ggufload.TensorIQ4_XS:
				block = 136
			case ggufload.TensorQ8_0:
				block, values = 34, 32
			case ggufload.TensorQ5_K:
				block = 176
			case ggufload.TensorQ6_K:
				block = 210
			default:
				t.Fatalf("unexpected fixture dtype %s", typ)
			}
			if n%values != 0 {
				t.Fatal("fixture has incomplete quant row")
			}
			raw = make([]byte, n/values*block)
			for i := range raw {
				raw[i] = byte(rng.Intn(256))
			}
			for i := 0; i < len(raw); i += block {
				if typ == ggufload.TensorQ6_K {
					binary.LittleEndian.PutUint16(raw[i+208:], 0x0800)
				} else {
					binary.LittleEndian.PutUint16(raw[i:], 0x0800)
					if typ == ggufload.TensorQ5_K {
						binary.LittleEndian.PutUint16(raw[i+2:], 0x0800)
					}
				}
			} // small finite nonzero f16 scales; old fixture formats unchanged
			if typ != ggufload.TensorQ8_0 {
				retained[canonical] = append([]byte(nil), raw...)
			}
		}
		file.Tensors = append(file.Tensors, ggufload.TensorInfo{Name: name, Dims: dims, Type: typ, FileOffset: int64(payload.Len())})
		payload.Write(raw)
	}
	add("token_embd.weight", "", ggufload.TensorF32, h, vocab)
	add("output_norm.weight", "", ggufload.TensorF32, h)
	add("output.weight", "lm_head.weight", kind, h, vocab)
	for l := 0; l < 4; l++ {
		p, c := fmt.Sprintf("blk.%d.", l), fmt.Sprintf("model.layers.%d.", l)
		add(p+"attn_norm.weight", "", ggufload.TensorF32, h)
		add(p+"post_attention_norm.weight", "", ggufload.TensorF32, h)
		for _, f := range []struct{ gguf, canonical string }{{"ffn_gate", "gate_proj"}, {"ffn_up", "up_proj"}, {"ffn_down", "down_proj"}} {
			add(p+f.gguf+".weight", c+"mlp."+f.canonical+".weight", kind, h, h)
		}
		if l < 3 {
			add(p+"attn_qkv.weight", "", ggufload.TensorF32, h, convDim)
			add(p+"attn_gate.weight", "", ggufload.TensorF32, h, h)
			add(p+"ssm_alpha.weight", "", ggufload.TensorF32, h, 4)
			add(p+"ssm_beta.weight", "", ggufload.TensorF32, h, 4)
			add(p+"ssm_out.weight", "", ggufload.TensorF32, h, h)
			add(p+"ssm_a", "", ggufload.TensorF32, 4)
			add(p+"ssm_dt.bias", "", ggufload.TensorF32, 4)
			add(p+"ssm_conv1d.weight", "", ggufload.TensorF32, 4, convDim)
			add(p+"ssm_norm.weight", "", ggufload.TensorF32, 64)
		} else {
			add(p+"attn_q.weight", "", ggufload.TensorF32, h, 2*qHeads*headDim)
			add(p+"attn_k.weight", "", ggufload.TensorF32, h, kvHeads*headDim)
			add(p+"attn_v.weight", c+"self_attn.v_proj.weight", kind, h, kvHeads*headDim)
			add(p+"attn_output.weight", c+"self_attn.o_proj.weight", kind, qHeads*headDim, h)
			add(p+"attn_q_norm.weight", "", ggufload.TensorF32, headDim)
			add(p+"attn_k_norm.weight", "", ggufload.TensorF32, headDim)
		}
	}
	ws, err := ggufload.NewWeightSource(file, bytes.NewReader(payload.Bytes()), int64(payload.Len()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ws.Close() })
	m, err := ws.QuantModelQ4KProfileOptions(nil, serveDenseKQuantOptions(be)...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.CloseWeights() })
	if m.Cfg.NumLayers != 4 || !m.Cfg.IsQwen35Hybrid() || m.Cfg.HiddenSize != h || m.Cfg.VocabSize != vocab {
		t.Fatalf("bad loaded hybrid fixture: %+v", m.Cfg)
	}
	return m, retained
}
