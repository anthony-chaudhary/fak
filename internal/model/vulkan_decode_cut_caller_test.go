package model

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type vulkanCutCallerEvent struct {
	kind, weight string
	w, x, y      compute.Tensor
}

// This is a CPU caller-selection capture, not a mock Vulkan execution result.
// The attention fused interfaces are intentionally present, as on Vulkan: the
// test cannot obtain plain Q/K/V by merely hiding those capabilities.
type vulkanCutCallerBackend struct {
	compute.Backend
	names  map[compute.Buffer]string
	events []vulkanCutCallerEvent
}

func (b *vulkanCutCallerBackend) BeginBatch() {
	b.events = append(b.events, vulkanCutCallerEvent{kind: "begin"})
}
func (b *vulkanCutCallerBackend) FlushBatch() {
	b.events = append(b.events, vulkanCutCallerEvent{kind: "flush"})
}
func (b *vulkanCutCallerBackend) UploadClass(x compute.Tensor, dtype compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	y := b.Backend.Upload(x, dtype)
	b.names[y.Buf()] = strings.TrimPrefix(site, "hal-weight ")
	b.events = append(b.events, vulkanCutCallerEvent{kind: "upload", weight: b.names[y.Buf()]})
	return y
}
func (b *vulkanCutCallerBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	y := b.Backend.MatMul(w, x)
	b.events = append(b.events, vulkanCutCallerEvent{kind: "matmul", weight: b.names[w.Buf()], w: w, x: x, y: y})
	return y
}
func (b *vulkanCutCallerBackend) AddBias(x, bias compute.Tensor) {
	b.events = append(b.events, vulkanCutCallerEvent{kind: "bias"})
	b.Backend.AddBias(x, bias)
}
func (b *vulkanCutCallerBackend) MatMul3(wq, wk, wv, x compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor) {
	b.events = append(b.events, vulkanCutCallerEvent{kind: "fused3"})
	return b.Backend.MatMul(wq, x), b.Backend.MatMul(wk, x), b.Backend.MatMul(wv, x)
}
func (b *vulkanCutCallerBackend) RMSNormMatMul3(wq, wk, wv, x, norm compute.Tensor, eps float32) (compute.Tensor, compute.Tensor, compute.Tensor) {
	b.events = append(b.events, vulkanCutCallerEvent{kind: "fused_norm3"})
	xn := b.Backend.RMSNorm(x, norm, eps)
	return b.Backend.MatMul(wq, xn), b.Backend.MatMul(wk, xn), b.Backend.MatMul(wv, xn)
}

// This tiny synthetic Session.Step witness executes the ordinary model caller on
// cpu-ref. It does not execute Vulkan, assign native ordinals, or establish GPU
// numerical/cache parity. Those remain separate candidate-bound obligations.
func TestVulkanBatchDecodeCutOrdinaryCaller(t *testing.T) {
	for _, bias := range []bool{true, false} {
		name := "bias_free_uses_fusion"
		if bias {
			name = "attention_bias_plain_qkv"
		}
		t.Run(name, func(t *testing.T) {
			cfg := Config{HiddenSize: 16, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 4,
				IntermediateSize: 32, VocabSize: 32, RMSNormEps: 1e-5, RopeTheta: 10000,
				TieWordEmbeddings: true, EOSTokenID: 31, AttentionBias: bias}
			extra := map[string][]int{}
			if bias {
				extra[layerName(0, "self_attn.q_proj.bias")] = []int{16}
				extra[layerName(0, "self_attn.k_proj.bias")] = []int{8}
				extra[layerName(0, "self_attn.v_proj.bias")] = []int{8}
			}
			b := &vulkanCutCallerBackend{Backend: compute.Pick("cpu-ref"), names: make(map[compute.Buffer]string)}
			s := newSyntheticExtra(cfg, extra).NewBackendSession(b)
			defer s.Close()
			s.Step(3) // Warm the actual model weight cache, not a hand-built call list.
			b.events = nil
			s.Step(7)
			qIndex := -1
			fused := false
			for i, e := range b.events {
				if e.kind == "fused_norm3" || e.kind == "fused3" {
					fused = true
				}
				if e.kind == "matmul" && e.weight == layerName(0, "self_attn.q_proj.weight") {
					qIndex = i
				}
			}
			if !bias {
				if !fused || qIndex != -1 {
					t.Fatal("bias-free caller did not retain fused attention")
				}
				return
			}
			if fused || qIndex < 0 || qIndex+2 >= len(b.events) {
				t.Fatal("ordinary attention-bias caller did not select plain Q/K/V")
			}
			q, k, v := b.events[qIndex], b.events[qIndex+1], b.events[qIndex+2]
			if k.kind != "matmul" || v.kind != "matmul" || k.weight != layerName(0, "self_attn.k_proj.weight") || v.weight != layerName(0, "self_attn.v_proj.weight") {
				t.Fatal("Q/K/V are not adjacent calls in warmed token")
			}
			if q.x.Buf() != k.x.Buf() || q.x.Buf() != v.x.Buf() || q.y.Buf() == k.y.Buf() || q.y.Buf() == v.y.Buf() || k.y.Buf() == v.y.Buf() {
				t.Fatal("Q/K/V owner-sharing contract drifted")
			}
			for _, e := range []vulkanCutCallerEvent{q, k, v} {
				if e.w.Dtype != compute.F32 || e.x.Dtype != compute.F32 || e.y.Dtype != compute.F32 {
					t.Fatal("caller was not all-F32")
				}
			}
			began, flushed, biasAfter := false, false, false
			for i, e := range b.events {
				if e.kind == "begin" && i < qIndex {
					began = true
				}
				if e.kind == "flush" && i > qIndex+2 {
					flushed = true
				}
				if e.kind == "bias" && i > qIndex+2 {
					biasAfter = true
				}
			}
			if !began || !flushed || !biasAfter {
				t.Fatal("batch boundary or bias ordering drifted")
			}
		})
	}
}
