package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type benchObservedBackend struct {
	compute.Backend
	matmuls int
}

type benchQuantBackend struct {
	benchObservedBackend
	q4Payload []int8
}

func (b *benchQuantBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true // cpu-ref implements Q8/Q4_K math; expose it for this routing witness.
	return c
}

func (b *benchQuantBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	if as == compute.Q4_K {
		b.q4Payload = append([]int8(nil), t.Buf().(compute.HostBuffer).I8()...)
	}
	return b.Backend.Upload(t, as)
}

func TestModelbenchMixedQuantBackendLoadAndForward(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	f := testCompleteBenchFlags()
	*f.gguf, *f.q4k, *f.backendName = benchMixedQuantGGUF(t), true, "vulkan"
	be := &benchQuantBackend{benchObservedBackend: benchObservedBackend{Backend: compute.Default()}}
	in := preflightInputFor(f, be)
	if in.Source != nil {
		defer in.Source.Close()
	}
	if in.Lean || in.Q4K {
		t.Fatal("HAL conversion fit must use the conservative F32 upper bound")
	}
	_, precision, report := describeEngine(f, be, nil)
	if precision != "resident Q4_K + dense non-Q4_K converted to Q8" || report["dense_non_q4k_load"] == nil {
		t.Fatal("HAL conversion is missing from execution identity")
	}
	for _, streamed := range []bool{false, true} {
		be.q4Payload = nil
		*f.streamQ4K = streamed
		m, _, err := loadModel(f, nil)
		if err != nil {
			t.Fatal(err)
		}
		func() {
			defer m.CloseWeights()
			s := newBenchSession(m, f, be)
			defer s.Close()
			prefill := append([]float32(nil), s.Prefill([]int{0, 1})...)
			decode := s.Step(2)
			if len(prefill) != 4 || len(decode) != 4 || !allFinite(prefill) || !allFinite(decode) || be.matmuls == 0 {
				t.Fatal("mixed-quant prefill/decode did not produce finite logits through HAL")
			}
			for _, name := range []string{"lm_head.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight"} {
				if m.HasKQuant(name) || !m.HasQ8(name) {
					t.Fatalf("%s must take the same dense Q8 conversion as serve", name)
				}
			}
			if !m.HasQ4K("model.layers.0.mlp.up_proj.weight") || len(be.q4Payload) != 256*144 || m.HasQ8("model.layers.0.mlp.up_proj.weight") {
				t.Fatal("eligible Q4_K did not retain its packed payload")
			}
			for i, got := range be.q4Payload {
				want := byte((i%144)*13 + i/144)
				if i%144 < 4 {
					want = 0
					if i%2 == 1 {
						want = 8
					}
				}
				if byte(got) != want {
					t.Fatalf("Q4_K source bytes changed at %d", i)
				}
			}
		}()
	}
	// The legacy resident path keeps its source representation.
	*f.backendName, *f.streamQ4K = "legacy", false
	m, _, err := loadModel(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	if !m.HasKQuant("model.layers.0.mlp.gate_proj.weight") {
		t.Fatal("legacy load lost native IQ2 residency")
	}
	// NewBackendSession wraps the checked constructor: unsupported hybrid
	// architectures must fail before a generic QKV operation is attempted.
	defer func() {
		if _, ok := recover().(*model.UnsupportedBackendForwardError); !ok {
			t.Fatal("unsupported hybrid backend did not return its typed refusal")
		}
	}()
	m.Cfg.ModelType = "qwen3_5"
	m.Cfg.LayerTypes = []string{"linear_attention"}
	m.NewBackendSession(be)
}

// A one-layer routing fixture, not a model-quality or hardware witness. It
// includes Q3_K/IQ3_S, unsupported resident IQ2_XXS, Q4_K and Q6_K projections.
func benchMixedQuantGGUF(t *testing.T) string {
	t.Helper()
	const dim, vocab = 256, 4
	type tensor struct {
		name   string
		dims   []uint64
		kind   ggufload.TensorType
		raw    []byte
		offset uint64
	}
	var tensors []tensor
	add := func(name string, shape []uint64, kind ggufload.TensorType, blockBytes int) {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		raw := make([]byte, n*4)
		if kind == ggufload.TensorF32 {
			for i := 0; i < n; i++ {
				v := float32((i*7)%19-9) / 128
				if len(shape) == 1 {
					v = 1
				}
				binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
			}
		} else {
			raw = make([]byte, n/256*blockBytes)
			for off := 0; off < len(raw); off += blockBytes {
				for i := 0; i < blockBytes; i++ {
					raw[off+i] = byte(i*13 + off/blockBytes)
				}
				scale := off
				if kind == ggufload.TensorQ3_K || kind == ggufload.TensorQ6_K {
					scale += blockBytes - 2
				}
				if kind == ggufload.TensorQ2_K {
					scale += blockBytes - 4
					binary.LittleEndian.PutUint16(raw[scale+2:], 0x0800)
				}
				binary.LittleEndian.PutUint16(raw[scale:], 0x0800)
				if kind == ggufload.TensorQ4_K {
					binary.LittleEndian.PutUint16(raw[off+2:], 0x0800)
				}
			}
		}
		tensors = append(tensors, tensor{name: name, dims: shape, kind: kind, raw: raw})
	}
	add("token_embd.weight", []uint64{dim, vocab}, ggufload.TensorF32, 0)
	add("output.weight", []uint64{dim, vocab}, ggufload.TensorQ2_K, 84)
	for _, name := range []string{"output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"} {
		add(name, []uint64{dim}, ggufload.TensorF32, 0)
	}
	for _, name := range []string{"blk.0.attn_q.weight", "blk.0.attn_k.weight"} {
		add(name, []uint64{dim, dim}, ggufload.TensorF32, 0)
	}
	add("blk.0.attn_v.weight", []uint64{dim, dim}, ggufload.TensorQ3_K, 110)
	add("blk.0.attn_output.weight", []uint64{dim, dim}, ggufload.TensorIQ3_S, 110)
	add("blk.0.ffn_gate.weight", []uint64{dim, dim}, ggufload.TensorIQ2_XXS, 66)
	add("blk.0.ffn_up.weight", []uint64{dim, dim}, ggufload.TensorQ4_K, 144)
	add("blk.0.ffn_down.weight", []uint64{dim, dim}, ggufload.TensorQ6_K, 210)
	var b bytes.Buffer
	write := func(v any) {
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	str := func(s string) { write(uint64(len(s))); b.WriteString(s) }
	write(uint32(0x46554747))
	write(uint32(3))
	write(uint64(len(tensors)))
	write(uint64(12))
	str("general.architecture")
	write(uint32(8))
	str("llama")
	for _, kv := range []struct {
		key string
		val uint32
	}{
		{"general.alignment", 32}, {"llama.context_length", 16}, {"llama.embedding_length", dim},
		{"llama.block_count", 1}, {"llama.feed_forward_length", dim}, {"llama.attention.head_count", 1},
		{"llama.attention.head_count_kv", 1}, {"llama.rope.dimension_count", dim},
	} {
		str(kv.key)
		write(uint32(4))
		write(kv.val)
	}
	str("llama.attention.layer_norm_rms_epsilon")
	write(uint32(6))
	write(float32(1e-5))
	str("llama.rope.freq_base")
	write(uint32(6))
	write(float32(10000))
	str("tokenizer.ggml.tokens")
	write(uint32(9))
	write(uint32(8))
	write(uint64(vocab))
	for _, tok := range []string{"a", "b", "c", "d"} {
		str(tok)
	}
	var offset uint64
	for i := range tensors {
		tn := &tensors[i]
		tn.offset = offset
		str(tn.name)
		write(uint32(len(tn.dims)))
		for _, d := range tn.dims {
			write(d)
		}
		write(uint32(tn.kind))
		write(offset)
		offset = (offset + uint64(len(tn.raw)) + 31) &^ 31
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	start := b.Len()
	for _, tn := range tensors {
		for b.Len() < start+int(tn.offset) {
			b.WriteByte(0)
		}
		b.Write(tn.raw)
	}
	path := filepath.Join(t.TempDir(), "mixed-native.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func (b *benchObservedBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}

func (b *benchObservedBackend) BatchedMatMul(w, x compute.Tensor, p int) compute.Tensor {
	b.matmuls++
	return b.Backend.BatchedMatMul(w, x, p)
}

func TestModelbenchSelectedBackendExecutesSmoke(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	ref, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU reference backend missing")
	}
	be := &benchObservedBackend{Backend: ref}
	compute.Register(be)
	t.Cleanup(func() { compute.Register(ref) })
	m := model.NewSynthetic(model.Config{
		HiddenSize: 16, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 4,
		IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	})
	f := testCompleteBenchFlags()
	*f.backendName = "cpu-ref"
	*f.out = filepath.Join(t.TempDir(), "smoke.json")
	*f.decodePrompt = 2
	runSmoke(f, m, "fixture", 0, m.Cfg.VocabSize)
	if be.matmuls == 0 {
		t.Fatal("smoke bypassed the selected backend: zero observed HAL matmuls")
	}
	raw, err := os.ReadFile(*f.out)
	if err != nil {
		t.Fatal(err)
	}
	var report struct {
		Backend struct{ Selected string } `json:"backend"`
		Status  string                    `json:"smoke_status"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report.Backend.Selected != "cpu-ref" || report.Status != smokeStatusOK {
		t.Fatalf("smoke execution identity: %+v", report)
	}
	// An independent prefill through the real HAL proves smoke included a decode
	// step as well, instead of claiming decode success from prefill alone.
	smokeMatmuls := be.matmuls
	be.matmuls = 0
	s := m.NewBackendSession(be)
	s.Prefill(lcgIDs(*f.decodePrompt, m.Cfg.VocabSize))
	s.Close()
	if smokeMatmuls <= be.matmuls {
		t.Fatalf("smoke omitted decode: smoke matmuls=%d, prefill-only=%d", smokeMatmuls, be.matmuls)
	}
	*f.q4k, *f.gguf, *f.backendName = true, "fixture.gguf", "vulkan"
	if err := validateFlagCombinations(f); err != nil {
		t.Fatalf("packed Vulkan session rejected: %v", err)
	}
	*f.metal = true
	if err := validateFlagCombinations(f); err == nil {
		t.Fatal("conflicting legacy Metal and compute backend accepted")
	}
}
