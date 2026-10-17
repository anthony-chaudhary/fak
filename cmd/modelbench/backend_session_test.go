package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type engProjGGUFTensor struct {
	name  string
	dims  []uint64
	dtype ggufload.TensorType
	data  []byte
}

func engProjPut(b *bytes.Buffer, value any) {
	if err := binary.Write(b, binary.LittleEndian, value); err != nil {
		panic(err)
	}
}
func engProjString(b *bytes.Buffer, value string) {
	engProjPut(b, uint64(len(value)))
	b.WriteString(value)
}
func engProjQ2(out, in int, seed uint64) []byte {
	raw := make([]byte, out*in/256*84)
	for i := range raw {
		seed = seed*6364136223846793005 + 1442695040888963407
		raw[i] = byte(seed >> 40)
	}
	for offset := 0; offset < len(raw); offset += 84 {
		// Modest finite scales and varied codes keep the Engram contribution visible without saturation.
		binary.LittleEndian.PutUint16(raw[offset+80:], 0x1400)
		binary.LittleEndian.PutUint16(raw[offset+82:], 0)
	}
	return raw
}
func engProjGGUF(t *testing.T) string {
	t.Helper()
	const H, I, E, cols, dim, hc, vocab = 64, 32, model.V41RouterExperts, 24, 256, 4, 8
	var tensors []engProjGGUFTensor
	seed := uint64(1366808)
	f32 := func(name string, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		data := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			value := (float32(seed>>40)/float32(1<<24)*2 - 1) * 0.05
			if strings.Contains(name, "norm") || strings.HasSuffix(name, "mhc_scale.weight") || strings.HasSuffix(name, "engram_q.weight") || strings.HasSuffix(name, "engram_k.weight") {
				value = 1
			}
			if strings.HasSuffix(name, "mhc_base.weight") {
				value = 0
			}
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
		}
		tensors = append(tensors, engProjGGUFTensor{name, dims, ggufload.TensorF32, data})
	}
	f32("token_embd.weight", H, vocab)
	f32("output.weight", H, vocab)
	f32("output_norm.weight", H)
	p := "blk.0."
	f32(p+"attn_norm.weight", H)
	f32(p+"ffn_norm.weight", H)
	f32(p+"mhc_mixes.weight", hc*H, 24)
	f32(p+"mhc_base.weight", 24)
	f32(p+"mhc_scale.weight", 3)
	f32(p+"attn_q_a.weight", H, 32)
	f32(p+"attn_q_b.weight", 32, 64)
	f32(p+"attn_kv.weight", H, 32)
	f32(p+"attn_kv_a_norm.weight", 32)
	f32(p+"attn_output_a.weight", 64, 16)
	f32(p+"attn_output_b.weight", 32, H)
	f32(p+"attn_sinks.weight", 2)
	f32(p+"ffn_gate_inp.weight", H, E)
	f32(p+"exp_probs_b.bias", E)
	f32(p+"ffn_gate_shexp.weight", H, I)
	f32(p+"ffn_up_shexp.weight", H, I)
	f32(p+"ffn_down_shexp.weight", I, H)
	f32(p+"ffn_gate_exps.weight", H, I, E)
	f32(p+"ffn_up_exps.weight", H, I, E)
	f32(p+"ffn_down_exps.weight", I, H, E)
	f32(p+"engram_q.weight", hc*H)
	f32(p+"engram_k.weight", hc*H)
	tensors = append(tensors,
		engProjGGUFTensor{p + "engram_wkv.weight", []uint64{cols * dim, (hc + 1) * H}, ggufload.TensorQ2_K, engProjQ2((hc+1)*H, cols*dim, 136681)},
		engProjGGUFTensor{p + "engram_embd.weight", []uint64{256, 72}, ggufload.TensorQ2_K, engProjQ2(72, 256, 136682)},
	)
	var metadata []func(*bytes.Buffer)
	u64 := func(key string, value uint64) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeUint64))
			engProjPut(b, value)
		})
	}
	u32 := func(key string, value uint32) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeUint32))
			engProjPut(b, value)
		})
	}
	floating := func(key string, value float32) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeFloat32))
			engProjPut(b, value)
		})
	}
	str := func(key, value string) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeString))
			engProjString(b, value)
		})
	}
	ints := func(key string, values []int32) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeArray))
			engProjPut(b, uint32(ggufload.TypeInt32))
			engProjPut(b, uint64(len(values)))
			for _, v := range values {
				engProjPut(b, v)
			}
		})
	}
	uints := func(key string, values []uint64) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeArray))
			engProjPut(b, uint32(ggufload.TypeUint64))
			engProjPut(b, uint64(len(values)))
			for _, v := range values {
				engProjPut(b, v)
			}
		})
	}
	str("general.architecture", "deepseek41")
	u32("general.alignment", 32)
	p = "deepseek41."
	for key, value := range map[string]uint64{
		"embedding_length": H, "block_count": 1, "attention.head_count": 2, "expert_count": E, "expert_used_count": model.V41RouterTopK,
		"expert_feed_forward_length": I, "expert_shared_count": 1, "expert_shared_feed_forward_length": I,
		"attention.q_lora_rank": 32, "attention.kv_lora_rank": 32, "attention.qk_nope_head_dim": 16, "attention.qk_rope_head_dim": 16,
		"attention.value_length_mla": 16, "attention.output_group_count": 2, "attention.output_lora_rank": 16, "hyper_connection.count": hc, "hyper_connection.sinkhorn_iterations": 11,
	} {
		u64(p+key, value)
	}
	u32(p+"engram.head_count", 8)
	u32(p+"engram.key_length", dim)
	u32(p+"engram.max_ngram_size", 4)
	u32(p+"engram.pad_id", 2)
	floating(p+"attention.layer_norm_rms_epsilon", 1e-6)
	floating(p+"hyper_connection.epsilon", 1e-6)
	floating(p+"expert_weights_scale", 1.5)
	ints(p+"attention.compress_ratios", []int32{0})
	ints(p+"engram.layer_ids", []int32{0})
	ints(p+"engram.token_map", []int32{0, 1, 2, 3, 4, 5, 6, 7})
	primes, offsets := make([]uint64, cols), make([]uint64, cols)
	for i := range primes {
		primes[i], offsets[i] = 3, uint64(3*i)
	}
	uints(p+"engram.primes", primes)
	uints(p+"engram.offsets", offsets)
	uints(p+"engram.multipliers", []uint64{1, 3, 5, 7})
	metadata = append(metadata, func(b *bytes.Buffer) {
		engProjString(b, "tokenizer.ggml.tokens")
		engProjPut(b, uint32(ggufload.TypeArray))
		engProjPut(b, uint32(ggufload.TypeString))
		engProjPut(b, uint64(vocab))
		for _, token := range []string{"a", "b", "c", "d", "e", "f", "g", "h"} {
			engProjString(b, token)
		}
	})
	var out bytes.Buffer
	out.WriteString("GGUF")
	engProjPut(&out, uint32(3))
	engProjPut(&out, uint64(len(tensors)))
	engProjPut(&out, uint64(len(metadata)))
	for _, write := range metadata {
		write(&out)
	}
	offset := uint64(0)
	for _, tensor := range tensors {
		engProjString(&out, tensor.name)
		engProjPut(&out, uint32(len(tensor.dims)))
		for _, d := range tensor.dims {
			engProjPut(&out, d)
		}
		engProjPut(&out, uint32(tensor.dtype))
		engProjPut(&out, offset)
		offset = (offset + uint64(len(tensor.data)) + 31) / 32 * 32
	}
	for out.Len()%32 != 0 {
		out.WriteByte(0)
	}
	for _, tensor := range tensors {
		out.Write(tensor.data)
		for out.Len()%32 != 0 {
			out.WriteByte(0)
		}
	}
	path := filepath.Join(t.TempDir(), "v41-engram-projection.gguf")
	if err := os.WriteFile(path, out.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

type benchObservedBackend struct {
	compute.Backend
	matmuls int
}

type benchQuantBackend struct {
	benchObservedBackend
	q4Payload []int8
	q2Payload []int8
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
	if as == compute.Q2_K {
		b.q2Payload = append([]int8(nil), t.Buf().(compute.HostBuffer).I8()...)
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
	if in.Lean || in.Q4K || !in.VulkanMixedQ4K {
		t.Fatal("Vulkan mixed-quant load must select the mixed Q4_K/Q2_K/Q8 preflight plan")
	}
	pf := ggufload.BuildModelPreflight(in)
	if pf.Refused() {
		t.Fatalf("mixed-quant preflight refused fixture: %s", pf.Reason)
	}
	if pf.EstReadBytes <= 0 || pf.EstHostResidentBytes <= 0 || pf.EstDeviceResidentBytes <= 0 || pf.EstLoadStagingBytes <= 0 {
		t.Fatalf("mixed-quant preflight omitted classed demand: %+v", pf)
	}
	if want := pf.EstHostResidentBytes + pf.EstDeviceResidentBytes + pf.EstLoadStagingBytes; pf.EstLoadBytes != want {
		t.Fatalf("mixed-quant total = %d, want host %d + device %d + staging %d = %d",
			pf.EstLoadBytes, pf.EstHostResidentBytes, pf.EstDeviceResidentBytes, pf.EstLoadStagingBytes, want)
	}
	*f.streamQ4K = true
	streamIn := preflightInputFor(f, be)
	if streamIn.Source != nil {
		defer streamIn.Source.Close()
	}
	if streamIn.Lean || streamIn.Q4K || !streamIn.VulkanMixedQ4K {
		t.Fatal("streaming Vulkan Q4_K must select its mixed-quant streamed lifecycle")
	}
	streamPF := ggufload.BuildModelPreflight(streamIn)
	if streamPF.Refused() {
		t.Fatalf("streamed mixed-quant preflight refused fixture: %s", streamPF.Reason)
	}
	if streamPF.EstReadBytes != pf.EstReadBytes || streamPF.EstDeviceResidentBytes != pf.EstDeviceResidentBytes ||
		streamPF.EstHostResidentBytes >= pf.EstHostResidentBytes {
		t.Fatalf("streamed preflight = %+v, resident = %+v; want same read/device and lower host residency", streamPF, pf)
	}
	*f.streamQ4K = false
	_, precision, report := describeEngine(f, be, nil)
	if precision != "resident Q4_K/Q2_K + unsupported dense formats converted to Q8" || report["dense_non_q4k_load"] == nil {
		t.Fatal("HAL conversion is missing from execution identity")
	}
	for _, streamed := range []bool{false, true} {
		be.q4Payload = nil
		be.q2Payload = nil
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
			if !m.HasKQuant("lm_head.weight") || m.HasQ8("lm_head.weight") || len(be.q2Payload) != 4*84 {
				t.Fatal("eligible Q2_K did not stay packed and reach the backend")
			}
			for _, name := range []string{"model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight"} {
				if m.HasKQuant(name) || !m.HasQ8(name) {
					t.Fatalf("unsupported dense format %s must take the Q8 conversion path", name)
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
func benchMixedQuantGGUF(t *testing.T, moe ...bool) string {
	t.Helper()
	const dim, vocab = 256, 4
	streamedMoE := len(moe) > 0 && moe[0]
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
	if streamedMoE {
		add("blk.0.ffn_gate_inp.weight", []uint64{dim, 3}, ggufload.TensorF32, 0)
		add("blk.0.ffn_gate_exps.weight", []uint64{dim, dim, 3}, ggufload.TensorQ2_K, 84)
		add("blk.0.ffn_up_exps.weight", []uint64{dim, dim, 3}, ggufload.TensorQ2_K, 84)
		add("blk.0.ffn_down_exps.weight", []uint64{dim, dim, 3}, ggufload.TensorQ3_K, 110)
	} else {
		add("blk.0.ffn_gate.weight", []uint64{dim, dim}, ggufload.TensorIQ2_XXS, 66)
		add("blk.0.ffn_up.weight", []uint64{dim, dim}, ggufload.TensorQ4_K, 144)
		add("blk.0.ffn_down.weight", []uint64{dim, dim}, ggufload.TensorQ6_K, 210)
	}
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
	metadataCount := uint64(12)
	arch := "llama"
	if streamedMoE {
		arch = "qwen3moe"
		metadataCount += 3
	}
	write(metadataCount)
	str("general.architecture")
	write(uint32(8))
	str(arch)
	for _, kv := range []struct {
		key string
		val uint32
	}{
		{"general.alignment", 32}, {"llama.context_length", 16}, {"llama.embedding_length", dim},
		{"llama.block_count", 1}, {"llama.feed_forward_length", dim}, {"llama.attention.head_count", 1},
		{"llama.attention.head_count_kv", 1}, {"llama.rope.dimension_count", dim},
	} {
		str(strings.Replace(kv.key, "llama.", arch+".", 1))
		write(uint32(4))
		write(kv.val)
	}
	if streamedMoE {
		for key, value := range map[string]uint32{"expert_count": 3, "expert_used_count": 1, "expert_feed_forward_length": dim} {
			str(arch + "." + key)
			write(uint32(4))
			write(value)
		}
	}
	str(arch + ".attention.layer_norm_rms_epsilon")
	write(uint32(6))
	write(float32(1e-5))
	str(arch + ".rope.freq_base")
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

func benchStreamedTuple(f *benchFlags, ring, dense int64) {
	for key, value := range map[string]int64{"expertRingBytes": ring, "streamedDenseBytes": dense} {
		field := reflect.ValueOf(f).Elem().FieldByName(key)
		if field.IsValid() {
			copyValue := value
			reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem().Set(reflect.ValueOf(&copyValue))
		}
	}
}

type benchColdKVBackend struct {
	benchObservedBackend
	newKV int
	mode  string
}

func (b *benchColdKVBackend) Name() string {
	if b.mode == "vulkan_identity" {
		return "vulkan"
	}
	return b.Backend.Name()
}

func (b *benchColdKVBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.newKV++
	store := b.Backend.NewKV(cfg)
	if b.mode == "warm_kv" {
		return benchWarmKV{KVStore: store}
	}
	return store
}

func (b *benchColdKVBackend) Read(t compute.Tensor) []float32 {
	if b.mode == "empty" {
		return nil
	}
	if b.mode == "nonfinite" {
		return []float32{float32(math.NaN())}
	}
	if b.mode == "refused_read" {
		panic(errors.New("test read refusal"))
	}
	return b.Backend.Read(t)
}

type benchWarmKV struct{ compute.KVStore }

func (b benchWarmKV) Pos() []int { return []int{0} }
func (b benchWarmKV) Len() int   { return 1 }

// fak-test:runtime fast est=2s lane=default
func TestModelbenchStreamedColdPrefill(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	t.Run("flags_and_fingerprint", func(t *testing.T) {
		f := testCompleteBenchFlags()
		*f.gguf, *f.q4k, *f.backendName = "fixture.gguf", true, "vulkan"
		before := modelbenchFingerprint(f, "fixture")
		benchStreamedTuple(f, 0, 0)
		if !reflect.DeepEqual(before, modelbenchFingerprint(f, "fixture")) {
			t.Fatal("inactive defaults changed legacy checkpoint identity")
		}
		for _, pair := range [][2]int64{{1 << 20, 0}, {0, 1 << 20}, {-1, 1 << 20}, {1 << 20, -1}} {
			benchStreamedTuple(f, pair[0], pair[1])
			if validateFlagCombinations(f) == nil {
				t.Fatalf("incomplete/invalid streamed bounds admitted: %v", pair)
			}
			_, _, err := loadModelContext(context.Background(), f, nil)
			var pathError *os.PathError
			if err == nil || errors.As(err, &pathError) {
				t.Fatalf("invalid tuple reached checkpoint IO: bounds=%v error=%v", pair, err)
			}
		}
		benchStreamedTuple(f, 1<<20, 1<<20)
		if err := validateFlagCombinations(f); err != nil {
			t.Fatal(err)
		}
		active := modelbenchFingerprint(f, "fixture")
		benchStreamedTuple(f, 2<<20, 1<<20)
		if reflect.DeepEqual(active, modelbenchFingerprint(f, "fixture")) {
			t.Fatal("active ring bound is absent from checkpoint identity")
		}
		benchStreamedTuple(f, 1<<20, 2<<20)
		if reflect.DeepEqual(active, modelbenchFingerprint(f, "fixture")) {
			t.Fatal("active dense bound is absent from checkpoint identity")
		}
		*f.backendName = "cpu-ref"
		if validateFlagCombinations(f) == nil {
			t.Fatal("streamed Vulkan tuple admitted on CPU flag")
		}
	})
	t.Run("mapped_loader_and_lifetime", func(t *testing.T) {
		f := testCompleteBenchFlags()
		*f.gguf, *f.q4k, *f.backendName = benchMixedQuantGGUF(t, true), true, "vulkan"
		benchStreamedTuple(f, 1<<20, 1<<20)
		in := preflightInputFor(f, &benchColdKVBackend{benchObservedBackend: benchObservedBackend{Backend: compute.Default()}, mode: "vulkan_identity"})
		if in.Source != nil {
			defer in.Source.Close()
		}
		if !in.StreamedExperts || in.StreamedExpertDeviceRingBytes != 1<<20 || in.StreamedDenseWorkingSetBytes != 1<<20 || !in.VulkanMixedQ4K || !in.StreamedDenseQ4K || in.Headroom != 0.15 {
			t.Fatal("benchmark preflight omitted the bounded streamed tuple or standard headroom")
		}
		if pf := ggufload.BuildModelPreflight(in); pf.Verdict != ggufload.PreflightReady || pf.EstHostResidentBytes <= 0 || pf.EstDeviceResidentBytes <= 0 {
			t.Fatalf("benchmark did not select admitted streamed preflight: verdict=%s reason=%s", pf.Verdict, pf.Reason)
		}
		m, _, err := loadModelContext(context.Background(), f, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer m.CloseWeights()
		stats := m.ExpertCheckpointStats()
		if !stats.Enabled || stats.BudgetBytes != 0 || stats.ResidentBytes != 0 || stats.Tensors != 9 {
			t.Fatalf("loader did not stream every expert with zero host retention: %+v", stats)
		}
		s := newBenchSession(m, f, compute.Default())
		if s.ExpertRingBytes != 1<<20 {
			t.Fatalf("session ring=%d want explicit bound", s.ExpertRingBytes)
		}
		if _, err := s.VerifyTokenLineage(nil); err != nil {
			t.Fatalf("new streamed session has prefix state: %v", err)
		}
		s.Close()
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		panicked := false
		func() { defer func() { panicked = recover() != nil }(); _ = newBenchSession(m, f, compute.Default()) }()
		if !panicked {
			t.Fatal("closed model accepted another benchmark session")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		cancelled, _, err := loadModelContext(ctx, f, nil)
		if cancelled != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled load returned model=%v error=%v", cancelled != nil, err)
		}
	})
	t.Run("first_smoke_prefill", func(t *testing.T) {
		ref, ok := compute.Lookup("cpu-ref")
		if !ok {
			t.Fatal("cpu-ref missing")
		}
		m := model.NewSynthetic(model.Config{HiddenSize: 16, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 4, IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1})
		defer m.CloseWeights()
		f := testCompleteBenchFlags()
		*f.backendName, *f.decodePrompt, *f.smokeDecodeSteps = "cpu-ref", 3, 1
		oracle := newBenchSession(m, f, ref)
		logits := oracle.Prefill(lcgIDs(3, m.Cfg.VocabSize))
		buf := make([]byte, 4*len(logits))
		for i, v := range logits {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
		}
		digest := sha256.Sum256(buf)
		expected := hex.EncodeToString(digest[:])
		oracle.Close()
		for _, mode := range []string{"", "warm_kv", "empty", "nonfinite", "refused_read", "closed_weights"} {
			t.Run("mode_"+mode, func(t *testing.T) {
				be := &benchColdKVBackend{benchObservedBackend: benchObservedBackend{Backend: ref}, mode: mode}
				compute.Register(be)
				t.Cleanup(func() { compute.Register(ref) })
				f := testCompleteBenchFlags()
				*f.backendName, *f.decodePrompt, *f.smokeDecodeSteps = "cpu-ref", 3, 1
				*f.out = filepath.Join(t.TempDir(), "smoke.json")
				exitCode := 0
				f.processExit = func(code int) { exitCode = code }
				target := m
				if mode == "closed_weights" {
					target = model.NewSynthetic(m.Cfg)
					if err := target.CloseWeights(); err != nil {
						t.Fatal(err)
					}
				}
				runSmoke(f, target, "fixture", 0, target.Cfg.VocabSize)
				var receipt map[string]any
				raw, err := os.ReadFile(*f.out)
				if err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(raw, &receipt); err != nil {
					t.Fatal(err)
				}
				if mode != "" {
					if receipt["smoke_status"] == smokeStatusOK || exitCode != 1 {
						t.Fatalf("invalid %s produced success: %v", mode, receipt)
					}
					if _, present := receipt["smoke_prefill_kv_reused_tokens"]; present {
						t.Fatal("failed cold proof emitted measured zero reuse")
					}
					return
				}
				if receipt["smoke_status"] != smokeStatusOK || exitCode != 0 || be.newKV != 1 {
					t.Fatalf("smoke execution status=%v exit=%d session constructions=%d", receipt["smoke_status"], exitCode, be.newKV)
				}
				if receipt["smoke_prefill_tokens"] != float64(3) || receipt["smoke_prefill_kv_reused_tokens"] != float64(0) || receipt["smoke_prefill_pass"] != "first_after_load" {
					t.Fatalf("fresh prefill evidence: %v", receipt)
				}
				if receipt["smoke_prefill_logits_sha256"] != expected {
					t.Fatal("receipt did not hash initial Float32LE logits before decode")
				}
				ms, ok := receipt["smoke_prefill_ms"].(float64)
				if !ok || ms <= 0 {
					t.Fatal("prefill wall absent/nonpositive")
				}
				rate, ok := receipt["smoke_prefill_tok_per_sec"].(float64)
				if !ok || rate <= 0 || math.Abs(rate-3000/ms) > 1e-9*rate {
					t.Fatal("prefill count/time/rate disagree")
				}
				if receipt["smoke_prefill_os_cache"] != "inherited_uncontrolled" {
					t.Fatal("OS cache regime undisclosed")
				}
			})
		}
	})
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
		Steps   int                       `json:"smoke_decode_steps"`
		Config  struct {
			NumLayers int `json:"num_hidden_layers"`
			HeadDim   int `json:"head_dim"`
		} `json:"model_config"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report.Backend.Selected != "cpu-ref" || report.Status != smokeStatusOK || report.Steps != 16 {
		t.Fatalf("smoke execution identity: %+v", report)
	}
	if report.Config.NumLayers != m.Cfg.NumLayers || report.Config.HeadDim != m.Cfg.HeadDim {
		t.Fatalf("smoke model geometry = layers %d head_dim %d, want source config layers %d head_dim %d", report.Config.NumLayers, report.Config.HeadDim, m.Cfg.NumLayers, m.Cfg.HeadDim)
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

// tierOverrideBackend presents a chosen Tier() over the cpu-ref oracle so preflightInputFor's
// integrated/discrete single-residency resolution can be witnessed without a Vulkan device.
type tierOverrideBackend struct {
	compute.Backend
	tier string
}

func (b *tierOverrideBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (b *tierOverrideBackend) Tier() string { return b.tier }

// TestPreflightInputForResolvesSingleResidencyOnIntegratedTier pins the caller-side fail-closed
// resolution: SingleResidencyQ4K reaches the estimator ONLY when the -single-residency-q4k knob
// (defaulting to FAK_Q4K_FREE_CPU=1) is on AND the Vulkan backend's physical-device tier is a
// unified-memory "integrated:" one. A discrete tier or a knob-off run never arms the release.
func TestPreflightInputForResolvesSingleResidencyOnIntegratedTier(t *testing.T) {
	f := testCompleteBenchFlags()
	*f.gguf, *f.q4k, *f.backendName = benchMixedQuantGGUF(t), true, "vulkan"
	*f.streamQ4K = false

	t.Setenv("FAK_Q4K_FREE_CPU", "1")
	*f.singleResidencyQ4K = true
	integrated := &tierOverrideBackend{Backend: compute.Default(), tier: "integrated:strix-halo"}
	in := preflightInputFor(f, integrated)
	if in.Source != nil {
		defer in.Source.Close()
	}
	if !in.VulkanMixedQ4K || !in.SingleResidencyQ4K {
		t.Fatalf("integrated tier with the knob on: vulkanMixed=%v singleResidency=%v, want both true", in.VulkanMixedQ4K, in.SingleResidencyQ4K)
	}

	*f.singleResidencyQ4K = false
	knobOff := preflightInputFor(f, integrated)
	if knobOff.Source != nil {
		defer knobOff.Source.Close()
	}
	if knobOff.SingleResidencyQ4K {
		t.Fatal("knob off still armed the single-residency release")
	}

	*f.singleResidencyQ4K = true
	discrete := preflightInputFor(f, &tierOverrideBackend{Backend: compute.Default(), tier: "discrete:nvidia-4090"})
	if discrete.Source != nil {
		defer discrete.Source.Close()
	}
	if discrete.SingleResidencyQ4K {
		t.Fatal("discrete tier armed the single-residency release")
	}
}

type benchPrefillObservationBackend struct {
	benchObservedBackend
	mode          string
	snapshotCalls []int
}

// fak-test:runtime fast est=2s lane=default
func TestModelbenchSmokeCheckpointSurvivesBlockedPrefill(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	ref, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU reference backend missing")
	}
	m := model.NewSynthetic(model.Config{HiddenSize: 16, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 4, IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1})
	defer m.CloseWeights()
	be := &benchPrefillObservationBackend{benchObservedBackend: benchObservedBackend{Backend: ref}, mode: "observed"}
	compute.Register(be)
	t.Cleanup(func() { compute.Register(ref) })

	f := testCompleteBenchFlags()
	*f.backendName, *f.decodePrompt, *f.smokeDecodeSteps = "cpu-ref", 3, 1
	*f.out = filepath.Join(t.TempDir(), "smoke.json")
	exitCode := 0
	f.processExit = func(code int) { exitCode = code }

	originalPrefill, originalWritten := smokeSessionPrefill, smokeCheckpointWritten
	t.Cleanup(func() {
		smokeSessionPrefill = originalPrefill
		smokeCheckpointWritten = originalWritten
	})
	entered := make(chan struct{})
	releasePrefill := make(chan struct{})
	smokeSessionPrefill = func(s *model.Session, ids []int) []float32 {
		close(entered)
		<-releasePrefill
		return originalPrefill(s, ids)
	}
	phases := make(chan string, 3)
	releaseCheckpoint := make(chan struct{}, 2)
	smokeCheckpointWritten = func(phase string) {
		phases <- phase
		if phase != "prefill_started" {
			<-releaseCheckpoint
		}
	}

	done := make(chan struct{})
	go func() {
		runSmoke(f, m, "fixture", 12345, m.Cfg.VocabSize)
		close(done)
	}()

	wait := func(ch <-chan string, want string) {
		t.Helper()
		select {
		case got := <-ch:
			if got != want {
				t.Fatalf("checkpoint phase=%q want=%q", got, want)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for checkpoint %q", want)
		}
	}
	readCheckpoint := func(wantPhase string, wantSequence float64) map[string]any {
		t.Helper()
		raw, err := os.ReadFile(*f.out)
		if err != nil {
			t.Fatal(err)
		}
		if len(raw) > smokeCheckpointMaxBytes {
			t.Fatalf("checkpoint bytes=%d exceed cap=%d", len(raw), smokeCheckpointMaxBytes)
		}
		var checkpoint map[string]any
		if err := json.Unmarshal(raw, &checkpoint); err != nil {
			t.Fatal(err)
		}
		if checkpoint["schema"] != smokeCheckpointSchema || checkpoint["phase"] != wantPhase || checkpoint["sequence"] != wantSequence || checkpoint["partial"] != true || checkpoint["qualifying"] != false || checkpoint["exclusive_request_attribution"] != false {
			t.Fatalf("checkpoint contract mismatch: %+v", checkpoint)
		}
		return checkpoint
	}

	wait(phases, "prefill_started")
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("Prefill did not enter")
	}
	started := readCheckpoint("prefill_started", 1)
	startedBackend := started["backend_execution"].(map[string]any)
	if startedBackend["status"] != "pending" || len(be.snapshotCalls) != 1 {
		t.Fatalf("blocked Prefill checkpoint=%+v snapshots=%v", startedBackend, be.snapshotCalls)
	}
	close(releasePrefill)

	wait(phases, "prefill_completed")
	completed := readCheckpoint("prefill_completed", 2)
	completedBackend := completed["backend_execution"].(map[string]any)
	if completedBackend["status"] != "observed" || len(be.snapshotCalls) != 2 {
		t.Fatalf("completed checkpoint=%+v snapshots=%v", completedBackend, be.snapshotCalls)
	}
	releaseCheckpoint <- struct{}{}

	wait(phases, "decode_started")
	decode := readCheckpoint("decode_started", 3)
	if decode["backend_execution"].(map[string]any)["status"] != "observed" {
		t.Fatal("decode boundary lost completed backend observation")
	}
	releaseCheckpoint <- struct{}{}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("smoke did not finish")
	}

	raw, err := os.ReadFile(*f.out)
	if err != nil {
		t.Fatal(err)
	}
	var report map[string]any
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if exitCode != 0 || report["smoke_status"] != smokeStatusOK || report["schema"] != nil || report["phase"] != nil || report["partial"] != nil {
		t.Fatalf("ordinary final report changed: exit=%d report=%+v", exitCode, report)
	}
}

func (b *benchPrefillObservationBackend) BackendExecutionSnapshot() (compute.BackendExecutionSnapshot, error) {
	b.snapshotCalls = append(b.snapshotCalls, b.matmuls)
	if b.mode == "error" {
		return compute.BackendExecutionSnapshot{}, errors.New("fixture observation unavailable")
	}
	n := uint64(b.matmuls)
	s := compute.BackendExecutionSnapshot{
		Identity:                 compute.BackendRuntimeIdentity{Backend: b.Name(), Device: "fixture-device", Driver: "fixture-driver", Runtime: "fixture-runtime"},
		TransferCountersObserved: b.mode != "transfer_unobserved" && b.mode != "zero_profile",
		Counters: compute.BackendCounterSnapshot{
			ComputeDispatches: 1000 + n, OtherDispatches: 1000 + n, DispatchSubmits: 2000 + n,
			H2DCount: 3000 + n, H2DBytes: 12000 + 4*n,
			Q4KStageCalls: 4000 + n, Q4KStageBytes: 16000 + 4*n,
			TensorHomeHits: 5000 + n, TensorHomeAdmissions: 6000 + n,
			TensorHomeBypasses: 7000 + n, TensorHomeCopiedBytes: 28000 + 4*n,
		},
	}
	if b.mode == "zero_profile" {
		s.Counters.ComputeDispatches, s.Counters.OtherDispatches, s.Counters.DispatchSubmits = 0, 0, 0
	}
	if b.mode == "zero_compute" {
		s.Counters.ComputeDispatches, s.Counters.OtherDispatches = 1000, 1000
	}
	if b.mode == "zero_submits" {
		s.Counters.DispatchSubmits = 2000
	}
	if len(b.snapshotCalls) == 2 {
		if b.mode == "after_error" {
			return compute.BackendExecutionSnapshot{}, errors.New("fixture closing observation unavailable")
		}
		if b.mode == "reset" {
			s.Counters.ComputeDispatches = 999
		}
		if b.mode == "identity" {
			s.Identity.Device = "different-device"
		}
	}
	return s, nil
}

// fak-test:runtime medium est=2s
func TestModelbenchInitialPrefillObservation(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	ref, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU reference backend missing")
	}
	m := model.NewSynthetic(model.Config{HiddenSize: 16, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 4, IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1})
	defer m.CloseWeights()
	oracleBackend := &benchObservedBackend{Backend: ref}
	oracleFlags := testCompleteBenchFlags()
	*oracleFlags.decodePrompt = 3
	oracle := newBenchSession(m, oracleFlags, oracleBackend)
	_ = oracle.Prefill(lcgIDs(3, m.Cfg.VocabSize))
	prefillCalls := oracleBackend.matmuls
	oracle.Close()
	if prefillCalls == 0 {
		t.Fatal("oracle fixture bypassed actual HAL")
	}
	for _, mode := range []string{"observed", "reset", "identity", "error", "after_error", "unsupported", "zero_profile", "zero_compute", "zero_submits", "transfer_unobserved"} {
		t.Run(mode, func(t *testing.T) {
			be := &benchPrefillObservationBackend{benchObservedBackend: benchObservedBackend{Backend: ref}, mode: mode}
			var selected compute.Backend = be
			if mode == "unsupported" {
				selected = &benchObservedBackend{Backend: ref}
			}
			compute.Register(selected)
			t.Cleanup(func() { compute.Register(ref) })
			f := testCompleteBenchFlags()
			*f.backendName, *f.decodePrompt, *f.smokeDecodeSteps = "cpu-ref", 3, 1
			*f.out = filepath.Join(t.TempDir(), "observation.json")
			exitCode := 0
			f.processExit = func(code int) { exitCode = code }
			runSmoke(f, m, "fixture", 12345, m.Cfg.VocabSize)
			raw, err := os.ReadFile(*f.out)
			if err != nil {
				t.Fatal(err)
			}
			var report map[string]any
			if err := json.Unmarshal(raw, &report); err != nil {
				t.Fatal(err)
			}
			if exitCode != 0 || report["smoke_status"] != smokeStatusOK {
				t.Fatal("optional observation broke successful smoke")
			}
			observation, ok := report["smoke_prefill_backend_execution"].(map[string]any)
			if !ok {
				t.Fatal("actual first Prefill emitted no backend observation status")
			}
			wantStatus := "observed"
			if mode == "unsupported" {
				wantStatus = "unsupported"
			}
			if mode == "reset" || mode == "identity" || mode == "error" || mode == "after_error" {
				wantStatus = "error"
			}
			if observation["status"] != wantStatus {
				t.Fatalf("observation status=%v want=%s", observation["status"], wantStatus)
			}
			if _, present := observation["v41_model_attribution"]; present {
				t.Fatal("non-V4.1 smoke published V4.1 model attribution")
			}
			if wantStatus != "observed" {
				if _, present := observation["counters"]; present {
					t.Fatal("unavailable observation published numeric counters")
				}
				return
			}
			if len(be.snapshotCalls) != 2 || be.snapshotCalls[0] != 0 || be.snapshotCalls[1] != prefillCalls || be.matmuls <= prefillCalls {
				t.Fatalf("snapshots failed to bracket initial Prefill before decode: positions=%v prefill=%d final=%d", be.snapshotCalls, prefillCalls, be.matmuls)
			}
			counters, ok := observation["counters"].(map[string]any)
			if !ok {
				t.Fatal("observed execution missing counters")
			}
			identity, ok := observation["identity"].(map[string]any)
			if !ok || identity["backend"] != "cpu-ref" || identity["device"] != "fixture-device" || identity["driver"] != "fixture-driver" || identity["runtime"] != "fixture-runtime" {
				t.Fatal("observation lost the actual backend identity")
			}
			for key, value := range map[string]float64{"q4k_stage_calls": float64(prefillCalls), "q4k_stage_bytes": float64(4 * prefillCalls), "fallbacks": 0, "tensor_home_hits": float64(prefillCalls), "tensor_home_admissions": float64(prefillCalls), "tensor_home_bypasses": float64(prefillCalls), "tensor_home_copied_bytes": float64(4 * prefillCalls)} {
				if counters[key] != value {
					t.Fatalf("%s=%v want prefill-only delta=%v", key, counters[key], value)
				}
			}
			dispatchKnown := mode != "zero_profile" && mode != "zero_compute" && mode != "zero_submits"
			if observation["dispatch_observed"] != dispatchKnown {
				t.Fatal("dispatch availability was fabricated")
			}
			for key, value := range map[string]float64{"compute_dispatches": float64(prefillCalls), "q4k_matmul_dispatches": 0, "other_dispatches": float64(prefillCalls), "dispatch_submits": float64(prefillCalls)} {
				got, present := counters[key]
				if !dispatchKnown {
					if present {
						t.Fatalf("unknown dispatch %s published numeric zero", key)
					}
					continue
				}
				if !present || got != value {
					t.Fatalf("%s=%v want prefill-only delta=%v", key, got, value)
				}
			}
			transferKnown := mode != "zero_profile" && mode != "transfer_unobserved"
			if observation["transfer_counters_observed"] != transferKnown {
				t.Fatal("transfer availability was fabricated")
			}
			for key, value := range map[string]float64{"h2d_count": float64(prefillCalls), "h2d_bytes": float64(4 * prefillCalls), "d2h_count": 0, "d2h_bytes": 0, "d2d_copies": 0, "d2d_bytes": 0} {
				got, present := counters[key]
				if !transferKnown {
					if present {
						t.Fatalf("unknown transfer %s published numeric zero", key)
					}
					continue
				}
				if !present || got != value {
					t.Fatalf("%s=%v want prefill-only delta=%v", key, got, value)
				}
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSmokePrefillV41AttributionRequiresObservedBackend(t *testing.T) {
	ref, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU reference backend missing")
	}
	m := model.NewSynthetic(model.Config{ModelType: "deepseek41", HiddenSize: 16, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 4, IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1})
	defer m.CloseWeights()
	for _, tc := range []struct {
		name, status string
		backend      compute.Backend
	}{
		{name: "unsupported", status: "unsupported", backend: &benchObservedBackend{Backend: ref}},
		{name: "error", status: "error", backend: &benchPrefillObservationBackend{benchObservedBackend: benchObservedBackend{Backend: ref}, mode: "error"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			observation := beginSmokePrefillObservation(tc.backend, m).end(tc.backend, m)
			if observation["status"] != tc.status {
				t.Fatalf("status=%v want=%s", observation["status"], tc.status)
			}
			if _, present := observation["v41_model_attribution"]; present {
				t.Fatal("unavailable HAL evidence published V4.1 model attribution")
			}
		})
	}
}

// fak-test:justify why=integration when=changed:cmd/modelbench/**,internal/model/v41_**
// fak-test:runtime medium est=8s lane=default
func TestModelbenchV41InitialPrefillAttribution(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	path := engProjGGUF(t)
	m, err := func() (*model.Model, error) {
		budget := ggufload.MaxEagerF32Bytes
		ggufload.MaxEagerF32Bytes = 4 << 20
		defer func() { ggufload.MaxEagerF32Bytes = budget }()
		return ggufload.LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	}()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	if !m.Cfg.IsDeepSeekV41() {
		t.Fatal("real GGUF loader lost V4.1 identity")
	}
	ref, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU reference backend missing")
	}
	be := &benchPrefillObservationBackend{benchObservedBackend: benchObservedBackend{Backend: ref}, mode: "observed"}
	compute.Register(be)
	defer compute.Register(ref)
	f := testCompleteBenchFlags()
	*f.backendName, *f.decodePrompt, *f.smokeDecodeSteps = "cpu-ref", 512, 1
	*f.out = filepath.Join(t.TempDir(), "v41-attribution.json")
	prime := newBenchSession(m, f, be)
	_ = prime.Prefill([]int{1, 2})
	prime.Close()
	primed := m.V41ExpertFaultAttribution()
	if primed.Prefill.Tokens == 0 {
		t.Fatal("real priming Prefill did not establish cumulative model attribution")
	}
	exitCode := 0
	f.processExit = func(code int) { exitCode = code }
	runSmoke(f, m, "v41-fixture", 12345, m.Cfg.VocabSize)
	raw, err := os.ReadFile(*f.out)
	if err != nil {
		t.Fatal(err)
	}
	if receipt := os.Getenv("FAK_MODELBENCH_V41_FIXTURE_OUT"); receipt != "" {
		if err := os.WriteFile(receipt, raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("real V4.1 smoke report: %d bytes (16 KiB cap)", len(raw))
	if len(raw) > 16<<10 {
		t.Fatalf("real V4.1 smoke report is %d bytes, exceeds appliance 16 KiB transport cap", len(raw))
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, raw); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, compact.Bytes()) {
		t.Fatal("smoke report spends the transport budget on JSON whitespace")
	}
	var report struct {
		Lineage     benchcli.Lineage           `json:"lineage"`
		Artifact    benchcli.BenchmarkArtifact `json:"benchmark_artifact"`
		Status      string                     `json:"smoke_status"`
		Observation struct {
			Status      string                    `json:"status"`
			Counters    map[string]any            `json:"counters"`
			Attribution *smokeV41ModelAttribution `json:"v41_model_attribution"`
		} `json:"smoke_prefill_backend_execution"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report.Lineage.Schema != benchcli.LineageSchema || report.Artifact.Schema != benchcli.BenchmarkArtifactSchema || report.Artifact.FAKCommit != report.Lineage.GitCommit {
		t.Fatal("compact smoke report lost its lineage or benchmark artifact")
	}
	if exitCode != 0 || report.Status != smokeStatusOK || report.Observation.Status != "observed" || report.Observation.Counters == nil {
		t.Fatalf("optional model observation changed smoke/backend status: exit=%d report=%+v", exitCode, report)
	}
	a := report.Observation.Attribution
	if a == nil || a.Schema != smokeV41ModelAttributionSchema || a.Scope != "model_lifetime_snapshots" || a.Phase != "prefill" || !a.Partial || a.ExclusiveRequestAttribution {
		t.Fatalf("V4.1 model attribution contract=%+v", a)
	}
	if !reflect.DeepEqual(a.Before, primed.Prefill) {
		t.Fatal("before snapshot did not freeze the primed model-lifetime ledger immediately before initial Prefill")
	}
	beforeHead := a.Before.HeadProjectionDeviceRows + a.Before.HeadProjectionHostRows
	afterHead := a.After.HeadProjectionDeviceRows + a.After.HeadProjectionHostRows
	beforeGrouped := a.Before.GroupedOutputDeviceRows + a.Before.GroupedOutputHostRows
	afterGrouped := a.After.GroupedOutputDeviceRows + a.After.GroupedOutputHostRows
	if a.After.Tokens <= a.Before.Tokens || afterHead <= beforeHead || afterGrouped <= beforeGrouped {
		t.Fatalf("real initial Prefill did not change tokens/head/grouped ledgers: before=%+v after=%+v", a.Before, a.After)
	}
	if bytes.Contains(raw, []byte(`"decode":`)) {
		t.Fatal("prefill-only model attribution serialized a decode ledger")
	}
	final := m.V41ExpertFaultAttribution()
	if !reflect.DeepEqual(a.After, final.Prefill) || final.Decode.Tokens <= primed.Decode.Tokens {
		t.Fatal("real smoke decode did not occur outside the captured Prefill interval")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestModelbenchReportFormattingAndMarshalFailure(t *testing.T) {
	t.Setenv("FAK_BENCH_UTC", "2026-10-17T00:00:00Z")
	t.Setenv("FAK_BENCH_COMMIT", "fixture-report-commit")
	dir := t.TempDir()
	stderr, err := os.Create(filepath.Join(dir, "stderr.txt"))
	if err != nil {
		t.Fatal(err)
	}
	originalStderr := os.Stderr
	os.Stderr = stderr
	t.Cleanup(func() {
		os.Stderr = originalStderr
		if err := stderr.Close(); err != nil {
			t.Error(err)
		}
	})
	f := testCompleteBenchFlags()
	*f.out = filepath.Join(dir, "report.json")
	exitCalls, exitCode := 0, 0
	f.processExit = func(code int) {
		exitCalls++
		exitCode = code
	}
	report := map[string]any{"model": "fixture", "results": map[string]any{"tokens": 1}}
	want, err := benchcli.MarshalReport(report)
	if err != nil {
		t.Fatal(err)
	}
	writeReport(f, report)
	raw, err := os.ReadFile(*f.out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, want) || !bytes.Contains(raw, []byte("\n")) {
		t.Fatal("non-smoke report changed its stamped, indented JSON format")
	}
	writeReport(f, map[string]any{"smoke_status": smokeStatusOK, "invalid": math.NaN()})
	if exitCalls != 1 || exitCode != 1 {
		t.Fatalf("marshal failure exit: calls=%d code=%d, want one exit with code 1", exitCalls, exitCode)
	}
	after, err := os.ReadFile(*f.out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, raw) {
		t.Fatal("marshal failure replaced the existing report")
	}
	diagnostic, err := os.ReadFile(stderr.Name())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(diagnostic, []byte("marshal report: json: unsupported value: NaN")) {
		t.Fatalf("marshal failure diagnostic missing: %s", diagnostic)
	}
}
