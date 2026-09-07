package ggufload

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// This synthetic llama fixture isolates quant dispatch from Qwen's architecture.
// Its matrices cover all 14 quantized encodings observed in the pinned UD header;
// F32 embeddings/norms complete the same 15-type inventory. It is not a model-quality
// witness. Each 512-wide row spans multiple blocks, with varying codes and scales.
func udMixedForwardFixture(t *testing.T) (string, []iq12LoaderFixtureTensor) {
	t.Helper()
	const dim, layers, vocab = 512, 3, 4
	types := []TensorType{TensorQ3_K, TensorIQ3_S, TensorIQ2_XXS, TensorIQ2_XS, TensorIQ1_S,
		TensorIQ2_S, TensorIQ1_M, TensorIQ3_XXS, TensorQ2_K, TensorIQ4_XS,
		TensorQ4_K, TensorQ5_K, TensorQ6_K, TensorQ8_0, TensorQ3_K}
	var tensors []iq12LoaderFixtureTensor
	add := func(name string, dims []uint64, typ TensorType, norm bool) {
		info := TensorInfo{Name: name, Dims: dims, Type: typ}
		n, err := tensorPayloadBytes(info)
		if err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, int(n))
		if typ == TensorF32 {
			for i := 0; i < len(raw)/4; i++ {
				v := float32((i*7+len(tensors)*3)%19-9) / 128
				if norm {
					v = 1
				}
				binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
			}
		} else {
			blockN, err := tensorPayloadBytes(TensorInfo{Name: name, Dims: []uint64{256}, Type: typ})
			if err != nil {
				t.Fatal(err)
			}
			blockBytes := int(blockN)
			if typ == TensorQ8_0 {
				blockBytes = blockQ8_0Bytes
			}
			for off := 0; off < len(raw); off += blockBytes {
				block := raw[off : off+blockBytes]
				for i := range block {
					block[i] = byte(i*37 + off/blockBytes*13 + len(tensors)*11)
				}
				// Keep every format's encoded f16 scale finite and small. IQ1_M
				// distributes that scale across the top nibbles of four u16 words.
				scale := uint16(0x0800 + (off/blockBytes)%4*0x0100)
				put := func(at int) { binary.LittleEndian.PutUint16(block[at:], scale) }
				switch typ {
				case TensorQ3_K, TensorQ6_K:
					put(blockBytes - 2)
				case TensorQ2_K:
					put(blockBytes - 4)
					put(blockBytes - 2)
				case TensorQ4_K, TensorQ5_K:
					put(0)
					put(2)
				case TensorIQ1_M:
					for j := 0; j < 4; j++ {
						at := 48 + 2*j
						v := binary.LittleEndian.Uint16(block[at:]) & 0x0fff
						binary.LittleEndian.PutUint16(block[at:], v|((scale>>uint(4*j))&15)<<12)
					}
				default:
					put(0)
				}
			}
		}
		tensors = append(tensors, iq12LoaderFixtureTensor{name: name, dims: dims, typ: typ, payload: raw})
	}
	add("token_embd.weight", []uint64{dim, vocab}, TensorF32, false)
	add("output_norm.weight", []uint64{dim}, TensorF32, true)
	add("output.weight", []uint64{dim, vocab}, TensorQ8_0, false)
	for layer := 0; layer < layers; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		add(prefix+"attn_norm.weight", []uint64{dim}, TensorF32, true)
		add(prefix+"ffn_norm.weight", []uint64{dim}, TensorF32, true)
		// Q/K keep the loader's existing normalization and Q8 conversion seam.
		add(prefix+"attn_q.weight", []uint64{dim, dim}, TensorF32, false)
		add(prefix+"attn_k.weight", []uint64{dim, dim}, TensorF32, false)
		for i, name := range []string{"attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"} {
			add(prefix+name+".weight", []uint64{dim, dim}, types[layer*5+i], false)
		}
	}
	var off uint64
	for i := range tensors {
		tensors[i].offset = off
		off = alignOffset(off+uint64(len(tensors[i].payload)), 32)
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 12)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint64(&b, "llama.context_length", 16)
	writeKVUint64(&b, "llama.embedding_length", dim)
	writeKVUint64(&b, "llama.block_count", layers)
	writeKVUint64(&b, "llama.feed_forward_length", dim)
	writeKVUint64(&b, "llama.rope.dimension_count", dim)
	writeKVUint64(&b, "llama.attention.head_count", 1)
	writeKVUint64(&b, "llama.attention.head_count_kv", 1)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVFloat32(&b, "llama.rope.freq_base", 10000)
	writeKVStringArray(&b, "tokenizer.ggml.tokens", []string{"a", "b", "c", "d"})
	for _, tn := range tensors {
		writeTensorInfoForTest(&b, tn.name, tn.dims, tn.typ, tn.offset)
	}
	padToAlignment(&b, 32)
	start := b.Len()
	for _, tn := range tensors {
		padToLen(&b, start+int(tn.offset))
		b.Write(tn.payload)
	}
	path := filepath.Join(t.TempDir(), "ud-mixed-native.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path, tensors
}

func TestUDMixedGGUFNativeForwardParity(t *testing.T) {
	path, tensors := udMixedForwardFixture(t)
	m, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	for _, tn := range tensors {
		if tn.typ == TensorF32 {
			continue
		}
		name, ok := CanonicalTensorNameArch(tn.name, "llama")
		if !ok {
			t.Fatalf("missing canonical name for %s", tn.name)
		}
		raw, resident := m.KQuantRaw(name)
		if tn.typ == TensorQ4_K {
			raw, resident = m.Q4KRaw(name)
		}
		if !resident || m.HasQ8(name) || !bytes.Equal(raw, tn.payload) {
			t.Fatalf("%s (%s): resident=%v Q8=%v; source bytes must remain verbatim", name, tn.typ, resident, m.HasQ8(name))
		}
	}
	ref, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.CloseWeights()
	s := m.NewSession()
	defer s.Close()
	s.Q4K = true
	// Step may reuse the session's logit buffer; preserve the prefill observation.
	prefill := append([]float32(nil), s.Prefill([]int{0, 1})...)
	decode := s.Step(2)
	want := ref.Forward([]int{0, 1, 2}).Logits
	for phase, got := range [][]float32{prefill, decode} {
		if len(got) != 4 || len(want) != 3 {
			t.Fatalf("phase %d: invalid logits shape", phase)
		}
		var peak, actualPeak, maxDiff float64
		for i, v := range got {
			w := want[phase+1][i]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				t.Fatalf("phase %d logit %d is nonfinite: got=%g want=%g", phase, i, v, w)
			}
			peak = math.Max(peak, math.Abs(float64(w)))
			actualPeak = math.Max(actualPeak, math.Abs(float64(v)))
			maxDiff = math.Max(maxDiff, math.Abs(float64(v-w)))
			// Nonresident Q/K still take the existing Q8 conversion path. Bound
			// that rounding plus different native reduction orders against F32.
			if diff := math.Abs(float64(v - w)); diff > 1e-3+0.01*math.Abs(float64(w)) {
				t.Fatalf("phase %d logit %d: got=%g F32=%g diff=%g", phase, i, v, w, diff)
			}
		}
		if peak < 1e-5 || actualPeak < 1e-5 {
			t.Fatalf("phase %d: vacuous near-zero logits: native peak=%g reference peak=%g", phase, actualPeak, peak)
		}
		t.Logf("phase %d: native peak=%g F32 peak=%g max absolute error=%g", phase, actualPeak, peak, maxDiff)
	}
}
