package ggufload

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type w3FixtureOptions struct {
	mlpType       TensorType
	omit          string
	attentionType TensorType
	extraNextNIQ3 bool
}

type w3FixtureTensor struct {
	name    string
	dims    []uint64
	typ     TensorType
	offset  uint64
	payload []byte
}

func w3CanonicalName(layer int, projection string) string {
	return fmt.Sprintf("model.layers.%d.mlp.%s_proj.weight", layer, projection)
}

func w3SourceName(layer int, projection string) string {
	return fmt.Sprintf("blk.%d.ffn_%s.weight", layer, projection)
}

func w3FixturePayload(t *testing.T, name string, dims []uint64, typ TensorType, marker byte) []byte {
	t.Helper()
	n, err := tensorPayloadBytes(TensorInfo{Name: name, Dims: dims, Type: typ})
	if err != nil {
		t.Fatalf("payload geometry for %s: %v", name, err)
	}
	payload := bytes.Repeat([]byte{marker}, int(n))
	var blockBytes int
	switch typ {
	case TensorIQ3_XXS:
		blockBytes = blockIQ3XXSBytes
	case TensorQ4_K:
		blockBytes = blockQ4KBytes
	}
	// Zero the f16 scale(s) in every block so dequantization is finite and produces zeros.
	for off := 0; blockBytes > 0 && off < len(payload); off += blockBytes {
		payload[off] = 0
		payload[off+1] = 0
		if typ == TensorQ4_K {
			payload[off+2] = 0
			payload[off+3] = 0
		}
	}
	return payload
}

func w3MLPGGUFFixture(t *testing.T, opts w3FixtureOptions) ([]byte, map[string][]byte) {
	t.Helper()
	const (
		H      = 256
		I      = 512
		layers = 4
		align  = 32
	)
	if opts.mlpType == 0 {
		opts.mlpType = TensorIQ3_XXS
	}
	if opts.attentionType == 0 {
		opts.attentionType = TensorQ4_K
	}
	var tensors []w3FixtureTensor
	expectedRaw := map[string][]byte{}
	add := func(name string, dims []uint64, typ TensorType) {
		if name == opts.omit {
			return
		}
		marker := byte(1 + len(tensors))
		payload := w3FixturePayload(t, name, dims, typ, marker)
		tensors = append(tensors, w3FixtureTensor{name: name, dims: dims, typ: typ, payload: payload})
		if typ == TensorIQ3_XXS && strings.Contains(name, ".ffn_") {
			canon, ok := CanonicalTensorNameArch(name, "qwen35")
			if !ok {
				t.Fatalf("canonical mapping missing for %s", name)
			}
			expectedRaw[canon] = append([]byte(nil), payload...)
		}
	}
	for layer := 0; layer < layers; layer++ {
		add(w3SourceName(layer, "gate"), []uint64{H, I}, opts.mlpType)
		add(w3SourceName(layer, "up"), []uint64{H, I}, opts.mlpType)
		add(w3SourceName(layer, "down"), []uint64{I, H}, opts.mlpType)
	}
	// Full-attention q is normalize-sensitive, so it remains on the established Q8 route and
	// also ensures QuantBuilder.Build has a q8 tensor in the all-resident W3 case.
	add("blk.3.attn_q.weight", []uint64{H, 2 * H}, opts.attentionType)
	if opts.extraNextNIQ3 {
		add("blk.4.nextn.eh_proj.weight", []uint64{H}, TensorIQ3_XXS)
	}

	var off uint64
	for i := range tensors {
		tensors[i].offset = off
		off = alignOffset(off+uint64(len(tensors[i].payload)), align)
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 10)
	writeKVUint32(&b, "general.alignment", align)
	writeKVString(&b, "general.architecture", "qwen35")
	writeKVUint64(&b, "qwen35.context_length", 32)
	writeKVUint64(&b, "qwen35.embedding_length", H)
	writeKVUint64(&b, "qwen35.block_count", layers)
	writeKVUint64(&b, "qwen35.feed_forward_length", I)
	writeKVUint64(&b, "qwen35.attention.head_count", 1)
	writeKVUint64(&b, "qwen35.attention.head_count_kv", 1)
	writeKVFloat32(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVUint64(&b, "qwen35.full_attention_interval", 4)
	for _, tensor := range tensors {
		writeTensorInfoForTest(&b, tensor.name, tensor.dims, tensor.typ, tensor.offset)
	}
	padToAlignment(&b, align)
	dataStart := b.Len()
	for _, tensor := range tensors {
		padToLen(&b, dataStart+int(tensor.offset))
		b.Write(tensor.payload)
	}
	return b.Bytes(), expectedRaw
}

func writeW3Fixture(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "w3-mlp.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestLoadW3MLPDefaultOffPreservesResidentIQ3(t *testing.T) {
	oldW3, hadW3 := os.LookupEnv("FAK_W3_MLP")
	if err := os.Unsetenv("FAK_W3_MLP"); err != nil {
		t.Fatalf("unset FAK_W3_MLP: %v", err)
	}
	t.Cleanup(func() {
		if hadW3 {
			_ = os.Setenv("FAK_W3_MLP", oldW3)
			return
		}
		_ = os.Unsetenv("FAK_W3_MLP")
	})
	data, expectedRaw := w3MLPGGUFFixture(t, w3FixtureOptions{mlpType: TensorIQ3_XXS, attentionType: TensorQ4_K})
	m, err := LoadModelQ4K(writeW3Fixture(t, data))
	if err != nil {
		t.Fatalf("LoadModelQ4K: %v", err)
	}
	if got := m.ResidentW3MLPCount(); got != 0 {
		t.Fatalf("ResidentW3MLPCount()=%d with default-off flag, want 0", got)
	}
	for name, wantRaw := range expectedRaw {
		if m.HasQ8(name) || !m.HasKQuant(name) || m.HasResidentW3MLP(name) {
			t.Errorf("%s route: q8=%v kquant=%v w3=%v, want untagged resident IQ3", name, m.HasQ8(name), m.HasKQuant(name), m.HasResidentW3MLP(name))
			continue
		}
		gotRaw, ok := m.KQuantRaw(name)
		if !ok || !bytes.Equal(gotRaw, wantRaw) {
			t.Errorf("%s raw IQ3 payload changed during default-off load", name)
		}
	}
	if !m.HasQ8("model.layers.3.self_attn.q_proj.weight") {
		t.Fatal("normalize-sensitive attention route changed: q_proj is not Q8")
	}
}

func TestLoadW3MLPFlagOnRoutesExactlyDenseMLP(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "1")
	data, expectedRaw := w3MLPGGUFFixture(t, w3FixtureOptions{mlpType: TensorIQ3_XXS, attentionType: TensorQ4_K})
	m, err := LoadModelQ4K(writeW3Fixture(t, data))
	if err != nil {
		t.Fatalf("LoadModelQ4K: %v", err)
	}
	if err := m.ValidateResidentW3MLP(); err != nil {
		t.Fatalf("ValidateResidentW3MLP: %v", err)
	}
	if got := m.ResidentW3MLPCount(); got != 12 {
		t.Fatalf("ResidentW3MLPCount()=%d, want 12", got)
	}
	if got := m.KQuantCount(); got != 12 {
		t.Fatalf("KQuantCount()=%d, want 12 (only dense W3 MLP)", got)
	}
	for name, wantRaw := range expectedRaw {
		if !m.HasResidentW3MLP(name) || m.HasQ8(name) {
			t.Errorf("%s route: w3=%v q8=%v, want tagged W3 only", name, m.HasResidentW3MLP(name), m.HasQ8(name))
			continue
		}
		gotRaw, ok := m.KQuantRaw(name)
		if !ok || !bytes.Equal(gotRaw, wantRaw) {
			t.Errorf("%s raw IQ3 payload changed during load", name)
		}
	}
	if !m.HasQ8("model.layers.3.self_attn.q_proj.weight") {
		t.Fatal("W3 selection changed attention route: q_proj is not Q8")
	}
}

func TestLoadW3MLPRejectsPartialBand(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "true")
	missingSource := w3SourceName(3, "down")
	data, _ := w3MLPGGUFFixture(t, w3FixtureOptions{mlpType: TensorIQ3_XXS, attentionType: TensorQ4_K, omit: missingSource})
	_, err := LoadModelQ4K(writeW3Fixture(t, data))
	missingCanon := w3CanonicalName(3, "down")
	if err == nil || !strings.Contains(err.Error(), missingCanon) || !strings.Contains(err.Error(), "11/12") {
		t.Fatalf("partial-band error=%v, want %s and 11/12", err, missingCanon)
	}
}

func TestLoadW3MLPRejectsOrdinaryQ4Artifact(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "on")
	data, _ := w3MLPGGUFFixture(t, w3FixtureOptions{mlpType: TensorQ4_K, attentionType: TensorQ4_K})
	_, err := LoadModelQ4K(writeW3Fixture(t, data))
	if err == nil || !strings.Contains(err.Error(), w3CanonicalName(0, "gate")) || !strings.Contains(err.Error(), "0/12") {
		t.Fatalf("ordinary-Q4 error=%v, want first missing W3 tensor and 0/12", err)
	}
}

func TestLoadW3MLPRejectsNonMLPIQ3(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "1")
	data, _ := w3MLPGGUFFixture(t, w3FixtureOptions{mlpType: TensorIQ3_XXS, attentionType: TensorIQ3_XXS})
	_, err := LoadModelQ4K(writeW3Fixture(t, data))
	if err == nil || !strings.Contains(err.Error(), "model.layers.3.self_attn.q_proj.weight") || !strings.Contains(err.Error(), "outside dense MLP W3 band") {
		t.Fatalf("non-MLP IQ3 error=%v, want attention name and band refusal", err)
	}
}

func TestLoadW3MLPRejectsMTPSelection(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "1")
	data, _ := w3MLPGGUFFixture(t, w3FixtureOptions{mlpType: TensorIQ3_XXS, attentionType: TensorQ4_K, extraNextNIQ3: true})
	_, err := LoadModelQ4K(writeW3Fixture(t, data))
	if err == nil || !strings.Contains(err.Error(), "blk.4.nextn.eh_proj.weight") || !strings.Contains(err.Error(), "outside dense MLP W3 band") {
		t.Fatalf("MTP IQ3 error=%v, want source name and band refusal", err)
	}
}

func TestLoadW3MLPRejectsWrongArchitecture(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "1")
	const dim = 256
	payload := w3FixturePayload(t, "blk.0.ffn_gate.weight", []uint64{dim, dim}, TensorIQ3_XXS, 1)
	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 8)
	writeKVUint32(&b, "general.alignment", 32)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint64(&b, "llama.context_length", 32)
	writeKVUint64(&b, "llama.embedding_length", dim)
	writeKVUint64(&b, "llama.block_count", 1)
	writeKVUint64(&b, "llama.feed_forward_length", dim)
	writeKVUint64(&b, "llama.attention.head_count", 1)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-5)
	writeTensorInfoForTest(&b, "blk.0.ffn_gate.weight", []uint64{dim, dim}, TensorIQ3_XXS, 0)
	padToAlignment(&b, 32)
	b.Write(payload)
	_, err := LoadModelQ4K(writeW3Fixture(t, b.Bytes()))
	if err == nil || !strings.Contains(err.Error(), "dense Qwen3.5-family hybrid") {
		t.Fatalf("wrong-architecture error=%v, want dense hybrid refusal", err)
	}
}
