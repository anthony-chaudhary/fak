package ggufload

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"testing"
)

const metadataHeavyTokenCount = 4096

type readCallCounter struct {
	r     io.Reader
	calls int
}

func (r *readCallCounter) Read(p []byte) (int, error) {
	r.calls++
	return r.r.Read(p)
}

func TestReadBuffersMetadataWithoutChangingHeaderParse(t *testing.T) {
	raw, dataOffset := metadataHeavyGGUFForBufferTest()

	want, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Read reference fixture: %v", err)
	}
	source := &readCallCounter{r: bytes.NewReader(raw)}
	got, err := Read(source)
	if err != nil {
		t.Fatalf("Read counted fixture: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buffered parse differs from reference:\n got: %#v\nwant: %#v", got, want)
	}
	if source.calls > 4 {
		t.Fatalf("underlying Read calls = %d, want <= 4 for %d metadata strings", source.calls, metadataHeavyTokenCount)
	}

	headerOnly, err := Read(bytes.NewReader(raw[:dataOffset]))
	if err != nil {
		t.Fatalf("Read header without tensor payload: %v", err)
	}
	if !reflect.DeepEqual(headerOnly, got) {
		t.Fatalf("header-only parse differs from payload-backed parse:\nheader: %#v\n  full: %#v", headerOnly, got)
	}
	if got.TensorDataOffset != int64(dataOffset) {
		t.Fatalf("TensorDataOffset = %d, want %d", got.TensorDataOffset, dataOffset)
	}
	if len(got.Tensors) != 2 || got.Tensors[0].FileOffset != int64(dataOffset) || got.Tensors[1].FileOffset != int64(dataOffset+64) {
		t.Fatalf("tensor offsets = %#v, want data offsets %d and %d", got.Tensors, dataOffset, dataOffset+64)
	}

	cfg, err := got.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "qwen2" || cfg.HiddenSize != 2048 || cfg.NumLayers != 24 || cfg.NumHeads != 16 || cfg.NumKVHeads != 4 || cfg.IntermediateSize != 5504 || cfg.VocabSize != metadataHeavyTokenCount || cfg.EOSTokenID != 7 {
		t.Fatalf("unexpected parsed config: %#v", cfg)
	}
}

func metadataHeavyGGUFForBufferTest() ([]byte, int) {
	tokens := make([]string, metadataHeavyTokenCount)
	for i := range tokens {
		tokens[i] = fmt.Sprintf("token-%04d", i)
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, 2, 12)
	writeKVString(&b, "general.architecture", "qwen2")
	writeKVString(&b, "general.name", "metadata-buffer-regression")
	writeKVUint32(&b, "general.alignment", 64)
	writeKVUint32(&b, "qwen2.embedding_length", 2048)
	writeKVUint32(&b, "qwen2.block_count", 24)
	writeKVUint32(&b, "qwen2.attention.head_count", 16)
	writeKVUint32(&b, "qwen2.attention.head_count_kv", 4)
	writeKVUint32(&b, "qwen2.feed_forward_length", 5504)
	writeKVFloat32(&b, "qwen2.attention.layer_norm_rms_epsilon", 0.00001)
	writeKVUint32(&b, "qwen2.context_length", 32768)
	writeKVUint32(&b, "tokenizer.ggml.eos_token_id", 7)
	writeKVStringArray(&b, "tokenizer.ggml.tokens", tokens)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{2048, metadataHeavyTokenCount}, TensorF32, 0)
	writeTensorInfoForTest(&b, "output.weight", []uint64{2048, metadataHeavyTokenCount}, TensorF32, 64)
	for b.Len()%64 != 0 {
		b.WriteByte(0)
	}
	dataOffset := b.Len()
	b.Write(bytes.Repeat([]byte{0xa5}, 128))
	return b.Bytes(), dataOffset
}
