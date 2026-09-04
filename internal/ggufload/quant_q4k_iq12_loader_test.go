package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type iq12LoaderFixtureTensor struct {
	name    string
	dims    []uint64
	typ     TensorType
	offset  uint64
	payload []byte
}

func iq12LoaderPayload(t *testing.T, name string, dims []uint64, typ TensorType, marker byte) []byte {
	t.Helper()
	n, err := tensorPayloadBytes(TensorInfo{Name: name, Dims: dims, Type: typ})
	if err != nil {
		t.Fatalf("payload geometry for %s (%s): %v", name, typ, err)
	}
	payload := bytes.Repeat([]byte{marker}, int(n))
	if typ == TensorQ4_K {
		for off := 0; off < len(payload); off += blockQ4KBytes {
			payload[off], payload[off+1] = 0, 0
			payload[off+2], payload[off+3] = 0, 0
		}
	}
	return payload
}

func iq12MixedLoaderGGUF(t *testing.T, truncateLast bool) ([]byte, map[string][]byte) {
	t.Helper()
	const (
		dim    = 256
		layers = 6
		align  = 32
	)
	types := []TensorType{TensorIQ2_XXS, TensorIQ2_XS, TensorIQ1_S, TensorIQ2_S, TensorIQ1_M, TensorQ2_K}
	tensors := make([]iq12LoaderFixtureTensor, 0, len(types)+1)
	wantRaw := make(map[string][]byte, len(types))
	for layer, typ := range types {
		name := "blk." + string(rune('0'+layer)) + ".ffn_down.weight"
		payload := iq12LoaderPayload(t, name, []uint64{dim, dim}, typ, byte(0x21+layer))
		tensors = append(tensors, iq12LoaderFixtureTensor{name: name, dims: []uint64{dim, dim}, typ: typ, payload: payload})
		canon, ok := CanonicalTensorNameArch(name, "qwen35")
		if !ok {
			t.Fatalf("canonical mapping missing for %s", name)
		}
		wantRaw[canon] = append([]byte(nil), payload...)
	}
	// A normalize-sensitive attention tensor follows the established Q8 route and lets Build
	// finish while the five XL-recipe constituents remain raw resident.
	qName := "blk.5.attn_q.weight"
	tensors = append(tensors, iq12LoaderFixtureTensor{
		name: qName, dims: []uint64{dim, 2 * dim}, typ: TensorQ4_K,
		payload: iq12LoaderPayload(t, qName, []uint64{dim, 2 * dim}, TensorQ4_K, 0x7f),
	})

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
	writeKVUint64(&b, "qwen35.embedding_length", dim)
	writeKVUint64(&b, "qwen35.block_count", layers)
	writeKVUint64(&b, "qwen35.feed_forward_length", dim)
	writeKVUint64(&b, "qwen35.attention.head_count", 1)
	writeKVUint64(&b, "qwen35.attention.head_count_kv", 1)
	writeKVFloat32(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVUint64(&b, "qwen35.full_attention_interval", layers)
	for _, tensor := range tensors {
		writeTensorInfoForTest(&b, tensor.name, tensor.dims, tensor.typ, tensor.offset)
	}
	padToAlignment(&b, align)
	dataStart := b.Len()
	for _, tensor := range tensors {
		padToLen(&b, dataStart+int(tensor.offset))
		b.Write(tensor.payload)
	}
	data := b.Bytes()
	if truncateLast {
		data = data[:len(data)-1]
	}
	return data, wantRaw
}

func writeIQ12LoaderFixture(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "qwen38-ud-q2-k-xl-synthetic.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestLoadQ2XLConstituentsStayResidentVerbatim(t *testing.T) {
	data, wantRaw := iq12MixedLoaderGGUF(t, false)
	m, err := LoadModelQ4K(writeIQ12LoaderFixture(t, data))
	if err != nil {
		t.Fatalf("LoadModelQ4K: %v", err)
	}
	if got := m.KQuantCount(); got != len(wantRaw) {
		t.Fatalf("KQuantCount()=%d, want %d XL-recipe constituents", got, len(wantRaw))
	}
	for name, want := range wantRaw {
		if !m.HasKQuant(name) || m.HasQ8(name) {
			t.Errorf("%s route: kquant=%v q8=%v, want raw resident only", name, m.HasKQuant(name), m.HasQ8(name))
			continue
		}
		got, ok := m.KQuantRaw(name)
		if !ok || !bytes.Equal(got, want) {
			t.Errorf("%s resident payload changed during load", name)
		}
	}
	if !m.HasQ8("model.layers.5.self_attn.q_proj.weight") {
		t.Fatal("normalize-sensitive Q4_K attention tensor did not follow the established Q8 route")
	}
}

func TestQ2XLResidentGeometryCoversAllIQ12Constituents(t *testing.T) {
	for _, tc := range []struct {
		typ   TensorType
		bytes int
	}{
		{TensorIQ2_XXS, blockIQ2XXSBytes},
		{TensorIQ2_XS, blockIQ2XSBytes},
		{TensorIQ1_S, blockIQ1SBytes},
		{TensorIQ2_S, blockIQ2SBytes},
		{TensorIQ1_M, blockIQ1MBytes},
		{TensorQ2_K, blockQ2KBytes},
	} {
		t.Run(tc.typ.String(), func(t *testing.T) {
			weights, blockBytes, ok := residentExpertBlockGeometry(tc.typ)
			if !ok || weights != qkK || blockBytes != tc.bytes {
				t.Fatalf("resident geometry=(%d,%d,%v), want (%d,%d,true)", weights, blockBytes, ok, qkK, tc.bytes)
			}
		})
	}
}

func TestLoadQ2XLTruncatedPayloadFailsClosed(t *testing.T) {
	data, _ := iq12MixedLoaderGGUF(t, true)
	_, err := LoadModelQ4K(writeIQ12LoaderFixture(t, data))
	if err == nil {
		t.Fatal("LoadModelQ4K accepted a truncated GGUF payload")
	}
	if got := err.Error(); !strings.Contains(got, "blk.5.attn_q.weight") {
		t.Fatalf("error %q does not name the malformed tensor", got)
	}
}
