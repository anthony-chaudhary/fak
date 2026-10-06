package ggufload

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// TestPQ2_0GGUFTag142LoadsGoldenBlock pins the on-disk Bonsai-2 PQ2_0 type
// independently of the loader's type constant. Prism's ggml block is one f16
// scale followed by 128 low-bit-first 2-bit codes (34 bytes). The valid trits
// 0, 1, 2 reconstruct (-1, 0, +1) * d.
func TestPQ2_0GGUFTag142LoadsGoldenBlock(t *testing.T) {
	const (
		name = "blk.0.ffn_up.weight"
		tag  = TensorType(142)
		cols = 128
	)
	raw := make([]byte, 34)
	binary.LittleEndian.PutUint16(raw[:2], 0x3800) // f16 0.5
	for i := 2; i < len(raw); i++ {
		raw[i] = 0x24 // low-to-high codes 0,1,2,0
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 2)
	writeKVString(&b, "general.architecture", "qwen35")
	writeKVUint32(&b, "general.alignment", 32)
	writeTensorInfoForTest(&b, name, []uint64{cols, 1}, tag, 0)
	padToAlignment(&b, 32)
	b.Write(raw)
	path := filepath.Join(t.TempDir(), "bonsai2-pq2-tag142.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	info, ok := ws.Tensor(name)
	if !ok || info.Type != tag || info.Type.String() != "PQ2_0" {
		t.Fatalf("GGUF tag 142 must identify PQ2_0, got info=%+v present=%t", info, ok)
	}
	if got, err := ws.EstimateLoadBytes(); err != nil || got != int64(len(raw)) {
		t.Fatalf("EstimateLoadBytes = %d, %v; want %d", got, err, len(raw))
	}
	payload, _, err := ws.TensorBytes(name)
	if err != nil || !bytes.Equal(payload, raw) {
		t.Fatalf("TensorBytes = %d bytes, %v; want exact 34-byte block", len(payload), err)
	}
	values, _, err := ws.TensorF32(name)
	if err != nil {
		t.Fatalf("TensorF32: %v", err)
	}
	if len(values) != cols {
		t.Fatalf("TensorF32 len = %d, want %d", len(values), cols)
	}
	want := []float32{-0.5, 0, 0.5, -0.5}
	for i, v := range values {
		if v != want[i%len(want)] {
			t.Fatalf("value[%d] = %g, want %g", i, v, want[i%len(want)])
		}
	}
}
