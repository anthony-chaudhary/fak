package model

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// estimate_test.go — issue #709 acceptance: the safetensors footprint estimate is callable
// off the HEADER alone (no tensor data, no full load). The synthetic file carries only the
// 8-byte header length + JSON header (no data section): EstimateSafetensorsLoadBytes reads
// the header and never dereferences data_offsets, so this runs under -short with no weights.

func writeSynthSafetensorsHeader(t *testing.T) string {
	t.Helper()
	entries := map[string]any{
		"tensor_a":     map[string]any{"dtype": "F32", "shape": []int{10, 10}, "data_offsets": []int{0, 400}},
		"tensor_b":     map[string]any{"dtype": "BF16", "shape": []int{8}, "data_offsets": []int{400, 416}},
		"__metadata__": map[string]any{"format": "pt"},
	}
	hdr, err := json.Marshal(entries)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "model.safetensors")
	f, err := os.Create(p)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(hdr)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		t.Fatalf("write len: %v", err)
	}
	if _, err := f.Write(hdr); err != nil {
		t.Fatalf("write header: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	return p
}

func TestEstimateSafetensorsLoadBytesIsHeaderArithmetic(t *testing.T) {
	p := writeSynthSafetensorsHeader(t)
	// tensor_a [10,10] F32 = 100 elems * 4 = 400; tensor_b [8] BF16 = 8 elems * 4 = 32.
	// Total resident f32 = 432, with no tensor data read and no model built.
	got, err := EstimateSafetensorsLoadBytes(p)
	if err != nil {
		t.Fatalf("EstimateSafetensorsLoadBytes: %v", err)
	}
	if want := int64(432); got != want {
		t.Fatalf("EstimateSafetensorsLoadBytes = %d, want %d", got, want)
	}
}
