package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_mmap_test.go gates the FAK_GGUF_MMAP shard-reader seam (gguf_mmap.go): the
// mmap-backed reader must be BYTE-IDENTICAL to the default os.Open + ReadAt path for
// every tensor, on both the single-file and the split-shard layout. On platforms
// without an mmap impl (native Windows: model.MmapOpen reports ok=false) the mmap leg
// is skipped — the fallback leg above it still gates that the gate wiring changes
// nothing when the map is unavailable.

// resetGGUFMmapGateForTest rearms the once-per-process FAK_GGUF_MMAP capture so a single
// test process can exercise both arms of the gate. Test-only: the production contract is
// exactly one capture per process (see ggufMmapEnabled).
func resetGGUFMmapGateForTest() {
	ggufMmapOnce = sync.Once{}
	ggufMmapOn = false
}

// mmapPatternPayload builds a distinct, recoverable payload for one fixture tensor:
// keyed on both the seed and the offset so tensor A's bytes can never be confused with
// tensor B's (the same idea as expertPatternByte in gguf_expert_shard_bytes_test.go).
// n must be a multiple of 4 so the bytes describe whole F32 elements.
func mmapPatternPayload(seed byte, n int) []byte {
	p := make([]byte, n)
	for i := range p {
		p[i] = byte((int(seed)*131 + i*7 + 17) & 0xff)
	}
	return p
}

// writeMmapSingleFileFixture writes a minimal single-file multi-tensor GGUF (no split.*
// keys, so OpenWeights takes the single-file path) and returns its tensor names.
func writeMmapSingleFileFixture(t *testing.T, path string) []string {
	t.Helper()
	const align = 32
	tensors := []struct {
		name string
		data []byte
	}{
		{"one.weight", mmapPatternPayload(0x11, 32)},
		{"two.weight", mmapPatternPayload(0x22, 64)},
		{"three.weight", mmapPatternPayload(0x33, 32)},
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 1)
	writeKVUint32(&b, "general.alignment", align)
	off := uint64(0)
	for _, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, []uint64{uint64(len(tt.data) / 4)}, TensorF32, off)
		off += uint64(len(tt.data))
		off = (off + align - 1) / align * align
	}
	padToAlignment(&b, align)
	for _, tt := range tensors {
		b.Write(tt.data)
		padToAlignment(&b, align)
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("write single-file fixture: %v", err)
	}
	names := make([]string, len(tensors))
	for i, tt := range tensors {
		names[i] = tt.name
	}
	return names
}

// writeMmapSplitFixture writes a 2-shard split checkpoint through the same shard
// builder the split tests use and returns shard 1's path.
func writeMmapSplitFixture(t *testing.T, dir string) string {
	t.Helper()
	shard1 := writeSplitShard(t, 1, 2, 2, true, []splitTensor{
		{name: "a.weight", dims: []uint64{8}, typ: TensorF32, data: mmapPatternPayload(0x44, 32)},
	})
	shard2 := writeSplitShard(t, 2, 2, 2, false, []splitTensor{
		{name: "b.weight", dims: []uint64{8}, typ: TensorF32, data: mmapPatternPayload(0x55, 32)},
	})
	p1 := filepath.Join(dir, "mmap-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "mmap-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	return p1
}

// readAllTensorBytes opens path through OpenWeights under the CURRENT gate state and
// returns every tensor's TensorBytes payload by name, closing the source before
// returning (so a retained-past-Close aliasing bug would surface as corrupt bytes).
func readAllTensorBytes(t *testing.T, path string) map[string][]byte {
	t.Helper()
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights(%s): %v", path, err)
	}
	out := make(map[string][]byte, len(ws.File.Tensors))
	for _, info := range ws.File.Tensors {
		raw, _, err := ws.TensorBytes(info.Name)
		if err != nil {
			_ = ws.Close()
			t.Fatalf("TensorBytes(%s): %v", info.Name, err)
		}
		if len(raw) == 0 {
			_ = ws.Close()
			t.Fatalf("TensorBytes(%s) returned no bytes", info.Name)
		}
		out[info.Name] = raw
	}
	if err := ws.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return out
}

func TestOpenWeightsMmapTensorBytesIdentical(t *testing.T) {
	dir := t.TempDir()
	singlePath := filepath.Join(dir, "single.gguf")
	singleNames := writeMmapSingleFileFixture(t, singlePath)
	splitShard1 := writeMmapSplitFixture(t, dir)

	// Leg 1 — the default os.Open + ReadAt path (gate off): the ground truth.
	t.Setenv("FAK_GGUF_MMAP", "")
	resetGGUFMmapGateForTest()
	wantSingle := readAllTensorBytes(t, singlePath)
	wantSplit := readAllTensorBytes(t, splitShard1)
	if len(wantSingle) != len(singleNames) || len(wantSplit) != 2 {
		t.Fatalf("baseline tensor counts = %d single / %d split, want %d / 2", len(wantSingle), len(wantSplit), len(singleNames))
	}

	// Leg 2 — the mmap path. Skipped where the platform cannot map (native Windows:
	// model.MmapOpen ok=false); the ReadAt fallback above still gated there.
	if data, closer, ok, err := model.MmapOpen(singlePath); err != nil {
		t.Fatalf("model.MmapOpen probe: %v", err)
	} else if !ok {
		t.Skip("mmap unsupported on this platform; skipping the FAK_GGUF_MMAP leg (ReadAt fallback leg passed)")
	} else {
		_ = data
		_ = closer.Close()
	}
	t.Setenv("FAK_GGUF_MMAP", "1")
	resetGGUFMmapGateForTest()

	// Witness the wiring is actually LIVE (not silently on the fallback): the retained
	// single-file reader must be an mmapReaderAt with the mapped region recorded, and
	// every split tensor's dataFor entry must carry its shard's map.
	ws, err := OpenWeights(singlePath)
	if err != nil {
		t.Fatalf("OpenWeights(single, mmap): %v", err)
	}
	if _, isMmap := ws.r.(*mmapReaderAt); !isMmap || ws.data == nil {
		_ = ws.Close()
		t.Fatalf("FAK_GGUF_MMAP=1 single-file reader is %T with data=%v, want *mmapReaderAt with a mapped region", ws.r, ws.data != nil)
	}
	if err := ws.Close(); err != nil {
		t.Fatalf("Close(single, mmap): %v", err)
	}
	wss, err := OpenWeights(splitShard1)
	if err != nil {
		t.Fatalf("OpenWeights(split, mmap): %v", err)
	}
	for i := range wss.File.Tensors {
		if i >= len(wss.dataFor) || wss.dataFor[i] == nil {
			_ = wss.Close()
			t.Fatalf("FAK_GGUF_MMAP=1 split tensor %d has no dataFor map entry", i)
		}
	}
	if err := wss.Close(); err != nil {
		t.Fatalf("Close(split, mmap): %v", err)
	}

	// The gate: every tensor's payload byte-identical across the two reader paths.
	gotSingle := readAllTensorBytes(t, singlePath)
	gotSplit := readAllTensorBytes(t, splitShard1)
	assertIdentical := func(label string, want, got map[string][]byte) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: mmap leg read %d tensors, want %d", label, len(got), len(want))
		}
		for name, w := range want {
			g, ok := got[name]
			if !ok {
				t.Fatalf("%s: mmap leg missing tensor %s", label, name)
			}
			if !bytes.Equal(g, w) {
				t.Fatalf("%s: tensor %s differs between mmap and ReadAt paths (len %d vs %d)", label, name, len(g), len(w))
			}
		}
	}
	assertIdentical("single-file", wantSingle, gotSingle)
	assertIdentical("split", wantSplit, gotSplit)

	// Restore the default arm for any later test in this process.
	t.Setenv("FAK_GGUF_MMAP", "")
	resetGGUFMmapGateForTest()
	t.Logf("mmap and ReadAt shard readers byte-identical across %d single-file + %d split tensors", len(gotSingle), len(gotSplit))
}
