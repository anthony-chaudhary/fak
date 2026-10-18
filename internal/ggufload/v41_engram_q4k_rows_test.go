package ggufload

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41Q4KBlockRow packs distinct 6-bit scales/minima, including their high bits.
// Expected values follow d*scale*q-dmin*min independently of the decoder.
func v41Q4KBlockRow(row int) ([]byte, []float32) {
	raw := make([]byte, blockQ4KBytes)
	d := float32(row%3 + 1)
	binary.LittleEndian.PutUint16(raw, f32ToF16BitsForTestQ2K(d))
	binary.LittleEndian.PutUint16(raw[2:], 0x3800) // dmin = 0.5
	var scales, mins [8]byte
	for i := range scales {
		scales[i], mins[i] = byte(7*(i+1)), byte(5*(i+1))
	}
	for i := 0; i < 4; i++ {
		raw[4+i] = scales[i] | (scales[i+4]>>4)<<6
		raw[8+i] = mins[i] | (mins[i+4]>>4)<<6
		raw[12+i] = (scales[i+4] & 15) | (mins[i+4]&15)<<4
	}
	codes, want := q4KFixtureCodes()
	copy(raw[16:], codes)
	for i := range want {
		want[i] = d*float32(scales[i/32])*want[i] - 0.5*float32(mins[i/32])
	}
	return raw, want
}

// writeV41Q4KEngramFixture replaces only the second shard of the admission
// fixture. The real 1017 header's type/width and noninitial-shard ownership are
// retained; the 72-row table and all model geometry remain synthetic.
func writeV41Q4KEngramFixture(t *testing.T) (string, []byte) {
	t.Helper()
	path := writeV41UnboundEngramFixture(t, true)
	var shard, raw bytes.Buffer
	writeMinimalHeader(&shard, 1, 4)
	writeKVUint32(&shard, "general.alignment", 32)
	writeKVUint32(&shard, "split.no", 1)
	writeKVUint32(&shard, "split.count", 2)
	writeKVUint32(&shard, "split.tensors.count", 2)
	writeTensorInfoForTest(&shard, "blk.1.engram_embd.weight", []uint64{256, 72}, TensorQ4_K, 0)
	padToAlignment(&shard, 32)
	for row := 0; row < 72; row++ {
		block, _ := v41Q4KBlockRow(row)
		raw.Write(block)
	}
	shard.Write(raw.Bytes())
	if err := os.WriteFile(filepath.Join(filepath.Dir(path), "engram-00002-of-00002.gguf"), shard.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path, raw.Bytes()
}

// TestV41EngramQ4KServingRows is the fak#13890 bounded loader/binding witness.
// It makes no real-checkpoint forward, quality, or device-performance claim.
// fak-test:runtime fast est=200ms lane=default
func TestV41EngramQ4KServingRows(t *testing.T) {
	path, raw := writeV41Q4KEngramFixture(t)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	inner, rows, rowBytes, err := openV41EngramRowSource(ws, 1)
	if err != nil || rows != 72 || rowBytes != 1024 {
		t.Fatalf("dispatch rows=%d rowBytes=%d err=%v", rows, rowBytes, err)
	}
	src, ok := inner.(*V41EngramKSource)
	if !ok || !src.q4K {
		t.Fatalf("dispatch returned %T, want Q4_K source", inner)
	}
	for _, row := range []int{71, 0, 14, 1} {
		_, want := v41Q4KBlockRow(row)
		backing := make([]byte, rowBytes+1)
		for _, dst := range [][]byte{backing[:rowBytes], backing[1:]} {
			if n, err := src.ReadRows(row, 1, dst); err != nil || n != rowBytes || !bytes.Equal(dst, f32BytesQ2KForTest(want)) {
				t.Fatalf("row %d: read=%d err=%v; numerical/stride mismatch", row, n, err)
			}
		}
	}
	counted := &v41EngramCountingQ2K{V41EngramQ2KSource: src}
	reader := newV41EngramRowReaderAt(counted, rows, rowBytes)
	_, w0 := v41Q4KBlockRow(0)
	_, w1 := v41Q4KBlockRow(1)
	wantSpan := append(f32BytesQ2KForTest(w0), f32BytesQ2KForTest(w1)...)
	partial := make([]byte, 1025)
	if n, err := reader.ReadAt(partial, 3); err != nil || n != len(partial) || !bytes.Equal(partial, wantSpan[3:1028]) {
		t.Fatalf("cross-row partial read=%d err=%v", n, err)
	}
	counted.reads.Store(0)
	size := int64(rows * rowBytes)
	shard := model.V41EngramShard{ID: "blk.1.engram_embd.weight", ExpectedID: "blk.1.engram_embd.weight", Size: size, Rows: rows}
	bind := model.V41EngramArtifactBinding{Shard: shard.ID, Size: size, Rows: rows, RowBytes: rowBytes}
	acquired, err := model.NewV41AcquiredEngramRowSource(reader, shard, bind)
	if err != nil {
		t.Fatal(err)
	}
	if counted.reads.Load() != 0 {
		t.Fatal("binding expanded Q4_K rows instead of hashing their encoded bytes")
	}
	last := make([]byte, rowBytes)
	_, wantLast := v41Q4KBlockRow(rows - 1)
	if n, err := acquired.ReadRows(rows-1, 1, last); err != nil || n != rowBytes || !bytes.Equal(last, f32BytesQ2KForTest(wantLast)) {
		t.Fatalf("verified source last row=%d err=%v", n, err)
	}
	got, ok := model.V41EngramBinding(acquired)
	// Independent single-leaf tree digest, including the Q4_K domain tag.
	root := sha256.New()
	root.Write([]byte("fak.v41.engram.tree1\x00q4_k\x00"))
	var hdr [16]byte
	binary.LittleEndian.PutUint64(hdr[:8], uint64(len(raw)))
	binary.LittleEndian.PutUint64(hdr[8:], 1<<20)
	root.Write(hdr[:])
	leaf := sha256.Sum256(raw)
	root.Write(leaf[:])
	wantDigest := "sha256t1-q4_k:" + hex.EncodeToString(root.Sum(nil))
	if !ok || got.Digest != wantDigest {
		t.Fatalf("binding=%+v ok=%t, want %s", got, ok, wantDigest)
	}
	if _, err := model.NewV41VerifiedEngramRowSource(reader, shard, got); err != nil {
		t.Fatalf("same Q4_K binding refused: %v", err)
	}
	other := append([]byte(nil), raw...)
	other[len(other)-1] ^= 1
	otherReader := newV41EngramRowReaderAt(&V41EngramKSource{r: bytes.NewReader(other), tableRows: rows, q4K: true}, rows, rowBytes)
	var mismatch *model.V41EngramStreamError
	if _, err := model.NewV41VerifiedEngramRowSource(otherReader, shard, got); !errors.As(err, &mismatch) || mismatch.Kind != model.V41EngramStreamBindingMismatch {
		t.Fatalf("mutated encoded bytes accepted: %v", err)
	}
	// Ordinary serving must attach this F32-served, Q4_K-bound source.
	m, err := LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	if m != nil {
		defer m.CloseWeights()
	}
	if err != nil || m == nil || !m.V41EngramAttached() {
		t.Fatalf("ordinary Q4_K Engram load/attach: %v", err)
	}
	bindings, ok := m.V41EngramStageBindings()
	if !ok || len(bindings) != 1 || bindings[0].Digest != wantDigest || bindings[0].RowBytes != rowBytes {
		t.Fatalf("serving binding=%+v ok=%t", bindings, ok)
	}

	t.Run("fail closed", func(t *testing.T) {
		if _, err := V41EngramQ2KOpen(ws, 1); err == nil {
			t.Fatal("Q2_K-specific API accepted Q4_K")
		}
		for _, span := range [][2]int{{-1, 1}, {72, 1}, {71, 2}, {0, -1}} {
			if _, err := src.ReadRows(span[0], span[1], make([]byte, 2048)); err == nil {
				t.Fatalf("invalid span %v accepted", span)
			}
		}
		if _, err := src.ReadRows(0, 1, make([]byte, rowBytes-1)); err == nil {
			t.Fatal("short destination accepted")
		}
		short := *src
		short.r = truncReaderAt{inner: src.r, keep: 10}
		var typed *V41EngramQ2KError
		if _, err := short.ReadRows(0, 1, make([]byte, rowBytes)); !errors.As(err, &typed) || typed.Kind != V41EngramQ2KRead {
			t.Fatalf("short read: %v", err)
		}
		huge := *src
		huge.tableRows = math.MaxInt
		if _, err := huge.ReadRows(0, math.MaxInt/rowBytes+1, nil); !errors.As(err, &typed) || typed.Kind != V41EngramQ2KRange {
			t.Fatalf("overflowing read: %v", err)
		}
		idx := ws.byName[shard.ID]
		original := ws.File.Tensors[idx]
		defer func() { ws.File.Tensors[idx] = original }()
		for _, bad := range []TensorInfo{
			{Name: original.Name, Type: TensorQ4_K, Dims: []uint64{128, 72}, FileOffset: original.FileOffset},
			{Name: original.Name, Type: TensorQ4_K, Dims: []uint64{256, math.MaxUint64}, FileOffset: original.FileOffset},
			{Name: original.Name, Type: TensorQ4_K, Dims: original.Dims, FileOffset: math.MaxInt64},
			{Name: original.Name, Type: TensorF16, Dims: original.Dims, FileOffset: original.FileOffset},
		} {
			ws.File.Tensors[idx] = bad
			if opened, err := V41EngramQ4KOpen(ws, 1); err == nil || opened != nil || !errors.As(err, &typed) {
				t.Fatalf("malformed tensor %+v: source=%v err=%v", bad, opened, err)
			}
		}
		ws.File.Tensors[idx] = original
		const rowsKey = "deepseek41.engram.rows"
		defer delete(ws.File.Metadata, rowsKey)
		for _, declared := range []uint64{0, 71} {
			ws.File.Metadata[rowsKey] = vcruzU64Array([]uint64{declared})
			if opened, err := V41EngramQ4KOpen(ws, 1); err == nil || opened != nil || !errors.As(err, &typed) || typed.Kind != V41EngramQ2KGeometry {
				t.Fatalf("declared rows %d: source=%v err=%v", declared, opened, err)
			}
		}
		ws.File.Metadata[rowsKey] = vcruzU64Array([]uint64{72})
		if _, err := V41EngramQ4KOpen(ws, 1); err != nil {
			t.Fatalf("matching declared rows refused: %v", err)
		}
	})
}
