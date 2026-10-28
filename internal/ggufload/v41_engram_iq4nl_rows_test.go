package ggufload

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The format is pinned independently by Strata's Engram decoder and its GGML:
// https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/src/dsv41/ds4_kernels.cu#L1506-L1517
// https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-quants.c#L2725-L2740
// Low nibbles fill j, high nibbles j+16 in each 32-element block. Alternating
// signed scales distinguish all eight blocks and adjacent rows without using
// the production decoder or codebook to derive the oracle.
func v41IQ4NLBlockRow(row int) ([]byte, []float32) {
	codes := []byte{0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
		0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F}
	values := []float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
		113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127}
	raw, want := make([]byte, 144), make([]float32, 256)
	for block := 0; block < 8; block++ {
		d := float32((row%3 + 1) * (block + 1))
		if block%2 != 0 {
			d = -d
		}
		binary.LittleEndian.PutUint16(raw[block*18:], f32ToF16BitsForTestQ2K(d))
		copy(raw[block*18+2:], codes)
		for i, v := range values {
			want[block*32+i] = d * v
		}
	}
	return raw, want
}

type v41IQ4NLReadProbe struct {
	inner io.ReaderAt
	base  int64
	bytes int
	last  int64
}

func (r *v41IQ4NLReadProbe) ReadAt(p []byte, off int64) (int, error) {
	r.bytes += len(p)
	r.last = off
	return r.inner.ReadAt(p, off-r.base)
}

// TestV41EngramIQ4NLServingRows covers the bounded format/binding prerequisite,
// not companion discovery, exact-checkpoint forward or device qualification.
// fak-test:runtime fast est=200ms lane=default
func TestV41EngramIQ4NLServingRows(t *testing.T) {
	path := writeV41UnboundEngramFixture(t, true)
	var second, raw bytes.Buffer
	writeMinimalHeader(&second, 1, 4)
	writeKVUint32(&second, "general.alignment", 32)
	writeKVUint32(&second, "split.no", 1)
	writeKVUint32(&second, "split.count", 2)
	writeKVUint32(&second, "split.tensors.count", 2)
	writeTensorInfoForTest(&second, "blk.1.engram_embd.weight", []uint64{256, 72}, TensorIQ4_NL, 0)
	padToAlignment(&second, 32)
	for row := 0; row < 72; row++ {
		block, _ := v41IQ4NLBlockRow(row)
		raw.Write(block)
	}
	second.Write(raw.Bytes())
	if err := os.WriteFile(filepath.Join(filepath.Dir(path), "engram-00002-of-00002.gguf"), second.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	inner, rows, rowBytes, err := openV41EngramRowSource(ws, 1)
	if err != nil || rows != 72 || rowBytes != 1024 {
		t.Fatalf("IQ4_NL dispatch rows=%d rowBytes=%d err=%v", rows, rowBytes, err)
	}
	src, ok := inner.(*V41EngramKSource)
	if !ok {
		t.Fatalf("dispatch returned %T, want bounded quantized source", inner)
	}
	probe := &v41IQ4NLReadProbe{inner: src.r}
	src.r = probe
	for _, row := range []int{71, 0, 14, 1} {
		_, want := v41IQ4NLBlockRow(row)
		backing := make([]byte, rowBytes+1)
		for _, dst := range [][]byte{backing[:rowBytes], backing[1:]} {
			before := probe.bytes
			if n, err := src.ReadRows(row, 1, dst); err != nil || n != rowBytes || !bytes.Equal(dst, f32BytesQ2KForTest(want)) {
				t.Fatalf("row %d: read=%d err=%v; numerical/stride mismatch", row, n, err)
			}
			if probe.bytes-before != 144 || probe.last != src.payloadOff+int64(row)*144 {
				t.Fatalf("row %d: read %d encoded bytes at %d", row, probe.bytes-before, probe.last)
			}
		}
	}
	counted := &v41EngramCountingQ2K{V41EngramQ2KSource: src}
	reader := newV41EngramRowReaderAt(counted, rows, rowBytes)
	_, w0 := v41IQ4NLBlockRow(0)
	_, w1 := v41IQ4NLBlockRow(1)
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
		t.Fatal("binding expanded IQ4_NL rows instead of hashing encoded bytes")
	}
	got, ok := model.V41EngramBinding(acquired)
	root := sha256.New()
	root.Write([]byte("fak.v41.engram.tree1\x00iq4_nl\x00"))
	var hdr [16]byte
	binary.LittleEndian.PutUint64(hdr[:8], uint64(raw.Len()))
	binary.LittleEndian.PutUint64(hdr[8:], 1<<20)
	root.Write(hdr[:])
	leaf := sha256.Sum256(raw.Bytes())
	root.Write(leaf[:])
	wantDigest := "sha256t1-iq4_nl:" + hex.EncodeToString(root.Sum(nil))
	if !ok || got.Digest != wantDigest {
		t.Fatalf("binding=%+v ok=%t, want %s", got, ok, wantDigest)
	}
	if _, err := model.NewV41VerifiedEngramRowSource(reader, shard, got); err != nil {
		t.Fatalf("same IQ4_NL binding refused: %v", err)
	}
	other := append([]byte(nil), raw.Bytes()...)
	other[len(other)-1] ^= 1
	changed := *src
	changed.r, changed.payloadOff = bytes.NewReader(other), 0
	var mismatch *model.V41EngramStreamError
	if _, err := model.NewV41VerifiedEngramRowSource(newV41EngramRowReaderAt(&changed, rows, rowBytes), shard, got); !errors.As(err, &mismatch) || mismatch.Kind != model.V41EngramStreamBindingMismatch {
		t.Fatalf("mutated IQ4_NL bytes accepted: %v", err)
	}
	wrongScheme := got
	wrongScheme.Digest = "sha256t1-q4_k:" + hex.EncodeToString(root.Sum(nil))
	if _, err := model.NewV41VerifiedEngramRowSource(reader, shard, wrongScheme); !errors.As(err, &mismatch) || mismatch.Kind != model.V41EngramStreamBindingMismatch {
		t.Fatalf("same-stride Q4_K identity accepted: %v", err)
	}
	// The ordinary tensor fits this budget; the table's f32 expansion does not.
	// Admission must use row serving, never eager table dequantization.
	priorBudget := MaxEagerF32Bytes
	MaxEagerF32Bytes = 4096
	defer func() { MaxEagerF32Bytes = priorBudget }()
	m, err := LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	if m != nil {
		defer m.CloseWeights()
	}
	if err != nil || m == nil || !m.V41EngramAttached() {
		t.Fatalf("ordinary IQ4_NL Engram load/attach: %v", err)
	}
	bindings, ok := m.V41EngramStageBindings()
	if !ok || len(bindings) != 1 || bindings[0].Digest != wantDigest || bindings[0].RowBytes != rowBytes {
		t.Fatalf("serving binding=%+v ok=%t", bindings, ok)
	}

	// A production-sized logical table still reads just the demanded final row.
	large := *src
	large.tableRows, large.payloadOff = 384006168, 0
	lastRaw, lastWant := v41IQ4NLBlockRow(large.tableRows - 1)
	lastProbe := &v41IQ4NLReadProbe{inner: bytes.NewReader(lastRaw), base: int64(large.tableRows-1) * 144}
	large.r = lastProbe
	dst := make([]byte, rowBytes)
	if n, err := large.ReadRows(large.tableRows-1, 1, dst); err != nil || n != rowBytes || lastProbe.bytes != 144 || !bytes.Equal(dst, f32BytesQ2KForTest(lastWant)) {
		t.Fatalf("large logical table: read=%d encoded=%d err=%v", n, lastProbe.bytes, err)
	}
	if _, err := V41EngramQ2KOpen(ws, 1); err == nil {
		t.Fatal("Q2_K-specific factory accepted IQ4_NL")
	}
	if _, err := V41EngramQ4KOpen(ws, 1); err == nil {
		t.Fatal("Q4_K-specific factory accepted IQ4_NL")
	}
	for _, span := range [][2]int{{-1, 1}, {72, 1}, {71, 2}, {0, -1}, {0, math.MaxInt/rowBytes + 1}} {
		if _, err := src.ReadRows(span[0], span[1], dst); err == nil {
			t.Fatalf("invalid span %v accepted", span)
		}
	}
	if _, err := src.ReadRows(0, 1, dst[:rowBytes-1]); err == nil {
		t.Fatal("short destination accepted")
	}
	short := *src
	short.r = truncReaderAt{inner: src.r, keep: 143}
	var typed *V41EngramQ2KError
	if _, err := short.ReadRows(0, 1, dst); !errors.As(err, &typed) || typed.Kind != V41EngramQ2KRead {
		t.Fatalf("short encoded read: %v", err)
	}
}
