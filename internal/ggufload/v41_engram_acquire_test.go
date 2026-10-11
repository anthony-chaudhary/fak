package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41EngramQ2KReferenceRows is the origin decode the in-place path must match:
// a serial scalar dequant into a scratch row, then one little-endian PutUint32
// per element.
func v41EngramQ2KReferenceRows(raw []byte, rows int) []byte {
	row := make([]float32, rows*qkK)
	dequantQ2KScalar(row, raw)
	out := make([]byte, rows*v41EngramQ2KRowBytes)
	for i, v := range row {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// TestV41EngramQ2KBindMatchesSerialReference is the byte-identity witness for
// the parallel Engram bind: the rows the source serves (in-place decode, aligned
// and misaligned destinations, row-aligned and unaligned ReadAt) equal the
// serial reference decode, and the acquired binding digest equals a direct
// sha256 over those reference rows.
// fak-test:runtime fast est=2s lane=default
func TestV41EngramQ2KBindMatchesSerialReference(t *testing.T) {
	const rows = 3000
	raw := make([]byte, rows*blockQ2KBytes)
	rand.New(rand.NewSource(13668)).Read(raw)
	src := &V41EngramQ2KSource{r: bytes.NewReader(raw), tableRows: rows, payloadOff: 0}
	want := v41EngramQ2KReferenceRows(raw, rows)

	backing := make([]byte, len(want)+1)
	for _, dst := range [][]byte{backing[:len(want)], backing[1:]} {
		n, err := src.ReadRows(0, rows, dst)
		if err != nil || n != len(want) || !bytes.Equal(dst[:n], want) {
			t.Fatalf("ReadRows (dst aligned=%t) = %d, %v; equal %t", &dst[0] == &backing[0], n, err, bytes.Equal(dst[:n], want))
		}
	}

	reader := &v41EngramRowReaderAt{src: src, rows: rows, rowBytes: v41EngramQ2KRowBytes}
	for _, span := range []struct{ off, n int64 }{{0, 1024}, {1024 * 7, 1024 * 900}, {3, 5000}, {1023, 1}, {0, int64(len(want))}} {
		p := make([]byte, span.n)
		if n, err := reader.ReadAt(p, span.off); err != nil || int64(n) != span.n || !bytes.Equal(p, want[span.off:span.off+span.n]) {
			t.Fatalf("ReadAt off=%d n=%d = %d, %v", span.off, span.n, n, err)
		}
	}

	size := int64(len(want))
	shard := model.V41EngramShard{ID: "blk.1.engram_embd.weight", ExpectedID: "blk.1.engram_embd.weight", Size: size, Rows: rows}
	bind := model.V41EngramArtifactBinding{Shard: shard.ID, Size: size, Rows: rows, RowBytes: v41EngramQ2KRowBytes}
	acquired, err := model.NewV41AcquiredEngramRowSource(reader, shard, bind)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	got, ok := model.V41EngramBinding(acquired)
	sum := sha256.Sum256(want)
	if !ok || got.Digest != "sha256:"+hex.EncodeToString(sum[:]) {
		t.Fatalf("acquired binding %+v ok=%t, want digest over the serial reference rows", got, ok)
	}
}
