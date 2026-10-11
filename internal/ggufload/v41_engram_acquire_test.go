package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"math/rand"
	"sync/atomic"
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

// v41EngramQ2KTreeReference re-derives scheme sha256t1-q2_k independently of
// the model package: sha256 per 1 MiB leaf of the Q2_K bytes, rooted under the
// scheme's domain, encoding, extent size and leaf size.
func v41EngramQ2KTreeReference(raw []byte) string {
	const leaf = 1 << 20
	root := sha256.New()
	root.Write([]byte("fak.v41.engram.tree1\x00q2_k\x00"))
	var hdr [16]byte
	binary.LittleEndian.PutUint64(hdr[:8], uint64(len(raw)))
	binary.LittleEndian.PutUint64(hdr[8:], leaf)
	root.Write(hdr[:])
	for at := 0; at < len(raw); at += leaf {
		end := min(at+leaf, len(raw))
		sum := sha256.Sum256(raw[at:end])
		root.Write(sum[:])
	}
	return "sha256t1-q2_k:" + hex.EncodeToString(root.Sum(nil))
}

// v41EngramCountingQ2K counts decoded-row reads through a Q2_K source while
// keeping its encoded-rows seam.
type v41EngramCountingQ2K struct {
	*V41EngramQ2KSource
	reads atomic.Int64
}

func (c *v41EngramCountingQ2K) ReadRows(start, count int, dst []byte) (int, error) {
	c.reads.Add(1)
	return c.V41EngramQ2KSource.ReadRows(start, count, dst)
}

// TestV41EngramQ2KBindMatchesSerialReference is the byte-identity witness for
// the Engram bind: the rows the source serves (in-place decode, aligned and
// misaligned destinations, row-aligned and unaligned ReadAt) equal the serial
// reference decode; acquisition binds the q2_k tree over exactly the Q2_K bytes
// those rows decode from without decoding a single row; a flat binding of the
// reference rows stays verifiable; and different Q2_K bytes are refused.
// fak-test:runtime fast est=2s lane=default
func TestV41EngramQ2KBindMatchesSerialReference(t *testing.T) {
	const rows = 3000
	raw := make([]byte, rows*blockQ2KBytes)
	rand.New(rand.NewSource(13668)).Read(raw)
	src := &v41EngramCountingQ2K{V41EngramQ2KSource: &V41EngramQ2KSource{r: bytes.NewReader(raw), tableRows: rows, payloadOff: 0}}
	want := v41EngramQ2KReferenceRows(raw, rows)

	backing := make([]byte, len(want)+1)
	for _, dst := range [][]byte{backing[:len(want)], backing[1:]} {
		n, err := src.ReadRows(0, rows, dst)
		if err != nil || n != len(want) || !bytes.Equal(dst[:n], want) {
			t.Fatalf("ReadRows (dst aligned=%t) = %d, %v; equal %t", &dst[0] == &backing[0], n, err, bytes.Equal(dst[:n], want))
		}
	}

	reader := newV41EngramRowReaderAt(src, rows, v41EngramQ2KRowBytes)
	for _, span := range []struct{ off, n int64 }{{0, 1024}, {1024 * 7, 1024 * 900}, {3, 5000}, {1023, 1}, {0, int64(len(want))}} {
		p := make([]byte, span.n)
		if n, err := reader.ReadAt(p, span.off); err != nil || int64(n) != span.n || !bytes.Equal(p, want[span.off:span.off+span.n]) {
			t.Fatalf("ReadAt off=%d n=%d = %d, %v", span.off, span.n, n, err)
		}
	}

	size := int64(len(want))
	shard := model.V41EngramShard{ID: "blk.1.engram_embd.weight", ExpectedID: "blk.1.engram_embd.weight", Size: size, Rows: rows}
	bind := model.V41EngramArtifactBinding{Shard: shard.ID, Size: size, Rows: rows, RowBytes: v41EngramQ2KRowBytes}
	src.reads.Store(0)
	acquired, err := model.NewV41AcquiredEngramRowSource(reader, shard, bind)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if n := src.reads.Load(); n != 0 {
		t.Fatalf("acquisition decoded %d row batches, want 0 (it binds the Q2_K bytes)", n)
	}
	got, ok := model.V41EngramBinding(acquired)
	if !ok || got.Digest != v41EngramQ2KTreeReference(raw) {
		t.Fatalf("acquired binding %+v ok=%t, want the q2_k tree over the Q2_K bytes %s", got, ok, v41EngramQ2KTreeReference(raw))
	}

	sum := sha256.Sum256(want)
	flat := bind
	flat.Digest = "sha256:" + hex.EncodeToString(sum[:])
	if _, err := model.NewV41VerifiedEngramRowSource(reader, shard, flat); err != nil {
		t.Fatalf("flat binding of the reference rows refused: %v", err)
	}
	other := append([]byte(nil), raw...)
	other[len(other)/2] ^= 0x10
	otherReader := newV41EngramRowReaderAt(&V41EngramQ2KSource{r: bytes.NewReader(other), tableRows: rows}, rows, v41EngramQ2KRowBytes)
	var typed *model.V41EngramStreamError
	if _, err := model.NewV41VerifiedEngramRowSource(otherReader, shard, got); !errors.As(err, &typed) || typed.Kind != model.V41EngramStreamBindingMismatch {
		t.Fatalf("different Q2_K bytes under the acquired binding = %v, want a binding mismatch", err)
	}
}

// BenchmarkV41EngramQ2KBind times binding one Q2_K Engram table. flat-f32 is
// the flat sha256 over the dequantized rows (the scheme acquisition bound
// before); tree-f32 is the tree over those rows; tree-q2k is acquisition as
// shipped, the tree over the Q2_K bytes. Throughput is in checkpoint bytes.
func BenchmarkV41EngramQ2KBind(b *testing.B) {
	const rows = 1 << 19
	raw := make([]byte, rows*blockQ2KBytes)
	rand.New(rand.NewSource(13668)).Read(raw)
	src := &V41EngramQ2KSource{r: bytes.NewReader(raw), tableRows: rows}
	size := int64(rows) * v41EngramQ2KRowBytes
	shard := model.V41EngramShard{ID: "blk.1.engram_embd.weight", ExpectedID: "blk.1.engram_embd.weight", Size: size, Rows: rows}
	bind := model.V41EngramArtifactBinding{Shard: shard.ID, Size: size, Rows: rows, RowBytes: v41EngramQ2KRowBytes}
	decoded := &v41EngramRowReaderAt{src: src, rows: rows, rowBytes: v41EngramQ2KRowBytes}
	b.Run("flat-f32", func(b *testing.B) {
		b.SetBytes(int64(len(raw)))
		for i := 0; i < b.N; i++ {
			if _, err := model.V41EngramDigestExtent(decoded, 0, size); err != nil {
				b.Fatal(err)
			}
		}
	})
	for _, arm := range []struct {
		name   string
		reader io.ReaderAt
	}{{"tree-f32", decoded}, {"tree-q2k", newV41EngramRowReaderAt(src, rows, v41EngramQ2KRowBytes)}} {
		b.Run(arm.name, func(b *testing.B) {
			b.SetBytes(int64(len(raw)))
			for i := 0; i < b.N; i++ {
				if _, err := model.NewV41AcquiredEngramRowSource(arm.reader, shard, bind); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
