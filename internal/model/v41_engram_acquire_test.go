package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"math/rand"
	"sync/atomic"
	"testing"
)

var errV41EngramAcquireTestRead = errors.New("injected read failure")

type v41EngramFailAtReader struct {
	inner  io.ReaderAt
	failAt int64
}

func (r v41EngramFailAtReader) ReadAt(p []byte, off int64) (int, error) {
	if off <= r.failAt && r.failAt < off+int64(len(p)) {
		return 0, errV41EngramAcquireTestRead
	}
	return r.inner.ReadAt(p, off)
}

type v41EngramCountingReader struct {
	inner io.ReaderAt
	reads atomic.Int64
}

func (r *v41EngramCountingReader) ReadAt(p []byte, off int64) (int, error) {
	r.reads.Add(1)
	return r.inner.ReadAt(p, off)
}

// TestV41EngramDigestExtentPipelinedMatchesSerial pins the parallel read-ahead
// digest to the serial reference and to a direct sha256 over the extent, across
// chunk-boundary sizes, offsets and pool sizes; a read failure reports the same
// first-failing chunk as the serial loop.
// fak-test:runtime fast est=1s lane=default
func TestV41EngramDigestExtentPipelinedMatchesSerial(t *testing.T) {
	const chunk = v41EngramDigestChunk
	data := make([]byte, 5*chunk+4096)
	rand.New(rand.NewSource(41)).Read(data)
	reader := bytes.NewReader(data)
	cases := []struct{ off, size int64 }{
		{0, 1}, {0, chunk - 1}, {0, chunk}, {0, chunk + 1}, {7, 3*chunk + 123}, {0, int64(len(data))}, {4096, 5 * chunk},
	}
	for _, c := range cases {
		sum := sha256.Sum256(data[c.off : c.off+c.size])
		want := "sha256:" + hex.EncodeToString(sum[:])
		serial, err := v41EngramDigestExtentSerial(reader, c.off, c.size)
		if err != nil || serial != want {
			t.Fatalf("serial off=%d size=%d = %q, %v; want %q", c.off, c.size, serial, err, want)
		}
		for _, workers := range []int{2, 3, 8, 16} {
			got, err := v41EngramDigestExtentWorkers(reader, c.off, c.size, workers)
			if err != nil || got != want {
				t.Fatalf("workers=%d off=%d size=%d = %q, %v; want %q", workers, c.off, c.size, got, err, want)
			}
		}
		if got, err := v41EngramDigestExtent(reader, c.off, c.size); err != nil || got != want {
			t.Fatalf("default pool off=%d size=%d = %q, %v; want %q", c.off, c.size, got, err, want)
		}
	}

	failing := v41EngramFailAtReader{inner: reader, failAt: 3*chunk + 10}
	_, serialErr := v41EngramDigestExtentSerial(failing, 0, int64(len(data)))
	_, pipeErr := v41EngramDigestExtentWorkers(failing, 0, int64(len(data)), 4)
	if !errors.Is(serialErr, errV41EngramAcquireTestRead) || !errors.Is(pipeErr, errV41EngramAcquireTestRead) {
		t.Fatalf("read failure not propagated: serial=%v pipelined=%v", serialErr, pipeErr)
	}
	if serialErr.Error() != pipeErr.Error() {
		t.Fatalf("pipelined error %q differs from serial %q", pipeErr, serialErr)
	}
	short := io.NewSectionReader(reader, 0, 2*chunk+5)
	_, serialErr = v41EngramDigestExtentSerial(short, 0, 4*chunk)
	_, pipeErr = v41EngramDigestExtentWorkers(short, 0, 4*chunk, 4)
	if serialErr == nil || pipeErr == nil || serialErr.Error() != pipeErr.Error() {
		t.Fatalf("short extent: serial=%v pipelined=%v, want the same refusal", serialErr, pipeErr)
	}
}

// TestV41AcquiredEngramRowSourceBindsObservedDigest proves the single-pass
// acquisition binds the same content address the digest-then-verify route
// admitted, refuses a caller-supplied digest, and refuses geometry before
// reading any payload byte.
// fak-test:runtime fast est=1s lane=default
func TestV41AcquiredEngramRowSourceBindsObservedDigest(t *testing.T) {
	const rows = 3000
	f32 := make([]byte, rows*V41EngramF32RowBytes)
	rand.New(rand.NewSource(7)).Read(f32)
	size := int64(len(f32))
	shard := V41EngramShard{ID: "blk.1.engram_embd.weight", ExpectedID: "blk.1.engram_embd.weight", Size: size, Rows: rows}
	bind := V41EngramArtifactBinding{Shard: shard.ID, Size: size, Rows: rows, RowBytes: V41EngramF32RowBytes}

	src, err := NewV41AcquiredEngramRowSource(bytes.NewReader(f32), shard, bind)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	got, ok := V41EngramBinding(src)
	if !ok {
		t.Fatal("acquired source carries no verified binding")
	}
	digest, err := v41EngramTreeDigest(bytes.NewReader(f32), 0, size, v41EngramEncodingRows, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got.Digest != digest {
		t.Fatalf("acquired digest %q, want the rows tree digest %q", got.Digest, digest)
	}
	bound := bind
	bound.Digest = digest
	if _, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), shard, bound); err != nil {
		t.Fatalf("verified route refuses the binding acquisition produced: %v", err)
	}
	flat := bind
	flat.Digest = v41EngramTestDigest(f32)
	if _, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), shard, flat); err != nil {
		t.Fatalf("verified route refuses a flat sha256 binding of the same bytes: %v", err)
	}
	row := make([]byte, V41EngramF32RowBytes)
	if _, err := src.ReadRows(rows-1, 1, row); err != nil || !bytes.Equal(row, f32[(rows-1)*V41EngramF32RowBytes:]) {
		t.Fatalf("acquired source row read = %v, bytes equal %t", err, bytes.Equal(row, f32[(rows-1)*V41EngramF32RowBytes:]))
	}

	var typed *V41EngramStreamError
	if _, err := NewV41AcquiredEngramRowSource(bytes.NewReader(f32), shard, bound); !errors.As(err, &typed) || typed.Kind != V41EngramStreamGeometry {
		t.Fatalf("caller-supplied digest = %v, want a typed geometry refusal", err)
	}
	counting := &v41EngramCountingReader{inner: bytes.NewReader(f32)}
	mismatched := shard
	mismatched.ID, mismatched.ExpectedID = "blk.2.engram_embd.weight", "blk.2.engram_embd.weight"
	if _, err := NewV41AcquiredEngramRowSource(counting, mismatched, bind); !errors.As(err, &typed) || typed.Kind != V41EngramStreamShardMismatch {
		t.Fatalf("mis-paired shard = %v, want a typed shard mismatch", err)
	}
	if n := counting.reads.Load(); n != 0 {
		t.Fatalf("geometry refusal read %d payload chunks, want 0", n)
	}
}
