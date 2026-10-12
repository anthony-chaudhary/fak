package model

// v41_engram_acquire.go — the load-time cost of binding a V4.1 Engram table.
//
// NewV41AcquiredEngramRowSource binds what it observes, once, under the tree
// scheme (v41_engram_digest_tree.go): over the Q2_K blocks when the reader names
// them, so the published table is hashed at checkpoint size on a bounded leaf
// pool and never dequantized to be bound. The flat "sha256:" form below is the scheme
// acquisition bound before; it stays the verifier for bindings that carry it.
// Its chunk reads run ahead on a bounded worker pool while one goroutine hashes
// them in extent order, so it is the serial form's byte-for-byte.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"runtime"
	"strings"
)

const (
	v41EngramDigestChunk      = 1 << 20
	v41EngramDigestMaxWorkers = 8
)

// v41EngramDigestWorkers sizes the read-ahead pool: one worker per available
// core up to a fixed cap, never more than the extent has chunks. Host memory is
// bounded by 2 chunks per worker (16 MiB at the cap).
func v41EngramDigestWorkers(size int64) int {
	chunks := (size + v41EngramDigestChunk - 1) / v41EngramDigestChunk
	w := runtime.GOMAXPROCS(0)
	if w > v41EngramDigestMaxWorkers {
		w = v41EngramDigestMaxWorkers
	}
	if int64(w) > chunks {
		w = int(chunks)
	}
	if w < 1 {
		w = 1
	}
	return w
}

type v41EngramDigestChunkResult struct {
	buf []byte
	got int
	err error
}

// v41EngramDigestExtentWorkers is v41EngramDigestExtent with an explicit pool
// size. Chunk k is read by worker k%workers into one of that worker's two
// buffers; the hasher consumes chunk results strictly in k order, so both the
// digest and the first reported read error are the serial loop's.
func v41EngramDigestExtentWorkers(reader io.ReaderAt, offset, size int64, workers int) (string, error) {
	if size <= 0 {
		return "", fmt.Errorf("extent size %d must be positive", size)
	}
	if workers <= 1 {
		return v41EngramDigestExtentSerial(reader, offset, size)
	}
	chunks := (size + v41EngramDigestChunk - 1) / v41EngramDigestChunk
	done := make(chan struct{})
	out := make([]chan v41EngramDigestChunkResult, workers)
	free := make([]chan []byte, workers)
	for w := 0; w < workers; w++ {
		out[w] = make(chan v41EngramDigestChunkResult, 2)
		free[w] = make(chan []byte, 2)
		free[w] <- make([]byte, v41EngramDigestChunk)
		free[w] <- make([]byte, v41EngramDigestChunk)
		go func(w int) {
			for k := int64(w); k < chunks; k += int64(workers) {
				var buf []byte
				select {
				case buf = <-free[w]:
				case <-done:
					return
				}
				at := offset + k*v41EngramDigestChunk
				n := size - k*v41EngramDigestChunk
				if n > v41EngramDigestChunk {
					n = v41EngramDigestChunk
				}
				got, err := reader.ReadAt(buf[:n], at)
				if int64(got) != n && err == nil {
					err = io.ErrUnexpectedEOF
				} else if int64(got) == n {
					err = nil
				}
				select {
				case out[w] <- v41EngramDigestChunkResult{buf: buf, got: got, err: err}:
				case <-done:
					return
				}
			}
		}(w)
	}
	defer close(done)
	h := sha256.New()
	for k := int64(0); k < chunks; k++ {
		w := int(k % int64(workers))
		r := <-out[w]
		if r.err != nil {
			at := offset + k*v41EngramDigestChunk
			n := size - k*v41EngramDigestChunk
			if n > v41EngramDigestChunk {
				n = v41EngramDigestChunk
			}
			return "", fmt.Errorf("hashing extent at offset %d: read %d want %d: %w", at, r.got, n, r.err)
		}
		h.Write(r.buf[:r.got])
		free[w] <- r.buf
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

// v41EngramDigestExtentSerial is the single-goroutine reference form.
func v41EngramDigestExtentSerial(reader io.ReaderAt, offset, size int64) (string, error) {
	if size <= 0 {
		return "", fmt.Errorf("extent size %d must be positive", size)
	}
	h := sha256.New()
	buf := make([]byte, v41EngramDigestChunk)
	remaining := size
	at := offset
	for remaining > 0 {
		n := int64(v41EngramDigestChunk)
		if n > remaining {
			n = remaining
		}
		got, err := reader.ReadAt(buf[:n], at)
		if int64(got) != n {
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return "", fmt.Errorf("hashing extent at offset %d: read %d want %d: %w", at, got, n, err)
		}
		h.Write(buf[:got])
		at += int64(got)
		remaining -= int64(got)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

// NewV41AcquiredEngramRowSource is the loader's acquisition-time form of
// NewV41VerifiedEngramRowSource: the binding names the shard, extent, row count
// and row width, and its Digest must be empty, because the content address is
// what this constructor observes. The extent is hashed exactly once, under the
// tree scheme (over the encoded extent when the reader names one), and the
// returned source carries the completed binding (V41EngramBinding), so the
// loader no longer hashes the extent to build a binding and again to verify it.
// Geometry is refused before any payload byte is read.
func NewV41AcquiredEngramRowSource(reader io.ReaderAt, shard V41EngramShard, binding V41EngramArtifactBinding) (V41EngramRowSource, error) {
	if strings.TrimSpace(binding.Digest) != "" {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("acquired binding must not carry a digest (got %q); use NewV41VerifiedEngramRowSource to check a known digest", binding.Digest)}
	}
	if err := binding.validateExtent(); err != nil {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("invalid artifact binding: %w", err)}
	}
	base, rowBytes, err := v41BoundEngramBase(reader, shard, binding)
	if err != nil {
		return nil, err
	}
	digest, err := v41EngramAcquireDigest(reader, binding, rowBytes)
	if err != nil {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamBindingMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("cannot hash shard bytes: %w", err)}
	}
	binding.Digest = digest
	return &v41VerifiedEngramRowSource{inner: base, binding: binding, rowBytes: rowBytes, encoded: v41EngramBoundEncoded(reader, digest)}, nil
}
