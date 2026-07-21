package model

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// weightsource_ranges_test.go — witnesses for ReadRanges (issue #3250):
//   1. byte-identity vs a sequential read of the same ranges, over both an in-memory blob and
//      a temp-file backed source (the two io.ReaderAt shapes the loader actually sees), and
//   2. first-error-cancels: a failing range stops the run and no ranges past the failure are read.
// No network: the source is always an in-process io.ReaderAt.

// makeGGUFLikeBlob builds a deterministic payload standing in for a GGUF tensor-data region:
// a header-sized preamble followed by several tensor spans of differing sizes, so the ranges
// exercised below mirror the uneven per-tensor spans a real load emits.
func makeGGUFLikeBlob(size int) []byte {
	b := make([]byte, size)
	for i := range b {
		// A non-repeating byte pattern so a mis-offset copy would corrupt visibly rather than
		// alias a neighbouring identical byte.
		b[i] = byte((i*31 + 7) & 0xff)
	}
	return b
}

// tensorSpans partitions [0,size) into contiguous, uneven ranges whose Dst sub-slices all view
// the same reconstruction buffer — the ZML per-shard-writer layout, reduced to one writer.
func tensorSpans(dst []byte) []Range {
	sizes := []int{4, 64, 1, 4096, 17, 1024, 333}
	var ranges []Range
	off := 0
	for _, sz := range sizes {
		if off >= len(dst) {
			break
		}
		if off+sz > len(dst) {
			sz = len(dst) - off
		}
		ranges = append(ranges, Range{Offset: int64(off), Dst: dst[off : off+sz]})
		off += sz
	}
	// Tail span to cover any remainder so the ranges tile the whole blob.
	if off < len(dst) {
		ranges = append(ranges, Range{Offset: int64(off), Dst: dst[off:]})
	}
	return ranges
}

// readRangesSequential is the reference: fill each range one at a time, in order, on the calling
// goroutine. ReadRanges must produce a byte-identical buffer.
func readRangesSequential(src io.ReaderAt, ranges []Range) error {
	for _, r := range ranges {
		if _, err := src.ReadAt(r.Dst, r.Offset); err != nil {
			return err
		}
	}
	return nil
}

func TestReadRangesMatchesSequential(t *testing.T) {
	const size = 5559
	blob := makeGGUFLikeBlob(size)

	// Case A: in-memory source (bytes.Reader satisfies io.ReaderAt).
	memSrc := bytes.NewReader(blob)

	// Case B: temp-file source (os.File satisfies io.ReaderAt) — the local-file loader path.
	dir := t.TempDir()
	path := filepath.Join(dir, "weights.bin")
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatalf("write temp blob: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open temp blob: %v", err)
	}
	defer f.Close()

	for _, tc := range []struct {
		name string
		src  io.ReaderAt
	}{
		{"memory", memSrc},
		{"tempfile", f},
	} {
		t.Run(tc.name, func(t *testing.T) {
			seqOut := make([]byte, size)
			if err := readRangesSequential(tc.src, tensorSpans(seqOut)); err != nil {
				t.Fatalf("sequential read: %v", err)
			}
			if !bytes.Equal(seqOut, blob) {
				t.Fatalf("sequential read did not reconstruct the blob")
			}

			for _, p := range []int{1, 2, 4, 8, 64} {
				parOut := make([]byte, size)
				if err := ReadRanges(tc.src, tensorSpans(parOut), p); err != nil {
					t.Fatalf("ReadRanges(parallelism=%d): %v", p, err)
				}
				if !bytes.Equal(parOut, seqOut) {
					t.Fatalf("parallelism=%d produced bytes differing from the sequential read", p)
				}
			}
		})
	}
}

func TestReadRangesEmptyIsNoop(t *testing.T) {
	if err := ReadRanges(bytes.NewReader([]byte("x")), nil, 4); err != nil {
		t.Fatalf("empty ranges should be a no-op success, got %v", err)
	}
}

// failAtReader is an io.ReaderAt that serves the first (failAt-1) reads correctly and fails the
// failAt-th read (1-based), counting every ReadAt call. It lets the cancel test observe exactly
// how many reads happened after the injected failure.
type failAtReader struct {
	blob   []byte
	failAt int64 // fail the failAt-th ReadAt call (1-based)
	calls  int64 // atomic
	boom   error
}

func (f *failAtReader) ReadAt(p []byte, off int64) (int, error) {
	n := atomic.AddInt64(&f.calls, 1)
	if n == f.failAt {
		return 0, f.boom
	}
	return bytes.NewReader(f.blob).ReadAt(p, off)
}

func TestReadRangesFirstErrorCancels(t *testing.T) {
	const size = 4096
	blob := makeGGUFLikeBlob(size)
	boom := errors.New("injected range failure")

	// Ten equal ranges; with parallelism=1 the worker claims them strictly in order, so failing
	// the 3rd read means reads 4..10 must NEVER be attempted (first-error-cancels the rest).
	fr := &failAtReader{blob: blob, failAt: 3, boom: boom}
	out := make([]byte, size)
	var ranges []Range
	step := size / 10
	for i := 0; i < 10; i++ {
		off := i * step
		end := off + step
		if i == 9 {
			end = size
		}
		ranges = append(ranges, Range{Offset: int64(off), Dst: out[off:end]})
	}

	err := ReadRanges(fr, ranges, 1)
	if err == nil {
		t.Fatalf("expected the injected error to propagate, got nil")
	}
	if !errors.Is(err, boom) {
		t.Fatalf("expected wrapped injected error, got %v", err)
	}
	if got := atomic.LoadInt64(&fr.calls); got != 3 {
		t.Fatalf("first-error-cancel: expected exactly 3 reads (1,2 ok + 3rd fails, rest cancelled), got %d", got)
	}
}

// TestReadRangesConcurrentErrorPropagates asserts that with several workers a failing range still
// makes ReadRanges return an error (not silently succeed), the multi-worker half of the semantic.
func TestReadRangesConcurrentErrorPropagates(t *testing.T) {
	const size = 8192
	blob := makeGGUFLikeBlob(size)
	boom := errors.New("injected range failure")
	fr := &failAtReader{blob: blob, failAt: 1, boom: boom} // first read served fails

	out := make([]byte, size)
	var ranges []Range
	step := size / 32
	for i := 0; i < 32; i++ {
		off := i * step
		end := off + step
		if i == 31 {
			end = size
		}
		ranges = append(ranges, Range{Offset: int64(off), Dst: out[off:end]})
	}

	if err := ReadRanges(fr, ranges, 8); !errors.Is(err, boom) {
		t.Fatalf("expected injected error to propagate under parallelism=8, got %v", err)
	}
}
