package model

import (
	"bytes"
	"errors"
	"io"
	"sync/atomic"
	"testing"
)

type mappedExpertReader struct {
	data    []byte
	reads   atomic.Int64
	ranges  atomic.Int64
	rangeFn func(int64, int64) ([]byte, error)
}

func (r *mappedExpertReader) ReadAt(p []byte, off int64) (int, error) {
	r.reads.Add(1)
	return bytes.NewReader(r.data).ReadAt(p, off)
}

func (r *mappedExpertReader) ReadOnlyRange(off, size int64) ([]byte, error) {
	r.ranges.Add(1)
	if r.rangeFn != nil {
		return r.rangeFn(off, size)
	}
	return r.data[off : off+size : off+size], nil
}

// fak-test:runtime fast est=100ms
func TestGGUFExpertReadOnlyRangeBorrowCopyAndValidation(t *testing.T) {
	const prefix, stride = 9, 84
	r := &mappedExpertReader{data: make([]byte, prefix+3*stride)}
	for i := range r.data {
		r.data[i] = byte(i*17 + 3)
	}
	desc := []ggufExpertTensorDesc{{Name: "experts", Offset: prefix, Experts: 3, Rows: 1, Cols: 256, BlockWeights: 256, BlockBytes: stride}}
	s, err := newGGUFExpertSource(r, int64(len(r.data)), desc)
	if err != nil {
		t.Fatal(err)
	}
	view, err := s.readExpertReadOnly("experts", 1)
	if err != nil || !bytes.Equal(view, r.data[prefix+stride:prefix+2*stride]) || &view[0] != &r.data[prefix+stride] || cap(view) != stride || r.reads.Load() != 0 {
		t.Fatalf("indexed mapping not borrowed exactly, err=%v reads=%d", err, r.reads.Load())
	}
	if allocs := testing.AllocsPerRun(100, func() { view, err = s.readExpertReadOnly("experts", 1) }); allocs != 0 || err != nil {
		t.Fatalf("borrow allocated %g times, err=%v", allocs, err)
	}
	copySource, err := newGGUFExpertSource(bytes.NewReader(r.data), int64(len(r.data)), desc)
	if err != nil {
		t.Fatal(err)
	}
	owned, err := copySource.readExpertReadOnly("experts", 1)
	if err != nil || !bytes.Equal(owned, view) || &owned[0] == &view[0] {
		t.Fatalf("missing capability lost byte-identical owned fallback, err=%v", err)
	}
	if allocs := testing.AllocsPerRun(100, func() { owned, err = copySource.readExpertReadOnly("experts", 1) }); allocs != 1 || err != nil {
		t.Fatalf("owned fallback allocated %g times, want 1, err=%v", allocs, err)
	}
	before := r.ranges.Load()
	for _, tc := range []struct {
		name   string
		expert int
		want   error
	}{
		{"missing", 0, ErrGGUFExpertNotFound}, {"experts", -1, ErrGGUFExpertOutOfRange}, {"experts", 3, ErrGGUFExpertOutOfRange},
	} {
		if got, err := s.readExpertReadOnly(tc.name, tc.expert); !errors.Is(err, tc.want) || got != nil {
			t.Fatalf("invalid index returned bytes=%d err=%v", len(got), err)
		}
	}
	if r.ranges.Load() != before {
		t.Fatal("invalid index reached the mapping")
	}
	r.rangeFn = func(off, size int64) ([]byte, error) { return r.data[off : off+size], nil }
	if bounded, err := s.readExpertReadOnly("experts", 1); err != nil || cap(bounded) != stride {
		t.Fatalf("adapter did not cap wider reader view: cap=%d err=%v", cap(bounded), err)
	}
	for _, tc := range []struct {
		name      string
		raw       []byte
		err, want error
	}{
		{"nil", nil, nil, ErrGGUFExpertMetadata}, {"short", view[:stride-1], nil, ErrGGUFExpertMetadata},
		{"long", r.data[:stride+1], nil, ErrGGUFExpertMetadata}, {"error", view, io.ErrUnexpectedEOF, io.ErrUnexpectedEOF},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r.rangeFn = func(int64, int64) ([]byte, error) { return tc.raw, tc.err }
			if got, err := s.readExpertReadOnly("experts", 1); !errors.Is(err, tc.want) || got != nil {
				t.Fatalf("malformed capability returned bytes=%d err=%v", len(got), err)
			}
			if r.reads.Load() != 0 {
				t.Fatal("malformed mapping silently fell back to ReadAt")
			}
		})
	}
}
