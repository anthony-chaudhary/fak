package model

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

type rangeFullEOFReader struct{ data []byte }

func (r rangeFullEOFReader) ReadAt(dst []byte, offset int64) (int, error) {
	if offset < 0 || offset >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(dst, r.data[offset:])
	if offset+int64(n) == int64(len(r.data)) {
		return n, io.EOF
	}
	return n, nil
}

type rangeErrorReader struct{ err error }

func (r rangeErrorReader) ReadAt(dst []byte, _ int64) (int, error) {
	return len(dst), r.err
}

// fak-test:runtime fast est=1ms lane=default
func TestReadRangesFullEOF(t *testing.T) {
	source := []byte("abcdefgh")
	for _, workers := range []int{1, 2, 8} {
		out := make([]byte, len(source))
		ranges := []Range{{Offset: 0, Dst: out[:3]}, {Offset: 3, Dst: out[3:]}}
		if err := ReadRanges(rangeFullEOFReader{source}, ranges, workers); err != nil || !bytes.Equal(out, source) {
			t.Fatalf("workers=%d full EOF read: output=%q err=%v", workers, out, err)
		}
	}
	if err := ReadRanges(rangeFullEOFReader{source}, []Range{{Offset: 6, Dst: make([]byte, 3)}}, 1); !errors.Is(err, io.EOF) {
		t.Fatalf("short EOF must still fail: %v", err)
	}
	want := errors.New("source integrity failure")
	if err := ReadRanges(rangeErrorReader{want}, []Range{{Dst: make([]byte, 2)}}, 1); !errors.Is(err, want) {
		t.Fatalf("full read with non-EOF error must still fail: %v", err)
	}
}
