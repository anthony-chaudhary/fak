package model

import (
	"errors"
	"io"
	"os"
	"testing"
	"unsafe"
)

// lazyDirectReadProbe records the actual destination of each bounded read. It
// synthesizes bytes instead of retaining a second large checkpoint fixture.
type lazyDirectReadProbe struct {
	calls int
	ptrs  [2]*byte
	sizes [2]int
	offs  [2]int64
	err   error
	short bool
}

func (r *lazyDirectReadProbe) ReadAt(p []byte, off int64) (int, error) {
	i := r.calls
	r.calls++
	if i < len(r.ptrs) {
		r.ptrs[i], r.sizes[i], r.offs[i] = &p[0], len(p), off
	}
	n := len(p)
	if r.short {
		n--
	}
	for j := 0; j < n; j++ {
		p[j] = byte((off+int64(j))*31 + 7)
	}
	return n, r.err
}

// fak-test:runtime medium est=1s lane=default
func TestLazyMaterializeDirectFinalRead(t *testing.T) {
	for _, tc := range []struct {
		name string
		run  func(*testing.T, *LazyQ4KRange, []byte) ([]byte, error)
	}{
		{"Q4_K", func(t *testing.T, lazy *LazyQ4KRange, resident []byte) ([]byte, error) {
			q := &q4kTensor{lazy: lazy, raw: resident}
			got, err := q.materializeRaw()
			if len(q.raw) != len(resident) {
				t.Fatal("materialize published a private read result")
			}
			return got, err
		}},
		{"k_quant", func(t *testing.T, lazy *LazyQ4KRange, resident []byte) ([]byte, error) {
			q := &kQuantTensor{lazy: lazy, raw: resident}
			got, err := q.materializeRaw()
			if len(q.raw) != len(resident) {
				t.Fatal("materialize published a private read result")
			}
			return got, err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Run("two_windows_use_final_destination", func(t *testing.T) {
				const offset = int64(19)
				const tail = 431
				r := &lazyDirectReadProbe{}
				got, err := tc.run(t, &LazyQ4KRange{Reader: r, Offset: offset, Bytes: q4kMaterializeWindowBytes + tail}, nil)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != q4kMaterializeWindowBytes+tail || uintptr(unsafe.Pointer(&got[0]))%uintptr(os.Getpagesize()) != 0 {
					t.Fatal("final output length or page alignment changed")
				}
				if r.calls != 2 || r.sizes != [2]int{q4kMaterializeWindowBytes, tail} || r.offs != [2]int64{offset, offset + q4kMaterializeWindowBytes} {
					t.Fatalf("read windows changed: calls=%d sizes=%v offsets=%v", r.calls, r.sizes, r.offs)
				}
				if r.ptrs[0] != &got[0] || r.ptrs[1] != &got[q4kMaterializeWindowBytes] {
					t.Fatal("ReadAt used a scratch destination instead of final page-aligned storage")
				}
				for i, b := range got {
					if b != byte((offset+int64(i))*31+7) {
						t.Fatalf("payload mismatch at %d", i)
					}
				}
				t.Logf("direct final-slice reads: %d payload bytes, %d-byte scratch and full payload copy avoided", len(got), q4kMaterializeWindowBytes)
			})
			t.Run("errors_return_unchanged_without_retry", func(t *testing.T) {
				sentinel := errors.New("checkpoint read failed")
				for _, failure := range []struct {
					err   error
					short bool
				}{{io.EOF, true}, {sentinel, true}, {io.EOF, false}} {
					r := &lazyDirectReadProbe{err: failure.err, short: failure.short}
					got, err := tc.run(t, &LazyQ4KRange{Reader: r, Offset: 19, Bytes: 144}, nil)
					if got != nil || err != failure.err || r.calls != 1 || r.sizes[0] != 144 || r.offs[0] != 19 {
						t.Fatalf("error contract changed: output=%d err=%v reads=%d", len(got), err, r.calls)
					}
				}
			})
			t.Run("mapped_and_resident_aliases_do_not_read", func(t *testing.T) {
				r := &lazyDirectReadProbe{err: errors.New("fast path must not read")}
				span := makePageAlignedResidentBytes(os.Getpagesize())
				lazy := &LazyQ4KRange{Reader: r, Bytes: 144, MappedSpan: span, MappedOffset: 32}
				got, err := tc.run(t, lazy, nil)
				if err != nil || len(got) != 144 || &got[0] != &span[32] || r.calls != 0 {
					t.Fatal("mapped alias fast path changed")
				}
				resident := []byte{1, 2, 3}
				got, err = tc.run(t, lazy, resident)
				if err != nil || len(got) != len(resident) || &got[0] != &resident[0] || r.calls != 0 {
					t.Fatal("resident alias priority changed")
				}
			})
		})
	}
}
