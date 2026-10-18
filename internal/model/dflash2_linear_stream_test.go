package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"testing"
)

type dflash2LinearRead struct {
	offset int64
	bytes  int
	buffer *byte
}

type dflash2LinearReader struct {
	*bytes.Reader
	reads  []dflash2LinearRead
	failAt int
	fault  error
	cancel context.CancelFunc
}

func (r *dflash2LinearReader) ReadAt(p []byte, offset int64) (int, error) {
	entry := dflash2LinearRead{offset: offset, bytes: len(p)}
	if len(p) > 0 {
		entry.buffer = &p[0]
	}
	r.reads = append(r.reads, entry)
	n, err := r.Reader.ReadAt(p, offset)
	if r.failAt > 0 && len(r.reads) == r.failAt {
		return max(0, n-1), r.fault
	}
	if r.cancel != nil {
		r.cancel()
	}
	return n, err
}

func dflash2LinearFixture(in, out int) (*dflash2Reference, *dflash2LinearReader, []byte, int64) {
	const offset = int64(14)
	data := make([]byte, int(offset)+2*in*out)
	for i := 0; i < in*out; i++ {
		bits := uint16(0x3e00 + (i*37)%0x200)
		if i%3 == 0 {
			bits |= 0x8000
		}
		binary.LittleEndian.PutUint16(data[int(offset)+2*i:], bits)
	}
	r := &dflash2LinearReader{Reader: bytes.NewReader(data)}
	w := &dflash2Weights{r: r, tensors: map[string]dflash2Tensor{
		"matrix": {shape: []int{out, in}, offset: offset, elements: int64(in * out)},
	}}
	return &dflash2Reference{w: w}, r, data, offset
}

// This oracle reads the raw fixture directly, never the production read helper.
// Its independent row/output/k nesting preserves scalar float32 dot order.
func dflash2LinearOracle(data []byte, offset int64, x []float32, in, out int) []float32 {
	y := make([]float32, len(x)/in*out)
	for row := 0; row < len(x)/in; row++ {
		for j := 0; j < out; j++ {
			var sum float32
			for k := 0; k < in; k++ {
				at := int(offset) + 2*(j*in+k)
				weight := math.Float32frombits(uint32(binary.LittleEndian.Uint16(data[at:])) << 16)
				sum += weight * x[row*in+k]
			}
			y[row*out+j] = sum
		}
	}
	return y
}

// fak-test:runtime fast est=100ms lane=default
func TestDFlash2LinearStreamScalarAndReadBounds(t *testing.T) {
	for _, tc := range []struct {
		name          string
		in, out, rows int
	}{
		{"small", 3, 5, 1},
		{"empty-input-still-validates-weights", 3, 5, 0},
		{"multiple-blocks-and-tail", 257, 515, 3},
		{"maximum-row", dflash2MaxReadElements, 3, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, reader, data, offset := dflash2LinearFixture(tc.in, tc.out)
			x := make([]float32, tc.rows*tc.in)
			for i := range x {
				x[i] = float32(math.Sin(float64(i + 1)))
			}
			want := dflash2LinearOracle(data, offset, x, tc.in, tc.out)
			got, err := f.linear(context.Background(), "matrix", x, tc.in, tc.out)
			if err != nil || len(got) != len(want) {
				t.Fatalf("linear length/error: got=%d want=%d err=%v", len(got), len(want), err)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("scalar mismatch at %d: got=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			blockRows := min(tc.out, dflash2MaxReadElements/tc.in)
			wantReads := (tc.out + blockRows - 1) / blockRows
			if len(reader.reads) != wantReads {
				t.Fatalf("ReadAt count=%d want=%d", len(reader.reads), wantReads)
			}
			for i, read := range reader.reads {
				first := i * blockRows
				count := min(blockRows, tc.out-first)
				if read.offset != offset+int64(2*first*tc.in) || read.bytes != 2*count*tc.in || read.bytes > 2*dflash2MaxReadElements || read.buffer != reader.reads[0].buffer {
					t.Fatalf("bounded contiguous scratch read %d: %+v", i, read)
				}
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestDFlash2LinearStreamDropsPartialResult(t *testing.T) {
	const in, out = 257, 515
	f, reader, _, _ := dflash2LinearFixture(in, out)
	marker := errors.New("short second block")
	reader.failAt, reader.fault = 2, marker
	got, err := f.linear(context.Background(), "matrix", make([]float32, in), in, out)
	if got != nil || !errors.Is(err, marker) || !errors.Is(err, io.ErrUnexpectedEOF) || len(reader.reads) != reader.failAt {
		t.Fatalf("partial output escaped after completed first block: got=%v err=%v reads=%d", got, err, len(reader.reads))
	}
	reader.failAt, reader.reads = 0, nil
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reader.cancel = cancel
	got, err = f.linear(ctx, "matrix", make([]float32, in), in, out)
	if got != nil || !errors.Is(err, context.Canceled) || len(reader.reads) == 0 {
		t.Fatalf("post-read cancellation: got=%v err=%v reads=%d", got, err, len(reader.reads))
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestDFlash2ScratchReadFailureAtomicity(t *testing.T) {
	for _, mode := range []string{"partial", "cancel", "short-scratch", "odd-scratch", "oversized-scratch", "negative-span", "closed"} {
		t.Run(mode, func(t *testing.T) {
			f, reader, _, _ := dflash2LinearFixture(3, 2)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			marker := errors.New("partial scratch read")
			dst := []float32{7, 11, 13}
			scratch := make([]byte, 2*len(dst))
			first := int64(0)
			wantRead := false
			switch mode {
			case "partial":
				reader.failAt, reader.fault, wantRead = 1, marker, true
			case "cancel":
				reader.cancel, wantRead = cancel, true
			case "short-scratch":
				scratch = scratch[:len(scratch)-2]
			case "odd-scratch":
				scratch = append(scratch, 0)
			case "oversized-scratch":
				scratch = make([]byte, 2*dflash2MaxReadElements+2)
			case "negative-span":
				first = -1
			case "closed":
				if err := f.w.Close(); err != nil {
					t.Fatal(err)
				}
			}
			err := f.w.readBF16WithScratch(ctx, "matrix", first, dst, scratch)
			if err == nil || (len(reader.reads) != 0) != wantRead || dst[0] != 7 || dst[1] != 11 || dst[2] != 13 {
				t.Fatalf("read error exposed partial destination: mode=%s dst=%v err=%v reads=%d", mode, dst, err, len(reader.reads))
			}
			if mode == "partial" && !errors.Is(err, marker) {
				t.Fatalf("primary read error lost: %v", err)
			}
			if mode == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation lost: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2LinearStreamRejectsGeometryBeforeRead(t *testing.T) {
	f, reader, _, _ := dflash2LinearFixture(3, 2)
	for _, shape := range []struct{ in, out int }{
		{0, 1},
		{dflash2MaxReadElements + 1, 1},
		{1, 0},
		{dflash2MaxReadElements, int(^uint(0) >> 1)},
	} {
		got, err := f.linear(context.Background(), "matrix", nil, shape.in, shape.out)
		if got != nil || err == nil || len(reader.reads) != 0 {
			t.Fatalf("invalid geometry touched weights: %+v got=%v err=%v", shape, got, err)
		}
	}
	// This geometry fits BF16 offset arithmetic but not float32 output bytes.
	out := int(^uint(0)>>1)/4 + 1
	got, err := f.linear(context.Background(), "matrix", []float32{1}, 1, out)
	if got != nil || err == nil || err.Error() != "dflash2: linear output exceeds addressable storage matrix" || len(reader.reads) != 0 {
		t.Fatalf("unaddressable output reached allocation or weights: got=%v err=%v reads=%d", got, err, len(reader.reads))
	}
}
