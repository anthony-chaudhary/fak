package ggufload

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"reflect"
	"testing"
)

type tensorRowReader struct {
	data    []byte
	reads   int
	bytes   int
	offsets []int64
	short   bool
}

func (r *tensorRowReader) ReadAt(p []byte, off int64) (int, error) {
	r.reads++
	r.offsets = append(r.offsets, off)
	if r.short && len(p) > 0 {
		p = p[:len(p)-1]
	}
	n := copy(p, r.data[off:])
	r.bytes += n
	if n != len(p) || r.short {
		return n, io.EOF
	}
	return n, nil
}

type tensorRowZeroReader struct {
	bytes   int
	offsets []int64
}

func (r *tensorRowZeroReader) ReadAt(p []byte, off int64) (int, error) {
	r.bytes += len(p)
	r.offsets = append(r.offsets, off)
	clear(p)
	return len(p), nil
}

func tensorRowSource(info TensorInfo, r io.ReaderAt, size int64) *WeightSource {
	return &WeightSource{
		File:   &File{Tensors: []TensorInfo{info}},
		r:      r,
		size:   size,
		byName: map[string]int{info.Name: 0},
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTensorRowF32ReadsOnlyRequestedF32Row(t *testing.T) {
	const offset = 13
	info := TensorInfo{Name: "table", Dims: []uint64{4, 3}, Type: TensorF32, FileOffset: offset}
	data := make([]byte, offset+4*3*4)
	for i := 0; i < 12; i++ {
		binary.LittleEndian.PutUint32(data[offset+i*4:], math.Float32bits(float32(i+1)))
	}
	r := &tensorRowReader{data: data}
	got, gotInfo, err := tensorRowSource(info, r, int64(len(data))).TensorRowF32("table", 1)
	if err != nil {
		t.Fatalf("TensorRowF32: %v", err)
	}
	if want := []float32{5, 6, 7, 8}; !reflect.DeepEqual(got, want) {
		t.Fatalf("row = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(gotInfo, info) {
		t.Fatalf("info = %+v, want %+v", gotInfo, info)
	}
	if r.reads != 1 || r.bytes != 16 || !reflect.DeepEqual(r.offsets, []int64{offset + 16}) {
		t.Fatalf("reads=%d bytes=%d offsets=%v, want one 16-byte read at %d", r.reads, r.bytes, r.offsets, offset+16)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTensorRowF32Q8MatchesWholeTensorDequant(t *testing.T) {
	const offset = 7
	info := TensorInfo{Name: "q8", Dims: []uint64{qk8_0, 2}, Type: TensorQ8_0, FileOffset: offset}
	raw := make([]byte, 2*blockQ8_0Bytes)
	for row := 0; row < 2; row++ {
		base := row * blockQ8_0Bytes
		binary.LittleEndian.PutUint16(raw[base:], 0x3c00) // f16(1)
		for i := 0; i < qk8_0; i++ {
			raw[base+2+i] = byte(int8(i - 16 + row))
		}
	}
	data := append(make([]byte, offset), raw...)
	got, _, err := tensorRowSource(info, bytes.NewReader(data), int64(len(data))).TensorRowF32("q8", 1)
	if err != nil {
		t.Fatalf("TensorRowF32: %v", err)
	}
	all, err := dequantF32(info, raw)
	if err != nil {
		t.Fatalf("dequantF32: %v", err)
	}
	if want := all[qk8_0:]; !reflect.DeepEqual(got, want) {
		t.Fatalf("row = %v, want %v", got, want)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTensorRowF32HugeVirtualTensorReadsOneRowFromShard(t *testing.T) {
	const (
		offset = int64(4096)
		rows   = uint64(1 << 30)
		row    = 1 << 29
	)
	info := TensorInfo{Name: "huge", Dims: []uint64{qk8_0, rows}, Type: TensorQ8_0, FileOffset: offset}
	primary := &tensorRowZeroReader{}
	shard := &tensorRowZeroReader{}
	ws := tensorRowSource(info, primary, offset+int64(rows*blockQ8_0Bytes))
	ws.readerFor = []io.ReaderAt{shard}
	ws.sizeFor = []int64{offset + int64(rows*blockQ8_0Bytes)}
	got, _, err := ws.TensorRowF32("huge", row)
	if err != nil {
		t.Fatalf("TensorRowF32: %v", err)
	}
	if len(got) != qk8_0 || primary.bytes != 0 || shard.bytes != blockQ8_0Bytes {
		t.Fatalf("len=%d primary bytes=%d shard bytes=%d", len(got), primary.bytes, shard.bytes)
	}
	wantOffset := offset + int64(row*blockQ8_0Bytes)
	if !reflect.DeepEqual(shard.offsets, []int64{wantOffset}) {
		t.Fatalf("shard offsets=%v, want [%d]", shard.offsets, wantOffset)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTensorRowF32RefusesInvalidRequestsBeforeAllocation(t *testing.T) {
	valid := TensorInfo{Name: "table", Dims: []uint64{4, 2}, Type: TensorF32}
	tests := []struct {
		name string
		ws   func(*tensorRowReader) *WeightSource
		row  int
	}{
		{name: "nil source", ws: func(*tensorRowReader) *WeightSource { return nil }},
		{name: "missing", ws: func(r *tensorRowReader) *WeightSource {
			return &WeightSource{File: &File{}, r: r, byName: map[string]int{}}
		}},
		{name: "not 2d", ws: func(r *tensorRowReader) *WeightSource {
			return tensorRowSource(TensorInfo{Name: "table", Dims: []uint64{4}, Type: TensorF32}, r, 0)
		}},
		{name: "negative row", ws: func(r *tensorRowReader) *WeightSource { return tensorRowSource(valid, r, 32) }, row: -1},
		{name: "high row", ws: func(r *tensorRowReader) *WeightSource { return tensorRowSource(valid, r, 32) }, row: 2},
		{name: "quant row not block aligned", ws: func(r *tensorRowReader) *WeightSource {
			return tensorRowSource(TensorInfo{Name: "table", Dims: []uint64{31, 2}, Type: TensorQ8_0}, r, 0)
		}},
		{name: "decoded row oversized", ws: func(r *tensorRowReader) *WeightSource {
			return tensorRowSource(TensorInfo{Name: "table", Dims: []uint64{maxTensorRowF32Bytes/4 + 1, 1}, Type: TensorF32}, r, 0)
		}},
		{name: "element count overflow", ws: func(r *tensorRowReader) *WeightSource {
			return tensorRowSource(TensorInfo{Name: "table", Dims: []uint64{4, math.MaxUint64}, Type: TensorF32}, r, math.MaxInt64)
		}},
		{name: "offset overflow", ws: func(r *tensorRowReader) *WeightSource {
			return tensorRowSource(TensorInfo{Name: "table", Dims: []uint64{4, 2}, Type: TensorF32, FileOffset: math.MaxInt64 - 16}, r, math.MaxInt64)
		}},
		{name: "file overrun", ws: func(r *tensorRowReader) *WeightSource { return tensorRowSource(valid, r, 31) }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := &tensorRowReader{}
			got, _, err := tc.ws(r).TensorRowF32("table", tc.row)
			if !errors.Is(err, ErrTensorRow) {
				t.Fatalf("error=%v, want ErrTensorRow", err)
			}
			if got != nil {
				t.Fatalf("row=%v, want nil on refusal", got)
			}
			if r.reads != 0 {
				t.Fatalf("reader called %d times before refusal", r.reads)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTensorRowF32RefusesShortRead(t *testing.T) {
	info := TensorInfo{Name: "table", Dims: []uint64{4, 2}, Type: TensorF32}
	r := &tensorRowReader{data: make([]byte, 32), short: true}
	got, _, err := tensorRowSource(info, r, 32).TensorRowF32("table", 0)
	if !errors.Is(err, ErrTensorRow) || !errors.Is(err, io.EOF) {
		t.Fatalf("error=%v, want ErrTensorRow wrapping io.EOF", err)
	}
	if got != nil {
		t.Fatalf("row=%v, want nil on short read", got)
	}
}
