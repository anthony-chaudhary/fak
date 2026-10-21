package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41EngramHostOwnedGather(t *testing.T) {
	// Keep this workload identical in parent/candidate receipts. The parent
	// records actual injection allocations before failing the owned-route check.
	const positions, cols, dim, H, hc = 2, 3, 8, 4, 4
	ids := []uint32{2, 0, 2, 1, 2, 0}
	for _, rowBytes := range []int{V41EngramPackedRowBytes, V41EngramF32RowBytes, v41EngramQ2KEncodedRowBytes} {
		for _, full := range []bool{false, true} {
			t.Run(fmt.Sprintf("row_bytes_%d/full_%t", rowBytes, full), func(t *testing.T) {
				src := newV41EngramBatchSource(3, rowBytes, 0)
				clear(src.data)
				for r := range src.rows {
					row := src.data[r*rowBytes : (r+1)*rowBytes]
					switch rowBytes {
					case V41EngramPackedRowBytes:
						for j := range 256 {
							row[j] = byte(0x30 + (r+j)%8)
						}
						for j := 256; j < rowBytes; j++ {
							row[j] = 127
						}
					case V41EngramF32RowBytes:
						for j := range 256 {
							binary.LittleEndian.PutUint32(row[4*j:], math.Float32bits(float32((r+1)*(j+1))/32))
						}
					case v41EngramQ2KEncodedRowBytes:
						for j := range 16 {
							row[j] = byte(1 + r)
						}
						for j := 16; j < 80; j++ {
							row[j] = byte(0x55 + 0x11*r)
						}
						binary.LittleEndian.PutUint16(row[80:], 0x2400)
					}
				}
				refSource := newV41EngramBatchSource(3, rowBytes, 0)
				copy(refSource.data, src.data)
				refCache := v41EngramBatchCache(t, refSource, 3*rowBytes)
				cache := v41EngramBatchCache(t, src, 3*rowBytes)
				observed := &v41EngramBatchObservedSource{src: src}
				cache.src = observed
				stage := &v41EngramStage{caches: []*V41EngramRowCache{cache}, rowBytes: rowBytes}
				geometry := v41EngramGeometry{stage: stage, H: H, dim: dim, cols: cols, hc: hc,
					qNorm: make([]float32, hc*H), kNorm: make([]float32, hc*H)}
				projection := make([]float32, (hc+1)*H)
				for i := range projection {
					projection[i] = 1
				}
				for i := range geometry.kNorm {
					geometry.kNorm[i] = 1
				}
				wantInput := make([][]float32, positions)
				for i := range wantInput {
					wantInput[i] = make([]float32, cols*dim)
				}
				calls := 0
				geometry.project = func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
					if layer != 0 || out != len(projection) || in != cols*dim || calls >= positions || len(row) != len(wantInput[calls]) {
						t.Fatal("actual injection changed projection geometry")
					}
					for i, value := range row {
						if math.Float32bits(value) != math.Float32bits(wantInput[calls][i]) {
							t.Fatalf("projection position %d input %d differs from public gather", calls, i)
						}
					}
					calls++
					return projection, v41ProjectionHandled, nil
				}
				x, streams := v41FullEngramAsymmetricStreams([]int{1, 2}, H)
				reset := func() {
					calls = 0
					for _, position := range streams {
						for _, stream := range position {
							for i := range stream {
								stream[i] = 1
							}
						}
					}
				}
				m := &Model{}
				run := func() {
					reset()
					n, err := m.v41EngramInjectPrepared(0, x, streams, full, positions, ids, 1e-5, geometry)
					if err != nil || n != len(ids) || calls != positions {
						t.Fatalf("actual injection rows=%d calls=%d error=%v", n, calls, err)
					}
					want := float32(3)
					if full {
						want = 1.5
					}
					for p := range positions {
						for _, value := range x[p] {
							if math.Float32bits(value) != math.Float32bits(want) {
								t.Fatalf("actual injection publication=%g want=%g", value, want)
							}
						}
						if full {
							for _, stream := range streams[p] {
								for _, value := range stream {
									if math.Float32bits(value) != math.Float32bits(want) {
										t.Fatalf("full stream publication=%g want=%g", value, want)
									}
								}
							}
						}
					}
				}
				for pass := range 2 {
					rows, err := GatherV41EngramRows([]*V41EngramRowCache{refCache}, ids, cols)
					if err != nil {
						t.Fatal(err)
					}
					for p := range positions {
						if err := v41EngramDecodePosition(0, stage, 0, rows, p, cols, dim, wantInput[p]); err != nil {
							t.Fatal(err)
						}
					}
					run()
					if cache.Stats() != refCache.Stats() || !reflect.DeepEqual(src.reads, refSource.reads) {
						t.Fatalf("cold/warm pass %d differs from public gather accounting", pass)
					}
				}
				// Both paths are resident now, so stack observation and reference
				// construction allocate nothing inside this measured host route.
				reads := len(src.reads)
				allocs := testing.AllocsPerRun(100, run)
				t.Logf("actual host injection allocations/op=%.0f positions=%d rows=%d rowBytes=%d full=%t", allocs, positions, len(ids), rowBytes, full)
				if len(src.reads) != reads || reads == 0 || !observed.prepared || !observed.gather {
					t.Fatal("measurement lost the prepared injection source boundary or issued warm I/O")
				}
				if !observed.owned {
					t.Fatal("actual host injection did not consume the canonical owned gather")
				}
			})
		}
	}
}
