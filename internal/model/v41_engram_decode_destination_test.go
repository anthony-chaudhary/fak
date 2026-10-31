package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The four fixtures use finite, exactly representable values. IQ4_NL's oracle
// below deliberately spells out its split-half lookup, not the production helper.
func v41DestinationFixture(rowBytes int) []byte {
	row := make([]byte, rowBytes)
	switch rowBytes {
	case V41EngramPackedRowBytes:
		for j := 0; j < 256; j++ {
			row[j] = byte(j%120) | byte((j%3)/2)*128
		}
		for j := 256; j < len(row); j++ {
			row[j] = byte(124 + j%5)
		}
	case v41EngramQ2KEncodedRowBytes:
		for j := 0; j < 80; j++ {
			row[j] = byte(17*j + 3)
		}
		binary.LittleEndian.PutUint16(row[80:], 0x3c00)
		binary.LittleEndian.PutUint16(row[82:], 0x3400)
	case v41EngramIQ4NLEncodedRowBytes:
		for b := 0; b < 8; b++ {
			binary.LittleEndian.PutUint16(row[b*18:], 0x3800)
			for j := 0; j < 16; j++ {
				row[b*18+2+j] = byte((j+b)%16) | byte((15-j+b)%16)<<4
			}
		}
	case V41EngramF32RowBytes:
		for j := 0; j < 256; j++ {
			binary.LittleEndian.PutUint32(row[j*4:], math.Float32bits(float32(j-128)/8))
		}
		binary.LittleEndian.PutUint32(row, 0x80000000) // retain negative zero bits
	}
	return row
}

func v41DestinationBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	for j := range want {
		if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
			t.Fatalf("value %d: bits %08x, want %08x", j, math.Float32bits(got[j]), math.Float32bits(want[j]))
		}
	}
}

// fak-test:runtime fast est=2s lane=default
func TestV41EngramDecodeDestination(t *testing.T) {
	widths := []int{V41EngramPackedRowBytes, v41EngramQ2KEncodedRowBytes, v41EngramIQ4NLEncodedRowBytes, V41EngramF32RowBytes}
	for _, width := range widths {
		t.Run(fmt.Sprint(width), func(t *testing.T) {
			row := v41DestinationFixture(width)
			before := bytes.Clone(row)
			for _, dim := range []int{1, 7, 32, 129, 256} {
				got := make([]float32, dim+2)
				for j := range got {
					got[j] = -9876
				}
				if err := decodeV41EngramRowInto(got[:dim], 0, row, dim, width); err != nil {
					t.Fatal(err)
				}
				alloc, err := decodeV41EngramRow(row, dim, width)
				if err != nil {
					t.Fatal(err)
				}
				v41DestinationBits(t, got[:dim], alloc)
				var scalar []float32
				switch width {
				case V41EngramPackedRowBytes:
					scalar, err = dequantV41EngramRow(row, dim)
				case v41EngramQ2KEncodedRowBytes:
					scalar, err = dequantV41EngramQ2KRow(row, dim)
				case v41EngramIQ4NLEncodedRowBytes:
					scalar, err = dequantV41EngramIQ4NLRow(row, dim)
				default:
					scalar = alloc
				}
				if err != nil {
					t.Fatal(err)
				}
				v41DestinationBits(t, scalar, alloc)
				if got[dim] != -9876 || got[dim+1] != -9876 {
					t.Fatal("destination tail changed")
				}
				if width == v41EngramIQ4NLEncodedRowBytes {
					table := [16]float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
					for j, value := range got[:dim] {
						b, k := j/32, j%32
						code := row[b*18+2+k%16] >> uint(4*(k/16)) & 15
						if math.Float32bits(value) != math.Float32bits(table[code]*0.5) {
							t.Fatalf("IQ4 split-half mismatch at %d", j)
						}
					}
				} else {
					encoding := compute.V41EngramRowPacked
					if width == v41EngramQ2KEncodedRowBytes {
						encoding = compute.V41EngramRowQ2K
					}
					if width == V41EngramF32RowBytes {
						encoding = compute.V41EngramRowF32
					}
					bits, status := v41EngramRowDecodeRef(row, encoding, dim)
					if status != 0 {
						t.Fatalf("reference status %#x", status)
					}
					for j := range bits {
						if math.Float32bits(got[j]) != bits[j] {
							t.Fatalf("independent reference mismatch at %d", j)
						}
					}
				}
				retained := append([]float32(nil), alloc...)
				got[0]++
				v41DestinationBits(t, alloc, retained)
				// Preserve the existing copy(dst, decoded) contract for a short
				// destination: validate the entire row, then copy only the prefix.
				short := make([]float32, dim-1)
				if err := decodeV41EngramRowInto(short, 0, row, dim, width); err != nil {
					t.Fatal(err)
				}
				v41DestinationBits(t, short, retained[:dim-1])
				offset := make([]float32, dim+2)
				offset[0], offset[dim+1] = -9876, -9876
				if err := decodeV41EngramRowInto(offset, 1, row, dim, width); err != nil {
					t.Fatal(err)
				}
				v41DestinationBits(t, offset[1:dim+1], retained)
				if offset[0] != -9876 || offset[dim+1] != -9876 {
					t.Fatal("offset boundaries changed")
				}
			}
			if !bytes.Equal(row, before) {
				t.Fatal("encoded borrower mutated")
			}

			bad := bytes.Clone(row)
			switch width {
			case V41EngramPackedRowBytes:
				bad[3] = 127
				bad[4] = 255
			case v41EngramQ2KEncodedRowBytes:
				binary.LittleEndian.PutUint16(bad[80:], 0x7c00)
			case v41EngramIQ4NLEncodedRowBytes:
				binary.LittleEndian.PutUint16(bad, 0x7c00)
			case V41EngramF32RowBytes:
				binary.LittleEndian.PutUint32(bad[12:], 0x7f800000)
			}
			cases := []struct {
				row        []byte
				dim, width int
			}{
				{row[:len(row)-1], 0, width}, {row, 0, width}, {row, 257, width}, {bad, 7, width}, {row, 7, 17},
			}
			for _, tc := range cases {
				dst := []float32{1, 2, 3, 4, 5, 6, 7, 8}
				keep := append([]float32(nil), dst...)
				alloc, wantErr := decodeV41EngramRow(tc.row, tc.dim, tc.width)
				err := decodeV41EngramRowInto(dst, 0, tc.row, tc.dim, tc.width)
				if alloc != nil || !errors.Is(err, ErrV41NativeUnsupported) || wantErr == nil || err.Error() != wantErr.Error() {
					t.Fatalf("refusal alloc=%v error=%v, want %v", alloc, err, wantErr)
				}
				v41DestinationBits(t, dst, keep)
				// Existing consumers report decode refusal before evaluating an
				// invalid output offset. Do not turn that into a new bounds panic.
				if err := decodeV41EngramRowInto(dst, len(dst)+1, tc.row, tc.dim, tc.width); err == nil || err.Error() != wantErr.Error() {
					t.Fatalf("offset/refusal order: %v", err)
				}
			}
			// A bad byte in a later decoded block/value must remain outside the
			// requested prefix; Q2_K's shared scales cannot isolate such a suffix.
			if width != v41EngramQ2KEncodedRowBytes {
				later := bytes.Clone(row)
				switch width {
				case V41EngramPackedRowBytes:
					later[32] = 127
				case v41EngramIQ4NLEncodedRowBytes:
					binary.LittleEndian.PutUint16(later[18:], 0x7c00)
				case V41EngramF32RowBytes:
					binary.LittleEndian.PutUint32(later[32*4:], 0x7fc00000)
				}
				dst := make([]float32, 7)
				if err := decodeV41EngramRowInto(dst, 0, later, 7, width); err != nil {
					t.Fatalf("observed suffix: %v", err)
				}
			}
		})
	}

	t.Run("actual_consumers", func(t *testing.T) {
		const dim, cols = 7, 2
		src := &v41EngramMemorySource{packed: v41EngramTestPackedRows(2), rows: 2}
		cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{TableRows: 2, RowBytes: V41EngramPackedRowBytes, BudgetBytes: 2 * V41EngramPackedRowBytes})
		if err != nil {
			t.Fatal(err)
		}
		rows, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, []uint32{0, 1, 1, 0}, cols)
		if err != nil {
			t.Fatal(err)
		}
		stage := &v41EngramStage{rowBytes: V41EngramPackedRowBytes, caches: []*V41EngramRowCache{cache}}
		g := &v41EngramGatheredRows{stage: stage, rows: rows, dim: dim}
		stop := enableV41BF16Witness()
		defer stop()
		v41HostEngramRowDecodes.Store(0)
		panel, err := g.decodeHost()
		if err != nil {
			t.Fatal(err)
		}
		if n := v41HostEngramRowDecodes.Load(); n != 4 {
			t.Fatalf("panel host witnesses %d", n)
		}
		dst := make([]float32, cols*dim)
		if err := v41EngramDecodePosition(0, stage, 0, rows, 1, cols, dim, dst); err != nil {
			t.Fatal(err)
		}
		if n := v41HostEngramRowDecodes.Load(); n != 6 {
			t.Fatalf("position host witnesses %d", n)
		}
		v41DestinationBits(t, dst, panel[cols*dim:])
		panel[0]++
		row, err := cache.Row(0)
		if err != nil || !bytes.Equal(row, rows[0]) {
			t.Fatalf("float panel aliases encoded cache: %v", err)
		}
		poison := bytes.Clone(rows[1])
		poison[0] = 127
		g.rows = [][]byte{rows[0], poison}
		if out, err := g.decodeHost(); out != nil || !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("partial panel escaped: %v %v", out, err)
		}
		if cache.inner.resident != 0 || len(cache.inner.cached) != 0 {
			t.Fatal("decode refusal retained cache rows")
		}
		if err := cache.warmRow(0); err != nil {
			t.Fatal(err)
		}
		for j := range dst {
			dst[j] = -9876
		}
		if err := v41EngramDecodePosition(0, stage, 0, g.rows, 0, cols, dim, dst); !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatal(err)
		}
		for _, value := range dst[dim:] {
			if value != -9876 {
				t.Fatal("faulting row partially published")
			}
		}
		if cache.inner.resident != 0 {
			t.Fatal("position refusal retained cache rows")
		}
	})

	t.Run("native_host_injection", func(t *testing.T) {
		m, layout := v41RectangularEngramModel(t, 7)
		ids := []int{1, 3}
		x, streams := v41FullEngramAsymmetricStreams(ids, m.Cfg.HiddenSize)
		eps := float32(m.Cfg.RMSNormEps)
		want := v41OracleRectangularEngramUpdates(t, m, layout, 0, streams, ids, eps)
		if err := m.v41EngramInject(0, x, streams, true, ids, eps); err != nil {
			t.Fatal(err)
		}
		for pos := range ids {
			for stream := range streams[pos] {
				for j, value := range streams[pos][stream] {
					if math.Abs(float64(value-want[pos][stream][j])) > cpuOracleTol {
						t.Fatalf("injection mismatch at %d/%d/%d", pos, stream, j)
					}
				}
			}
			v41DestinationBits(t, x[pos], streams[pos][0])
		}
	})
}

// Both forms use the same fixture, output storage and decode equations. This
// isolates transient row-result allocation, not source I/O, model RAM or latency.
func BenchmarkV41EngramDecodeDestination(b *testing.B) {
	for _, width := range []int{V41EngramPackedRowBytes, v41EngramQ2KEncodedRowBytes, v41EngramIQ4NLEncodedRowBytes, V41EngramF32RowBytes} {
		for _, into := range []bool{false, true} {
			b.Run(fmt.Sprintf("width%d/into%t", width, into), func(b *testing.B) {
				row, dst := v41DestinationFixture(width), make([]float32, 256)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if into {
						if err := decodeV41EngramRowInto(dst, 0, row, len(dst), width); err != nil {
							b.Fatal(err)
						}
					} else {
						values, err := decodeV41EngramRow(row, len(dst), width)
						if err != nil {
							b.Fatal(err)
						}
						copy(dst, values)
					}
				}
				runtime.KeepAlive(dst)
			})
		}
	}
}
