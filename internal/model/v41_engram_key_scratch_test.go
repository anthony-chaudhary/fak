package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=15s lane=default
// The measured boundary is actual injection after a selected projection. It
// excludes fixture setup and restores identical input storage before each call.
func TestV41EngramInjectionKeyScratch(t *testing.T) {
	m, _ := v41FullEngramModelWith(t, func(c *Config) {
		c.HiddenSize, c.NumExperts, c.NumExpertsPerTok = 128, 1, 1
	})
	t.Cleanup(func() { _ = m.CloseWeights() })
	ids := []int{1}
	geometry, rows, err := m.v41EngramLayerRows(0, ids)
	if err != nil || geometry.hc < 3 {
		t.Fatalf("actual Engram geometry: %+v, %v", geometry, err)
	}
	stage := geometry.stage
	gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[geometry.cacheIdx]}, rows, geometry.cols)
	if err != nil {
		t.Fatal(err)
	}
	row := make([]float32, geometry.cols*geometry.dim)
	if err := v41EngramDecodePosition(0, stage, geometry.cacheIdx, gathered, 0, geometry.cols, geometry.dim, row); err != nil {
		t.Fatal(err)
	}
	projection, err := m.v41EngramProjector(0, (geometry.hc+1)*geometry.H, len(row), nil)(row)
	if err != nil || len(projection) != (geometry.hc+1)*geometry.H {
		t.Fatalf("actual host projection width=%d error=%v", len(projection), err)
	}
	bitsEqual := func(a, b []float32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				return false
			}
		}
		return true
	}
	for _, full := range []bool{true, false} {
		name := "full"
		if !full {
			name = "reduced"
		}
		t.Run(name, func(t *testing.T) {
			x, streams := v41FullEngramAsymmetricStreams(ids, geometry.H)
			original := make([][]float32, len(streams[0]))
			for i := range original {
				original[i] = append([]float32(nil), streams[0][i]...)
			}
			reset := func() {
				for i := range original {
					copy(streams[0][i], original[i])
				}
			}
			matches := func(want [][]float32) bool {
				for i := range want {
					if !bitsEqual(streams[0][i], want[i]) {
						return false
					}
				}
				return true
			}
			wantX, wantStreams := v41FullEngramAsymmetricStreams(ids, geometry.H)
			eps := float32(m.Cfg.RMSNormEps)
			if n, err := m.v41EngramInjectPrepared(0, wantX, wantStreams, full, len(ids), rows, eps, geometry); err != nil || n != len(rows) {
				t.Fatalf("actual host injection rows=%d error=%v", n, err)
			}
			if bitsEqual(wantX[0], original[0]) {
				t.Fatal("host injection did not change the residual")
			}
			selectedError := errors.New("selected projection allocation control")
			fail, calls := false, 0
			selected := geometry
			selected.project = func(layer int, input []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
				calls++
				if layer != 0 || out != len(projection) || in != len(row) || !bitsEqual(input, row) {
					panic("allocation control changed its actual projection input")
				}
				if fail {
					return nil, v41ProjectionError, selectedError
				}
				return projection, v41ProjectionHandled, nil
			}
			run := func() (int, error) {
				return m.v41EngramInjectPrepared(0, x, streams, full, len(ids), rows, eps, selected)
			}
			// Warm both identical cache/read paths and establish numerical and
			// publication behavior before interpreting any allocation difference.
			for _, refused := range []bool{false, true} {
				reset()
				fail, calls = refused, 0
				n, err := run()
				if n != len(rows) || calls != len(ids) {
					t.Fatalf("projection boundary rows=%d calls=%d", n, calls)
				}
				if refused {
					if !errors.Is(err, selectedError) || !matches(original) {
						t.Fatalf("selected refusal mutated streams or changed error: %v", err)
					}
				} else if err != nil || !matches(wantStreams[0]) || !bitsEqual(x[0], wantX[0]) {
					t.Fatalf("selected injection differs from actual host projection: %v", err)
				}
			}
			measure := func(refused bool) int64 {
				fail = refused
				result := testing.Benchmark(func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						reset()
						n, err := run()
						if n != len(rows) || refused && !errors.Is(err, selectedError) || !refused && err != nil {
							b.Fatalf("measured injection rows=%d error=%v", n, err)
						}
					}
				})
				return result.AllocedBytesPerOp()
			}
			refusedBytes, successBytes := measure(true), measure(false)
			if !matches(wantStreams[0]) || !bitsEqual(x[0], wantX[0]) {
				t.Fatal("measured successful injection lost parity")
			}
			// Success needs value[H], plus acc[H] only in the reduced path.
			// Allow one additional row for allocator/closure overhead. The old
			// implementation also allocated hc key rows that never escaped.
			rowsAllowed := 2
			if !full {
				rowsAllowed++
			}
			allowance := int64(rowsAllowed * len(ids) * geometry.H * 4)
			delta := successBytes - refusedBytes
			if successBytes <= 0 || refusedBytes <= 0 || delta <= 0 || delta > allowance {
				t.Fatalf("Engram injection key scratch full=%t success=%d refused=%d delta=%d allowance=%d unused-key-rows=%d", full, successBytes, refusedBytes, delta, allowance, geometry.hc*len(ids))
			}
		})
	}
}
