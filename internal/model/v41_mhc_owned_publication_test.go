package model

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41MHCOwnedPublication(t *testing.T) {
	const width = 64
	const layer = 3
	streams := make([][]float32, 4)
	for h := range streams {
		streams[h] = make([]float32, width)
		for d := range streams[h] {
			streams[h][d] = float32((h+1)*(d%11-5)) / 13
		}
	}
	mix := v41MHCMix{
		pre:  []float32{0.125, -0.25, 0.5, 0.75},
		post: []float32{0.25, 0.5, 0.75, 1},
		comb: []float32{1, 0.25, 0, 0, 0, 1, 0.5, 0, 0, 0, 1, 0.75, 0.125, 0, 0, 1},
	}
	// Deliberately alias a borrowed residual row. Neither rounding nor mixing
	// may change it, and every published stream must remain independent.
	x := streams[0]
	before := make([][]float32, len(streams))
	for h := range streams {
		before[h] = append([]float32(nil), streams[h]...)
	}
	assertRows := func(t *testing.T, got, want [][]float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("rows=%d want=%d", len(got), len(want))
		}
		for h := range want {
			if len(got[h]) != len(want[h]) {
				t.Fatalf("row %d width=%d want=%d", h, len(got[h]), len(want[h]))
			}
			for d := range want[h] {
				if math.Float32bits(got[h][d]) != math.Float32bits(want[h][d]) {
					t.Fatalf("row %d value %d=%g want=%g", h, d, got[h][d], want[h][d])
				}
			}
		}
	}
	preControl := func() ([][]float32, error) {
		row, err := v41MHCPre(streams, mix.pre)
		if err == nil {
			row, err = v41MHCBF16(layer, row)
		}
		return [][]float32{row}, err
	}
	postControl := func() ([][]float32, error) {
		row, err := v41MHCBF16(layer, x)
		if err != nil {
			return nil, err
		}
		out, err := v41MHCPost(row, streams, mix.post, mix.comb)
		if err != nil {
			return nil, err
		}
		for h := range out {
			out[h], err = v41MHCBF16(layer, out[h])
			if err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	initialControl := func() ([][]float32, error) {
		row, err := v41MHCBF16(-1, x)
		if err != nil {
			return nil, err
		}
		out := make([][]float32, len(streams))
		for h := range out {
			out[h] = append([]float32(nil), row...)
		}
		return out, nil
	}
	for _, tc := range []struct {
		name         string
		run, control func() ([][]float32, error)
	}{
		{"pre", func() ([][]float32, error) {
			row, err := v41MHCPreBF16On(nil, layer, streams, mix.pre)
			return [][]float32{row}, err
		}, preControl},
		{"post", func() ([][]float32, error) { return v41MHCPostBF16On(nil, layer, x, streams, mix) }, postControl},
		{"initial", func() ([][]float32, error) { return v41FullInitialStreamsOn(nil, x) }, initialControl},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.run()
			if err != nil {
				t.Fatal(err)
			}
			want, err := tc.control()
			if err != nil {
				t.Fatal(err)
			}
			assertRows(t, got, want)
			assertRows(t, streams, before)
			for h := range got {
				saved := got[h][0]
				got[h][0] = saved + 100
				assertRows(t, streams, before)
				for other := range got {
					if other != h && math.Float32bits(got[other][0]) != math.Float32bits(want[other][0]) {
						t.Fatal("published streams alias each other")
					}
				}
				got[h][0] = saved
			}
			var sink [][]float32
			var runErr error
			actual := testing.AllocsPerRun(50, func() { sink, runErr = tc.run() })
			if runErr != nil {
				t.Fatal(runErr)
			}
			assertRows(t, sink, want)
			control := testing.AllocsPerRun(50, func() { sink, runErr = tc.control() })
			if runErr != nil {
				t.Fatal(runErr)
			}
			assertRows(t, sink, want)
			assertRows(t, streams, before)
			if actual >= control {
				t.Fatalf("owned mHC publication still copies: path=%s actual_allocs=%g allocating_control=%g", tc.name, actual, control)
			}
		})
	}
	t.Run("refusal", func(t *testing.T) {
		empty, err := v41FullInitialStreamsOn(nil, nil)
		if err != nil || len(empty) != len(streams) {
			t.Fatalf("empty initial streams=%v err=%v", empty, err)
		}
		for _, row := range empty {
			if row != nil {
				t.Fatal("empty initial publication changed nil-row representation")
			}
		}
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32} {
			input := []float32{0.25, bad, float32(math.NaN())}
			saved := append([]float32(nil), input...)
			borrowed := [][]float32{input, make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))}
			for index, run := range []func() error{
				func() error { _, err := v41FullInitialStreamsOn(nil, input); return err },
				func() error { _, err := v41MHCPreBF16On(nil, layer, borrowed, []float32{1, 0, 0, 0}); return err },
				func() error { _, err := v41MHCPostBF16On(nil, layer, input, borrowed, mix); return err },
				func() error {
					_, err := v41MHCPostBF16On(nil, layer, make([]float32, len(input)), borrowed, mix)
					return err
				},
			} {
				err := run()
				wantLayer := layer
				if index == 0 {
					wantLayer = -1
				}
				var stage *V41ForwardError
				if !errors.As(err, &stage) || stage.Stage != v41StageMHC || stage.Layer != wantLayer || !errors.Is(err, ErrV41ForwardStage) || !strings.HasSuffix(err.Error(), "non-finite full mHC BF16 value 1") {
					t.Fatalf("first invalid publication changed: %v", err)
				}
				assertRows(t, [][]float32{input}, [][]float32{saved})
			}
		}
	})
}
