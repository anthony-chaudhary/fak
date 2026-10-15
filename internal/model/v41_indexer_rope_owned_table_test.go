package model

import (
	"errors"
	"math"
	"testing"
)

func v41OwnedTableEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length: %d != %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("bits[%d]: %08x != %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41IndexerOwnedRotaryTable(t *testing.T) {
	const dim, rd, heads = 128, 64, 2
	m := &Model{Cfg: Config{HeadDim: dim, IndexHeadDim: dim, QKRopeHeadDim: rd, RopeTheta: 10000}}
	seed := make([]float32, heads*dim)
	for i := range seed {
		seed[i] = float32(i+1) / 256
	}
	seed[0] = math.Float32frombits(0x80000000)
	cos, sin := v41RopeTableForLayer(m.Cfg, 0, 17)
	sentinel := errors.New("selected rotary failure")
	for _, mode := range []string{"success", "error", "short", "prefix", "nonfinite"} {
		t.Run(mode, func(t *testing.T) {
			values := append([]float32(nil), seed...)
			want := append([]float32(nil), seed...)
			if mode == "success" {
				for h := 0; h < heads; h++ {
					applyRopeTailInterleaved(want[h*dim:(h+1)*dim], cos, sin, rd)
				}
				if err := v41IndexTailsBF16(0, want, heads, dim, rd, nil); err != nil {
					t.Fatal(err)
				}
			}
			var heldCos, heldSin, heldInput []float32
			callback := func(l int, x, c, s []float32, h, d, r int) ([]float32, error) {
				v41OwnedTableEqual(t, c, cos)
				v41OwnedTableEqual(t, s, sin)
				heldCos, heldSin, heldInput = c, s, x
				for n := 0; n < h; n++ {
					applyRopeTailInterleaved(x[n*d:(n+1)*d], c, s, r)
				}
				c[0], s[0] = 999, -999
				switch mode {
				case "error":
					return nil, sentinel
				case "short":
					return x[:1], nil
				case "prefix":
					x[0] = 1
				case "nonfinite":
					x[d-1] = float32(math.Inf(1))
				}
				return x, nil
			}
			err := m.v41IndexRoPEPublished(0, 17, values, heads, callback, nil)
			if mode == "success" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				var typed *V41ProjectionOperationError
				if !errors.As(err, &typed) || typed.Layer != 0 || typed.Leaf != "indexer.wq_b.weight" {
					t.Fatalf("wrong failure: %v", err)
				}
				if mode == "error" && !errors.Is(err, sentinel) {
					t.Fatal(err)
				}
				if mode != "error" && !errors.Is(err, errV41TailRoPEResult) {
					t.Fatal(err)
				}
			}
			v41OwnedTableEqual(t, values, want)
			// A callback may retain its tables/input. Later writes cannot corrupt the
			// returned caller buffer or another invocation's fresh table generation.
			for i := range heldInput {
				heldInput[i] = 123
			}
			for i := range heldCos {
				heldCos[i] = 123
				heldSin[i] = 123
			}
			v41OwnedTableEqual(t, values, want)
			c, s := v41RopeTableForLayer(m.Cfg, 0, 17)
			v41OwnedTableEqual(t, c, cos)
			v41OwnedTableEqual(t, s, sin)
		})
	}
	for _, withKey := range []bool{false, true} {
		for _, mode := range []string{"success", "error", "short"} {
			latent := append([]float32(nil), seed[:dim]...)
			var key []float32
			if withKey {
				key = append([]float32(nil), seed[dim:]...)
			}
			wantLatent := append([]float32(nil), latent...)
			wantKey := append([]float32(nil), key...)
			if mode == "success" {
				for _, v := range [][]float32{wantLatent, wantKey} {
					if v == nil {
						continue
					}
					applyRopeTailInterleaved(v, cos, sin, rd)
					for i := len(v) - rd; i < len(v); i++ {
						v[i] = v41RoundBF16(v[i])
					}
				}
			}
			var heldCos, heldSin, heldInput []float32
			callback := func(l int, x, c, s []float32, rows, r int) ([]float32, error) {
				v41OwnedTableEqual(t, c, cos)
				v41OwnedTableEqual(t, s, sin)
				heldCos, heldSin, heldInput = c, s, x
				for row := 0; row < rows; row++ {
					applyRopeTailInterleaved(x[row*r:(row+1)*r], c, s, r)
				}
				for i := range x {
					x[i] = v41RoundBF16(x[i])
				}
				c[0], s[0] = 999, -999
				if mode == "error" {
					return nil, sentinel
				}
				if mode == "short" {
					return x[:1], nil
				}
				return x, nil
			}
			err := m.v41CompressedPublicationRoPEWithDevice(0, 17, latent, key, callback)
			if mode == "success" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				var typed *V41TailRoPEOperationError
				if !errors.As(err, &typed) || typed.Layer != 0 {
					t.Fatalf("wrong failure: %v", err)
				}
				cause := sentinel
				if mode == "short" {
					cause = errV41TailRoPEResult
				}
				if !errors.Is(err, cause) {
					t.Fatal(err)
				}
			}
			v41OwnedTableEqual(t, latent, wantLatent)
			v41OwnedTableEqual(t, key, wantKey)
			for i := range heldInput {
				heldInput[i] = 123
			}
			for i := range heldCos {
				heldCos[i] = 123
				heldSin[i] = 123
			}
			v41OwnedTableEqual(t, latent, wantLatent)
			v41OwnedTableEqual(t, key, wantKey)
			c, s := v41RopeTableForLayer(m.Cfg, 0, 17)
			v41OwnedTableEqual(t, c, cos)
			v41OwnedTableEqual(t, s, sin)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41IndexerOwnedRotaryTableAllocs(t *testing.T) {
	const dim, rd = 128, 64
	m := &Model{Cfg: Config{HeadDim: dim, IndexHeadDim: dim, QKRopeHeadDim: rd, RopeTheta: 10000}}
	values := make([]float32, dim)
	latent := make([]float32, dim)
	key := make([]float32, dim)
	rotate := func(_ int, x, c, s []float32, _, _, _ int) ([]float32, error) { return x, nil }
	compress := func(_ int, x, c, s []float32, _, _ int) ([]float32, error) { return x, nil }
	var err error
	query := testing.AllocsPerRun(50, func() { err = m.v41IndexRoPEPublished(0, 17, values, 1, rotate, nil) })
	if err != nil {
		t.Fatal(err)
	}
	publication := testing.AllocsPerRun(50, func() { err = m.v41CompressedPublicationRoPEWithDevice(0, 17, latent, key, compress) })
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("selected query allocations=%g; compressed publication=%g", query, publication)
	// Fresh cos/sin + defensive input/output + one BF16 scratch for query;
	// fresh cos/sin + packed owned tails for compressed publication.
	if query > 5 || publication > 3 {
		t.Fatalf("redundant rotary table copies: query=%g publication=%g", query, publication)
	}
}
