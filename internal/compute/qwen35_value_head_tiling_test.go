package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=5ms
func TestQwen35TiledValueHeadSourceIsPermutation(t *testing.T) {
	for _, g := range []struct{ nK, nV int }{{16, 48}, {2, 4}, {4, 4}, {1, 3}, {3, 12}} {
		src, err := Qwen35TiledValueHeadSource(g.nK, g.nV)
		if err != nil {
			t.Fatalf("nK=%d nV=%d: %v", g.nK, g.nV, err)
		}
		ratio := g.nV / g.nK
		seen := make([]bool, g.nV)
		for c, h := range src {
			if h < 0 || h >= g.nV || seen[h] {
				t.Fatalf("nK=%d nV=%d: position %d reads head %d twice or out of range", g.nK, g.nV, c, h)
			}
			seen[h] = true
			// Tiled position c = r*nK + k holds grouped head k*ratio + r: both share key head k.
			r, k := c/g.nK, c%g.nK
			if h != k*ratio+r || h/ratio != c%g.nK {
				t.Fatalf("nK=%d nV=%d: position %d reads head %d, want %d", g.nK, g.nV, c, h, k*ratio+r)
			}
		}
		if ratio == 1 {
			for c, h := range src {
				if c != h {
					t.Fatalf("nV == nK must be the identity, position %d reads %d", c, h)
				}
			}
		}
	}
}

// fak-test:runtime fast est=5ms
func TestQwen35TiledValueHeadSourceRefusesBadGeometry(t *testing.T) {
	for _, g := range []struct{ nK, nV int }{{0, 4}, {3, 4}, {2, -2}, {-1, 3}} {
		if _, err := Qwen35TiledValueHeadSource(g.nK, g.nV); !errors.Is(err, ErrQwen35ValueHeadTiling) {
			t.Fatalf("nK=%d nV=%d: err=%v, want ErrQwen35ValueHeadTiling", g.nK, g.nV, err)
		}
	}
	if _, err := TileQwen35ValueHeads(make([]float32, 31), 1, 2, 4, 8); !errors.Is(err, ErrQwen35ValueHeadTiling) {
		t.Fatalf("short panel: err=%v, want ErrQwen35ValueHeadTiling", err)
	}
	if _, err := TileQwen35ValueHeads(nil, 0, 2, 4, 8); !errors.Is(err, ErrQwen35ValueHeadTiling) {
		t.Fatalf("zero tokens: err=%v, want ErrQwen35ValueHeadTiling", err)
	}
}

// fak-test:runtime fast est=5ms
func TestTileQwen35ValueHeadsMovesWholeHeadsPerToken(t *testing.T) {
	const tokens, nK, nV, vHd = 3, 2, 6, 4
	width := nV * vHd
	core := make([]float32, tokens*width)
	for i := range core {
		core[i] = float32(i)
	}
	snapshot := append([]float32(nil), core...)
	got, err := TileQwen35ValueHeads(core, tokens, nK, nV, vHd)
	if err != nil {
		t.Fatal(err)
	}
	src, _ := Qwen35TiledValueHeadSource(nK, nV)
	for row := 0; row < tokens; row++ {
		for c, h := range src {
			for d := 0; d < vHd; d++ {
				if want := snapshot[row*width+h*vHd+d]; got[row*width+c*vHd+d] != want {
					t.Fatalf("token %d position %d dim %d = %v, want %v", row, c, d, got[row*width+c*vHd+d], want)
				}
			}
		}
	}
	for i := range core {
		if core[i] != snapshot[i] {
			t.Fatal("tiling mutated its input panel")
		}
	}
}
