//go:build darwin && arm64 && cgo && metal

package metalgemm

import (
	"fmt"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// runPanel is mlxAttentionFixture.run without its MLX-dispatch expectation, which the tiled
// panel path deliberately bypasses; it returns the gated attention rows and the MLX dispatch delta.
func (f *mlxAttentionFixture) runPanel(t *testing.T, device bool) ([]float32, uint64) {
	t.Helper()
	before := AttentionMLXDispatchCount()
	g, err := BeginProjectionGraph(nil, f.codes, f.scales, f.rows, f.in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	result := make([]*GraphResult, 4)
	for i, w := range f.weights {
		if result[i], err = g.EncodeQ8(w); err != nil {
			t.Fatal(err)
		}
	}
	norm := make([]float32, f.hd)
	for i := range norm {
		norm[i] = 1
	}
	cos := make([]float32, f.rows)
	sin := make([]float32, f.rows)
	for i := range cos {
		cos[i] = 1
	}
	scale := 1 / float32(math.Sqrt(float64(f.hd)))
	var a Qwen35GraphAttentionResult
	if device {
		kv := NewDeviceKV(1, f.base+f.rows, f.kw)
		if kv == nil {
			t.Fatal("KV allocation failed")
		}
		defer kv.Close()
		if err = kv.Upload(1, f.pk); err != nil {
			t.Fatal(err)
		}
		if err = kv.Upload(2, f.pv); err != nil {
			t.Fatal(err)
		}
		a, err = g.FullAttentionDevice(result[0], result[2], result[3], result[1], kv, 0, norm, norm, cos, sin, f.base, f.heads, f.kv, f.hd, 2, scale, 1e-6, false, false)
	} else {
		a, err = g.FullAttention(result[0], result[2], result[3], result[1], norm, norm, cos, sin, f.pk, f.pv, f.base, f.heads, f.kv, f.hd, 2, scale, 1e-6, false, false)
	}
	if err != nil {
		t.Fatal(err)
	}
	out, r, err := g.FinishRead(a.Output)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Committed || !r.CompletedWait {
		t.Fatalf("missing physical completion: %+v", r)
	}
	return out[0], AttentionMLXDispatchCount() - before
}

// fak-test:runtime medium est=20s
func TestQwen35GraphTiledPanelAttentionOracle(t *testing.T) {
	lease, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if !Available() {
		t.Fatal("physical Metal device required")
	}
	t.Setenv("FAK_QWEN35_ATTN_MLX", "")
	t.Setenv("FAK_QWEN35_ATTN_SPLIT", "")
	owners, buffers := graphLiveOwnerCount(), graphLiveBufferCount()
	type geom struct{ hd, heads, kv int }
	cases := []struct {
		g          geom
		rows, base int
	}{
		{geom{256, 16, 4}, 8, 0},
		{geom{256, 16, 4}, 37, 200},
		{geom{256, 16, 4}, 128, 200},
		{geom{128, 8, 2}, 37, 0},
		{geom{128, 8, 2}, 128, 200},
		{geom{64, 4, 1}, 37, 200},
		// Past 4096 keys the per-row path falls back to qg_attn_online; the tiled kernel has no bound.
		{geom{128, 8, 2}, 37, 4100},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("d%d_h%d_kv%d_p%d_b%d", c.g.hd, c.g.heads, c.g.kv, c.rows, c.base), func(t *testing.T) {
			f := newMLXAttentionFixture(t, c.g.hd, c.rows, c.base, c.g.heads, c.g.kv, false)
			defer f.close()
			want := f.oracle()
			for _, device := range []bool{false, true} {
				t.Setenv("FAK_QWEN35_ATTN_TILED", "")
				tiled, mlx := f.runPanel(t, device)
				if mlx != 0 {
					t.Fatalf("device=%v tiled panel still dispatched %d per-row MLX encoders", device, mlx)
				}
				mlxAttentionAssert(t, tiled, want)
				t.Setenv("FAK_QWEN35_ATTN_TILED", "0")
				rowwise, _ := f.runPanel(t, device)
				mlxAttentionAssert(t, rowwise, want)
				mlxAttentionAssert(t, tiled, rowwise)
			}
		})
	}
	t.Run("decode_rows_keep_rowwise_path", func(t *testing.T) {
		t.Setenv("FAK_QWEN35_ATTN_TILED", "")
		f := newMLXAttentionFixture(t, 256, 4, 7, 4, 2, false)
		defer f.close()
		got, mlx := f.runPanel(t, true)
		if mlx != 1 {
			t.Fatalf("rows<8 MLX dispatches=%d want 1", mlx)
		}
		mlxAttentionAssert(t, got, f.oracle())
	})
	if graphLiveOwnerCount() != owners || graphLiveBufferCount() != buffers {
		t.Fatal("tiled attention fixtures leaked native graph resources")
	}
}
