package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41Q2KParityBackend stands in for Vulkan's two Q2_K kernels: one-row MatMul
// is the per-token reference, while a Q2_K BatchedMatMul is perturbed by one
// ulp, as q2k_matmul's sequential order differs from q2k_matvec (Ref #12640).
// The optional token panel reproduces the per-row MatMul bits.
type v41Q2KParityBackend struct {
	*v41DenseTestBackend
	tile                  bool
	tileCalls, batchedQ2K int
	tileFlatInput         bool
}

func (b *v41Q2KParityBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	y := b.v41DenseTestBackend.BatchedMatMul(w, x, rows)
	if w.Dtype != compute.Q2_K {
		return y
	}
	b.batchedQ2K++
	values := append([]float32(nil), b.v41DenseTestBackend.Read(y)...)
	for i, v := range values {
		values[i] = math.Nextafter32(v, float32(math.Inf(1)))
	}
	return compute.NewF32(b.Backend, y.Shape, values)
}

func (b *v41Q2KParityBackend) SupportsV41Q2KTokenPanel(w compute.Tensor, rows int) bool {
	return b.tile && w.Dtype == compute.Q2_K && rows >= 2 && rows <= 64
}

func (b *v41Q2KParityBackend) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.tileCalls++
	b.tileFlatInput = len(x.Shape) == 1
	in, out := w.Shape[1], w.Shape[0]
	host := b.Backend.Read(x)
	values := make([]float32, 0, rows*out)
	for row := 0; row < rows; row++ {
		xr := compute.NewF32(b.Backend, []int{in}, append([]float32(nil), host[row*in:(row+1)*in]...))
		values = append(values, b.Backend.Read(b.Backend.MatMul(w, xr))...)
	}
	return compute.NewF32(b.Backend, []int{rows * out}, values), nil
}

// v41Q2KParityNoTile hides the token-panel interface entirely.
type v41Q2KParityNoTile struct{ *v41Q2KParityBackend }

func (v41Q2KParityNoTile) SupportsV41Q2KTokenPanel() {}

// A multi-row attn.wkv Q2_K panel must equal the per-token projection bit for
// bit, via the token panel when admitted and per-row MatMul otherwise, and must
// never reach the Q2_K BatchedMatMul whose order differs (Ref #12640). wq_a
// stays on BatchedMatMul: its goldens are pinned on that order.
// fak-test:runtime fast est=300ms lane=default
func TestV41Q2KTokenParityKVPanelMatchesPerToken(t *testing.T) {
	for _, tc := range []struct {
		name      string
		tile      bool
		hideTile  bool
		wantTiles int
	}{
		{name: "token-panel", tile: true, wantTiles: 1},
		{name: "panel-declined-per-row", tile: false},
		{name: "no-panel-interface-per-row", hideTile: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41DenseTestFixture(t, false, true)
			kvName := layerName(0, "attn.wkv.weight")
			shape := v41ProjectionShape(t, m, "attn.wkv.weight")
			out, in := shape[0], shape[1]
			if in%qkK != 0 {
				t.Fatalf("fixture KV input width %d is not Q2_K-blocked", in)
			}
			m.kqw[kvName] = q2kFixtureTensor(out, in, 0x12640)
			delete(m.manifest, kvName)
			delete(m.q8w, kvName)
			pb := &v41Q2KParityBackend{v41DenseTestBackend: newV41DenseTestBackend(), tile: tc.tile}
			var be compute.Backend = pb
			if tc.hideTile {
				be = v41Q2KParityNoTile{pb}
			}
			s := v41DenseTestSession(t, m, be)
			project := s.v41DenseProjectionFunc()
			if project == nil {
				t.Fatal("device dense projection was not bound")
			}
			const rows = 3
			panel := make([]float32, rows*in)
			for i := range panel {
				panel[i] = float32((i*7)%23-11) / 16
			}
			want := make([]float32, 0, rows*out)
			for row := 0; row < rows; row++ {
				y, outcome, err := project(0, "attn.wkv.weight", panel[row*in:(row+1)*in], out, in, 1)
				if err != nil || outcome != v41ProjectionHandled {
					t.Fatalf("per-token KV row %d: outcome=%v err=%v", row, outcome, err)
				}
				want = append(want, y...)
			}
			got, outcome, err := project(0, "attn.wkv.weight", panel, out, in, rows)
			if err != nil || outcome != v41ProjectionHandled {
				t.Fatalf("KV panel: outcome=%v err=%v", outcome, err)
			}
			if len(got) != len(want) {
				t.Fatalf("KV panel length=%d, want %d", len(got), len(want))
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("KV panel index %d = %08x, per-token %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			if pb.batchedQ2K != 0 {
				t.Fatalf("KV panel reached Q2_K BatchedMatMul %d times", pb.batchedQ2K)
			}
			if pb.tileCalls != tc.wantTiles || (tc.wantTiles > 0 && !pb.tileFlatInput) {
				t.Fatalf("token panel calls=%d flat=%t, want %d flat", pb.tileCalls, pb.tileFlatInput, tc.wantTiles)
			}

			qShape := v41ProjectionShape(t, m, "attn.wq_a.weight")
			qPanel := make([]float32, rows*qShape[1])
			if _, outcome, err := project(0, "attn.wq_a.weight", qPanel, qShape[0], qShape[1], rows); err != nil || outcome != v41ProjectionHandled {
				t.Fatalf("wq_a panel: outcome=%v err=%v", outcome, err)
			}
			if pb.batchedQ2K != 1 {
				t.Fatalf("wq_a Q2_K panel BatchedMatMul calls=%d, want 1 (scope widened without re-baseline)", pb.batchedQ2K)
			}
		})
	}
}
