package model

import (
	"errors"
	"reflect"
	"testing"
)

// Exercise the live QKV producer, including callback outputs that alias each
// other and a later-row refusal after earlier rows have been published.
func TestV41QKVPanelReusePublication(t *testing.T) {
	for _, failSecond := range []bool{false, true} {
		name := "complete"
		if failSecond {
			name = "partial refusal"
		}
		t.Run(name, func(t *testing.T) {
			cfg := v41FullGeometryConfig(t)
			m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{
				layerName(0, "attn.wq_a.weight"): {Shape: []int{cfg.QLoraRank, cfg.HiddenSize}},
				layerName(0, "attn.wq_b.weight"): {Shape: []int{cfg.NumHeads * cfg.HeadDim, cfg.QLoraRank}},
				layerName(0, "attn.wkv.weight"):  {Shape: []int{v41KVLoraRank, cfg.HiddenSize}},
			}}
			var projectionBuffers [][]float32
			project := func(_ int, _ string, _ []float32, out, _, n int) ([]float32, v41DenseProjectionOutcome, error) {
				values := make([]float32, n*out)
				for i := range values {
					values[i] = float32(i%7+1) / 8
				}
				projectionBuffers = append(projectionBuffers, values)
				return values, v41ProjectionHandled, nil
			}
			identity := func(_ int, values []float32) ([]float32, error) { return values, nil }
			stop := errors.New("second row rotation refused")
			calls := 0
			var callbackBuffers [][]float32
			rotate := func(_ int, q, _, _, _ []float32, _, _, _ int) ([]float32, []float32, error) {
				calls++
				if failSecond && calls == 2 {
					return nil, nil, stop
				}
				for i := range q {
					q[i] += 0.25
				}
				callbackBuffers = append(callbackBuffers, q)
				return q, q, nil // Deliberately alias Q and KV results.
			}
			scratch := &v41ProjScratch{
				projectionRoute: newV41ProjectionRouteBinding(project, nil),
				queryNorm:       &v41QueryNormBinding{ordinary: identity},
				kvNorm:          &v41KVNormBinding{ordinary: identity},
				tailRoPE:        rotate,
			}
			const seq = 3
			rows := make([][]float32, seq)
			for i := range rows {
				rows[i] = make([]float32, cfg.HiddenSize)
			}
			got, err := m.v41LayerProjectQKV(0, seq, rows, true, cfg.HeadDim, cfg.NumHeads, cfg.HiddenSize, 1e-5, scratch, nil)
			published := seq
			if failSecond {
				published = 1
				if !errors.Is(err, stop) {
					t.Fatalf("error = %v, want stop", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			// Neither projection nor rotation callback storage may escape through the
			// returned rows, including the successful prefix of an interrupted call.
			for _, buffers := range [][][]float32{projectionBuffers, callbackBuffers} {
				for _, values := range buffers {
					for i := range values {
						values[i] = -999
					}
				}
			}
			for row := 0; row < published; row++ {
				want := make([]float32, cfg.HeadDim)
				for i := range want {
					want[i] = float32((row*cfg.HeadDim+i)%7+1)/8 + 0.25
				}
				if !reflect.DeepEqual(got.qHeads[row], want) || !reflect.DeepEqual(got.kvRows[row], want) {
					t.Fatalf("published row %d differs from independently published dyadic values", row)
				}
				if cap(got.qHeads[row]) != len(got.qHeads[row]) || cap(got.kvRows[row]) != len(got.kvRows[row]) {
					t.Fatalf("row %d can append into another row", row)
				}
				got.qHeads[row][0] = -5
				if got.kvRows[row][0] != want[0] {
					t.Fatal("Q and KV rows alias")
				}
			}
			for row := published; row < seq; row++ {
				if got.qHeads[row] != nil || got.kvRows[row] != nil || got.qLatRows[row] != nil {
					t.Fatalf("failed or unvisited row %d was published", row)
				}
			}
		})
	}
}
