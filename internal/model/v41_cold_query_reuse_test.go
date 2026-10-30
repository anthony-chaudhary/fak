package model

import (
	"errors"
	"reflect"
	"testing"
)

// The cold layer owns its query rows; ordinary callbacks do not promise to
// release them. Exercise both attention layouts and a callback replacement at
// the timed dispatch boundary. This is a host ownership witness, not GPU proof.
// fak-test:runtime fast est=100ms lane=default
func TestV41ColdAttentionQueryReuse(t *testing.T) {
	for _, mode := range []string{"host-plain", "native-plain", "native-combined", "retaining-rebind", "selected-failure", "borrowed"} {
		t.Run(mode, func(t *testing.T) {
			cfg := v41FullGeometryConfig(t)
			cfg.Window = []int{2}
			m := &Model{Cfg: cfg}
			const seq = 3
			width := cfg.NumHeads * cfg.HeadDim
			panel := make([]float32, seq*width)
			qkv := v41LayerQKV{qHeads: make([][]float32, seq), kvRows: make([][]float32, seq)}
			for row := 0; row < seq; row++ {
				qkv.qHeads[row] = panel[row*width : (row+1)*width : (row+1)*width]
				qkv.kvRows[row] = make([]float32, cfg.HeadDim)
				for i := range qkv.qHeads[row] {
					qkv.qHeads[row][i] = float32((row+i)%7+1) / 16
				}
				for i := range qkv.kvRows[row] {
					qkv.kvRows[row][i] = float32((2*row+i)%5+1) / 8
				}
			}
			before := append([]float32(nil), panel...)
			src := v41LayerSources{}
			if mode == "native-combined" {
				src.plan.Ratio = 2
				src.sharedKV = [][]float32{append([]float32(nil), qkv.kvRows[0]...)}
			}
			st, scratch := &v41ForwardState{}, &v41ProjScratch{}
			var binding *v41SharedAttentionBinding
			var backend *v41SharedAttentionTestBackend
			if mode != "host-plain" {
				backend = newV41SharedAttentionTestBackend()
				s := &Session{M: m, Backend: backend}
				t.Cleanup(s.Close)
				binding = s.v41BindSharedAttention()
				if binding == nil {
					t.Fatal("canonical attention did not bind")
				}
				scratch.sharedAttentionBinding = binding
			}
			// The borrowed legacy entry supplies the numerical control and must
			// never overwrite a query even when the selected core is canonical.
			want, err := m.v41LayerAttentionContract(0, seq, cfg.NumHeads, cfg.HeadDim, src, qkv, nil, nil, st, scratch)
			if err != nil || !reflect.DeepEqual(panel, before) {
				t.Fatalf("borrowed control: err=%v query changed=%v", err, !reflect.DeepEqual(panel, before))
			}
			stop := errors.New("second selected attention call failed")
			if mode == "selected-failure" {
				backend.fault, backend.cause, backend.failCall = "dispatch", stop, len(backend.calls)+2
			}
			var retained []float32
			if mode == "retaining-rebind" {
				// A clock hook replaces the original canonical binding immediately
				// before dispatch. The wrapper retains Q and then restores canonical
				// binding: a post-call provenance check would wrongly recycle Q.
				installed := false
				m.v41SetExpertTimeClock(func() int64 {
					if !installed {
						installed = true
						scratch.sharedAttentionBinding = newV41SharedAttentionBinding(func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
							retained = request.q
							scratch.sharedAttentionBinding = binding
							return binding.attentionFunc()(layer, request)
						})
					}
					return 1
				})
				defer m.v41SetExpertTimeClock(nil)
			}
			var got [][]float32
			if mode == "borrowed" {
				got, err = m.v41LayerAttentionContract(0, seq, cfg.NumHeads, cfg.HeadDim, src, qkv, nil, nil, st, scratch)
			} else {
				got, err = m.v41LayerAttentionContractWithQueryScratch(0, seq, cfg.NumHeads, cfg.HeadDim, src, qkv, nil, nil, st, scratch, true)
			}
			if mode == "selected-failure" {
				if got != nil || !errors.Is(err, stop) || !reflect.DeepEqual(panel[width:], before[width:]) {
					t.Fatalf("failed or unvisited query was overwritten: out=%v err=%v", got != nil, err)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("query recycling changed output: err=%v", err)
			}
			for row := range got {
				reused := mode != "borrowed" && !(mode == "retaining-rebind" && row == 0)
				if (&got[row][0] == &qkv.qHeads[row][0]) != reused {
					t.Fatalf("row %d reuse=%v, want %v", row, &got[row][0] == &qkv.qHeads[row][0], reused)
				}
				if reused && cap(got[row]) != width {
					t.Fatal("recycled row can append into its successor")
				}
			}
			if mode == "retaining-rebind" && !reflect.DeepEqual(retained, before[:width]) {
				t.Fatal("ordinary callback's retained query changed")
			}
		})
	}
}
