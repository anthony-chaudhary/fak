package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// This is metadata-only fixture construction. The ordinary CPU backend cannot
// pass valid's sealed backend gate, and no device admission is claimed here.
func v41InputPanelScopeFixture() (*v41AttentionInputPanel, *v41ProjScratch) {
	m := &Model{Cfg: Config{HiddenSize: 4, NumLayers: 1}}
	backend := compute.Default()
	s := &Session{M: m, Backend: backend, modelWeightsHeld: true}
	state := &v41ForwardState{callbackOwner: s}
	s.v41Forward = state
	scope := &backendExecutionScope{owner: &sync.Mutex{}, sessions: []*Session{s}, live: true}
	s.backendExecutionScope = scope
	publish := func(_ int, _ string, x []float32) ([]float32, error) { return x, nil }
	norm := &v41AttentionNormBinding{normalize: func(_ int, x []float32) ([]float32, error) { return x, nil }, publish: publish,
		native: &v41BoundAttentionNorm{owner: s, model: m, backend: backend}}
	dense := func(_ int, _ string, _ []float32, _, _, _ int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionDeclined, nil
	}
	route := newV41ProjectionRouteBinding(dense, nil)
	route.nativeProjection = v41BoundDenseProjection{ordinary: dense, owner: s, model: m, backend: backend,
		core: func(_ int, _ string, _ []float32, _, _, _ int, _ *v41ProjectionContinuation) ([]float32, v41DenseProjectionOutcome, error) {
			return nil, v41ProjectionDeclined, nil
		}}
	publication := &v41BF16PublicationBinding{ordinary: publish, owner: s, model: m, backend: backend,
		core: func(_ int, _ string, x []float32, _ *v41BF16ResidentInput) ([]float32, error) { return x, nil }}
	gather := &v41DeviceGatherOps{owner: s, bf16Publish: publication}
	query := &v41QueryNormBinding{ordinary: func(_ int, x []float32) ([]float32, error) { return x, nil }, owner: s, model: m, backend: backend, publication: publication}
	state.attentionNormBinding, state.projectionRoute, state.queryNorm, state.gatherOps = norm, route, query, gather
	scratch := &v41ProjScratch{attentionNormBinding: norm, projectionRoute: route, queryNorm: query, gatherOps: gather}
	p := &v41AttentionInputPanel{owner: s, model: m, backend: backend, state: state, norm: norm, route: route, query: query, gather: gather, publication: publication, scope: scope, layer: 0, width: 4, rows: 2}
	scratch.attentionInputPanel = p
	return p, scratch
}

// fak-test:runtime fast est=10ms lane=default
func TestV41AttentionInputPanelSnapshot(t *testing.T) {
	t.Run("chunk identity and tails", func(t *testing.T) {
		for _, count := range []int{2, 64, 65, 129} {
			p := &v41AttentionInputPanel{layer: 2, width: 4, rows: count}
			rows := make([][]float32, count)
			for lo := 0; lo < count; lo += v41ProjPanelRowChunk {
				hi := min(lo+v41ProjPanelRowChunk, count)
				chunk := make([]float32, (hi-lo)*4)
				p.chunks = append(p.chunks, chunk)
				for row := lo; row < hi; row++ {
					at := (row - lo) * 4
					rows[row] = chunk[at : at+4 : at+4]
				}
			}
			if !p.matches(2, 4, rows) || p.matches(3, 4, rows) || p.matches(2, 3, rows) {
				t.Fatalf("identity/tail rows=%d", count)
			}
			old := rows[count-1]
			rows[count-1] = append([]float32(nil), old...)
			if p.matches(2, 4, rows) {
				t.Fatal("replacement alias admitted")
			}
			rows[count-1] = old
			p.chunks[len(p.chunks)-1] = p.chunks[len(p.chunks)-1][:len(p.chunks[len(p.chunks)-1])-1]
			if p.matches(2, 4, rows) {
				t.Fatal("short chunk admitted")
			}
		}
	})
	t.Run("whole interval metadata", func(t *testing.T) {
		p, scratch := v41InputPanelScopeFixture()
		if !p.scopeValid(scratch) || p.valid(scratch) {
			t.Fatal("metadata baseline or sealed CPU refusal")
		}
		if p.norm.newInputPanel(p.model, p.state, scratch, 0, 4, 2) != nil {
			t.Fatal("CPU minted producer loan")
		}
		original := scratch.queryNorm
		scratch.queryNorm = &v41QueryNormBinding{ordinary: original.ordinary}
		if p.scopeValid(scratch) {
			t.Fatal("custom query accepted")
		}
		scratch.queryNorm = original
		p.gather.setBF16Publication(p.publication.ordinary)
		if p.scopeValid(scratch) {
			t.Fatal("custom publication accepted")
		}
		p, scratch = v41InputPanelScopeFixture()
		scratch.activationTrace = &v41ActivationTraceState{}
		if p.scopeValid(scratch) {
			t.Fatal("tracer accepted")
		}
		scratch.activationTrace = nil
		computetrace.SetObserver(func(computetrace.Event) {})
		observed := p.scopeValid(scratch)
		computetrace.SetObserver(nil)
		if observed {
			t.Fatal("observer-only route accepted")
		}
		clockCalls := 0
		p.model.v41SetExpertTimeClock(func() int64 { clockCalls++; return 0 })
		if p.scopeValid(scratch) || clockCalls != 0 {
			t.Fatal("custom clock invoked or admitted")
		}
		p.model.v41SetExpertTimeClock(nil)
		p.owner.cacheGeometryEpoch++
		if p.scopeValid(scratch) {
			t.Fatal("epoch drift admitted")
		}
		p.owner.cacheGeometryEpoch--
		p.scope.live = false
		if p.scopeValid(scratch) {
			t.Fatal("retired execution scope admitted")
		}
		p.scope.live = true
		p.state.callbackOwner = &Session{}
		if p.scopeValid(scratch) {
			t.Fatal("owner drift admitted")
		}
	})
	for _, site := range []string{"query", "observer", "clock", "projection"} {
		t.Run("mutating "+site+" keeps snapshot", func(t *testing.T) {
			m := &Model{Cfg: Config{NumLayers: 1, HiddenSize: 4, QLoraRank: 2, QKRopeHeadDim: 2, RMSNormEps: 1e-5}, manifest: map[string]tensorMeta{
				layerName(0, "attn.wq_a.weight"): {Shape: []int{2, 4}},
				layerName(0, "attn.wq_b.weight"): {Shape: []int{v41KVLoraRank, 2}},
				layerName(0, "attn.wkv.weight"):  {Shape: []int{v41KVLoraRank, 4}},
			}}
			rows := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
			want := []float32{1, 2, 3, 4, 5, 6, 7, 8}
			mutated := false
			mutate := func() { rows[0][0] = 99; mutated = true }
			if site == "observer" {
				computetrace.SetObserver(func(computetrace.Event) { mutate() })
				defer computetrace.SetObserver(nil)
			}
			if site == "clock" {
				m.v41SetExpertTimeClock(func() int64 { mutate(); return 0 })
				defer m.v41SetExpertTimeClock(nil)
			}
			seenKV := false
			project := func(_ int, leaf string, input []float32, out, in, n int) ([]float32, v41DenseProjectionOutcome, error) {
				if leaf == "attn.wq_a.weight" {
					switch site {
					case "observer":
						computetrace.Record(computetrace.Event{})
					case "clock":
						m.v41NowNanos()
					case "projection":
						mutate()
					}
				}
				if leaf == "attn.wkv.weight" {
					seenKV = true
					if !reflect.DeepEqual(input, want) {
						t.Fatalf("snapshot changed: %v", input)
					}
				}
				values := make([]float32, n*out)
				for i := range values {
					values[i] = 1
				}
				return values, v41ProjectionHandled, nil
			}
			stop := errors.New("stop after both projections")
			scratch := &v41ProjScratch{projectionRoute: newV41ProjectionRouteBinding(project, nil), queryNorm: &v41QueryNormBinding{ordinary: func(_ int, x []float32) ([]float32, error) {
				if site == "query" {
					mutate()
				}
				return x, nil
			}}, kvNorm: &v41KVNormBinding{ordinary: func(int, []float32) ([]float32, error) { return nil, stop }}}
			_, err := m.v41LayerProjectQKV(0, 2, rows, true, v41KVLoraRank, 1, 4, 1e-5, scratch, nil)
			if !errors.Is(err, stop) || !seenKV || !mutated || scratch.attentionInputPanel != nil {
				t.Fatalf("err=%v seenKV=%t mutated=%t", err, seenKV, mutated)
			}
		})
	}
	t.Run("selected drift never replays", func(t *testing.T) {
		p, scratch := v41InputPanelScopeFixture()
		stop := errors.New("owner already failed")
		p.owner.halFailure, p.owner.halClosed = stop, true
		calls := 0
		project := func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			calls++
			return nil, v41ProjectionDeclined, nil
		}
		_, err := p.model.v41ProjPanelWithInput(0, "attn.wkv.weight", nil, 2, 4, 2, project, p, scratch)
		if !errors.Is(err, stop) || calls != 0 {
			t.Fatalf("err=%v calls=%d", err, calls)
		}
	})
}
