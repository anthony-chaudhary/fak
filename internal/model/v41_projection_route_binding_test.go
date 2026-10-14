package model

import (
	"errors"
	"reflect"
	"testing"
)

// Replacing an ordinary seam must preserve its opaque wrapper's output and
// captured refusal. The old binding remains usable by an already-bound scratch.
// fak-test:runtime fast est=10ms lane=default
func TestV41ProjectionRouteBindingEquivalence(t *testing.T) {
	var calls []string
	projection := v41DenseProjectionFunc(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		calls = append(calls, "projection")
		return []float32{2, 3}, v41ProjectionHandled, nil
	})
	router := v41RouterTopKFunc(func(_ int, logits, _ []float32, _ v41RouterConfig, _ bool) ([]routePick, error) {
		calls = append(calls, "router")
		return []routePick{{expert: 7, weight: logits[0]}}, nil
	})
	run := func(t *testing.T, st *v41ForwardState) ([]routePick, error) {
		t.Helper()
		x, outcome, err := st.denseProjection()(0, "ffn.gate.weight", []float32{1}, 2, 1, 1)
		if err != nil {
			return nil, err
		}
		if outcome != v41ProjectionHandled {
			t.Fatalf("unexpected projection outcome %d", outcome)
		}
		return st.routerTopK()(0, x, nil, v41DefaultRouterConfig(), true)
	}

	t.Run("wrappers-see-and-change-results", func(t *testing.T) {
		st := &v41ForwardState{projectionRoute: newV41ProjectionRouteBinding(projection, router)}
		old := &v41ProjScratch{projectionRoute: st.projectionRoute}
		originalProjection := st.denseProjection()
		st.setDenseProjection(func(layer int, leaf string, input []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			x, outcome, err := originalProjection(layer, leaf, input, out, in, rows)
			if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(x, []float32{2, 3}) {
				t.Fatal("projection wrapper did not see the original host result")
			}
			calls = append(calls, "projection-wrapper")
			x[0] += 4
			return x, outcome, err
		})
		originalRouter := st.routerTopK()
		st.setRouterTopK(func(layer int, logits, bias []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
			picks, err := originalRouter(layer, logits, bias, cfg, full)
			if err != nil || len(picks) != 1 || picks[0].weight != 6 {
				t.Fatal("router wrapper did not see the modified projection result")
			}
			calls = append(calls, "router-wrapper")
			picks[0].weight += 5
			return picks, err
		})
		calls = nil
		got, err := run(t, st)
		if err != nil || !reflect.DeepEqual(got, []routePick{{expert: 7, weight: 11}}) ||
			!reflect.DeepEqual(calls, []string{"projection", "projection-wrapper", "router", "router-wrapper"}) {
			t.Fatalf("wrapped result=%v err=%v calls=%v", got, err, calls)
		}
		if old.projectionRoute == st.projectionRoute {
			t.Fatal("replacement mutated the copied binding")
		}
		calls = nil
		x, _, err := old.denseProjection()(0, "ffn.gate.weight", nil, 2, 1, 1)
		picks, routeErr := old.routerTopK()(0, x, nil, v41DefaultRouterConfig(), true)
		if err != nil || routeErr != nil || len(picks) != 1 || picks[0].weight != 2 || !reflect.DeepEqual(calls, []string{"projection", "router"}) {
			t.Fatal("replacement changed the original scratch callbacks")
		}
	})

	t.Run("captured-refusals-are-not-retried", func(t *testing.T) {
		st := &v41ForwardState{projectionRoute: newV41ProjectionRouteBinding(projection, router)}
		refusal := errors.New("captured projection refusal")
		st.setDenseProjection(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			calls = append(calls, "projection-refusal")
			return nil, v41ProjectionError, refusal
		})
		calls = nil
		if got, err := run(t, st); got != nil || err != refusal || !reflect.DeepEqual(calls, []string{"projection-refusal"}) {
			t.Fatalf("projection refusal=%v err=%v calls=%v", got, err, calls)
		}
		st.setDenseProjection(projection)
		refusal = errors.New("captured router refusal")
		st.setRouterTopK(func(int, []float32, []float32, v41RouterConfig, bool) ([]routePick, error) {
			calls = append(calls, "router-refusal")
			return nil, refusal
		})
		calls = nil
		if got, err := run(t, st); got != nil || err != refusal || !reflect.DeepEqual(calls, []string{"projection", "router-refusal"}) {
			t.Fatalf("router refusal=%v err=%v calls=%v", got, err, calls)
		}
	})

	t.Run("nil-copy-and-snapshot-semantics", func(t *testing.T) {
		st := &v41ForwardState{}
		if st.denseProjection() != nil || st.routerTopK() != nil || newV41ProjectionRouteBinding(nil, nil) != nil {
			t.Fatal("empty binding changed ordinary nil semantics")
		}
		st.setDenseProjection(projection)
		st.setRouterTopK(router)
		clone := *st
		st.setDenseProjection(nil)
		if st.denseProjection() != nil || st.routerTopK() == nil || clone.denseProjection() == nil || clone.routerTopK() == nil {
			t.Fatal("clearing one callback changed its companion or a copied binding")
		}
		restored := captureV41ForwardSnapshot(&clone).clone().restore()
		if restored.projectionRoute != nil || restored.denseProjection() != nil || restored.routerTopK() != nil {
			t.Fatal("continuation snapshot retained callback authority")
		}
		st.setRouterTopK(nil)
		if st.projectionRoute != nil {
			t.Fatal("clearing both callbacks retained an empty binding")
		}
	})
}
