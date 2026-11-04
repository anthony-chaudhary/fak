package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=15s lane=default
func TestV41CheckpointSplitEquivalence(t *testing.T) {
	t.Setenv(v41ActivationTraceEnvVar, "")
	t.Setenv(v41SuffixRefoldMaxHistoryEnv, "")
	t.Setenv(v41SuffixUbatchEnv, "")
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("production cpu-ref backend is not registered")
	}
	for _, profile := range []string{"plain", "compressed_engram"} {
		for _, route := range []string{"host", "cpu-ref"} {
			t.Run(profile+"/"+route, func(t *testing.T) {
				var m *Model
				if profile == "plain" {
					m = v41IncrementalPlainModel(t, 2)
				} else {
					m, _ = v41IncrementalEngramFixture(t, true)
					if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full {
						t.Fatalf("compressed Engram fixture did not admit full geometry: %t/%v", full, err)
					}
				}
				defer m.CloseWeights()
				newSession := func() *Session {
					var s *Session
					if route == "host" {
						s = m.NewSession()
					} else {
						var err error
						s, err = m.NewBackendSessionChecked(backend)
						if err != nil {
							t.Fatal(err)
						}
					}
					s.Quant, s.Q4K = true, true
					return s
				}
				cold, split := newSession(), newSession()
				defer cold.Close()
				defer split.Close()
				ids := v41ParityPrompt(m.Cfg.VocabSize, 8)
				ids[0] = 0
				want := append([]float32(nil), cold.Prefill(ids)...)
				split.Prefill(ids[:1])
				if !split.v41IncrementalEligible() {
					t.Fatal("checkpoint must select the production incremental suffix path")
				}
				got := split.Prefill(ids[1:])
				check := func(phase string, got, want []float32) {
					t.Helper()
					if len(got) != m.Cfg.VocabSize || len(want) != len(got) {
						t.Fatalf("%s: expected complete vocabulary rows, got %d/%d", phase, len(got), len(want))
					}
					var squaredError, squaredReference float64
					for i := range want {
						a, b := float64(got[i]), float64(want[i])
						if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
							t.Fatalf("%s: nonfinite logit at %d", phase, i)
						}
						squaredError += (a - b) * (a - b)
						squaredReference += b * b
					}
					if squaredReference == 0 {
						t.Fatalf("%s: fixture has no output signal", phase)
					}
					t.Logf("%s relative_l2=%g argmax=%d/%d", phase, math.Sqrt(squaredError/squaredReference), argmax(got), argmax(want))
					assertV41RowsClose(t, phase, got, want, 1e-6)
					if argmax(got) != argmax(want) {
						t.Fatalf("%s: checkpoint split changed greedy token", phase)
					}
				}
				check("prefill", got, want)
				next := argmax(want)
				want = append([]float32(nil), cold.Step(next)...)
				got = split.Step(next)
				check("continuation", got, want)
				history := append(append([]int(nil), ids...), next)
				for _, s := range []*Session{cold, split} {
					if !reflect.DeepEqual(s.v41Forward.history, history) {
						t.Fatal("continuation history differs from consumed tokens")
					}
					for layer, pos := range v41AllNextWindowPos(s.v41Forward) {
						if pos != len(history) {
							t.Fatalf("layer %d cursor=%d history=%d", layer, pos, len(history))
						}
					}
				}
			})
		}
	}
}
