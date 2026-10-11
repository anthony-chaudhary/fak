package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// Shared gate parameters must be valid before this injection call publishes any
// position. Later arithmetic overflow and whole-session rollback are separate
// contracts. All APIs below exist on the unchanged parent.
// fak-test:runtime medium est=1s lane=default
func TestV41EngramGateInputFiniteBoundary(t *testing.T) {
	t.Parallel()
	for _, mode := range []struct {
		name  string
		full  bool
		layer int
		model func(*testing.T) (*Model, V41EngramLayout)
	}{
		{"full", true, 0, v41FullEngramModel},
		{"reduced", false, 1, v41ReducedEngramModel},
	} {
		t.Run(mode.name, func(t *testing.T) {
			m, _ := mode.model(t)
			H, ids := m.Cfg.HiddenSize, []int{1, 3}
			stage := m.v41EngramStageFor()
			for _, route := range []string{"hash_entry", "prepared_entry"} {
				cases := []struct {
					name, leaf string
					value      float32
					valid      bool
				}{
					{"positive_epsilon", "epsilon", 1e-5, true},
					{"zero_epsilon", "epsilon", 0, true},
					{"negative_epsilon", "epsilon", -.5, true},
				}
				for _, leaf := range []string{"epsilon", "engram_q_norm.weight", "engram_k_norm.weight"} {
					for _, fault := range []struct {
						name  string
						value float32
					}{{"nan", float32(math.NaN())}, {"positive_inf", float32(math.Inf(1))}, {"negative_inf", float32(math.Inf(-1))}} {
						cases = append(cases, struct {
							name, leaf string
							value      float32
							valid      bool
						}{leaf + "/" + fault.name, leaf, fault.value, false})
					}
				}
				for _, tc := range cases {
					t.Run(route+"/"+tc.name, func(t *testing.T) {
						q, k := make([]float32, stage.hc*H), make([]float32, stage.hc*H)
						for i := range q {
							// Both learned gains retain valid zero/negative values.
							// Their product is zero, so finite controls have a fixed gate.
							if i%2 == 0 {
								q[i], k[i] = 0, -1
							} else {
								q[i], k[i] = -1, 0
							}
						}
						eps := float32(1e-5)
						switch tc.leaf {
						case "epsilon":
							eps = tc.value
						case "engram_q_norm.weight":
							q[len(q)-1] = tc.value // final stream: earlier streams must not move
						case "engram_k_norm.weight":
							k[len(k)-1] = tc.value
						}
						v41WriteTensorF32(t, m, layerName(mode.layer, "engram_q_norm.weight"), q)
						v41WriteTensorF32(t, m, layerName(mode.layer, "engram_k_norm.weight"), k)
						x, streams := v41FullEngramAsymmetricStreams(ids, H)
						for _, position := range streams {
							for _, stream := range position {
								for i := range stream {
									stream[i] = 1
								}
							}
						}
						if !mode.full {
							streams = nil
						}
						beforeX := append([]float32(nil), sysFlatten(x)...)
						beforeStreams := make([][]float32, len(streams))
						for i := range streams {
							beforeStreams[i] = append([]float32(nil), sysFlatten(streams[i])...)
						}
						cache := stage.caches[stage.cacheIndex(mode.layer)]
						beforeCache, calls := cache.Stats(), 0
						project := func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
							calls++
							if layer != mode.layer || len(row) != in || out != (stage.hc+1)*H {
								t.Fatal("projection fixture lost its geometry")
							}
							values := make([]float32, out)
							for i := range values {
								values[i] = 1
							}
							return values, v41ProjectionHandled, nil
						}
						var err error
						if route == "hash_entry" {
							err = m.v41EngramInjectWithProjection(mode.layer, x, streams, mode.full, ids, eps, project)
						} else {
							geometry, geometryErr := m.v41EngramInjectionGeometry(mode.layer, len(ids))
							if geometryErr != nil {
								t.Fatal(geometryErr)
							}
							geometry.project = project
							_, err = m.v41EngramInjectPrepared(mode.layer, x, streams, mode.full, len(ids), make([]uint32, len(ids)*stage.columns), eps, geometry)
						}
						if tc.valid {
							if err != nil || calls != len(ids) {
								t.Fatalf("finite control err=%v projection calls=%d", err, calls)
							}
							// Unit hidden/key rows keep norm operands positive for eps=-.5.
							// Zero dot gives sigmoid(.001), whose BF16 publications yield
							// 1.5 per full stream, or 1+4*.5=3 for the reduced reduction.
							want := float32(3)
							if mode.full {
								want = 1.5
								for _, position := range streams {
									for _, value := range sysFlatten(position) {
										if value != want {
											t.Fatalf("finite full publication=%g, want %g", value, want)
										}
									}
								}
							}
							for _, value := range sysFlatten(x) {
								if value != want {
									t.Fatalf("finite hidden publication=%g, want %g", value, want)
								}
							}
							return
						}
						var attributed *V41ForwardError
						if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != mode.layer {
							t.Errorf("nonfinite gate input err=%v, want typed Engram refusal", err)
						}
						if calls != 0 || cache.Stats() != beforeCache {
							t.Errorf("nonfinite gate parameters reached retrieval/projection: calls=%d", calls)
						}
						if !reflect.DeepEqual(beforeX, sysFlatten(x)) {
							t.Error("invalid shared parameter partially published hidden positions")
						}
						for i := range streams {
							if !reflect.DeepEqual(beforeStreams[i], sysFlatten(streams[i])) {
								t.Errorf("invalid shared parameter partially published stream position %d", i)
							}
						}
					})
				}
			}
		})
	}
}
