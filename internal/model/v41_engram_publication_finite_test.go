package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// This exercises the real host hash/gather/decode/projection/injection path.
// Only final residual publication is checked: finite extreme keys and the
// established intermediate arithmetic remain admitted. Earlier finite writes
// are transient and need not roll back within this private injection helper.
// fak-test:runtime medium est=1s lane=default
func TestV41EngramFinalPublicationFinite(t *testing.T) {
	maxBF16 := math.Float32frombits(0x7f7f0000)
	gate := float32(1) / (1 + float32(math.Exp(-float64(float32(math.Sqrt(1e-6))))))
	if !(gate > .5 && gate < 1) {
		t.Fatal("zero-dot gate control is not bounded")
	}
	for _, mode := range []struct {
		name  string
		full  bool
		layer int
		model func(*testing.T) (*Model, V41EngramLayout)
	}{{"full", true, 0, v41FullEngramModel}, {"reduced", false, 1, v41ReducedEngramModel}} {
		t.Run(mode.name, func(t *testing.T) {
			m, layout := mode.model(t)
			t.Cleanup(func() { _ = m.CloseWeights() })
			H, ids := m.Cfg.HiddenSize, []int{1}
			stage := m.v41EngramStageFor()
			cacheIndex := stage.cacheIndex(mode.layer)
			in, out := stage.columns*stage.headDim, (stage.hc+1)*H
			// Every gathered packed row is [1,0,...]. A literal column-zero
			// matrix makes the actual host contraction return the chosen finite
			// keys/value without replacing any production projection callback.
			sources := make([]V41EngramRowSource, len(layout.Rows))
			for layer, rows := range layout.Rows {
				packed := make([]byte, int(rows)*V41EngramPackedRowBytes)
				for row := 0; row < int(rows); row++ {
					packed[row*V41EngramPackedRowBytes] = 0x38
					packed[row*V41EngramPackedRowBytes+256] = 127
				}
				sources[layer] = &v41EngramMemorySource{packed: packed, rows: int(rows)}
			}
			if err := m.wireV41Engram(layout, sources, int64(V41EngramPackedRowBytes)*4); err != nil {
				t.Fatal(err)
			}
			stage = m.v41EngramStageFor()
			zeros := make([]float32, stage.hc*H)
			v41WriteTensorF32(t, m, layerName(mode.layer, "engram_q_norm.weight"), zeros)
			v41WriteTensorF32(t, m, layerName(mode.layer, "engram_k_norm.weight"), zeros)
			cases := []struct {
				name           string
				initial, value float32
				fail, aboveBF  bool
			}{
				{"positive_f32_overflow", maxBF16, maxBF16, true, false},
				{"negative_f32_overflow", -maxBF16, -maxBF16, true, false},
				{"ordinary", 1, 1, false, false},
				{"maximum_finite", 0, maxBF16 / 8, false, false},
			}
			if mode.full {
				cases = append(cases, struct {
					name           string
					initial, value float32
					fail, aboveBF  bool
				}{"positive_bf16_overflow", maxBF16, math.Float32frombits(0x7b800000), true, false})
			} else {
				cases = append(cases, struct {
					name           string
					initial, value float32
					fail, aboveBF  bool
				}{"finite_f32_above_bf16", maxBF16, math.Float32frombits(0x7a800000), false, true})
			}
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					projected := make([]float32, out)
					for i := range projected {
						projected[i] = 1
					}
					projected[len(projected)-1] = tc.value
					if _, bad := v41BF16PublishFault(projected); bad {
						t.Fatal("fixture projection is not finite BF16 input")
					}
					weights := make([]float32, out*in)
					for i, value := range projected {
						weights[i*in] = value
					}
					v41WriteTensorF32(t, m, layerName(mode.layer, "engram_kv.weight"), weights)
					geometry, rows, err := m.v41EngramLayerRows(mode.layer, ids)
					if err != nil {
						t.Fatal(err)
					}
					gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[cacheIndex]}, rows, stage.columns)
					if err != nil {
						t.Fatal(err)
					}
					input := make([]float32, in)
					if err = v41EngramDecodePosition(mode.layer, stage, cacheIndex, gathered, 0, stage.columns, stage.headDim, input); err != nil || input[0] != 1 {
						t.Fatalf("real row decode did not produce unit source: %v", err)
					}
					actualProjection, err := m.v41EngramProjector(mode.layer, out, in, nil)(input)
					if err != nil || !reflect.DeepEqual(actualProjection, projected) || geometry.project != nil {
						t.Fatalf("actual host projection differs from literal control: %v", err)
					}
					x, streams := v41FullEngramAsymmetricStreams(ids, H)
					for _, stream := range streams[0] {
						for i := range stream {
							stream[i] = 1
						}
					}
					target := x[0]
					if mode.full {
						target = streams[0][stage.hc-1]
					} else {
						streams = nil
					}
					target[H-1] = tc.initial
					beforeFirst := target[0]
					// Positive epsilon, unit keys and zero gains give exactly the
					// bounded zero-dot gate even when a residual square overflows.
					err = m.v41EngramInject(mode.layer, x, streams, mode.full, ids, 1e-5)
					nonfinite := 0
					all := x[0]
					if mode.full {
						all = sysFlatten(streams[0])
					}
					for _, value := range all {
						if !finite32(value) {
							nonfinite++
						}
					}
					if tc.fail {
						var attributed *V41ForwardError
						if err == nil || nonfinite != 0 || math.Float32bits(target[H-1]) != math.Float32bits(tc.initial) {
							t.Fatalf("nonfinite final residual published: mode=%s error_nil=%v nonfinite=%d offending_unchanged=%v host_projection_exact=true", mode.name, err == nil, nonfinite, math.Float32bits(target[H-1]) == math.Float32bits(tc.initial))
						}
						if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != mode.layer {
							t.Fatalf("final publication lost Engram attribution: %v", err)
						}
						if target[0] == beforeFirst {
							t.Fatal("fixture did not reach the final component after finite transient writes")
						}
						return
					}
					if err != nil || nonfinite != 0 {
						t.Fatalf("finite final publication refused: %v nonfinite=%d", err, nonfinite)
					}
					want := v41OracleBF16(tc.initial + gate*v41OracleBF16(tc.value))
					if !mode.full {
						var sum float32
						for range stage.hc {
							sum += v41OracleBF16(gate * v41OracleBF16(tc.value))
						}
						want = tc.initial + sum
					}
					if math.Float32bits(target[H-1]) != math.Float32bits(want) || target[0] == beforeFirst {
						t.Fatalf("finite publication changed equation: got=%g want=%g", target[H-1], want)
					}
					if tc.aboveBF {
						if _, bad := v41BF16PublishFault([]float32{target[H-1]}); !bad {
							t.Fatal("reduced F32 control did not cross the BF16-only boundary")
						}
					}
				})
			}
		})
	}
}
