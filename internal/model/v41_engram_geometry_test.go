package model

import (
	"errors"
	"reflect"
	"testing"
)

// Existing injection APIs must refuse malformed hidden rows before any earlier
// position is published. Full injection used to silently truncate short x rows;
// missing positions and reduced short rows instead panicked after partial writes.
// fak-test:runtime fast est=100ms lane=default
func TestV41EngramHiddenGeometry(t *testing.T) {
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
			H := m.Cfg.HiddenSize
			ids := []int{1, 3}
			eps := float32(m.Cfg.RMSNormEps)
			for _, shape := range []string{"missing_position", "short_later_row", "wide_later_row", "valid_extra_position"} {
				t.Run(shape, func(t *testing.T) {
					x, streams := v41FullEngramAsymmetricStreams(ids, H)
					// Independently owned full x must be synchronized too; the malformed
					// cases retain the production stream-0 alias to catch partial writes.
					valid := shape == "valid_extra_position"
					switch shape {
					case "missing_position":
						x = x[:1]
					case "short_later_row":
						x[1] = x[1][:H-1]
					case "wide_later_row":
						x[1] = append(append([]float32(nil), x[1]...), 7)
					case "valid_extra_position":
						for i := range x {
							x[i] = append([]float32(nil), x[i]...)
						}
						// Only the requested prefix is consumed; unused rows need not
						// have HiddenSize width and must remain untouched.
						x = append(x, []float32{7})
					}
					if !mode.full {
						streams = nil
					}
					beforeX := sysFlatten(x)
					beforeStreams := make([][]float32, len(streams))
					for i := range streams {
						beforeStreams[i] = sysFlatten(streams[i])
					}
					cache := m.v41EngramStageFor().caches[0]
					beforeCache := cache.Stats()
					calls := 0
					var err error
					func() {
						defer func() {
							if p := recover(); p != nil {
								t.Errorf("injection panicked instead of returning a typed geometry error: %v", p)
							}
						}()
						err = m.v41EngramInjectWithProjection(mode.layer, x, streams, mode.full, ids, eps,
							func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
								calls++
								if layer != mode.layer || out != 5*H || len(row) != in {
									t.Fatal("projection lost its exact geometry")
								}
								projected := make([]float32, out)
								for i := range projected {
									projected[i] = 1
								}
								return projected, v41ProjectionHandled, nil
							})
					}()
					if valid {
						if err != nil || calls != len(ids) || reflect.DeepEqual(beforeX, sysFlatten(x)) {
							t.Fatalf("valid prefix: err=%v projection calls=%d; want %d and a real update", err, calls, len(ids))
						}
						if !reflect.DeepEqual(x[len(ids)], []float32{7}) {
							t.Fatal("injection mutated the unused position")
						}
						if mode.full {
							for i := range ids {
								if !reflect.DeepEqual(x[i], streams[i][0]) {
									t.Fatalf("independently owned x[%d] was not synchronized", i)
								}
							}
						}
						return
					}
					var stageErr *V41ForwardError
					if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &stageErr) ||
						stageErr.Stage != v41StageEngram || stageErr.Layer != mode.layer {
						t.Errorf("geometry error=%v, want typed Engram refusal at layer %d", err, mode.layer)
					}
					if calls != 0 || cache.Stats() != beforeCache {
						t.Errorf("refused geometry reached row gather or projection: calls=%d", calls)
					}
					if !reflect.DeepEqual(beforeX, sysFlatten(x)) {
						t.Error("refused geometry partially mutated x")
					}
					for i := range streams {
						if !reflect.DeepEqual(beforeStreams[i], sysFlatten(streams[i])) {
							t.Errorf("refused geometry partially mutated streams[%d]", i)
						}
					}
				})
			}
		})
	}
}
