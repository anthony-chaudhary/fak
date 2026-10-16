package model

import (
	"errors"
	"math"
	"reflect"
	"runtime"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=15s lane=default
func TestV41EngramColdLayerHash(t *testing.T) {
	m, _ := v41IncrementalEngramFixture(t, false)
	t.Cleanup(func() { _ = m.CloseWeights() })
	stage := m.v41EngramStageFor()
	if stage == nil || len(stage.layerIDs) < 2 {
		t.Fatal("fixture requires distinct Engram layers")
	}
	selectRows := func(all []uint32, positions, layerIndex int) []uint32 {
		rows := make([]uint32, 0, positions*stage.columns)
		for pos := 0; pos < positions; pos++ {
			lo := (pos*len(stage.layerIDs) + layerIndex) * stage.columns
			rows = append(rows, all[lo:lo+stage.columns]...)
		}
		return rows
	}
	prototypeTail := append([]int64(nil), stage.hashPrototype.tail...)

	t.Run("rows", func(t *testing.T) {
		for _, tokens := range [][]int{nil, {1}, {1, 3, 2, 5, 7, 1, 2, 3}} {
			want := v41EngramReference(stage.layout, tokens, nil)
			for layerIndex, layer := range stage.layerIDs {
				geometry, got, err := m.v41EngramLayerRows(layer, tokens)
				if err != nil || geometry.cacheIdx != layerIndex || !reflect.DeepEqual(got, selectRows(want, len(tokens), layerIndex)) {
					t.Fatalf("layer=%d tokens=%d rows=%v err=%v", layer, len(tokens), got, err)
				}
				if len(got) > 0 {
					got[0] ^= 1
					_, again, err := m.v41EngramLayerRows(layer, tokens)
					if err != nil || !reflect.DeepEqual(again, selectRows(want, len(tokens), layerIndex)) {
						t.Fatal("caller mutation changed a later hash result")
					}
				}
			}
		}
	})

	t.Run("allocation", func(t *testing.T) {
		// Compare the actual cold producer with a full-layout hash under the
		// identical geometry admission. This is a byte-allocation observation,
		// not a timing or physical-device throughput assertion. The old producer
		// allocated that whole layout AND copied the selected layer out of it.
		tokens := make([]int, 64)
		for i := range tokens {
			tokens[i] = (i*3 + 1) % m.Cfg.VocabSize
		}
		layer := stage.layerIDs[len(stage.layerIDs)-1]
		if _, _, err := m.v41EngramLayerRows(layer, tokens); err != nil {
			t.Fatal(err)
		}
		var sink []uint32
		measure := func(full bool) int64 {
			result := testing.Benchmark(func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					var err error
					if full {
						if _, err = m.v41EngramInjectionGeometry(layer, len(tokens)); err == nil {
							sink, err = stage.hashPrototype.Clone().Hash(tokens, nil)
						}
					} else {
						_, sink, err = m.v41EngramLayerRows(layer, tokens)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			})
			return result.AllocedBytesPerOp()
		}
		fullBytes, selectedBytes := measure(true), measure(false)
		runtime.KeepAlive(sink)
		unusedBytes := int64(len(tokens)*stage.columns*(len(stage.layerIDs)-1)) * 4
		if unusedBytes <= 0 || fullBytes <= 0 || selectedBytes <= 0 || selectedBytes >= fullBytes {
			t.Fatalf("cold layer hash allocated selected=%d full-layout=%d bytes; unrequested output=%d bytes", selectedBytes, fullBytes, unusedBytes)
		}
	})

	t.Run("injection", func(t *testing.T) {
		tokens := []int{1, 3, 2}
		all := v41EngramReference(stage.layout, tokens, nil)
		for layerIndex, layer := range stage.layerIDs {
			gotX, gotStreams := v41FullEngramAsymmetricStreams(tokens, m.Cfg.HiddenSize)
			wantX, wantStreams := v41FullEngramAsymmetricStreams(tokens, m.Cfg.HiddenSize)
			_, beforeStreams := v41FullEngramAsymmetricStreams(tokens, m.Cfg.HiddenSize)
			geometry, err := m.v41EngramInjectionGeometry(layer, len(tokens))
			if err != nil {
				t.Fatal(err)
			}
			rows := selectRows(all, len(tokens), layerIndex)
			readBefore := stage.caches[layerIndex].Stats().RowsServed
			if err := m.v41EngramInject(layer, gotX, gotStreams, true, tokens, float32(m.Cfg.RMSNormEps)); err != nil {
				t.Fatal(err)
			}
			gathered, err := m.v41EngramInjectPrepared(layer, wantX, wantStreams, true, len(tokens), rows, float32(m.Cfg.RMSNormEps), geometry)
			if err != nil || gathered != len(rows) || stage.caches[layerIndex].Stats().RowsServed-readBefore != int64(2*len(rows)) {
				t.Fatalf("actual and oracle injection did not consume every requested row: gathered=%d err=%v", gathered, err)
			}
			changed := false
			for pos := range tokens {
				for stream := range gotStreams[pos] {
					for col, value := range gotStreams[pos][stream] {
						changed = changed || math.Float32bits(value) != math.Float32bits(beforeStreams[pos][stream][col])
						if math.Float32bits(value) != math.Float32bits(wantStreams[pos][stream][col]) {
							t.Fatalf("layer=%d position=%d stream=%d col=%d changed bits", layer, pos, stream, col)
						}
					}
				}
			}
			if !changed {
				t.Fatal("fixture injection did not change any residual")
			}
		}
	})

	t.Run("refusal", func(t *testing.T) {
		before := v41IncrementalEngramRequests(m)
		for _, tokens := range [][]int{{-1}, {1, m.Cfg.VocabSize}} {
			for _, layer := range stage.layerIDs {
				_, rows, err := m.v41EngramLayerRows(layer, tokens)
				if rows != nil || !errors.Is(err, ErrV41NativeUnsupported) {
					t.Fatalf("invalid tokens returned rows=%v err=%v", rows, err)
				}
			}
		}
		if _, rows, err := m.v41EngramLayerRows(m.Cfg.NumLayers, []int{1}); rows != nil || !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("undeclared layer returned rows=%v err=%v", rows, err)
		}
		if v41IncrementalEngramRequests(m) != before || !reflect.DeepEqual(prototypeTail, stage.hashPrototype.tail) {
			t.Fatal("refused input performed a row read or changed prototype history")
		}
	})

	t.Run("session", func(t *testing.T) {
		tokens := []int{1, 3, 2, 5, 7}
		cold, incremental := m.NewSession(), m.NewSession()
		defer cold.Close()
		defer incremental.Close()
		got := cold.Prefill(tokens)
		want := incremental.Prefill(tokens[:1])
		for _, token := range tokens[1:] {
			want = incremental.Step(token)
		}
		if len(got) != m.Cfg.VocabSize || cold.v41Forward == nil || incremental.v41Forward == nil || !reflect.DeepEqual(cold.v41Forward.history, tokens) || !reflect.DeepEqual(incremental.v41Forward.history, tokens) {
			t.Fatal("actual cold/incremental sessions did not publish the complete token history")
		}
		assertV41RowsClose(t, "cold selected-layer hash versus incremental all-layer hash", got, want, 1e-6)
	})
	if !reflect.DeepEqual(prototypeTail, stage.hashPrototype.tail) {
		t.Fatal("cold hashing mutated the immutable prototype tail")
	}
}
