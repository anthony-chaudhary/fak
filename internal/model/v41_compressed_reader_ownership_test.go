package model

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// Pins the declared source/reader ownership in the reference Attention and
// Indexer: only a KV producer compresses, an index-only source queries that
// producer's keys, and an ordinary reader reuses the latest selection. The
// legacy no-declared-source fixture and window-plus-compressed composition are
// separate contracts; this is not a full-model or quantization parity claim.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41CompressedReaderOwnership(t *testing.T) {
	t.Parallel()
	for _, ownQuery := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(ownQuery)), func(t *testing.T) {
			m := v41CompressorTestFixtureIndex(t, 128)
			t.Cleanup(func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			if ownQuery {
				m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
			}
			// These tensors do not exist on a reference reader. Their absence
			// must be admitted, and both cold and incremental execution must work.
			for _, leaf := range []string{"attn.compressor.wkv.weight", "attn.compressor.wgate.weight", "attn.compressor.norm.weight", "indexer.wk.weight", "indexer.k_norm.weight"} {
				delete(m.manifest, layerName(1, leaf))
			}
			if !ownQuery {
				delete(m.manifest, layerName(1, "indexer.wq_b.weight"))
				delete(m.manifest, layerName(1, "indexer.weights_proj.weight"))
			}
			if err := m.v41ForwardAdmitted(); err != nil {
				t.Fatalf("reader admission required producer-only tensors: %v", err)
			}
			boom := errors.New("late reader selected projection failure")
			fail := false
			keyRows, readerQueries, faults := 0, 0, 0
			ownerKeys := map[int][]float32{}
			bind := func(st *v41ForwardState) {
				st.denseProjection = func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					if layer == 1 && (strings.HasPrefix(leaf, "attn.compressor.") || leaf == "indexer.wk.weight" || (!ownQuery && strings.HasPrefix(leaf, "indexer."))) {
						t.Fatalf("reader projected producer-only tensor %s", leaf)
					}
					if leaf == "indexer.wk.weight" {
						keyRows += rows
					}
					if layer == 1 && leaf == "indexer.wq_b.weight" {
						readerQueries += rows
					}
					if fail && layer == 1 && leaf == "ffn.gate.weight" {
						faults++
						published, _ := st.attn.KVSourceRows(0)
						if len(published) != 2 {
							t.Fatal("late reader failure did not follow the source publication")
						}
						return nil, v41ProjectionError, boom
					}
					return nil, v41ProjectionDeclined, nil
				}
				st.indexerScore = func(layer int, q, keys, weights []float32, heads, dim, rows int) ([]float32, error) {
					if layer == 0 {
						ownerKeys[rows] = append([]float32(nil), keys...)
					} else if !ownQuery || (rows > 0 && !reflect.DeepEqual(keys, ownerKeys[rows])) {
						t.Fatal("index-only reader did not query the owner's already-rotated key bytes")
					}
					return V41IndexerScore(q, keys, weights, heads, dim, rows)
				}
			}
			checkReader := func(st *v41ForwardState, positions int) {
				t.Helper()
				reader := st.layerState(1)
				if reader == nil || reader.nextWindowPos != positions || reader.nextCompressRow != positions {
					t.Fatal("reader continuation cursor did not advance")
				}
				if len(reader.partialInputs) != 0 || len(reader.partialPositions) != 0 || len(reader.partialKV) != 0 || len(reader.kvPublications) != 0 || len(reader.indexPublications) != 0 {
					t.Fatal("reader retained private compressor or key history")
				}
				if _, ok := st.attn.KVSourceRows(1); ok {
					t.Fatal("reader published private shared KV rows")
				}
				if _, ok := st.attn.IndexKeys(1); ok {
					t.Fatal("index-only reader published private index keys")
				}
			}
			tiny := &v41ForwardState{}
			bind(tiny)
			if _, err := m.forwardV41([]int{1}, tiny); err != nil {
				t.Fatalf("empty compressed source group: %v", err)
			}
			checkReader(tiny, 1)
			if keyRows != 0 {
				t.Fatal("incomplete source group projected a key")
			}
			keyRows, readerQueries = 0, 0
			st := &v41ForwardState{}
			bind(st)
			prefix := []int{1, 2, 3}
			if _, err := m.forwardV41(prefix, st); err != nil {
				t.Fatal(err)
			}
			checkReader(st, 3)
			if keyRows != 1 || (ownQuery && readerQueries != 3) || (!ownQuery && readerQueries != 0) {
				t.Fatal("cold ownership projection counts differ")
			}
			snapshot := captureV41ForwardSnapshot(st)
			restored := snapshot.clone().restore()
			bind(restored)
			fail = true
			failure := v41CompressorTestRecover(func() { _, _, _ = m.forwardV41Step(4, restored, nil) })
			failedErr, isError := failure.(error)
			var selected *V41ProjectionOperationError
			if !isError || !errors.Is(failedErr, boom) || !errors.As(failedErr, &selected) || selected.Layer != 1 || faults != 1 {
				t.Fatalf("selected reader failure lost its cause or retried: %v", failure)
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(restored), snapshot) {
				t.Fatal("late reader failure changed the restored source/history state")
			}
			fail = false
			beforeKeys := keyRows
			got, stats, err := m.forwardV41Step(4, restored, nil)
			if err != nil || !stats.Committed || keyRows != beforeKeys+1 {
				t.Fatalf("restored source completion replayed or failed: %v", err)
			}
			checkReader(restored, 4)
			if !reflect.DeepEqual(captureV41ForwardSnapshot(st), snapshot) {
				t.Fatal("restored continuation mutated the original owner")
			}
			cold, err := m.forwardV41([]int{1, 2, 3, 4}, nil)
			if err != nil {
				t.Fatalf("stateless cold forward did not carry source ownership: %v", err)
			}
			assertV41LogitsClose(t, got, lastLogits(cold), "reader restore/retry vs stateless cold")
			beforeKeys = keyRows
			if _, stats, err := m.forwardV41Step(5, restored, nil); err != nil || !stats.Committed || keyRows != beforeKeys {
				t.Fatalf("incomplete group reprojected source keys: %v", err)
			}
			checkReader(restored, 5)

			// Zero completed rows are legitimate; missing completed source rows
			// or source-owned keys must fail closed rather than synthesize a cache.
			plan, err := m.v41AttentionPlan(1)
			if err != nil {
				t.Fatal(err)
			}
			empty, _ := NewV41AttentionState(m.Cfg.HeadDim, 8)
			empty.indexHeadDim = m.Cfg.IndexHeadDim
			if _, _, err := m.v41CompressedReaderRows(plan, empty, 1); err != nil {
				t.Fatalf("incomplete source group was refused: %v", err)
			}
			if _, _, err := m.v41CompressedReaderRows(plan, empty, 2); !errors.Is(err, ErrV41ForwardStage) {
				t.Fatal("missing completed source publication was accepted")
			}
			if ownQuery {
				missing := snapshot.clone().restore().attn
				missing.indexPublications = nil
				missing.indexPublishedEnd = nil
				if _, _, err := m.v41CompressedReaderRows(plan, missing, 3); !errors.Is(err, ErrV41ForwardStage) {
					t.Fatal("index-only reader accepted missing source-owned keys")
				}
			}
		})
	}
}
