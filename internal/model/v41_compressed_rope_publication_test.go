package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// This witness covers the existing ratio-two assembly's publication boundary,
// not reference FP4 quantization or ratio-one compressor execution.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41CompressedRotaryPublication(t *testing.T) {
	t.Parallel()

	// With rotary width two, the sole pair's phase is the absolute position.
	// Keep the independent expected arithmetic explicit instead of invoking a
	// production rotary helper or deriving position from the emitted row count.
	rotate := func(row []float32, pos int) []float32 {
		out := append([]float32(nil), row...)
		i := len(out) - 2
		c, s := float32(math.Cos(float64(pos))), float32(math.Sin(float64(pos)))
		a, b := out[i], out[i+1]
		out[i] = float32(a*c) - float32(b*s)
		out[i+1] = float32(b*c) + float32(a*s)
		return out
	}
	rotateLatent := func(row []float32, pos int) []float32 {
		out := rotate(row, pos)
		for i := len(out) - 2; i < len(out); i++ {
			out[i] = v41CompressorNormRefCast(out[i])
		}
		return out
	}
	fixture := func() *Model {
		m := v41CompressorTestFixtureIndex(t, 128)
		m.Cfg.QKRopeHeadDim, m.Cfg.QKNopeHeadDim = 2, 510
		m.Cfg.RopeScaling = ""
		t.Cleanup(func() {
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
		})
		return m
	}

	t.Run("cold-and-midgroup-restore", func(t *testing.T) {
		m := fixture()
		type records struct {
			latent, key map[int][][]float32
			projections int
		}
		bind := func(st *v41ForwardState, r *records) {
			st.compressorNorm = func(l int, input, gain []float32, eps float32) ([]float32, error) {
				out := v41CompressorNormRefTail(input, gain, eps, "")
				r.latent[l] = append(r.latent[l], append([]float32(nil), out...))
				return out, nil
			}
			st.indexKeyNorm = func(l int, input, gain []float32, eps float32) ([]float32, error) {
				out := v41IndexKeyNormReference(input, gain, eps)
				r.key[l] = append(r.key[l], append([]float32(nil), out...))
				return out, nil
			}
			st.denseProjection = func(l int, leaf string, input []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
				if leaf == "indexer.wk.weight" {
					if rows != 1 || !reflect.DeepEqual(input, r.latent[l][len(r.key[l])]) {
						t.Fatal("index projection did not receive the unrotated normalized latent")
					}
					r.projections++
				}
				return nil, v41ProjectionDeclined, nil
			}
		}
		check := func(st *v41ForwardState, r *records, groups int) {
			t.Helper()
			for layer := 0; layer < m.Cfg.NumLayers; layer++ {
				state := st.layerState(layer)
				own, ok := state.KVSourceRows(layer)
				if layer != 0 {
					key, keyOK := state.IndexKeys(layer)
					if ok || keyOK || len(own) != 0 || len(key) != 0 || len(r.latent[layer]) != 0 || len(r.key[layer]) != 0 || len(state.partialInputs) != 0 || len(state.partialPositions) != 0 {
						t.Fatal("reader retained or recomputed its own compressor/index history")
					}
					continue
				}
				if !ok || len(own) != groups || len(r.latent[layer]) != groups {
					t.Fatal("own latent publication count differs from completed groups")
				}
				for group := range own {
					want := rotateLatent(r.latent[layer][group], group*2)
					if !reflect.DeepEqual(own[group], want) {
						t.Fatalf("layer %d group %d: owner retained the wrong rotary position", layer, group)
					}
					if group == 1 && reflect.DeepEqual(want, rotateLatent(r.latent[layer][group], 3)) {
						t.Fatal("fixture cannot distinguish closing position 3 from group-first position 2")
					}
				}
			}
			ownKV, _ := st.layerState(0).KVSourceRows(0)
			sharedKV, _ := st.attn.KVSourceRows(0)
			ownK, _ := st.layerState(0).IndexKeys(0)
			sharedK, _ := st.attn.IndexKeys(0)
			if !reflect.DeepEqual(ownKV, sharedKV) || !reflect.DeepEqual(ownK, sharedK) || len(ownK) != groups {
				t.Fatal("owner and registry published different latent/key pairs")
			}
			for group := range ownK {
				key := append([]float32(nil), r.key[0][group]...)
				for i := range key {
					key[i] = v41CompressorNormRefCast(key[i])
				}
				if !reflect.DeepEqual(ownK[group], rotateLatent(key, group*2)) {
					t.Fatal("index key was not rotated after normalization at group-first position")
				}
			}
			if r.projections != groups {
				t.Fatal("reader, restore or incomplete group reprojected a completed index key")
			}
		}
		newRecords := func() *records {
			return &records{latent: map[int][][]float32{}, key: map[int][][]float32{}}
		}
		st, r := &v41ForwardState{}, newRecords()
		bind(st, r)
		if _, err := m.forwardV41([]int{1, 2, 3}, st); err != nil {
			t.Fatal(err)
		}
		check(st, r, 1)
		snapshot := captureV41ForwardSnapshot(st)
		restored := snapshot.clone().restore()
		if !reflect.DeepEqual(captureV41ForwardSnapshot(restored), snapshot) {
			t.Fatal("restore changed published rows or pending unrotated inputs")
		}
		bind(restored, r)
		if _, stats, err := m.forwardV41Step(4, restored, nil); err != nil || !stats.Committed {
			t.Fatalf("restored group completion: %v", err)
		}
		check(restored, r, 2)
		if !reflect.DeepEqual(captureV41ForwardSnapshot(st), snapshot) {
			t.Fatal("restored continuation mutated the source snapshot owner")
		}
		before, _ := restored.attn.KVSourceRows(0)
		beforeK, _ := restored.attn.IndexKeys(0)
		if _, stats, err := m.forwardV41Step(5, restored, nil); err != nil || !stats.Committed {
			t.Fatalf("incomplete next group: %v", err)
		}
		check(restored, r, 2)
		after, _ := restored.attn.KVSourceRows(0)
		afterK, _ := restored.attn.IndexKeys(0)
		if !reflect.DeepEqual(before, after) || !reflect.DeepEqual(beforeK, afterK) {
			t.Fatal("incomplete group rotated a previously published row twice")
		}
		cold, coldRecords := &v41ForwardState{}, newRecords()
		bind(cold, coldRecords)
		if _, err := m.forwardV41([]int{1, 2, 3, 4}, cold); err != nil {
			t.Fatal(err)
		}
		check(cold, coldRecords, 2)
	})

	t.Run("failed-pair-is-unpublished", func(t *testing.T) {
		for _, fault := range []string{"compressor-norm", "index-projection", "index-norm", "rotary", "registry-width"} {
			t.Run(fault, func(t *testing.T) {
				m := &Model{Cfg: Config{HeadDim: 4, IndexHeadDim: 4, QKRopeHeadDim: 2, RopeTheta: 10000}}
				owner, _ := NewV41AttentionState(4, 2)
				registry, _ := NewV41AttentionState(4, 2)
				pool := mustPool(t, 2, 4)
				project := func(input []float32) ([]float32, error) { return append([]float32(nil), input...), nil }
				score := func([]float32) ([]float32, error) { return make([]float32, 4), nil }
				gain := []float32{1, 1, 1, 1}
				input := []float32{1, 2, 3, 4}
				boom := errors.New("publication seam fault")
				fail := false
				publication := v41CompressorPublication{
					registry: registry,
					ref:      V41AttentionStateRef{LayerID: 0, Ratio: 2, IsKVSource: true, IsIndexSource: true},
					projectIndex: func(latent []float32) ([]float32, error) {
						if fail && fault == "index-projection" {
							return nil, boom
						}
						return m.v41IndexKeyNorm(0, append([]float32(nil), latent...), gain, 1e-5,
							func(_ int, value, gain []float32, eps float32) ([]float32, error) {
								if fail && fault == "index-norm" {
									return nil, boom
								}
								return v41IndexKeyNormReference(value, gain, eps), nil
							})
					},
					finalize: func(pos int, latent, key []float32) error {
						if fail && fault == "rotary" {
							latent[0], key[0] = 99, 99
							return boom
						}
						return m.v41CompressedPublicationRoPE(0, pos, latent, key)
					},
				}
				pool.normalize = func(value, gain []float32, eps float32) ([]float32, error) {
					if fail && fault == "compressor-norm" {
						return nil, boom
					}
					return v41CompressorNormRefTail(value, gain, eps, ""), nil
				}
				appendRow := func(pos int) error {
					_, _, _, _, _, err := owner.appendCompressorSourcePublication(0, 2, pos, input, pool, project, score, gain, 1e-5, publication)
					return err
				}
				for pos := 0; pos < 3; pos++ {
					if err := appendRow(pos); err != nil {
						t.Fatal(err)
					}
				}
				if !reflect.DeepEqual(owner.partialInputs, [][]float32{input}) || !reflect.DeepEqual(owner.partialPositions, []int{2}) {
					t.Fatal("incomplete group did not retain its unrotated input and absolute position")
				}
				if fault == "registry-width" {
					registry.indexHeadDim = 3
				}
				before, sharedBefore := owner.clone(), registry.clone()
				fail = true
				if err := appendRow(3); err == nil {
					t.Fatal("injected failure was accepted")
				}
				if !reflect.DeepEqual(owner, before) || !reflect.DeepEqual(registry, sharedBefore) {
					t.Fatal("failed pair changed a publication, cursor or pending group")
				}
				fail, registry.indexHeadDim = false, 4
				if err := appendRow(3); err != nil {
					t.Fatal(err)
				}
				own, _ := owner.KVSourceRows(0)
				shared, _ := registry.KVSourceRows(0)
				ownK, _ := owner.IndexKeys(0)
				sharedK, _ := registry.IndexKeys(0)
				if len(own) != 2 || len(ownK) != 2 || !reflect.DeepEqual(own, shared) || !reflect.DeepEqual(ownK, sharedK) {
					t.Fatal("successful retry did not publish exactly one matched pair")
				}
			})
		}
	})

	t.Run("query-uses-query-position", func(t *testing.T) {
		m := fixture()
		m.Cfg.IndexTopK = 1
		dim := m.Cfg.IndexHeadDim
		keys := [][]float32{make([]float32, dim), make([]float32, dim)}
		keys[0][dim-2], keys[1][dim-1] = -1, 1
		before := cloneV41Rows(keys)
		project := func(_ int, leaf string, _ []float32, out, _, _ int) ([]float32, v41DenseProjectionOutcome, error) {
			value := make([]float32, out)
			switch leaf {
			case "indexer.wq_b.weight":
				value[dim-2] = 1
			case "indexer.weights_proj.weight":
				value[0] = 1
			default:
				t.Fatal("cached key was reprojected")
			}
			return value, v41ProjectionHandled, nil
		}
		for _, tc := range []struct {
			pos int
			row int32
		}{{2, 1}, {3, 0}} {
			rows, err := m.v41IndexRowsProjected(0, tc.pos, make([]float32, m.Cfg.QLoraRank), make([]float32, m.Cfg.HiddenSize), keys, project)
			if err != nil || !reflect.DeepEqual(rows, []int32{tc.row}) {
				t.Fatalf("query position %d selected %v: %v", tc.pos, rows, err)
			}
		}
		if !reflect.DeepEqual(keys, before) {
			t.Fatal("query scoring rotated cached keys again")
		}
		m.Cfg.QKRopeHeadDim = dim + 2
		if err := m.v41CompressIndexForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatal("oversize index rotary width was not refused at admission")
		}
	})
}
