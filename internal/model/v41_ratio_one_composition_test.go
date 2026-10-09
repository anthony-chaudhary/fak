package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// This projected-input witness leaves public admission closed. It exercises
// composition and transaction semantics, not FP8/FP4 or device qualification.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=30ms lane=default
func TestV41RatioOneComposition(t *testing.T) {
	t.Parallel()

	t.Run("one-denominator-duplicates-and-source-width", func(t *testing.T) {
		window := [][]float32{{2, 4}, {6, 8}}
		compressed := [][]float32{{2, 4}, {10, 12}, {100, 200}}
		ids := []int32{0, 0, 1, 2, -1}
		calls := 0
		selected := func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
			calls++
			if layer != 3 || r.plain.N != 5 || r.plain.TopK != 7 || r.plain.Inverse != nil ||
				!reflect.DeepEqual(r.idx, []int32{0, 1, 2, 2, 3, -1, -1}) {
				t.Fatalf("wrong combined request: layer=%d options=%+v indices=%v", layer, r.plain, r.idx)
			}
			return V41SparseAttentionSink(r.q, r.kv, r.sink, r.idx, r.plain)
		}
		for _, attend := range []v41SharedAttentionFunc{nil, selected} {
			// At position 3, source ratio 2 exposes groups 0 and 1. Both
			// copies of group 0 and its equal-valued window row must survive.
			out, err := v41CombinedAttention(3, 3, 2, 2, 2, make([]float32, 4), []float32{0, 0}, window, compressed, ids, 1, attend)
			if err != nil || len(out) != 4 {
				t.Fatalf("combined contraction: %v, %v", out, err)
			}
			for i, want := range []float32{22.0 / 6, 32.0 / 6, 22.0 / 6, 32.0 / 6} {
				if math.Abs(float64(out[i]-want)) > 1e-6 {
					t.Fatalf("output[%d]=%g, want %g with one sink", i, out[i], want)
				}
			}
		}
		if calls != 1 {
			t.Fatalf("selected contraction calls=%d, want one", calls)
		}
		// Validation precedes the causal mask, even for an invisible row/ID.
		compressed[2][0] = float32(math.NaN())
		if out, err := v41CombinedAttention(3, 0, 2, 1, 2, []float32{0, 0}, nil, window, compressed, ids, 1, selected); err == nil || out != nil || calls != 1 {
			t.Fatal("invisible non-finite source reached selected execution")
		}
		compressed[2][0], ids[3] = 100, 99
		if out, err := v41CombinedAttention(3, 0, 2, 1, 2, []float32{0, 0}, nil, window, compressed, ids, 1, selected); err == nil || out != nil || calls != 1 {
			t.Fatal("invisible invalid canonical ID was hidden by masking")
		}
	})

	t.Run("selected-output-and-host-finiteness", func(t *testing.T) {
		borrowed := []float32{7, 9}
		calls := 0
		attend := func(int, v41SharedAttentionRequest) ([]float32, error) {
			calls++
			return borrowed, nil
		}
		invoke := func() ([]float32, error) {
			return v41CombinedAttention(0, 0, 1, 1, 2, []float32{0, 0}, nil, [][]float32{{1, 2}}, nil, nil, 1, attend)
		}
		out, err := invoke()
		if err != nil || !reflect.DeepEqual(out, borrowed) {
			t.Fatalf("selected output: %v, %v", out, err)
		}
		out[0] = -1
		if borrowed[0] != 7 {
			t.Fatal("returned output aliases the selected callback")
		}
		for _, bad := range [][]float32{{1}, {float32(math.Inf(1)), 0}} {
			borrowed = bad
			out, err = invoke()
			var selected *V41SharedAttentionOperationError
			if out != nil || !errors.As(err, &selected) || selected.Layer != 0 {
				t.Fatalf("invalid selected output lost its no-replay error: %v, %v", out, err)
			}
		}
		if calls != 3 {
			t.Fatal("selected failure retried the contraction")
		}
		// Finite inputs overflow the existing scalar score. The composer must
		// refuse its non-finite output without relying on a future sink fix.
		out, err = v41CombinedAttention(0, 0, 1, 1, 1, []float32{math.MaxFloat32}, nil, [][]float32{{2}}, nil, nil, 1, nil)
		if err == nil || out != nil {
			t.Fatalf("non-finite scalar result accepted: %v, %v", out, err)
		}
	})

	newState := func() *V41AttentionState {
		s, err := NewV41AttentionState(2, 1)
		if err != nil {
			t.Fatal(err)
		}
		s.windowSize, s.window, s.indexHeadDim = 2, s.window[:2], 1
		return s
	}
	requestAt := func(pos int) v41RatioOneStep {
		return v41RatioOneStep{
			Layer: 0, KVSourceLayer: 0, IndexKeySourceLayer: 0, Pos: pos, Window: 2, Heads: 1,
			Input: []float32{float32(pos + 1), 2}, WindowKV: []float32{float32(50 + pos), float32(-50 - pos)},
			NormWeight: []float32{1, 0.5}, Epsilon: 1e-5,
			ProjectKV:    func(input []float32) ([]float32, error) { return input, nil },
			ProjectIndex: func(latent []float32) ([]float32, error) { return []float32{latent[0] - latent[1]}, nil },
			Finalize:     func(int, []float32, []float32) error { return nil },
			Consume:      func(window, compressed, keys [][]float32) ([]float32, error) { return window[len(window)-1], nil },
		}
	}

	t.Run("owner-reader-ring-and-clone", func(t *testing.T) {
		owner, reader, registry := newState(), newState(), newState()
		projections, keysProjected, finalized := 0, 0, 0
		for pos := 0; pos < 4; pos++ {
			r := requestAt(pos)
			wantLatent := v41CompressorNormRefTail(r.Input, r.NormWeight, r.Epsilon, "")
			r.ProjectKV = func(input []float32) ([]float32, error) { projections++; return input, nil }
			r.ProjectIndex = func(latent []float32) ([]float32, error) {
				keysProjected++
				if !reflect.DeepEqual(latent, wantLatent) {
					t.Fatal("index projection did not receive normalized, unfinalized latent")
				}
				return []float32{latent[0] - latent[1]}, nil
			}
			r.Finalize = func(at int, latent, key []float32) error {
				finalized++
				if at != pos {
					t.Fatalf("ratio-one finalize position=%d, want %d", at, pos)
				}
				latent[0] += float32(10 * pos)
				key[0] -= float32(pos)
				return nil
			}
			var retainedOutput []float32
			r.Consume = func(window, compressed, keys [][]float32) ([]float32, error) {
				if len(window) != min(pos+1, 2) || len(compressed) != pos+1 || len(keys) != pos+1 ||
					window[0][0] != float32(50+max(0, pos-1)) || compressed[pos][0] != wantLatent[0]+float32(10*pos) {
					t.Fatal("consumer saw wrong window history, source width or finalized latent")
				}
				retainedOutput = window[len(window)-1]
				compressed[0][0], keys[0][0] = -999, -999
				return retainedOutput, nil
			}
			out, err := owner.stepRatioOne(registry, r)
			if err != nil || !reflect.DeepEqual(out, r.WindowKV) {
				t.Fatalf("owner position %d: %v, %v", pos, out, err)
			}
			retainedOutput[0], out[1] = -999, -999
			own, _ := owner.KVSourceRows(0)
			shared, _ := registry.KVSourceRows(0)
			ownKeys, _ := owner.IndexKeys(0)
			sharedKeys, _ := registry.IndexKeys(0)
			if !reflect.DeepEqual(own, shared) || !reflect.DeepEqual(ownKeys, sharedKeys) || shared[0][0] == -999 || sharedKeys[0][0] == -999 ||
				!reflect.DeepEqual(owner.window[pos%2], r.WindowKV) || owner.nextWindowPos != pos+1 || owner.nextCompressRow != pos+1 ||
				owner.retainedWindowRows != min(pos+1, 2) || owner.retainedCopies != min(pos+1, 2) {
				t.Fatal("publication, alias ownership, retained ring or single-advance invariant failed")
			}
			before := registry.clone()
			r = requestAt(pos)
			r.Layer, r.Input, r.NormWeight = 1, nil, nil
			r.ProjectKV = func([]float32) ([]float32, error) { panic("reader projected KV") }
			r.ProjectIndex = func([]float32) ([]float32, error) { panic("reader projected index") }
			r.Finalize = func(int, []float32, []float32) error { panic("reader finalized source") }
			r.Consume = func(window, compressed, keys [][]float32) ([]float32, error) {
				if !reflect.DeepEqual(compressed, shared) || !reflect.DeepEqual(keys, sharedKeys) {
					t.Fatal("reader did not resolve the owner's current source")
				}
				compressed[0][0], keys[0][0] = -888, -888
				return window[len(window)-1], nil
			}
			if _, err := reader.stepRatioOne(registry, r); err != nil || !reflect.DeepEqual(registry, before) || reader.nextCompressRow != pos+1 || reader.nextWindowPos != pos+1 || len(reader.kvPublications) != 0 {
				t.Fatalf("reader mutated registry or published/projected its own source: %v", err)
			}
			// Clone the actual data-only state mid-prefix, then continue from
			// the clones on the next loop without carrying any callback.
			if pos == 1 {
				owner, reader, registry = owner.clone(), reader.clone(), registry.clone()
			}
		}
		if projections != 4 || keysProjected != 4 || finalized != 4 {
			t.Fatalf("producer call counts=%d/%d/%d, want 4 each", projections, keysProjected, finalized)
		}
	})

	t.Run("transactional-failure-and-retry", func(t *testing.T) {
		faults := []string{"kv-error", "kv-panic", "index-error", "index-panic", "finalize-error", "finalize-panic", "consume-error", "consume-panic", "short-output", "nonfinite-output", "nonfinite-latent", "nonfinite-key", "nonfinite-finalized", "wrong-key-source", "negative-source-end", "negative-key-end", "source-hole", "missing-window", "output-byte-overflow"}
		for _, fault := range faults {
			t.Run(fault, func(t *testing.T) {
				owner, registry := newState(), newState()
				if _, err := owner.stepRatioOne(registry, requestAt(0)); err != nil {
					t.Fatal(err)
				}
				validOwner, validRegistry := owner.clone(), registry.clone()
				r := requestAt(1)
				boom := errors.New(fault)
				raise := func() error {
					if fault == "kv-panic" || fault == "index-panic" || fault == "finalize-panic" || fault == "consume-panic" {
						panic(boom)
					}
					return boom
				}
				switch fault {
				case "kv-error", "kv-panic":
					r.ProjectKV = func(input []float32) ([]float32, error) { input[0] = -99; return nil, raise() }
				case "index-error", "index-panic":
					r.ProjectIndex = func(latent []float32) ([]float32, error) { latent[0] = -99; return nil, raise() }
				case "finalize-error", "finalize-panic":
					r.Finalize = func(_ int, latent, key []float32) error { latent[0], key[0] = -99, -99; return raise() }
				case "consume-error", "consume-panic":
					r.Consume = func(window, compressed, keys [][]float32) ([]float32, error) {
						window[0][0], compressed[0][0], keys[0][0] = -99, -99, -99
						return nil, raise()
					}
				case "short-output":
					r.Consume = func(_, _, _ [][]float32) ([]float32, error) { return []float32{1}, nil }
				case "nonfinite-output":
					r.Consume = func(_, _, _ [][]float32) ([]float32, error) { return []float32{float32(math.NaN()), 0}, nil }
				case "nonfinite-latent":
					r.ProjectKV = func([]float32) ([]float32, error) { return []float32{math.MaxFloat32, 0}, nil }
					r.ProjectIndex = func([]float32) ([]float32, error) { t.Fatal("bad latent reached index projection"); return nil, nil }
				case "nonfinite-key":
					r.ProjectIndex = func([]float32) ([]float32, error) { return []float32{float32(math.Inf(1))}, nil }
					r.Finalize = func(int, []float32, []float32) error { t.Fatal("bad key reached finalization"); return nil }
				case "nonfinite-finalized":
					r.Finalize = func(_ int, latent, _ []float32) error { latent[0] = float32(math.NaN()); return nil }
				case "wrong-key-source":
					r.Layer, r.IndexKeySourceLayer = 1, 1
				case "negative-source-end":
					registry.kvPublishedEnd[0] = -1
				case "negative-key-end":
					registry.indexPublishedEnd[0] = -1
				case "source-hole":
					delete(registry.kvPublications, v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1})
				case "missing-window":
					owner.retainedWindowRows, owner.retainedCopies = 0, 0
				case "output-byte-overflow":
					r.Heads = int(^uint(0)>>1)/8 + 1
				}
				before, sharedBefore := owner.clone(), registry.clone()
				var out []float32
				var err error
				var caught any
				func() {
					defer func() { caught = recover() }()
					out, err = owner.stepRatioOne(registry, r)
				}()
				wantPanic := fault == "kv-panic" || fault == "index-panic" || fault == "finalize-panic" || fault == "consume-panic"
				if out != nil || (wantPanic && caught != boom) || (!wantPanic && (caught != nil || err == nil)) ||
					!reflect.DeepEqual(owner, before) || !reflect.DeepEqual(registry, sharedBefore) || r.Input[0] != 2 {
					t.Fatalf("failure mutated state/input or lost its outcome: output=%v err=%v panic=%v", out, err, caught)
				}
				if _, err := validOwner.stepRatioOne(validRegistry, requestAt(1)); err != nil || validOwner.nextWindowPos != 2 || validOwner.nextCompressRow != 2 || validRegistry.kvPublishedEnd[0] != 2 || validRegistry.indexPublishedEnd[0] != 2 {
					t.Fatalf("retry failed to publish/advance exactly once: %v", err)
				}
			})
		}
	})

	t.Run("public-fence", func(t *testing.T) {
		m := &Model{Cfg: Config{NumLayers: 1, DeepSeekV41: &DeepSeekV41Config{CompressRatios: []int{1}, KVSourceLayerIDs: []int{0}}}}
		if !errors.Is(m.v41CompressIndexForwardAdmitted(), ErrV41ForwardStage) || !errors.Is(m.v41KVSourceForwardAdmitted(), ErrV41ForwardStage) {
			t.Fatal("internal composition widened ratio-one public admission")
		}
		m.Cfg.DeepSeekV41.CandidateSourceLayerID = 0
		if !errors.Is(v41CandidateSourceForwardAdmitted(m.Cfg), ErrV41ForwardStage) {
			t.Fatal("internal composition widened candidate-source public admission")
		}
	})
}
