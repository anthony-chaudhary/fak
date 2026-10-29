package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=50ms

// v41CompressedKVRowLogRecorder is the fake compute.V41RowLogBackend the #13914
// witness drives: it materializes V4.1's kept-KV rows on a device row log and
// accounts the bytes those allocations occupy.
type v41CompressedKVRowLogRecorder struct {
	allocs int64
}

func (b *v41CompressedKVRowLogRecorder) V41RowLogAvailable() bool { return true }

func (b *v41CompressedKVRowLogRecorder) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	if err := compute.ValidateV41RowLogAlloc(rows, width); err != nil {
		return compute.Tensor{}, err
	}
	b.allocs += int64(rows) * int64(width) * 4
	return compute.NewF32(compute.Default(), []int{rows, width}, make([]float32, rows*width)), nil
}

func (b *v41CompressedKVRowLogRecorder) V41RowLogCopy(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	_, err := compute.ValidateV41RowLogCopy(dst, dstRow, src, srcRow, rows)
	return err
}

func (b *v41CompressedKVRowLogRecorder) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	rows, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	b.allocs += int64(rows) * int64(width) * 4
	return compute.NewF32(compute.Default(), []int{rows, width}, make([]float32, rows*width)), nil
}

// v41RealizedMirrorBytes drives the published V4.1 retention schedule through
// the real attention-state machinery at `tokens` cached positions and
// materializes every kept row on rec, returning the bytes rec allocated.
//
//   - One real V41AttentionState ring is advanced through Prefill/Step so the
//     per-layer device row mirror's populated slots are what the ring actually
//     holds; the 40 decoder layers share that geometry (one v41WindowSize-row
//     HeadDim-f32 ring each), so its populated count replicates verbatim.
//   - One shared state receives the source layers' publications at the real
//     group-completion cadence (a ratio-r source completes one group — and
//     publishes one compressed row plus, when it is an index source, one index
//     key — every r-th position), so the kept compressed rows and index keys
//     are the state's own storage, not a re-derived count.
//   - The compressor in-flight peak — at most ratio-1 pre-attention input
//     carriers plus ratio-1 projected rows per compressing layer, the bound
//     V41AttentionState's append guards enforce — is materialized as fixed
//     buffers.
func v41RealizedMirrorBytes(t *testing.T, cfg Config, rec *v41CompressedKVRowLogRecorder, tokens int) int64 {
	t.Helper()
	headDim, indexDim, hidden := cfg.HeadDim, cfg.IndexHeadDim, cfg.HiddenSize
	kvSources := map[int]bool{}
	for _, l := range v41PublishedKVSourceLayerIDs {
		kvSources[l] = true
	}
	indexSources := map[int]bool{}
	for _, l := range v41PublishedIndexSourceLayerIDs {
		indexSources[l] = true
	}

	// Per-layer device row mirror: advance one real ring to `tokens` positions.
	ring, err := NewV41AttentionState(headDim, 2)
	if err != nil {
		t.Fatal(err)
	}
	chunk := min(v41WindowSize, tokens)
	prefill := make([][]float32, chunk)
	for i := range prefill {
		prefill[i] = make([]float32, headDim)
	}
	if err := ring.Prefill(prefill, nil); err != nil {
		t.Fatal(err)
	}
	step := make([]float32, headDim)
	for pos := chunk; pos < tokens; pos++ {
		if err := ring.Step(step, nil); err != nil {
			t.Fatal(err)
		}
	}
	populated := min(v41WindowSize, ring.nextWindowPos)
	if want := min(v41WindowSize, tokens); populated != want {
		t.Fatalf("ring populated %d slots at %d positions, want %d", populated, tokens, want)
	}
	for l := 0; l < v41PublishedNumLayers; l++ {
		if _, err := rec.V41RowLogAlloc(populated, headDim); err != nil {
			t.Fatal(err)
		}
	}

	// Shared compressed rows and index keys, published by the real sources.
	shared, err := NewV41AttentionState(headDim, 2)
	if err != nil {
		t.Fatal(err)
	}
	shared.indexHeadDim = indexDim
	sources := make([]int, 0, len(indexSources))
	for l := range indexSources {
		sources = append(sources, l)
	}
	for i := 0; i+1 < len(sources); i++ {
		for j := i + 1; j < len(sources); j++ {
			if sources[j] < sources[i] {
				sources[i], sources[j] = sources[j], sources[i]
			}
		}
	}
	row := make([]float32, headDim)
	key := make([]float32, indexDim)
	for pos := 0; pos < tokens; pos++ {
		var updates []V41AttentionStateUpdate
		for _, l := range sources {
			ratio := cfg.CompressRatios[l]
			if ratio <= 0 || (pos+1)%ratio != 0 {
				continue
			}
			upd := V41AttentionStateUpdate{
				Ref: V41AttentionStateRef{
					LayerID:       l,
					Ratio:         ratio,
					IsKVSource:    kvSources[l],
					IsIndexSource: indexSources[l],
				},
			}
			if kvSources[l] {
				upd.Latent = row
			}
			if indexSources[l] {
				upd.IndexKey = key
			}
			updates = append(updates, upd)
		}
		if len(updates) > 0 {
			if err := shared.publishUpdates(updates); err != nil {
				t.Fatal(err)
			}
		}
	}
	// A ratio-r source completes one group every r-th position, so it has
	// published exactly tokens/r rows at `tokens` positions (zero before its
	// first completion). The state retains publications until Reset, so the
	// retained count is the cadence count verbatim.
	for _, l := range v41PublishedKVSourceLayerIDs {
		rows, ok := shared.KVSourceRows(l)
		want := tokens / cfg.CompressRatios[l]
		if ok != (want > 0) || len(rows) != want {
			t.Fatalf("source layer %d retained %d compressed rows at %d positions, want %d", l, len(rows), tokens, want)
		}
		if len(rows) > 0 { // the row log refuses a zero-row alloc; zero rows keep zero bytes
			if _, err := rec.V41RowLogAlloc(len(rows), headDim); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, l := range v41PublishedIndexSourceLayerIDs {
		keys, ok := shared.IndexKeys(l)
		want := tokens / cfg.CompressRatios[l]
		if ok != (want > 0) || len(keys) != want {
			t.Fatalf("index source layer %d retained %d index keys at %d positions, want %d", l, len(keys), tokens, want)
		}
		if len(keys) > 0 { // the row log refuses a zero-row alloc; zero keys keep zero bytes
			if _, err := rec.V41RowLogAlloc(len(keys), indexDim); err != nil {
				t.Fatal(err)
			}
		}
	}

	// Compressor in-flight peak: ratio-1 carriers plus ratio-1 projected rows
	// per compressing layer.
	for l := 0; l < v41PublishedNumLayers; l++ {
		ratio := cfg.CompressRatios[l]
		if ratio <= 1 {
			continue
		}
		if _, err := rec.V41RowLogAlloc(ratio-1, hidden); err != nil {
			t.Fatal(err)
		}
		if _, err := rec.V41RowLogAlloc(ratio-1, headDim); err != nil {
			t.Fatal(err)
		}
	}
	return rec.allocs
}

// TestV41CompressedKVChargeBoundsRealizedMirror is the #13914 witness: under
// FAK_HYBRID_KV the V4.1 context plan must be sized from the kept KV — the
// per-layer device row mirror, the owner layers' compressed rows, the index
// keys, and the compressor in-flight rows — and charge at least what a real
// session retains at t positions. Gate-off stays byte-identical to the uniform
// projection.
func TestV41CompressedKVChargeBoundsRealizedMirror(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	t.Setenv("FAK_HYBRID_KV", "0")
	off := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
	if len(off.CompressedKV) != 0 {
		t.Fatalf("gate-off V4.1 projection carries a compressed schedule: %#v", off.CompressedKV)
	}
	t.Setenv("FAK_HYBRID_KV", "1")
	on := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
	if len(on.CompressedKV) == 0 {
		t.Fatal("V4.1 kept-KV schedule uncharged under FAK_HYBRID_KV")
	}
	if got, want := on.KVBytes(2048), off.KVBytes(2048); got >= want {
		t.Fatalf("gate-on V4.1 charge %d did not discount the uniform %d", got, want)
	}
	inflight := false
	for _, demand := range on.SessionState {
		if demand.Detail == "deepseek-v41-compressor-inflight-rows" &&
			demand.Class == compute.MemoryKVCache && demand.DType == compute.F32.String() {
			inflight = true
		}
	}
	if !inflight {
		t.Errorf("compressor in-flight rows uncharged: %#v", on.SessionState)
	}
	for _, tokens := range []int{1, 127, 128, 129, 2048} {
		rec := &v41CompressedKVRowLogRecorder{}
		realized := v41RealizedMirrorBytes(t, cfg, rec, tokens)
		planned := on.KVBytes(tokens) + on.SessionState.Total()
		if planned < realized {
			t.Errorf("tokens=%d planned %d B < realized %d B", tokens, planned, realized)
		}
	}
	t.Setenv("FAK_HYBRID_KV", "0")
	again := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
	if !reflect.DeepEqual(off, again) {
		t.Fatal("gate-off V4.1 projection changed after gated projection")
	}
}
