package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Observer snapshots own their rows after the synchronous callback returns.
// Keep this independent of the production row lookup and selection helpers.
func v41TestAttentionFlatKV(r v41SharedAttentionRequest) []float32 {
	if r.rowKV == nil {
		return append([]float32(nil), r.kv...)
	}
	var out []float32
	for _, stream := range [][][]float32{r.rowKV.window, r.rowKV.compressed} {
		for _, row := range stream {
			out = append(out, row...)
		}
	}
	return out
}

// CPU-storage contract witness; this does not qualify hardware throughput.
// fak-test:runtime medium est=1s lane=default
func TestV41CombinedResidentBorrowedRows(t *testing.T) {
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	for _, rowLog := range []bool{false, true} {
		t.Run(fmt.Sprintf("row_log_%t", rowLog), func(t *testing.T) {
			recorder := newV41SharedAttentionTestBackend()
			var backend compute.Backend = recorder
			if rowLog {
				resident := newV41ResidentKVTestBackend()
				backend, recorder = resident, resident.v41SharedAttentionTestBackend
			}
			s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), backend)
			selected := s.v41SharedAttentionFunc()
			if selected == nil {
				t.Fatal("shared attention was not bound")
			}
			owner := &V41AttentionState{}
			window := [][]float32{{0.25, -0.5}, {0.75, 0.125}}
			compressed := [][]float32{{-0.25, 0.5}, {0.125, 0.25}}
			for pos := 0; pos < 2; pos++ {
				view := &v41ResidentKVView{owner: owner, pos: pos, ringRows: 2, windowRows: pos + 1, compressedRows: pos + 1}
				ids := []int32{0, 0, -1}
				if pos == 1 {
					ids = []int32{1, 0, 1, -1}
				}
				before := recorder.uploadBytes
				var expected v41SharedAttentionTestCall
				attend := func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
					if r.kv != nil || r.rowKV == nil || &r.rowKV.window[0][0] != &window[0][0] || &r.rowKV.compressed[0][0] != &compressed[0][0] {
						t.Fatal("resident seam copied or flattened borrowed history")
					}
					flat := v41TestAttentionFlatKV(r)
					expected = v41SharedAttentionTestCall{q: append([]float32(nil), r.q...), sink: append([]float32(nil), r.sink...), heads: 1, dim: 2, scale: 0.5, mode: r.mode, hasSink: true}
					for _, id := range r.idx {
						if id >= 0 {
							expected.kv = append(expected.kv, flat[int(id)*2:(int(id)+1)*2]...)
							expected.rows++
						}
					}
					out, err := selected(layer, r)
					if !reflect.DeepEqual(v41TestAttentionFlatKV(r), flat) {
						t.Fatal("callback mutated borrowed source rows")
					}
					return out, err
				}
				got, err := v41CombinedAttentionResident(0, pos, 1, 1, 2, []float32{0.5, -0.25}, []float32{0.125}, window[:pos+1], compressed[:pos+1], ids, 0.5, view, attend)
				if err != nil {
					t.Fatal(err)
				}
				want, err := v41SharedAttentionRoundedControl(expected)
				if err != nil || !reflect.DeepEqual(got, want) || !reflect.DeepEqual(recorder.calls[pos], expected) {
					t.Fatalf("row payload/parity mismatch: got=%v want=%v err=%v", got, want, err)
				}
				if pos == 1 {
					wantBytes := 4 * (2 + 5*2 + 1)
					if rowLog {
						wantBytes = 4 * (2 + 2 + 2 + 1) // Q, new window/log rows, transient sink
					}
					if delta := recorder.uploadBytes - before; delta != wantBytes {
						t.Fatalf("steady upload bytes=%d want=%d", delta, wantBytes)
					}
				}
			}
		})
	}
	// Invalid future rows must still fail before masking or backend dispatch.
	called := false
	_, err := v41CombinedAttentionResident(0, 0, 2, 1, 2, []float32{1, 0}, nil,
		[][]float32{{0, 1}}, [][]float32{{0, float32(math.NaN())}}, []int32{0}, 1,
		&v41ResidentKVView{}, func(int, v41SharedAttentionRequest) ([]float32, error) {
			called = true
			return []float32{0, 0}, nil
		})
	var stage *V41ForwardError
	if called || !errors.As(err, &stage) || stage.Stage != v41StageAttention {
		t.Fatalf("invisible invalid row: called=%t err=%v", called, err)
	}

	t.Run("restored_step_borrows_source_publications", func(t *testing.T) {
		m := v41CompressorTestFixtureIndex(t, 128)
		// The reader owns its query but must consume layer 0's KV and keys.
		m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
		source := m.NewSession()
		t.Cleanup(source.Close)
		source.Prefill([]int{1, 2, 3})
		cached, err := source.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(cached.Close)
		cached.PrepareForCache()
		pristine := cached.v41.clone()
		restore := func(shared bool) *Session {
			t.Helper()
			var snapshot *PrefixSnapshot
			var err error
			if shared {
				snapshot, err = cached.CloneForRestore()
			} else {
				snapshot, err = cached.Clone()
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(snapshot.Close)
			s := m.NewSession()
			t.Cleanup(s.Close)
			if err := snapshot.Restore(s); err != nil {
				t.Fatal(err)
			}
			return s
		}
		branch, sibling, control := restore(true), restore(true), restore(false)
		st := branch.v41Forward
		oldKey := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
		newKey := v41AttentionPublicationKey{sourceLayer: 0, start: 1, end: 2}
		if st.attn.publicationPrefix == nil || st.layers[0].publicationPrefix == nil ||
			control.v41Forward.attn.publicationPrefix != nil {
			t.Fatal("witness requires sealed and deep-owned restored branches")
		}
		base, ownerBase := st.attn.publicationPrefix, st.layers[0].publicationPrefix
		before := captureV41ForwardSnapshot(st)
		failure := errors.New("selected reader after borrowed source publication")
		fail := true
		calls, scores := [2]int{}, [2]int{}
		var retained [2][]float32 // Only owned copies may outlive each callback.
		storeFor := func(layer int) *V41AttentionState {
			if layer == 0 {
				return st.layers[0]
			}
			return st.attn
		}
		st.setIndexerScore(func(layer int, q, flat, weights []float32, heads, dim, rows int) ([]float32, error) {
			store := storeFor(layer)
			keys, ok := store.borrowIndexKeys(0)
			if !ok || len(keys) != 2 || rows != 2 || len(flat) != 2*dim {
				t.Fatal("Step scorer lost completed source keys")
			}
			for group, key := range keys {
				published, ok := store.indexPublication(v41AttentionPublicationKey{sourceLayer: 0, start: group, end: group + 1})
				if !ok || &key[0] != &published[0] || &flat[group*dim] == &key[0] ||
					!reflect.DeepEqual(flat[group*dim:(group+1)*dim], key) {
					t.Fatal("keys were copied before packing, aliased the scorer, or changed source")
				}
			}
			scores[layer]++
			return V41IndexerScore(q, flat, weights, heads, dim, rows)
		})
		st.setSharedAttention(func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
			store := storeFor(layer)
			if r.kv != nil || r.rowKV == nil || len(r.rowKV.compressed) != 2 ||
				len(store.publicationPrefix.kv[oldKey]) != m.Cfg.HeadDim || len(store.kvPublications[newKey]) != m.Cfg.HeadDim {
				t.Fatal("Step did not expose a sealed row and a private append")
			}
			for group, row := range r.rowKV.compressed {
				published, ok := store.kvPublication(v41AttentionPublicationKey{sourceLayer: 0, start: group, end: group + 1})
				if !ok || &row[0] != &published[0] {
					t.Fatalf("layer %d group %d copied its source publication", layer, group)
				}
			}
			flat := v41TestAttentionFlatKV(r)
			out, err := V41SparseAttentionSink(r.q, flat, r.sink, r.idx, r.plain)
			if !reflect.DeepEqual(v41TestAttentionFlatKV(r), flat) {
				t.Fatal("synchronous consumer mutated borrowed publications")
			}
			retained[layer] = flat
			calls[layer]++
			if layer == 1 && fail {
				return nil, failure
			}
			return out, err
		})
		_, stats, err := m.forwardV41Step(4, st, nil)
		var selected *V41SharedAttentionOperationError
		if !errors.Is(err, failure) || !errors.As(err, &selected) || selected.Layer != 1 || stats.Committed ||
			calls != [2]int{1, 1} || scores != [2]int{1, 1} ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(st), before) ||
			len(st.attn.kvPublications) != 0 || len(st.layers[0].kvPublications) != 0 ||
			st.attn.publicationPrefix != base || st.layers[0].publicationPrefix != ownerBase {
			t.Fatalf("selected reader failure lost rollback or borrowed-prefix ownership: %v", err)
		}
		fail = false
		got, stats, err := m.forwardV41Step(4, st, nil)
		want, controlStats, controlErr := m.forwardV41Step(4, control.v41Forward, nil)
		if err != nil || controlErr != nil || !stats.Committed || !controlStats.Committed ||
			calls != [2]int{2, 2} || scores != [2]int{2, 2} || !reflect.DeepEqual(got, want) ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(control.v41Forward)) ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(sibling.v41Forward), pristine) || !reflect.DeepEqual(cached.v41.clone(), pristine) {
			t.Fatalf("borrowed Step differs from owned continuation or changed a sibling: %v / %v", err, controlErr)
		}
		for layer := 0; layer < 2; layer++ {
			store := storeFor(layer)
			rows, _ := store.KVSourceRows(0)
			keys, _ := store.IndexKeys(0)
			for group := range rows {
				row, _ := store.kvPublication(v41AttentionPublicationKey{sourceLayer: 0, start: group, end: group + 1})
				key, _ := store.indexPublication(v41AttentionPublicationKey{sourceLayer: 0, start: group, end: group + 1})
				rows[group][0], keys[group][0] = row[0]+1, key[0]+1
				if row[0] == rows[group][0] || key[0] == keys[group][0] {
					t.Fatal("public reader exposed a writable publication")
				}
			}
			compressed := retained[layer][len(retained[layer])-2*m.Cfg.HeadDim:]
			if !reflect.DeepEqual(compressed, append(append([]float32(nil), store.publicationPrefix.kv[oldKey]...), store.kvPublications[newKey]...)) {
				t.Fatal("borrowed publications changed after callback return")
			}
		}

		plan, err := m.v41AttentionPlan(1)
		if err != nil || plan.KVSourceLayer != 0 || plan.IndexSourceLayer != 1 {
			t.Fatalf("witness must separate KV ownership from query ownership: %v", err)
		}
		ownedRows, ownedKeys, err := m.v41CompressedReaderRows(plan, st.attn, 4)
		borrowedRows, borrowedKeys, borrowErr := m.v41CompressedReaderStepRows(plan, st.attn, 4)
		if err != nil || borrowErr != nil || !reflect.DeepEqual(ownedRows, borrowedRows) || !reflect.DeepEqual(ownedKeys, borrowedKeys) ||
			&ownedRows[0][0] == &borrowedRows[0][0] || &ownedKeys[0][0] == &borrowedKeys[0][0] {
			t.Fatal("full-prefill resolver lost its independent writable result")
		}
		for _, fault := range []string{"nil", "short", "stale", "missing", "hole", "keys", "empty"} {
			registry := st.attn.clone()
			positions := 4
			switch fault {
			case "nil":
				registry = nil
			case "short":
				positions = 6
			case "stale":
				positions = 2
			case "missing":
				registry.kvPublications = nil
			case "hole":
				delete(registry.kvPublications, oldKey)
			case "keys":
				registry.indexPublications = nil
			case "empty":
				registry.Reset()
				positions = 1
			}
			_, _, ownedErr := m.v41CompressedReaderRows(plan, registry, positions)
			_, _, borrowedErr := m.v41CompressedReaderStepRows(plan, registry, positions)
			if fault == "empty" {
				if ownedErr != nil || borrowedErr != nil {
					t.Fatal("incomplete source group was refused")
				}
				continue
			}
			var ownedStage, borrowedStage *V41ForwardError
			if !errors.As(ownedErr, &ownedStage) || !errors.As(borrowedErr, &borrowedStage) ||
				!errors.Is(borrowedErr, ErrV41ForwardStage) || ownedErr.Error() != borrowedErr.Error() || ownedStage.Stage != borrowedStage.Stage {
				t.Fatalf("%s changed the exact source refusal: %v / %v", fault, ownedErr, borrowedErr)
			}
		}
		ordinary := &Model{Cfg: m.Cfg}
		d41 := *m.Cfg.DeepSeekV41
		d41.IndexSourceLayerIDs = []int{0}
		ordinary.Cfg.DeepSeekV41 = &d41
		noKeys := st.attn.clone()
		noKeys.indexPublications = nil
		for _, resolve := range []func(V41AttentionPlan, *V41AttentionState, int) ([][]float32, [][]float32, error){
			ordinary.v41CompressedReaderRows, ordinary.v41CompressedReaderStepRows,
		} {
			rows, keys, err := resolve(plan, noKeys, 4)
			if err != nil || len(rows) != 2 || keys != nil {
				t.Fatalf("ordinary reader required index keys: %v", err)
			}
		}
	})
}

// v41ResidentKVTestBackend adds a CPU-storage row log to the shared-attention
// recorder. It tests mirror routing, transfer volume and failure typing only;
// physical device behavior is TestV41DecodeResidentH2DVulkan.
type v41ResidentKVTestBackend struct {
	*v41SharedAttentionTestBackend
	gathers  int
	rowFault string
}

func newV41ResidentKVTestBackend() *v41ResidentKVTestBackend {
	return &v41ResidentKVTestBackend{v41SharedAttentionTestBackend: newV41SharedAttentionTestBackend()}
}

func (b *v41ResidentKVTestBackend) own(shape []int, values []float32) compute.Tensor {
	out := compute.NewF32(b, append([]int(nil), shape...), values)
	b.allocations[out.Buf()] = true
	b.managed[out.Buf()] = true
	return out
}

func (b *v41ResidentKVTestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 resident ") && !strings.HasPrefix(site, "V4.1 shared attention ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 {
		panic("unexpected resident upload dtype")
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	b.uploads++
	b.uploadBytes += 4 * len(values)
	return b.own(x.Shape, values)
}

// V41SharedAttention repeats the recorder's rounded control over this
// wrapper's own tensors.
func (b *v41ResidentKVTestBackend) V41SharedAttention(q, kv, sink compute.Tensor, rows, heads, dim int, scale float32, mode compute.V41SharedAttentionMode, hasSink bool) (compute.Tensor, error) {
	sinkWidth := 1
	if hasSink {
		sinkWidth = heads
	}
	for _, input := range []struct {
		x     compute.Tensor
		shape []int
	}{{q, []int{heads, dim}}, {kv, []int{rows, dim}}, {sink, []int{sinkWidth}}} {
		if input.x.Backend() != b || !b.allocations[input.x.Buf()] || !reflect.DeepEqual(input.x.Shape, input.shape) {
			return compute.Tensor{}, errors.New("invalid tensor or dead binding")
		}
	}
	call := v41SharedAttentionTestCall{
		q: append([]float32(nil), b.Backend.Read(q)...), kv: append([]float32(nil), b.Backend.Read(kv)...), sink: append([]float32(nil), b.Backend.Read(sink)...),
		rows: rows, heads: heads, dim: dim, scale: scale, mode: mode, hasSink: hasSink,
	}
	b.calls = append(b.calls, call)
	values, err := v41SharedAttentionRoundedControl(call)
	if err != nil {
		return compute.Tensor{}, err
	}
	return b.own([]int{heads, dim}, values), nil
}

func (b *v41ResidentKVTestBackend) V41RowLogAvailable() bool { return true }

func (b *v41ResidentKVTestBackend) fault(op string) error {
	if b.rowFault == op {
		return &compute.BackendError{Backend: "cpu-ref-row-log", Site: op, Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
	}
	return nil
}

func (b *v41ResidentKVTestBackend) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	if err := compute.ValidateV41RowLogAlloc(rows, width); err != nil {
		return compute.Tensor{}, err
	}
	if err := b.fault("alloc"); err != nil {
		return compute.Tensor{}, err
	}
	return b.own([]int{rows, width}, make([]float32, rows*width)), nil
}

func (b *v41ResidentKVTestBackend) V41RowLogCopy(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	rowBytes, err := compute.ValidateV41RowLogCopy(dst, dstRow, src, srcRow, rows)
	if err != nil {
		return err
	}
	if !b.allocations[dst.Buf()] || !b.allocations[src.Buf()] {
		return errors.New("row-log copy on a dead buffer")
	}
	if err := b.fault("copy"); err != nil {
		return err
	}
	w := rowBytes / 4
	copy(b.Backend.Read(dst)[dstRow*w:(dstRow+rows)*w], b.Backend.Read(src)[srcRow*w:(srcRow+rows)*w])
	return nil
}

func (b *v41ResidentKVTestBackend) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	total, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	if err := b.fault("gather"); err != nil {
		return compute.Tensor{}, err
	}
	out := make([]float32, 0, total*width)
	for _, span := range spans {
		if !b.allocations[span.Src.Buf()] {
			return compute.Tensor{}, errors.New("row-log gather from a dead buffer")
		}
		out = append(out, b.Backend.Read(span.Src)[span.Row*width:(span.Row+span.Rows)*width]...)
	}
	b.gathers++
	return b.own([]int{total, width}, out), nil
}

type v41ResidentKVRun struct {
	logits  [][]float32
	h2d     []int // attention upload bytes per decode step
	gathers int
}

// fak-test:runtime fast est=10ms lane=default
func TestV41StepWindowKVOwnedGather(t *testing.T) {
	for _, pos := range []int{0, 2, 3, 7, 8} {
		t.Run(fmt.Sprintf("pos_%d", pos), func(t *testing.T) {
			state := &V41AttentionState{windowSize: 3, nextWindowPos: pos, window: make([][]float32, 3)}
			for key := max(0, pos-3); key < pos; key++ {
				state.window[key%3] = []float32{float32(key), -float32(key)}
			}
			current := []float32{float32(pos), -float32(pos)}
			first := max(0, pos-3)
			if pos == 8 {
				first = pos - 1 // a configured window narrower than the retained ring
			}
			keys := make([]int32, 0, 4)
			for key := first; key <= pos; key++ {
				keys = append(keys, int32(key))
			}
			// The previous gather first obtained owned inspection rows.
			legacy := func() []float32 {
				rows := state.retainedTailRows()
				out := make([]float32, 0, len(keys)*len(current))
				for _, key := range keys {
					if int(key) == pos {
						out = append(out, current...)
					} else {
						out = append(out, rows[int(key)-max(0, pos-3)]...)
					}
				}
				return out
			}
			want := legacy()
			got, err := state.stepWindowKV(0, pos, keys, current)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("gather=%v err=%v want=%v", got, err, want)
			}
			for i := range got {
				got[i] = -999
			}
			if !reflect.DeepEqual(legacy(), want) {
				t.Fatal("attention payload aliases the live ring or current row")
			}
			var output []float32
			allocs := testing.AllocsPerRun(10, func() {
				output, err = state.stepWindowKV(0, pos, keys, current)
			})
			if err != nil || !reflect.DeepEqual(output, want) || allocs != 1 {
				t.Fatalf("owned gather allocs=%g err=%v, want one flat allocation", allocs, err)
			}
			if pos > 3 {
				if _, err := state.stepWindowKV(0, pos, []int32{int32(pos - 4)}, current); !errors.Is(err, ErrV41ForwardStage) {
					t.Fatalf("dropped key error=%v, want typed refusal", err)
				}
			}
		})
	}
}

func v41ResidentKVDecode(t *testing.T, build func(*testing.T) *Model, ctx int, disabled bool) v41ResidentKVRun {
	t.Helper()
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = disabled
	defer func() { v41ResidentKVDisabled = prior }()
	m := build(t)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	b := newV41ResidentKVTestBackend()
	s := v41DenseTestSession(t, m, b)
	v41SharedAttentionOnly(s)
	if s.v41State().sharedAttentionFunc() == nil {
		t.Fatal("shared attention was not bound")
	}
	var run v41ResidentKVRun
	logits := append([]float32(nil), s.Prefill(v41DecodeSubmitsPrompt(ctx, m.Cfg.VocabSize))...)
	run.logits = append(run.logits, logits)
	for i := 0; i < v41DecodeSubmitsSteps; i++ {
		before := b.uploadBytes
		logits = append([]float32(nil), s.Step(argmaxF32(logits))...)
		run.logits = append(run.logits, logits)
		run.h2d = append(run.h2d, b.uploadBytes-before)
	}
	run.gathers = b.gathers
	return run
}

// TestV41DecodeResidentH2D is the cpu-ref half of the #13821 witness: with the
// session KV mirror on, steady per-step attention upload volume no longer
// depends on context length, and logits equal the per-step upload path bit for
// bit. Each fail-closed refusal keeps its sentinel.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=10s lane=default
func TestV41DecodeResidentH2D(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("compressed_%t", compressed), func(t *testing.T) {
			build := func(t *testing.T) *Model { return v41DecodeSubmitsLayeredFixture(t, 1, compressed) }
			runs := map[int]v41ResidentKVRun{}
			for _, ctx := range []int{4, 64} {
				on, off := v41ResidentKVDecode(t, build, ctx, false), v41ResidentKVDecode(t, build, ctx, true)
				if on.gathers == 0 || off.gathers != 0 {
					t.Fatalf("ctx=%d mirror gathers on=%d off=%d", ctx, on.gathers, off.gathers)
				}
				if !reflect.DeepEqual(on.logits, off.logits) {
					t.Fatalf("ctx=%d logits differ between the mirror and the per-step upload path", ctx)
				}
				t.Logf("ctx=%d attention h2d bytes per step: mirror=%v upload=%v", ctx, on.h2d, off.h2d)
				runs[ctx] = on
			}
			// Step 0 seeds the mirror; later steps carry only the query and new rows.
			if a, b := runs[4].h2d[1:], runs[64].h2d[1:]; !reflect.DeepEqual(a, b) {
				t.Fatalf("steady mirror upload bytes depend on context: ctx4=%v ctx64=%v", a, b)
			}
		})
	}

	t.Run("sentinels", func(t *testing.T) {
		const dim = 4
		request := func(owner *V41AttentionState, pos, ringRows, windowRows, compressedRows int) v41SharedAttentionRequest {
			rows := windowRows + compressedRows
			idx := make([]int32, rows+1)
			for i := range idx {
				idx[i] = int32(i)
			}
			idx[rows] = -1
			kv := make([]float32, rows*dim)
			for i := range kv {
				kv[i] = float32(i%7) / 8
			}
			view := &v41ResidentKVView{owner: owner, pos: pos, ringRows: ringRows, firstKey: pos - windowRows + 1, windowRows: windowRows, compressedRows: compressedRows}
			return v41SharedAttentionRequest{
				mode: compute.V41SharedAttentionPlain, q: []float32{1, 0.5, -0.25, 0.125}, kv: kv, idx: idx,
				plain:    V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: dim, N: rows, TopK: rows + 1, Softmax: 0.5},
				resident: view,
			}
		}
		for _, tc := range []struct {
			name     string
			rowFault string
			calls    func(*V41AttentionState) []v41SharedAttentionRequest
			want     error
		}{
			{"capacity", "", func(o *V41AttentionState) []v41SharedAttentionRequest {
				return []v41SharedAttentionRequest{request(o, 5, 2, 4, 0)}
			}, errV41ResidentKVCapacity},
			{"desync", "", func(o *V41AttentionState) []v41SharedAttentionRequest {
				return []v41SharedAttentionRequest{request(o, 0, 4, 1, 2), request(o, 1, 4, 2, 1)}
			}, errV41ResidentKVDesync},
			{"backend", "gather", func(o *V41AttentionState) []v41SharedAttentionRequest {
				return []v41SharedAttentionRequest{request(o, 0, 4, 1, 0)}
			}, errV41ResidentKVBackend},
		} {
			t.Run(tc.name, func(t *testing.T) {
				b := newV41ResidentKVTestBackend()
				b.rowFault = tc.rowFault
				s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), b)
				attend := s.v41SharedAttentionFunc()
				if attend == nil {
					t.Fatal("shared attention was not bound")
				}
				var err error
				calls := tc.calls(&V41AttentionState{})
				for i, call := range calls {
					if _, err = attend(0, call); err != nil && i < len(calls)-1 {
						t.Fatalf("setup call %d: %v", i, err)
					}
				}
				var closed *BackendForwardOperationError
				if !errors.Is(err, tc.want) || !errors.As(err, &closed) {
					t.Fatalf("err=%v, want %v through the forward operation error", err, tc.want)
				}
			})
		}
	})
}
