package model

import (
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RoutedIDTestBackend serves the id stages by composing the per-pick
// recorder's operations, so a session through it witnesses the batched
// layer's id wiring: slot map, verdict order, sum order and the dispatch
// counter. The Vulkan kernels' bit parity is TestVulkanV41RoutedIDMatchesPerPick.
type v41RoutedIDTestBackend struct {
	*v41RoutedLayerTestBackend
	idCalls      map[string]int
	multiPending map[compute.Buffer][]compute.V41DeferredStatus
}

func newV41RoutedIDTestBackend() *v41RoutedIDTestBackend {
	return &v41RoutedIDTestBackend{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(),
		idCalls: map[string]int{}, multiPending: map[compute.Buffer][]compute.V41DeferredStatus{}}
}

var _ compute.V41RoutedIDBackend = (*v41RoutedIDTestBackend)(nil)

func (b *v41RoutedIDTestBackend) V41RoutedIDAvailable() bool                   { return true }
func (b *v41RoutedIDTestBackend) SupportsV41RoutedIDWeight(compute.Dtype) bool { return true }

func (b *v41RoutedIDTestBackend) V41RoutedIDs(slots []int, weights []float32, order []int) (compute.Tensor, error) {
	b.idCalls["ids"]++
	words, err := compute.V41RoutedIDWords(slots, weights, order)
	if err != nil {
		return compute.Tensor{}, err
	}
	bits := make([]float32, len(words))
	for i, w := range words {
		bits[i] = math.Float32frombits(w)
	}
	return b.Backend.Upload(compute.NewF32(compute.Default(), []int{len(bits)}, bits), compute.F32), nil
}

func (b *v41RoutedIDTestBackend) words(ids compute.Tensor) []uint32 {
	raw := b.Backend.Read(ids)
	out := make([]uint32, len(raw))
	for i, v := range raw {
		out[i] = math.Float32bits(v)
	}
	return out
}

func (b *v41RoutedIDTestBackend) rows(t compute.Tensor, picks int) []compute.Tensor {
	all := b.Backend.Read(t)
	n := len(all) / picks
	out := make([]compute.Tensor, picks)
	for p := range out {
		out[p] = b.Backend.Upload(compute.NewF32(compute.Default(), []int{n}, append([]float32(nil), all[p*n:(p+1)*n]...)), compute.F32)
	}
	return out
}

func (b *v41RoutedIDTestBackend) concat(parts []compute.Tensor) compute.Tensor {
	var all []float32
	for _, p := range parts {
		all = append(all, b.Backend.Read(p)...)
		b.Backend.Free(p)
	}
	return b.Backend.Upload(compute.NewF32(compute.Default(), []int{len(all)}, all), compute.F32)
}

func (b *v41RoutedIDTestBackend) V41MatVecID(weights []compute.Tensor, ids compute.Tensor, picks int, x compute.Tensor) (compute.Tensor, error) {
	b.idCalls["matvec"]++
	words := b.words(ids)
	in := weights[0].Shape[1]
	xs := []compute.Tensor{x}
	if x.Shape[0] == picks*in {
		xs = b.rows(x, picks)
	}
	out := make([]compute.Tensor, picks)
	for p := 0; p < picks; p++ {
		xp := xs[0]
		if len(xs) > 1 {
			xp = xs[p]
		}
		out[p] = b.MatMul(weights[words[p]], xp)
	}
	if len(xs) > 1 {
		for _, xp := range xs {
			b.Backend.Free(xp)
		}
	}
	return b.concat(out), nil
}

// record keeps the per-pick statuses under one multi-record status.
func (b *v41RoutedIDTestBackend) record(kind compute.V41DeferredStatusKind, count int, site uint32, parts []compute.V41DeferredStatus) compute.V41DeferredStatus {
	t := b.Backend.Upload(compute.NewF32(compute.Default(), []int{4 * len(parts)}, make([]float32, 4*len(parts))), compute.F32)
	b.multiPending[t.Buf()] = parts
	return compute.V41DeferredStatus{Kind: kind, Count: count, Site: site, Records: len(parts), Record: t}
}

func (b *v41RoutedIDTestBackend) V41RoutedSwiGLUIDDeferred(gate, up, ids compute.Tensor, picks int, limit float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.idCalls["swiglu"]++
	words := b.words(ids)
	gs, us := b.rows(gate, picks), b.rows(up, picks)
	out := make([]compute.Tensor, picks)
	parts := make([]compute.V41DeferredStatus, picks)
	for p := 0; p < picks; p++ {
		o, st, err := b.v41RoutedLayerTestBackend.V41RoutedSwiGLUDeferred(gs[p], us[p], limit, math.Float32frombits(words[picks+p]))
		if err != nil {
			return compute.Tensor{}, compute.V41DeferredStatus{}, err
		}
		out[p], parts[p] = o, st
		b.Backend.Free(gs[p])
		b.Backend.Free(us[p])
	}
	return b.concat(out), b.record(compute.V41DeferredRoutedSwiGLU, gate.Shape[0]/picks, 0, parts), nil
}

func (b *v41RoutedIDTestBackend) V41RoutedPublishSumDeferred(down, ids compute.Tensor, picks int, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.idCalls["publish_sum"]++
	words := b.words(ids)
	ds := b.rows(down, picks)
	published := make([]compute.Tensor, picks)
	parts := make([]compute.V41DeferredStatus, picks)
	for p := 0; p < picks; p++ {
		row, st, err := b.V41BF16PublishDeferred(ds[p], site)
		if err != nil {
			return compute.Tensor{}, compute.V41DeferredStatus{}, err
		}
		published[p], parts[p] = row, st
		b.Backend.Free(ds[p])
	}
	width := down.Shape[0] / picks
	acc := b.Backend.Upload(compute.NewF32(compute.Default(), []int{width}, make([]float32, width)), compute.F32)
	for j := 0; j < picks; j++ {
		b.AddInPlace(acc, published[words[2*picks+j]])
	}
	for _, row := range published {
		b.Backend.Free(row)
	}
	return acc, b.record(compute.V41DeferredBF16Publish, width, site, parts), nil
}

// V41ReadDeferred expands a multi-record status into its per-pick verdicts.
func (b *v41RoutedIDTestBackend) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	var flat []compute.V41DeferredStatus
	for _, st := range statuses {
		parts, ok := b.multiPending[st.Record.Buf()]
		if !ok {
			flat = append(flat, st)
			continue
		}
		delete(b.multiPending, st.Record.Buf())
		b.Backend.Free(st.Record)
		flat = append(flat, parts...)
	}
	return b.v41RoutedLayerTestBackend.V41ReadDeferred(flat, rows)
}

// The id arm of the batched routed layer is the per-pick arm bit for bit:
// identical decode logits, the same activations and publications, one fence
// per layer, five recorded routed dispatches per layer instead of six per
// pick, and the counter attributes every decode layer to the id arm.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=default
func TestV41RoutedLayerIDMatchesPerPick(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "1")
	const steps = 3
	for _, fx := range []struct {
		name  string
		build func(t *testing.T) *Model
	}{
		{"pinned_slate_plain", func(t *testing.T) *Model { return v41PinnedSlateFixture(t, false) }},
		{"pinned_slate_compressed", func(t *testing.T) *Model { return v41PinnedSlateFixture(t, true) }},
		{"mixed_quant_role", func(t *testing.T) *Model { return v41IncrementalExpertFixture(t, true, false) }},
	} {
		t.Run(fx.name, func(t *testing.T) {
			idModel, perPickModel := fx.build(t), fx.build(t)
			idb := newV41RoutedIDTestBackend()
			si, err := idModel.NewBackendSessionChecked(idb)
			if err != nil {
				t.Fatal(err)
			}
			defer si.Close()
			perPick := newV41RoutedLayerTestBackend()
			sp, err := perPickModel.NewBackendSessionChecked(perPick)
			if err != nil {
				t.Fatal(err)
			}
			defer sp.Close()
			prompt := []int{1, 2, 3}
			idLast := si.Prefill(prompt)
			perPickLast := sp.Prefill(prompt)
			if !reflect.DeepEqual(idLast, perPickLast) {
				t.Fatal("prefill logits differ between the id arm and the per-pick arm")
			}
			if len(idb.pending) != 0 || len(idb.multiPending) != 0 || len(perPick.pending) != 0 {
				t.Fatalf("prefill left deferred records unread: id=%d/%d per-pick=%d", len(idb.pending), len(idb.multiPending), len(perPick.pending))
			}
			idb.deferredReads, idb.calls, idb.publishes = 0, 0, 0
			idb.idCalls = map[string]int{}
			perPick.deferredReads, perPick.calls, perPick.publishes = 0, 0, 0
			decodeSteps := func(s *Session, last []float32) [][]float32 {
				out := make([][]float32, 0, steps+1)
				for i := 0; i < steps; i++ {
					out = append(out, append([]float32(nil), last...))
					last = s.Step(argmaxF32(last))
				}
				return append(out, append([]float32(nil), last...))
			}
			got := decodeSteps(si, idLast)
			want := decodeSteps(sp, perPickLast)
			for i := range want {
				if !reflect.DeepEqual(got[i], want[i]) {
					t.Fatalf("window %d logits differ between the id arm and the per-pick arm", i)
				}
			}
			layers, k := idModel.Cfg.NumLayers, idModel.Cfg.NumExpertsPerTok
			if idb.deferredReads != perPick.deferredReads || idb.calls != perPick.calls || idb.publishes != perPick.publishes {
				t.Fatalf("id arm reads=%d activations=%d publishes=%d, per-pick %d/%d/%d",
					idb.deferredReads, idb.calls, idb.publishes, perPick.deferredReads, perPick.calls, perPick.publishes)
			}
			if idb.idCalls["ids"] != steps*layers || idb.idCalls["matvec"] != 3*idb.idCalls["ids"] || idb.idCalls["swiglu"] != idb.idCalls["ids"] || idb.idCalls["publish_sum"] != idb.idCalls["ids"] {
				t.Fatalf("id stages per layer: %v, want one ids upload, three GEMVs, one SwiGLU and one publish+sum", idb.idCalls)
			}
			if len(idb.pending) != 0 || len(idb.multiPending) != 0 {
				t.Fatalf("%d/%d deferred status records were never read", len(idb.pending), len(idb.multiPending))
			}
			idDecode, perPickDecode := idModel.V41ExpertFaultAttribution().Decode, perPickModel.V41ExpertFaultAttribution().Decode
			if idDecode.RoutedIDLayers != steps*layers || idDecode.RoutedLayers != steps*layers {
				t.Fatalf("decode id layers=%d routed layers=%d, want %d", idDecode.RoutedIDLayers, idDecode.RoutedLayers, steps*layers)
			}
			if idDecode.RoutedLayerDispatches != steps*layers*v41RoutedIDLayerDispatches {
				t.Fatalf("id arm recorded %d routed dispatches, want %d", idDecode.RoutedLayerDispatches, steps*layers*v41RoutedIDLayerDispatches)
			}
			if perPickDecode.RoutedIDLayers != 0 || perPickDecode.RoutedLayerDispatches != steps*layers*(v41RoutedPerPickDispatches+1)*k {
				t.Fatalf("per-pick arm recorded %d routed dispatches over %d id layers, want %d over 0",
					perPickDecode.RoutedLayerDispatches, perPickDecode.RoutedIDLayers, steps*layers*(v41RoutedPerPickDispatches+1)*k)
			}
		})
	}
}

// The id arm is opt-in: unset, empty, "0" or any value but "1" keeps the batched
// layer on its per-pick dispatches even on an id-capable backend.
// fak-test:runtime slow est=20s lane=default
func TestV41RoutedIDDefaultsOffUntilOptedIn(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	for _, c := range []struct {
		name  string
		value string
		set   bool
		id    bool
	}{
		{name: "unset"},
		{name: "empty", value: "", set: true},
		{name: "zero", value: "0", set: true},
		{name: "other", value: "true", set: true},
		{name: "opted_in", value: "1", set: true, id: true},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv(v41RoutedIDEnv, "")
			if !c.set {
				os.Unsetenv(v41RoutedIDEnv)
			} else {
				t.Setenv(v41RoutedIDEnv, c.value)
			}
			m := v41IncrementalExpertFixture(t, true, false)
			b := newV41RoutedIDTestBackend()
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			picks, x := v41RoutedLayerFaultPicks(m)
			routed := s.v41RoutedLayerFunc()
			if routed == nil {
				t.Fatal("explicitly enabled batched routed layer did not bind")
			}
			sum, outcome, err := routed(0, picks, x)
			if err != nil || outcome != v41RoutedExpertHandled || len(sum) != m.Cfg.HiddenSize {
				t.Fatalf("result width=%d outcome=%v err=%v", len(sum), outcome, err)
			}
			if b.deferredReads != 1 || b.calls != len(picks) || b.publishes != len(picks) {
				t.Fatalf("reads=%d activations=%d publishes=%d, want 1/%d/%d", b.deferredReads, b.calls, b.publishes, len(picks), len(picks))
			}
			wantIDCalls := map[string]int{}
			if c.id {
				wantIDCalls = map[string]int{"ids": 1, "matvec": 3, "swiglu": 1, "publish_sum": 1}
			}
			if !reflect.DeepEqual(b.idCalls, wantIDCalls) {
				t.Fatalf("id stages=%v, want %v", b.idCalls, wantIDCalls)
			}
		})
	}
}

// A fault in pick k reaches the id arm as the same typed error the per-pick
// arm raises: the lowest faulting pick wins, SwiGLU before publication.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41RoutedLayerIDFaultAttribution(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "1")
	cases := []struct {
		name       string
		activation map[int]compute.V41RoutedSwiGLUStage
		publishAt  int
	}{
		{name: "activation at pick 3", activation: map[int]compute.V41RoutedSwiGLUStage{3: compute.V41RoutedSwiGLUStageActivation}},
		{name: "lowest slot wins", activation: map[int]compute.V41RoutedSwiGLUStage{4: compute.V41RoutedSwiGLUStageActivation, 1: compute.V41RoutedSwiGLUStageDownOperand}},
		{name: "publish at pick 2", publishAt: 3},
		{name: "activation before a later publish", activation: map[int]compute.V41RoutedSwiGLUStage{1: compute.V41RoutedSwiGLUStageActivation}, publishAt: 5},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			run := func(id bool) (error, bool) {
				m := v41IncrementalExpertFixture(t, true, false)
				picks, x := v41RoutedLayerFaultPicks(m)
				arm := func(b *v41RoutedLayerTestBackend) {
					b.activationFaults = map[float32]*compute.V41RoutedSwiGLUFault{}
					for slot, stage := range c.activation {
						b.activationFaults[picks[slot].weight] = &compute.V41RoutedSwiGLUFault{Stage: stage, Index: 2}
					}
					if c.publishAt > 0 {
						b.publishFaultAt = map[int]bool{c.publishAt: true}
					}
				}
				var be compute.Backend
				if id {
					b := newV41RoutedIDTestBackend()
					arm(b.v41RoutedLayerTestBackend)
					be = b
				} else {
					b := newV41RoutedLayerTestBackend()
					arm(b)
					be = b
				}
				s, err := m.NewBackendSessionChecked(be)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				routed := s.v41RoutedLayerFunc()
				if routed == nil {
					t.Fatal("explicitly enabled batched routed layer did not bind")
				}
				sum, outcome, err := routed(0, picks, x)
				if sum != nil || outcome != v41RoutedExpertError {
					t.Fatalf("id=%t faulting layer published=%v outcome=%v", id, sum != nil, outcome)
				}
				if id {
					calls := be.(*v41RoutedIDTestBackend).idCalls
					want := map[string]int{"ids": 1, "matvec": 3, "swiglu": 1, "publish_sum": 1}
					if !reflect.DeepEqual(calls, want) {
						t.Fatalf("faulting id stages=%v, want %v", calls, want)
					}
				}
				return err, s.BackendSessionClosed()
			}
			idErr, idClosed := run(true)
			perPickErr, perPickClosed := run(false)
			var idOp, perPickOp *V41ExpertOperationError
			if !errors.As(idErr, &idOp) || !errors.As(perPickErr, &perPickOp) || idOp.Stage != perPickOp.Stage || idOp.Layer != perPickOp.Layer ||
				!errors.Is(idErr, errV41ExpertResult) || idClosed || perPickClosed {
				t.Fatalf("id arm err=%v closed=%t, per-pick err=%v closed=%t", idErr, idClosed, perPickErr, perPickClosed)
			}
		})
	}
}
