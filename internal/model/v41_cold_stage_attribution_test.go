package model

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41StageCounterBackend struct {
	compute.Backend
	name     string
	counters compute.BackendCounterSnapshot
	submits  uint64
	work     compute.DeviceWork
	err      error
}

func (b *v41StageCounterBackend) Name() string { return b.name }
func (b *v41StageCounterBackend) BackendCounters() (compute.BackendCounterSnapshot, error) {
	return b.counters, b.err
}
func (b *v41StageCounterBackend) DeviceSubmits() uint64 { return b.submits }

type v41StageGatedBackend struct{ *v41StageCounterBackend }

func (v41StageGatedBackend) DispatchCountsLive() bool { return false }

func v41StageCounterBackendFor(t *testing.T, name string) *v41StageCounterBackend {
	t.Helper()
	be := &v41StageCounterBackend{name: name}
	t.Cleanup(compute.RegisterDeviceWorkReader(name, func() compute.DeviceWork { return be.work }))
	return be
}

// TestV41ColdStageRecorderPartition pins the closed partition: every counter
// movement and host nanosecond between two reads lands in exactly one row.
// fak-test:runtime fast est=20ms lane=default
func TestV41ColdStageRecorderPartition(t *testing.T) {
	be := v41StageCounterBackendFor(t, "v41-stage-counter")
	now := int64(1000)
	s := &Session{Backend: be}
	s.EnableV41ColdStageAttribution()
	r := s.v41ColdStages
	r.now = func() int64 { return now }
	r.lastAt = now
	work := func(submits, readbacks, ns uint64) {
		be.counters.DispatchSubmits += submits
		be.submits += submits
		be.counters.D2HCount += readbacks
		be.counters.D2HBytes += 4 * readbacks
		be.work.Dispatches += 2 * submits
		be.work.SubmitWaitNS += ns / 2
		now += int64(ns)
	}
	row := func(entries, submits, readbacks, ns uint64) V41ColdStageCounters {
		return V41ColdStageCounters{
			Entries: entries, HostWallNanos: int64(ns),
			Counters:      compute.BackendCounterSnapshot{DispatchSubmits: submits, D2HCount: readbacks, D2HBytes: 4 * readbacks},
			DeviceSubmits: submits, DeviceDispatches: 2 * submits, SubmitWaitNanos: ns / 2,
		}
	}

	work(1, 0, 6) // before any region: unassigned
	r.enter(V41ColdStageQKV)
	work(3, 1, 70)
	r.enter(V41ColdStageRouted)
	work(5, 2, 110)
	r.enter(V41ColdStageRouted)
	work(7, 0, 130)
	r.leave()
	work(11, 3, 170) // between forwards: unassigned
	got, ok := s.V41ColdStageAttribution()
	if !ok || !got.CountersObserved || !got.DispatchCountsLive || !got.DeviceSubmitsObserved || !got.DeviceWorkObserved || got.SubmitWaitComplete {
		t.Fatalf("attribution observability: ok=%t %+v", ok, got)
	}
	want := map[V41ColdStage]V41ColdStageCounters{
		V41ColdStageQKV:    row(1, 3, 1, 70),
		V41ColdStageRouted: row(2, 12, 2, 240),
	}
	for stage := V41ColdStage(0); stage < V41ColdStageCount; stage++ {
		if got.Stages[stage] != want[stage] {
			t.Errorf("%s row=%+v want %+v", stage, got.Stages[stage], want[stage])
		}
	}
	if wantU := row(0, 12, 3, 176); got.Unassigned != wantU {
		t.Errorf("unassigned=%+v want %+v", got.Unassigned, wantU)
	}
	if total := got.Total(); total.Counters != be.counters || total.DeviceSubmits != be.submits || total.DeviceDispatches != be.work.Dispatches ||
		total.SubmitWaitNanos != be.work.SubmitWaitNS || total.HostWallNanos != now-1000 {
		t.Errorf("rows do not close on the backend aggregate: total=%+v aggregate=%+v work=%+v wall=%d", total, be.counters, be.work, now-1000)
	}

	before := got
	r.enter(V41ColdStageHead)
	work(2, 1, 10)
	r.leave()
	after, _ := s.V41ColdStageAttribution()
	window, err := after.Sub(before)
	if err != nil {
		t.Fatal(err)
	}
	if head := window.Stages[V41ColdStageHead]; head != row(1, 2, 1, 10) {
		t.Fatalf("window head=%+v", head)
	}
	if w := window.Total(); w != row(1, 2, 1, 10) {
		t.Fatalf("window total=%+v", w)
	}
	if _, err := before.Sub(after); !errors.Is(err, ErrV41ColdStageWindow) {
		t.Fatalf("reversed window err=%v", err)
	}

	// A rebound backend fails closed: no row may keep a partial total.
	s.Backend = &v41StageCounterBackend{name: "other"}
	r.enter(V41ColdStageQKV)
	work(1, 1, 4)
	r.leave()
	dropped, _ := s.V41ColdStageAttribution()
	if dropped.CountersObserved || dropped.DeviceSubmitsObserved || dropped.DeviceWorkObserved || dropped.Total().Counters != (compute.BackendCounterSnapshot{}) ||
		dropped.Total().DeviceSubmits != 0 || dropped.Total().DeviceDispatches != 0 || dropped.Total().SubmitWaitNanos != 0 {
		t.Fatalf("rebound backend kept counters: %+v", dropped)
	}
	if dropped.Stages[V41ColdStageQKV].Entries != 2 || dropped.Total().HostWallNanos != now-1000 {
		t.Fatalf("host wall and entries must survive a counter drop: %+v", dropped)
	}
	if _, err := dropped.Sub(after); !errors.Is(err, ErrV41ColdStageWindow) {
		t.Fatalf("observed/unobserved window err=%v", err)
	}

	for name, fail := range map[string]func(*v41StageCounterBackend){
		"read_error":    func(b *v41StageCounterBackend) { b.err = compute.ErrBackendCounterWrap },
		"counter_wrap":  func(b *v41StageCounterBackend) { b.counters.DispatchSubmits-- },
		"wait_wrap":     func(b *v41StageCounterBackend) { b.work.SubmitWaitNS-- },
		"dispatch_wrap": func(b *v41StageCounterBackend) { b.work.Dispatches-- },
		"submit_wrap":   func(b *v41StageCounterBackend) { b.submits-- },
	} {
		b := v41StageCounterBackendFor(t, "v41-stage-"+name)
		b.counters.DispatchSubmits, b.submits, b.work.Dispatches, b.work.SubmitWaitNS = 9, 9, 9, 9
		fs := &Session{Backend: b}
		fs.EnableV41ColdStageAttribution()
		fs.v41ColdStages.enter(V41ColdStageAttention)
		fail(b)
		fs.v41ColdStages.leave()
		got, _ := fs.V41ColdStageAttribution()
		countersKept := name == "wait_wrap" || name == "dispatch_wrap" || name == "submit_wrap"
		submitsKept := name != "read_error" && name != "submit_wrap"
		workKept := name == "counter_wrap" || name == "submit_wrap"
		if got.CountersObserved != countersKept || got.DeviceSubmitsObserved != submitsKept || got.DeviceWorkObserved != workKept {
			t.Fatalf("%s: observability after a failed read: %+v", name, got)
		}
		if total := got.Total(); total.Counters != (compute.BackendCounterSnapshot{}) || total.DeviceSubmits != 0 || total.DeviceDispatches != 0 || total.SubmitWaitNanos != 0 {
			t.Fatalf("%s: partial totals kept: %+v", name, got)
		}
	}

	gated := v41StageGatedBackend{v41StageCounterBackendFor(t, "v41-stage-gated")}
	gs := &Session{Backend: gated}
	gs.EnableV41ColdStageAttribution()
	gated.submits = 5
	gs.v41ColdStages.enter(V41ColdStageRouted)
	gated.submits += 3
	gs.v41ColdStages.leave()
	if got, _ := gs.V41ColdStageAttribution(); !got.CountersObserved || got.DispatchCountsLive || !got.DeviceSubmitsObserved ||
		got.Stages[V41ColdStageRouted].DeviceSubmits != 3 || got.Total().DeviceSubmits != 8 {
		t.Fatalf("profile-off submits must stay attributed while dispatch counts are flagged: %+v", got)
	}

	if got, ok := (&Session{}).V41ColdStageAttribution(); ok || got != (V41ColdStageAttribution{}) {
		t.Fatalf("disabled session published attribution: %+v", got)
	}
	host := &Session{}
	host.EnableV41ColdStageAttribution()
	if got, ok := host.V41ColdStageAttribution(); !ok || got.CountersObserved || got.DeviceSubmitsObserved || got.DeviceWorkObserved {
		t.Fatalf("host session must report unobserved counters: ok=%t %+v", ok, got)
	}
	for stage := V41ColdStage(0); stage <= V41ColdStageCount; stage++ {
		if name := stage.String(); name == "" || strings.ContainsAny(name, " -") {
			t.Fatalf("stage %d name %q", stage, name)
		}
	}
}

// TestV41ColdStageAllocs: the disabled brackets are nil-receiver no-ops and an
// enabled recorder folds into fixed rows, so neither allocates per region.
// fak-test:runtime fast est=500ms lane=default
func TestV41ColdStageAllocs(t *testing.T) {
	var disabled *v41ColdStageRecorder
	st := &v41ForwardState{callbackOwner: &Session{}}
	if st.coldStageRecorder() != nil || (*v41ForwardState)(nil).coldStageRecorder() != nil {
		t.Fatal("a session without attribution resolved a recorder")
	}
	cycle := func(r *v41ColdStageRecorder) func() {
		return func() {
			for stage := V41ColdStage(0); stage < V41ColdStageCount; stage++ {
				r.enter(stage)
			}
			r.leave()
		}
	}
	if allocs := testing.AllocsPerRun(100, cycle(disabled)); allocs != 0 {
		t.Fatalf("disabled brackets allocate %.1f per layer", allocs)
	}
	be := v41StageCounterBackendFor(t, "v41-stage-allocs")
	s := &Session{Backend: be}
	s.EnableV41ColdStageAttribution()
	enabled := cycle(s.v41ColdStages)
	if allocs := testing.AllocsPerRun(100, func() {
		be.counters.DispatchSubmits++
		enabled()
		_, _ = s.V41ColdStageAttribution()
	}); allocs != 0 {
		t.Fatalf("enabled brackets allocate %.1f per layer", allocs)
	}

	// Whole cold prefill: installing a recorder adds no allocation anywhere.
	m := v41DecodeSubmitsLayeredFixture(t, 3, true)
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	prompt := v41DecodeSubmitsPrompt(4, m.Cfg.VocabSize)
	var counters compute.BackendCounterRead
	rec := newV41ColdStageRecorder(func() (compute.BackendCounterRead, bool) { return counters, true }, v41ColdStageNow)
	prefill := func(r *v41ColdStageRecorder) func() {
		return func() {
			s := &Session{M: m}
			s.v41ColdStages = r
			s.Prefill(prompt)
		}
	}
	off := testing.AllocsPerRun(5, prefill(nil))
	on := testing.AllocsPerRun(5, prefill(rec))
	if on != off {
		t.Fatalf("cold prefill allocations: recorder=%.1f disabled=%.1f", on, off)
	}
	if got := rec.snapshot(); got.Stages[V41ColdStageQKV].Entries != 6*3 {
		t.Fatalf("recorder missed prefill layers: qkv entries=%d", got.Stages[V41ColdStageQKV].Entries)
	}
}

// v41StageFakeDevice stands in for a device under every cold-forward seam
// that can decline or reproduce the host result. Each call charges one
// stage-tagged unit of work, so a bracket entered on the wrong side of a seam
// moves that unit into the wrong row.
type v41StageFakeDevice struct {
	counters compute.BackendCounterSnapshot
	submits  uint64
	ns       int64
	cold     bool
	expected [V41ColdStageCount + 1]V41ColdStageCounters
	calls    map[string]int
	unknown  []string
}

func v41StageForLeaf(leaf string) (V41ColdStage, bool) {
	switch {
	case leaf == "attn.wq_a.weight" || leaf == "attn.wq_b.weight" || leaf == "attn.wkv.weight":
		return V41ColdStageQKV, true
	case strings.HasPrefix(leaf, "attn.compressor.") || leaf == "attn.indexer.wk.weight" || leaf == "indexer.wk.weight":
		return V41ColdStageCompressor, true
	case strings.HasPrefix(leaf, "attn.indexer.") || strings.HasPrefix(leaf, "indexer."):
		return V41ColdStageIndexer, true
	case leaf == "ffn.gate.weight":
		return V41ColdStageRouter, true
	case strings.HasPrefix(leaf, "ffn.shared_experts."):
		return V41ColdStageSharedPost, true
	case leaf == "lm_head.weight":
		return V41ColdStageHead, true
	}
	return 0, false
}

// charge simulates one device operation: one submit, its dispatches, one
// upload and one readback, plus 1000 ns of wall. A cold call is expected in
// stage, an incremental one in Unassigned.
func (d *v41StageFakeDevice) charge(seam string, stage V41ColdStage, in, out int) {
	work := compute.BackendCounterSnapshot{
		ComputeDispatches: 2, OtherDispatches: 2, DispatchSubmits: 1,
		H2DCount: 1, H2DBytes: uint64(4 * in), D2HCount: 1, D2HBytes: uint64(4 * out),
	}
	d.counters = compute.AddBackendCounters(d.counters, work)
	d.submits++
	d.ns += 1000
	d.calls[seam]++
	slot := int(V41ColdStageCount)
	if d.cold {
		slot = int(stage)
	}
	row := &d.expected[slot]
	row.Counters = compute.AddBackendCounters(row.Counters, work)
	row.DeviceSubmits++
	row.HostWallNanos += 1000
}

func (d *v41StageFakeDevice) read() (compute.BackendCounterRead, bool) {
	return compute.BackendCounterRead{Counters: d.counters, DispatchCountsLive: true, Submits: d.submits, SubmitsObserved: true}, true
}

// install binds the declining or host-exact fakes on m's session state; the
// host computes every value, so logits stay bit-identical to a seamless run.
func (d *v41StageFakeDevice) install(m *Model, st *v41ForwardState) {
	d.calls = map[string]int{}
	st.gatherOps = &v41DeviceGatherOps{embedGather: func(ids []int) ([]float32, error) {
		d.charge("embedding_gather", V41ColdStageEmbedding, len(ids), len(ids)*m.Cfg.HiddenSize)
		return m.v41EmbeddingPanelOn(ids, nil)
	}}
	st.engramProjection = func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
		d.charge("engram_projection", V41ColdStageEngram, in, out)
		return nil, v41ProjectionDeclined, nil
	}
	st.mhcProjection = func(layer int, input []float32, H int, eps float32, full, transposed bool) ([]float32, v41DenseProjectionOutcome, error) {
		d.charge("mhc_projection", V41ColdStageMHCPre, len(input), v41MHCMixWidth)
		return nil, v41ProjectionDeclined, nil
	}
	st.mhcFFNProjection = func(layer int, input []float32, H int, eps float32, full, transposed bool) ([]float32, v41DenseProjectionOutcome, error) {
		d.charge("mhc_ffn_projection", V41ColdStageFFNIn, len(input), v41MHCMixWidth)
		return nil, v41ProjectionDeclined, nil
	}
	st.setDenseProjection(func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		stage, ok := v41StageForLeaf(leaf)
		if !ok {
			d.unknown = append(d.unknown, leaf)
			return nil, v41ProjectionDeclined, nil
		}
		d.charge("dense:"+stage.String(), stage, in*rows, out*rows)
		return nil, v41ProjectionDeclined, nil
	})
	st.setIndexerScore(func(layer int, q, keys, weights []float32, heads, headDim, rows int) ([]float32, error) {
		d.charge("indexer_score", V41ColdStageIndexer, len(q)+len(keys), rows)
		return V41IndexerScore(q, keys, weights, heads, headDim, rows)
	})
	st.setSharedAttention(func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
		d.charge("shared_attention", V41ColdStageAttention, len(r.q), len(r.q))
		if r.mode == compute.V41SharedAttentionCompressed {
			return V41AttentionCompressedForward(r.q, r.values, r.compressed)
		}
		return V41SparseAttentionSink(r.q, v41TestAttentionFlatKV(r), r.sink, r.idx, r.plain)
	})
	st.groupedOutput = v41GroupedOutputFunc(func(layer int, o []float32, heads, headDim, groups, rank, dim int) ([]float32, v41DenseProjectionOutcome, error) {
		d.charge("grouped_output", V41ColdStageOutProj, len(o), dim)
		return nil, v41ProjectionDeclined, nil
	})
	st.routedExpert = func(layer int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
		d.charge("routed_expert", V41ColdStageRouted, len(x), len(x))
		return nil, v41RoutedExpertDeclined, nil
	}
	st.expertGateUp = func(layer int, stem string, x []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		d.charge("expert_gate_up", V41ColdStageRouted, len(x), len(x))
		return nil, v41GateUpDeclined, nil
	}
	st.expertDown = func(layer int, stem string, fused []float32) ([]float32, v41ExpertDownOutcome, error) {
		d.charge("expert_down", V41ColdStageRouted, len(fused), len(fused))
		return nil, v41DownDeclined, nil
	}
}

// v41StageWitnessSeams are the seams the CPU witness must see a cold prefill
// offer across its fixtures. expert_down is absent: it runs only after a
// handled gate/up, which every fake declines.
var v41StageWitnessSeams = []string{
	"embedding_gather", "engram_projection", "mhc_projection", "dense:qkv", "dense:compressor", "indexer_score",
	"dense:indexer", "shared_attention", "grouped_output", "mhc_ffn_projection", "dense:router", "routed_expert",
	"expert_gate_up", "dense:shared_post", "dense:head",
}

func v41StageWitnessFixtures() []v41DecodeSubmitsFixture {
	return append(v41DecodeSubmitsFixtures(), v41DecodeSubmitsFixture{"indexer_sources2_compressed_true", func(t *testing.T) *Model {
		m := v41IndexerTestFixture(t)
		m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
		return m
	}})
}

// TestV41ColdStageAttributionCPURef is the CPU-reference witness: with a
// declining or host-exact fake device under every cold-forward seam, each
// cold prefill stage row equals the stage-tagged work its seams charged,
// exactly, so a bracket on the wrong side of a seam fails. Rows close on the
// fake's aggregate, incremental decode lands in Unassigned, and the recorder
// leaves every logit bit-identical to a seamless host session.
// fak-test:justify why=contract when=changed:internal/model/v41_forward*.go,internal/model/v41_cold_stage_attribution*.go
// fak-test:runtime medium est=5s lane=default
func TestV41ColdStageAttributionCPURef(t *testing.T) {
	seen := map[string]int{}
	for _, fx := range v41StageWitnessFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			plainModel, attributedModel := fx.build(t), fx.build(t)
			defer func() {
				for _, mm := range []*Model{plainModel, attributedModel} {
					if err := mm.CloseWeights(); err != nil {
						t.Error(err)
					}
				}
			}()
			layers := attributedModel.Cfg.NumLayers
			prompt := v41DecodeSubmitsPrompt(4, attributedModel.Cfg.VocabSize)

			plain := &Session{M: plainModel}
			defer plain.Close()
			s := &Session{M: attributedModel}
			defer s.Close()
			device := &v41StageFakeDevice{cold: true}
			device.install(attributedModel, s.v41State())
			s.v41ColdStages = newV41ColdStageRecorder(device.read, func() int64 { return device.ns })

			before, _ := s.V41ColdStageAttribution()
			aggregateBefore, submitsBefore, nsBefore := device.counters, device.submits, device.ns
			got := append([]float32(nil), s.Prefill(prompt)...)
			want := plain.Prefill(prompt)
			after, _ := s.V41ColdStageAttribution()
			window, err := after.Sub(before)
			if err != nil {
				t.Fatal(err)
			}
			v41ColdStageRequireBitIdentical(t, "prefill", got, want)
			if len(device.unknown) != 0 {
				t.Fatalf("dense leaves without a pinned stage: %v", device.unknown)
			}
			aggregate, err := compute.SubBackendCounters(device.counters, aggregateBefore)
			if err != nil {
				t.Fatal(err)
			}
			total := window.Total()
			if !window.CountersObserved || !window.DeviceSubmitsObserved || total.Counters != aggregate ||
				total.DeviceSubmits != device.submits-submitsBefore || total.HostWallNanos != device.ns-nsBefore {
				t.Fatalf("prefill rows do not close on the aggregate:\n rows=%+v\n aggregate=%+v submits=%d wall=%d", total, aggregate, device.submits-submitsBefore, device.ns-nsBefore)
			}
			for stage := V41ColdStage(0); stage <= V41ColdStageCount; stage++ {
				row, label := window.Unassigned, "unassigned"
				if stage < V41ColdStageCount {
					row, label = window.Stages[stage], stage.String()
				}
				want := device.expected[stage]
				if row.Counters != want.Counters || row.DeviceSubmits != want.DeviceSubmits || row.HostWallNanos != want.HostWallNanos {
					t.Errorf("%s row=%+v want counters=%+v submits=%d wall=%d", label, row, want.Counters, want.DeviceSubmits, want.HostWallNanos)
				}
			}
			perLayer := []V41ColdStage{V41ColdStageLayerSetup, V41ColdStageEngram, V41ColdStageMHCPre, V41ColdStageQKV, V41ColdStageCompressor,
				V41ColdStageIndexer, V41ColdStagePublish, V41ColdStageAttention, V41ColdStageOutProj, V41ColdStageFFNIn,
				V41ColdStageRouter, V41ColdStageRouted, V41ColdStageSharedPost, V41ColdStageLayerHidden}
			for _, stage := range perLayer {
				if window.Stages[stage].Entries != uint64(layers) {
					t.Errorf("%s entries=%d want one per layer (%d)", stage, window.Stages[stage].Entries, layers)
				}
			}
			for _, stage := range []V41ColdStage{V41ColdStageEmbedding, V41ColdStageHead} {
				if window.Stages[stage].Entries != 1 {
					t.Errorf("%s entries=%d want 1", stage, window.Stages[stage].Entries)
				}
			}
			for seam, n := range device.calls {
				seen[seam] += n
			}
			v41ColdStageLog(t, fx.name+" phase=prefill", window)

			device.cold = false
			next := argmaxF32(got)
			for step := 0; step < 2; step++ {
				before, _ = s.V41ColdStageAttribution()
				aggregateBefore, submitsBefore = device.counters, device.submits
				got = append([]float32(nil), s.Step(next)...)
				want = plain.Step(next)
				after, _ = s.V41ColdStageAttribution()
				v41ColdStageRequireBitIdentical(t, "decode", got, want)
				if window, err = after.Sub(before); err != nil {
					t.Fatal(err)
				}
				aggregate, _ = compute.SubBackendCounters(device.counters, aggregateBefore)
				if window.Total().Counters != aggregate || window.Total().DeviceSubmits != device.submits-submitsBefore {
					t.Fatalf("decode step %d rows do not close: rows=%+v aggregate=%+v", step, window.Total(), aggregate)
				}
				if window.Stages[V41ColdStageLayerSetup].Entries == 0 && window.Unassigned.Counters != aggregate {
					t.Fatalf("incremental step %d work left the unassigned row: %+v", step, window)
				}
				v41ColdStageLog(t, fx.name+" phase=decode", window)
				next = argmaxF32(got)
			}
		})
	}
	for _, seam := range v41StageWitnessSeams {
		if seen[seam] == 0 {
			t.Errorf("no fixture's cold prefill offered seam %s, so its bracket is unpinned", seam)
		}
	}
}

func v41ColdStageRequireBitIdentical(t *testing.T, phase string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("%s logits len=%d want %d", phase, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s logit %d = %v, want bit-identical %v", phase, i, got[i], want[i])
		}
	}
}

func v41ColdStageLog(t *testing.T, label string, a V41ColdStageAttribution) {
	t.Helper()
	for stage := V41ColdStage(0); stage <= V41ColdStageCount; stage++ {
		row := a.Unassigned
		if stage < V41ColdStageCount {
			row = a.Stages[stage]
		}
		if row.HostWallNanos == 0 && row.Counters == (compute.BackendCounterSnapshot{}) && row.DeviceSubmits == 0 && row.DeviceDispatches == 0 {
			continue
		}
		c := row.Counters
		t.Logf("v41_cold_stage %s stage=%s entries=%d host_wall_ns=%d submits=%d dispatches=%d d2h_count=%d d2h_bytes=%d h2d_count=%d h2d_bytes=%d device_submits=%d device_dispatches=%d submit_wait_ns=%d",
			label, stage, row.Entries, row.HostWallNanos, c.DispatchSubmits, c.ComputeDispatches, c.D2HCount, c.D2HBytes, c.H2DCount, c.H2DBytes, row.DeviceSubmits, row.DeviceDispatches, row.SubmitWaitNanos)
	}
}
