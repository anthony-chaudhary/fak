package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RoutedLayerTestBackend adds a cpu-ref deferred-status recorder to the
// fused routed seam's recorder. Statuses are host verdicts, so a session
// through it is a wiring and ordering witness only (#13820).
type v41RoutedLayerTestBackend struct {
	*v41RoutedSwiGLUTestBackend
	deferredReads    int
	publishes        int
	pending          map[compute.Buffer]error
	activationFaults map[float32]*compute.V41RoutedSwiGLUFault
	publishFaultAt   map[int]bool
	verdictFailure   map[float32]error
	readFailure      error
}

func newV41RoutedLayerTestBackend() *v41RoutedLayerTestBackend {
	return &v41RoutedLayerTestBackend{v41RoutedSwiGLUTestBackend: newV41RoutedSwiGLUTestBackend(), pending: map[compute.Buffer]error{}}
}

func (b *v41RoutedLayerTestBackend) SupportsV41DeferredStatus(count int) bool { return count > 0 }

// V41RoutedSwiGLU keeps the per-pick seam's weight-keyed faults equal to the
// deferred ones, so both routes see the same injected verdicts.
func (b *v41RoutedLayerTestBackend) V41RoutedSwiGLU(gate, up compute.Tensor, limit, weight float32) (compute.Tensor, error) {
	if fault := b.activationFaults[weight]; fault != nil {
		b.calls++
		return compute.Tensor{}, fault
	}
	return b.v41RoutedSwiGLUTestBackend.V41RoutedSwiGLU(gate, up, limit, weight)
}

func (b *v41RoutedLayerTestBackend) record(verdict error) compute.V41DeferredStatus {
	t := b.Backend.Upload(compute.NewF32(compute.Default(), []int{4}, make([]float32, 4)), compute.F32)
	b.pending[t.Buf()] = verdict
	return compute.V41DeferredStatus{Record: t}
}

func (b *v41RoutedLayerTestBackend) V41RoutedSwiGLUDeferred(gate, up compute.Tensor, limit, weight float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if b.failure != nil {
		b.calls++
		return compute.Tensor{}, compute.V41DeferredStatus{}, b.failure
	}
	out, err := b.v41RoutedSwiGLUTestBackend.V41RoutedSwiGLU(gate, up, limit, weight)
	if err != nil {
		return compute.Tensor{}, compute.V41DeferredStatus{}, err
	}
	var verdict error
	if fault := b.activationFaults[weight]; fault != nil {
		verdict = fault
	} else if failure := b.verdictFailure[weight]; failure != nil {
		verdict = failure
	}
	st := b.record(verdict)
	st.Kind, st.Count = compute.V41DeferredRoutedSwiGLU, gate.Shape[0]
	return out, st, nil
}

func (b *v41RoutedLayerTestBackend) V41BF16PublishDeferred(input compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.publishes++
	values := append([]float32(nil), b.Backend.Read(input)...)
	var verdict error
	if index, bad := v41BF16PublishFault(values); bad || b.publishFaultAt[b.publishes] {
		verdict = &compute.BackendError{Backend: "test", Site: "V41BF16Publish", Err: compute.ErrVulkanExecutionFailed,
			Recovered: &compute.V41BF16CheckedArithmeticError{Site: site, Index: index}}
	}
	for i, v := range values {
		values[i] = v41RoundBF16(v)
	}
	st := b.record(verdict)
	st.Kind, st.Count, st.Site = compute.V41DeferredBF16Publish, len(values), site
	return b.Backend.Upload(compute.NewF32(compute.Default(), []int{len(values)}, values), compute.F32), st, nil
}

func (b *v41RoutedLayerTestBackend) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.deferredReads++
	verdicts := make([]error, len(statuses))
	for i, st := range statuses {
		verdict, ok := b.pending[st.Record.Buf()]
		if !ok {
			return nil, nil, errors.New("deferred status read twice or never issued")
		}
		delete(b.pending, st.Record.Buf())
		b.Backend.Free(st.Record)
		verdicts[i] = verdict
	}
	if b.readFailure != nil {
		return nil, nil, b.readFailure
	}
	values := make([][]float32, len(rows))
	for i, row := range rows {
		values[i] = append([]float32(nil), b.Backend.Read(row)...)
	}
	return verdicts, values, nil
}

// v41DeviceParityBlockAligned widens the hidden and expert widths to one k-quant
// super-block (256), the narrowest geometry that carries real Q2_K/Q3_K experts.
func v41DeviceParityBlockAligned(cfg *Config) {
	cfg.HiddenSize = qkK
	cfg.MoEIntermediateSize = qkK
}

// v41DeviceParityPinnedExperts replaces every routed expert with the pinned
// DeepSeek-V4.1-Flash Q2_K slate's dtypes: Q2_K gate/up and Q3_K down. The checkpoint has
// no F32 experts, and the device-only routed seam stages only the quantized kinds a
// device kernel serves, so an F32 slate would test a route the real model never takes.
//
// Quantized GEMMs and BF16 publications drift by about one BF16 ulp between host and
// device, and the random router had near-ties at its top-k boundary: one flipped
// selection (layer 2, expert 287 vs 321) moved later logits by 5%. That is tie-breaking,
// not device arithmetic, so the selection-only correction bias spaces experts a unit
// apart, which no score drift can cross; routing weights still come from the scores.
func v41DeviceParityPinnedExperts(t *testing.T, m *Model) {
	t.Helper()
	H, I := m.Cfg.HiddenSize, m.Cfg.MoEIntermediateSize
	bias := make([]float32, m.Cfg.NumExperts)
	for e := range bias {
		bias[e] = -float32(e)
	}
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		v41WriteTensorF32(t, m, layerName(l, "ffn.gate.e_score_correction_bias"), bias)
		for e := 0; e < m.Cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			w1, w3, w2 := layerName(l, stem+".w1.weight"), layerName(l, stem+".w3.weight"), layerName(l, stem+".w2.weight")
			m.kqw[w1] = q2kFixtureTensor(I, H, uint64(0x13827+3*e))
			m.kqw[w3] = q2kFixtureTensor(I, H, uint64(0x13827+3*e+1))
			m.kqw[w2] = q3kFixtureTensor(H, I)
			delete(m.manifest, w1)
			delete(m.manifest, w3)
			delete(m.manifest, w2)
		}
	}
}

// v41PinnedSlateFixture is the b98e317bb18 whole-forward slate: three layers,
// Engram on 0 and 2, block-aligned Q2_K gate/up and Q3_K down experts.
func v41PinnedSlateFixture(t *testing.T, compressed bool) *Model {
	t.Helper()
	m, _ := v41IncrementalEngramFixtureWith(t, compressed, v41DeviceParityBlockAligned)
	v41DeviceParityPinnedExperts(t, m)
	return m
}

func v41RoutedLayerSteps(t *testing.T, s *Session, prompt []int, steps int) [][]float32 {
	t.Helper()
	var out [][]float32
	last := s.Prefill(prompt)
	for i := 0; i < steps; i++ {
		last = append([]float32(nil), last...)
		out = append(out, last)
		last = s.Step(argmaxF32(last))
	}
	return append(out, append([]float32(nil), last...))
}

// The batched routed layer is the per-pick fused seam and routed sum bit for
// bit: every decode step's logits are identical, the batched session issues
// the same activations, and it fences once per layer per step.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=default
func TestV41RoutedLayerBatchMatchesPerPick(t *testing.T) {
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
			batchedModel, perPickModel := fx.build(t), fx.build(t)
			batched := newV41RoutedLayerTestBackend()
			sb, err := batchedModel.NewBackendSessionChecked(batched)
			if err != nil {
				t.Fatal(err)
			}
			defer sb.Close()
			perPick := newV41RoutedSwiGLUTestBackend()
			sp, err := perPickModel.NewBackendSessionChecked(perPick)
			if err != nil {
				t.Fatal(err)
			}
			defer sp.Close()
			if sb.v41State().routedLayer == nil || sp.v41State().routedLayer != nil {
				t.Fatal("only the deferred-status backend binds the batched routed layer")
			}
			incremental := 0
			prior := v41IncrementalStepProbe
			v41IncrementalStepProbe = func() { incremental++ }
			defer func() { v41IncrementalStepProbe = prior }()
			prompt := []int{1, 2, 3}
			got := v41RoutedLayerSteps(t, sb, prompt, steps)
			want := v41RoutedLayerSteps(t, sp, prompt, steps)
			if incremental != 2*steps {
				t.Fatalf("incremental decode steps=%d want %d", incremental, 2*steps)
			}
			for i := range want {
				if !reflect.DeepEqual(got[i], want[i]) {
					t.Fatalf("window %d logits differ between the batched layer and the per-pick seam", i)
				}
				for j, v := range got[i] {
					if math.IsNaN(float64(v)) {
						t.Fatalf("window %d logit %d is NaN", i, j)
					}
				}
			}
			layers := batchedModel.Cfg.NumLayers
			if batched.deferredReads != steps*layers {
				t.Fatalf("deferred reads=%d want one per layer per step (%d)", batched.deferredReads, steps*layers)
			}
			if batched.calls != perPick.calls || batched.publishes != steps*layers*batchedModel.Cfg.NumExpertsPerTok {
				t.Fatalf("activations batched=%d per-pick=%d, publishes=%d", batched.calls, perPick.calls, batched.publishes)
			}
			if len(batched.pending) != 0 {
				t.Fatalf("%d deferred status records were never read", len(batched.pending))
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41RoutedLayerEnvDisables(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "0")
	m := v41IncrementalExpertFixture(t, true, false)
	s, err := m.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.v41RoutedLayerFunc() != nil || s.v41RoutedExpertFunc() == nil {
		t.Fatalf("%s=0 must leave only the per-pick fused seam bound", v41LayerExpertBatchEnv)
	}
}

// v41RoutedLayerFaultPicks gives every slot a distinct route weight, so a
// weight-keyed fault names exactly one pick.
func v41RoutedLayerFaultPicks(m *Model) ([]routePick, []float32) {
	picks := make([]routePick, m.Cfg.NumExpertsPerTok)
	for slot := range picks {
		picks[slot] = routePick{expert: (3*slot + 1) % m.Cfg.NumExperts, weight: .25 + float32(slot)/8}
	}
	x := make([]float32, m.Cfg.HiddenSize)
	for i := range x {
		x[i] = v41OracleBF16(float32(i%7-3) * .19)
	}
	return picks, x
}

// v41RoutedLayerSerial runs the per-pick fused seam over picks in slot order
// and returns the first error and the slot that raised it.
func v41RoutedLayerSerial(t *testing.T, m *Model, s *Session, picks []routePick, x []float32) (int, error) {
	t.Helper()
	routed := s.v41RoutedExpertFunc()
	for slot, pick := range picks {
		stem := "ffn.experts." + itoa(pick.expert)
		_, err := m.v41FullRoutedExpertWithDevice(0, stem, x, pick.weight, m.Cfg, routed, nil, nil,
			func() ([]float32, []float32, []float32, error) {
				t.Fatal("serial fused seam loaded the host triple")
				return nil, nil, nil, nil
			},
			func() ([]float32, error) { t.Fatal("serial fused seam loaded the host down"); return nil, nil }, false)
		if err != nil {
			return slot, err
		}
	}
	return -1, nil
}

// A fault in pick k surfaces as the typed error the per-pick loop raises when
// it stops at pick k; with several faulting picks the lowest slot wins, and a
// numeric verdict never closes the session while a backend verdict does.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41RoutedLayerFaultAttribution(t *testing.T) {
	sentinel := &compute.BackendError{Backend: "test", Site: "V41RoutedSwiGLU", Err: errors.New("status protocol")}
	cases := []struct {
		name       string
		activation map[int]compute.V41RoutedSwiGLUStage
		publishAt  int
		protocolAt int
		readFails  bool
		slot       int
		stage      string
		closed     string
	}{
		{name: "activation at pick 3", activation: map[int]compute.V41RoutedSwiGLUStage{3: compute.V41RoutedSwiGLUStageActivation}, slot: 3, stage: "gate/up"},
		{name: "down operand at pick 0", activation: map[int]compute.V41RoutedSwiGLUStage{0: compute.V41RoutedSwiGLUStageDownOperand}, slot: 0, stage: "down"},
		{name: "lowest slot wins", activation: map[int]compute.V41RoutedSwiGLUStage{4: compute.V41RoutedSwiGLUStageActivation, 1: compute.V41RoutedSwiGLUStageDownOperand}, slot: 1, stage: "down"},
		{name: "publish at pick 2", publishAt: 3, slot: 2, stage: "down"},
		{name: "activation before a later publish", activation: map[int]compute.V41RoutedSwiGLUStage{1: compute.V41RoutedSwiGLUStageActivation}, publishAt: 5, slot: 1, stage: "gate/up"},
		{name: "protocol verdict at pick 2", protocolAt: 3, slot: 2, closed: "activation"},
		{name: "readback failure", readFails: true, slot: -1, closed: "readback"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
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
				if c.protocolAt > 0 {
					b.verdictFailure = map[float32]error{picks[c.protocolAt-1].weight: sentinel}
				}
				if c.readFails {
					b.readFailure = sentinel
				}
			}
			b := newV41RoutedLayerTestBackend()
			arm(b)
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			sum, outcome, err := s.v41RoutedLayerFunc()(0, picks, x)
			if sum != nil || outcome != v41RoutedExpertError || err == nil {
				t.Fatalf("faulting layer published=%v outcome=%v err=%v", sum != nil, outcome, err)
			}
			if b.calls != len(picks) || b.deferredReads != 1 || len(b.pending) != 0 {
				t.Fatalf("batched layer activations=%d reads=%d pending=%d, want %d/1/0", b.calls, b.deferredReads, len(b.pending), len(picks))
			}
			if c.closed != "" {
				var closed *BackendForwardOperationError
				if !errors.As(err, &closed) || !errors.Is(err, sentinel) || closed.Stage != c.closed || closed.Layer != 0 || !s.BackendSessionClosed() {
					t.Fatalf("backend verdict err=%v closed=%t, want closed stage %q", err, s.BackendSessionClosed(), c.closed)
				}
				return
			}
			var operation *V41ExpertOperationError
			if !errors.As(err, &operation) || operation.Stage != c.stage || operation.Layer != 0 || !errors.Is(err, errV41ExpertResult) || s.BackendSessionClosed() {
				t.Fatalf("numeric verdict err=%v closed=%t, want stage %q", err, s.BackendSessionClosed(), c.stage)
			}
			if c.publishAt > 0 {
				// The per-pick seam rounds the down output on the host for this
				// backend, so a publication verdict has no serial counterpart.
				return
			}
			// The per-pick loop over the same injected verdicts stops at the same
			// pick with the same typed error.
			serialModel := v41IncrementalExpertFixture(t, true, false)
			serial := newV41RoutedLayerTestBackend()
			arm(serial)
			ss, err := serialModel.NewBackendSessionChecked(serial)
			if err != nil {
				t.Fatal(err)
			}
			defer ss.Close()
			slot, serialErr := v41RoutedLayerSerial(t, serialModel, ss, picks, x)
			var serialOperation *V41ExpertOperationError
			if slot != c.slot || !errors.As(serialErr, &serialOperation) || serialOperation.Stage != operation.Stage || serialOperation.Layer != operation.Layer {
				t.Fatalf("per-pick loop stopped at pick %d with %v, batched reported %v (want pick %d)", slot, serialErr, err, c.slot)
			}
		})
	}
}

// A ring that pins one expert at a time but not a whole layer declines the
// batch before any device work, and the per-pick loop still runs.
// fak-test:runtime medium est=5s lane=default
func TestV41RoutedLayerDeclinesWhenRingCannotPinLayer(t *testing.T) {
	m := v41IncrementalExpertFixture(t, true, false)
	picks, x := v41RoutedLayerFaultPicks(m)
	stem := "ffn.experts." + itoa(picks[0].expert)
	var expert int64
	for _, leaf := range []string{".w1.weight", ".w3.weight", ".w2.weight"} {
		expert += kQuantResidentBytes(m.kqw[layerName(0, stem+leaf)])
	}
	for _, c := range []struct {
		ring    int64
		handled bool
	}{{ring: expert + expert/2}, {ring: int64(len(picks)+1) * expert, handled: true}} {
		b := newV41RoutedLayerTestBackend()
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		s.ExpertRingBytes = c.ring
		sum, outcome, err := s.v41RoutedLayerFunc()(0, picks, x)
		if err != nil || (outcome == v41RoutedExpertHandled) != c.handled || (sum != nil) != c.handled {
			t.Fatalf("ring=%d: outcome=%v err=%v, want handled=%t", c.ring, outcome, err, c.handled)
		}
		if !c.handled && (b.calls != 0 || b.deferredReads != 0) {
			t.Fatalf("ring=%d: a declined layer recorded %d activations and %d reads", c.ring, b.calls, b.deferredReads)
		}
		if _, serialErr := v41RoutedLayerSerial(t, m, s, picks, x); serialErr != nil {
			t.Fatalf("ring=%d: per-pick loop after the decline: %v", c.ring, serialErr)
		}
		s.Close()
	}
}
