package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41StepMHCCounts(stages []string) map[string]int {
	counts := map[string]int{}
	for _, stage := range stages {
		counts[stage]++
	}
	return counts
}

func v41StepMHCPhaseFacts(stages []string) []string {
	var facts []string
	for _, stage := range stages {
		if stage == "mhc-streams" || stage == "mhc-projection" {
			facts = append(facts, stage)
		}
	}
	return facts
}

// fak-test:runtime fast est=0.2s lane=default
func TestV41HostArmStepMHCContinuation(t *testing.T) {
	for _, tc := range []struct {
		name          string
		build         func(*testing.T) *Model
		full, decline bool
	}{
		{"full-plain", v41FullStepPatchedNorms, true, false},
		{"full-role", func(t *testing.T) *Model { return v41IncrementalExpertFixture(t, true, false) }, true, false},
		{"reduced-plain", func(t *testing.T) *Model { return v41IncrementalPlainModel(t, 2) }, false, false},
		{"projection-decline", v41FullStepPatchedNorms, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.build(t)
			s, control := m.NewSession(), m.NewSession()
			t.Cleanup(s.Close)
			t.Cleanup(control.Close)
			if full, err := v41ForwardGeometry(m.Cfg); err != nil || full != tc.full {
				t.Fatalf("fixture geometry full=%t: %v", full, err)
			}
			v41HostArmMHCProjectionBits(t, s.Prefill([]int{1, 2}), control.Prefill([]int{1, 2}))
			if tc.decline {
				decline := func(int, []float32, int, float32, bool, bool) ([]float32, v41DenseProjectionOutcome, error) {
					return nil, v41ProjectionDeclined, nil
				}
				s.v41Forward.mhcProjection, control.v41Forward.mhcProjection = decline, decline
			}
			var stages []string
			s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
			for _, token := range []int{3, 4} {
				stages = nil
				v41HostArmMHCProjectionBits(t, s.Step(token), control.Step(token))
				if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(control.v41Forward)) {
					t.Fatal("observed Step changed continuation arithmetic")
				}
				counts := v41StepMHCCounts(stages)
				if counts["mhc-streams"] == 0 || counts["mhc-projection"] == 0 {
					t.Fatalf("owned Step caller facts = %v", counts)
				}
				if tc.full && counts["bf16-boundary"] == 0 {
					t.Fatal("full scalar Step omitted actual BF16 publication")
				}
			}
		})
	}
}

// fak-test:runtime fast est=0.2s lane=default
func TestV41HostArmStepMHCPrefixRebind(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	source, branch := m.NewSession(), m.NewSession()
	t.Cleanup(source.Close)
	t.Cleanup(branch.Close)
	source.Prefill([]int{1, 2})
	var sourceStages, branchStages []string
	source.hostStageObserver = func(stage string) { sourceStages = append(sourceStages, stage) }
	prefix, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(prefix.Close)
	branch.hostStageObserver = func(stage string) { branchStages = append(branchStages, stage) }
	if err := prefix.Restore(branch); err != nil {
		t.Fatal(err)
	}
	if branch.v41Forward.callbackOwner != branch || branch.v41Forward.gatherOps.owner != branch {
		t.Fatal("prefix restore did not rebind destination owner")
	}
	for _, token := range []int{3, 4} {
		branchStages = nil
		want := source.Step(token)
		beforeSource := append([]string(nil), sourceStages...)
		v41HostArmMHCProjectionBits(t, branch.Step(token), want)
		if !reflect.DeepEqual(sourceStages, beforeSource) {
			t.Fatal("restored Step observed through source owner")
		}
		if v41StepMHCCounts(branchStages)["mhc-streams"] == 0 || v41StepMHCCounts(branchStages)["mhc-projection"] == 0 {
			t.Fatalf("restored caller facts = %v", branchStages)
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(source.v41Forward), captureV41ForwardSnapshot(branch.v41Forward)) {
			t.Fatal("restored observer changed arithmetic state")
		}
	}
}

// fak-test:runtime fast est=0.2s lane=default
func TestV41HostArmStepMHCOwnedPhases(t *testing.T) {
	for _, full := range []bool{false, true} {
		t.Run(map[bool]string{false: "reduced", true: "full"}[full], func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 1)
			if full {
				m = v41FullStepPatchedNorms(t)
			}
			s := m.NewSession()
			t.Cleanup(s.Close)
			s.Prefill([]int{1, 2})
			state := s.v41State()
			scratch := &v41ProjScratch{}
			scratch.bindV41StepState(state)
			x, _ := v41LayerStepInputs(t, m, 3, full)
			var stages []string
			s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
			streams, resident, err := m.v41StepInitialStreams(state, x, full, scratch)
			if err != nil || resident != nil {
				t.Fatalf("host initializer: %v", err)
			}
			var initial []string
			if full {
				initial = []string{"bf16-boundary", "mhc-streams"}
			}
			if !reflect.DeepEqual(stages, initial) {
				t.Fatalf("isolated initial facts = %v", stages)
			}
			plain := &v41PlainLayerStep{m: m, l: 0, x: x, streams: streams, scratch: scratch, full: full, eps: float32(m.Cfg.RMSNormEps)}
			stages = nil
			if err := plain.collapseMHC(m.v41LeafTensor(0, v41LeafAttnNorm), m.v41LeafTensor(0, v41LeafMHCScale), m.v41LeafTensor(0, v41LeafMHCBase)); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(v41StepMHCPhaseFacts(stages), []string{"mhc-projection", "mhc-streams", "mhc-streams"}) {
				t.Fatalf("isolated projection/split/pre facts = %v", stages)
			}
			plain.attnOut = make([]float32, m.Cfg.HiddenSize)
			plain.ffnResidual, plain.ffnMix = streams, plain.mix
			stages = nil
			if _, err := plain.postMix(x, false); err != nil {
				t.Fatal(err)
			}
			post := []string{"mhc-streams"}
			if full {
				post = []string{"bf16-boundary", "mhc-streams", "bf16-boundary", "bf16-boundary", "bf16-boundary", "bf16-boundary"}
			}
			if !reflect.DeepEqual(stages, post) {
				t.Fatalf("isolated post facts = %v", stages)
			}
		})
	}
}

// fak-test:runtime fast est=0.2s lane=default
func TestV41HostArmStepMHCRolePhases(t *testing.T) {
	m := v41IncrementalExpertFixture(t, true, false)
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Prefill([]int{1, 2})
	state := s.v41State()
	scratch := &v41ProjScratch{}
	scratch.bindV41StepState(state)
	scratch.mhcCarry = newV41MHCCarry(1)
	x, streams := v41LayerStepInputs(t, m, 3, true)
	plan, err := v41AttentionPlanFor(m.Cfg, 0, m.v41AttentionRolesCached())
	if err != nil {
		t.Fatal(err)
	}
	var stages []string
	s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
	attention, err := m.prepareV41RoleAttention(0, plan, x, streams, len(state.history), state.layers[0], state.attn, scratch)
	if err != nil {
		t.Fatal(err)
	}
	defer attention.close()
	if !reflect.DeepEqual(v41StepMHCPhaseFacts(stages), []string{"mhc-projection", "mhc-streams", "mhc-streams"}) {
		t.Fatalf("isolated role projection/split/pre facts = %v", stages)
	}
	work, err := m.prepareV41LayerStepRoleFinish(0, len(state.history), x, streams, attention.mix, make([]float32, m.Cfg.HiddenSize), scratch)
	if err != nil {
		t.Fatal(err)
	}
	// FFN preparation is outside this observation bracket; only the returned
	// completion runs, so later FFN-helper adoption cannot satisfy this post fact.
	stages = nil
	if err := work.complete(nil); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(v41StepMHCPhaseFacts(stages), []string{"mhc-streams"}) {
		t.Fatalf("isolated role final post facts = %v", stages)
	}
}

// fak-test:runtime fast est=0.2s lane=default
func TestV41HostArmStepMHCSelected(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	b := newV41MHCTestBackend(true)
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2})
	var stages []string
	s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
	beforeInitial := b.calls["initial"]
	s.Step(3)
	if b.calls["initial"] != beforeInitial+1 || v41StepMHCCounts(stages)["mhc-streams"] != 0 {
		t.Fatalf("resident selected Step facts = %v, initial calls %d", stages, b.calls["initial"])
	}
	// Drive the actual plain-layer nonresident entry with its selected seam.
	st := s.v41State()
	scratch := &v41ProjScratch{}
	scratch.bindV41StepState(st)
	scratch.mhcCarry = newV41MHCCarry(1)
	scratch.mhcCarry.device = st.mhcDevice
	x, streams := v41LayerStepInputs(t, m, 4, true)
	plain := &v41PlainLayerStep{m: m, l: 0, x: x, streams: streams, scratch: scratch, full: true, eps: float32(m.Cfg.RMSNormEps), mhcDevice: st.mhcDevice}
	stages = nil
	if err := plain.collapseMHC(m.v41LeafTensor(0, v41LeafAttnNorm), m.v41LeafTensor(0, v41LeafMHCScale), m.v41LeafTensor(0, v41LeafMHCBase)); err != nil {
		t.Fatal(err)
	}
	counts := v41StepMHCCounts(stages)
	if counts["mhc-streams"] != 0 || counts["mhc-projection"] != 0 || counts["rms-norm"] != 1 {
		t.Fatalf("selected primitives versus actual projection host RMS = %v", counts)
	}
	// A selected primitive failure never replays scalar MHC in the Step path.
	b.failOp, b.failCall = "initial", b.calls["initial"]+1
	b.cause = &compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
	stages = nil
	func() {
		defer func() {
			err, ok := recover().(error)
			var selected *V41MHCOperationError
			if !ok || !errors.As(err, &selected) || !errors.Is(err, compute.ErrVulkanExecutionFailed) {
				t.Fatalf("selected Step failure = %v", err)
			}
		}()
		s.Step(4)
	}()
	if !s.BackendSessionClosed() || v41StepMHCCounts(stages)["mhc-streams"] != 0 {
		t.Fatalf("selected failure replayed MHC: %v", stages)
	}
}

// fak-test:runtime fast est=0.1s lane=default
func TestV41HostArmStepMHCPartialPublication(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	s := m.NewSession()
	t.Cleanup(s.Close)
	st := s.v41State()
	scratch := &v41ProjScratch{}
	scratch.bindV41StepState(st)
	var stages []string
	s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
	_, _, err := m.v41StepInitialStreams(st, []float32{1, float32(math.NaN())}, true, scratch)
	var typed *V41ForwardError
	if !errors.As(err, &typed) || typed.Stage != v41StageMHC || !reflect.DeepEqual(stages, []string{"bf16-boundary"}) || scratch.mhcCarry != nil {
		t.Fatalf("partial initializer = %v, facts %v", err, stages)
	}
	stages = nil
	plain := &v41PlainLayerStep{m: m, scratch: scratch, full: true}
	_, err = plain.postMix([]float32{1, float32(math.NaN())}, false)
	if !errors.As(err, &typed) || typed.Stage != v41StageMHC || !reflect.DeepEqual(stages, []string{"bf16-boundary"}) {
		t.Fatalf("partial final post = %v, facts %v", err, stages)
	}
}
