package agent

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_expert_spill_test.go — witnesses for the served graded-spill seam (#5612).
//
// The placement ARITHMETIC and the split predicate are witnessed in internal/model
// (expert_spill_placement_test.go). What these pin is the part only this package can get wrong:
// that a resolved grade actually reaches the session a request decodes on, that the ungraded
// default reaches it byte-for-byte unchanged, and that the three operator mistakes worth refusing
// are refused instead of silently degrading.

// tinyMoECfg is a 4-MoE-layer synthetic with 2 routed experts per layer — small enough to build in
// memory, and enough layers that a partial spill (some layers host, some device) is distinguishable
// from both endpoints.
func tinyMoECfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 63,
		NumExperts: 2, NumExpertsPerTok: 1,
	}
}

// moeSpillBytes reports the model's measured expert residency the way ExpertSpillBudgetFor does, so
// a witness can state a budget in units of REAL layers ("the dense base plus room for two expert
// layers") instead of a magic byte count that silently stops meaning that if the fixture changes.
func moeSpillBytes(t *testing.T, m *model.Model) (base, perLayer int64, layers int) {
	t.Helper()
	replicated, expert, ok := m.MoEResidentWeightBytes()
	if !ok || expert <= 0 {
		t.Fatalf("fixture has no routed-expert residency: replicated=%d expert=%d ok=%v", replicated, expert, ok)
	}
	l := m.MoEExpertLayers()
	if len(l) == 0 {
		t.Fatal("fixture reports no MoE layers")
	}
	n := int64(len(l))
	return replicated, (expert + n - 1) / n, len(l)
}

func TestSetExpertSpillInstallsTheResolvedPlacementOnEverySession(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	base, perLayer, layers := moeSpillBytes(t, m)
	if layers != 4 {
		t.Fatalf("MoE layers = %d, want 4", layers)
	}
	p := &InKernelPlanner{m: m}

	// A budget with room for the dense base plus exactly two expert layers: auto-fit must spill the
	// other two, and the ring gets precisely the two layers' worth that stay on the device.
	if err := p.SetExpertSpill(ExpertSpillAuto, base+2*perLayer); err != nil {
		t.Fatalf("SetExpertSpill(auto): %v", err)
	}
	plan, ok := p.ExpertSpillPlacement()
	if !ok {
		t.Fatal("ExpertSpillPlacement reports no plan after a successful auto resolve")
	}
	if plan.Fit.SpillLayers != 2 {
		t.Fatalf("SpillLayers = %d, want 2", plan.Fit.SpillLayers)
	}
	if !plan.Fit.Fits {
		t.Fatalf("Fits = false for a budget sized to hold the plan: %+v", plan.Fit)
	}
	if plan.RingBytes != 2*perLayer {
		t.Fatalf("RingBytes = %d, want %d (the two device-resident expert layers)", plan.RingBytes, 2*perLayer)
	}

	// The seam that matters: the plan reaches the session a request would decode on. Two sessions,
	// because a served planner builds one per request and the install must not be first-only.
	for i := 0; i < 2; i++ {
		s := m.NewSession()
		p.applyExpertSpill(s)
		if s.ExpertSpillLayers != 2 {
			t.Fatalf("session %d: ExpertSpillLayers = %d, want 2", i, s.ExpertSpillLayers)
		}
		if s.ExpertRingBytes != 2*perLayer {
			t.Fatalf("session %d: ExpertRingBytes = %d, want %d", i, s.ExpertRingBytes, 2*perLayer)
		}
		if !s.CPUOffloadExperts {
			t.Fatalf("session %d: a spill of 2 layers left CPUOffloadExperts off, so the split kernel never runs", i)
		}
		s.Close()
	}
}

func TestUngradedPlannerLeavesSessionPlacementUntouched(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	p := &InKernelPlanner{m: m} // no SetExpertSpill: the default every serve runs today
	s := m.NewSession()
	defer s.Close()
	p.applyExpertSpill(s)
	if s.ExpertSpillLayers != 0 || s.ExpertRingBytes != 0 || s.CPUOffloadExperts {
		t.Fatalf("ungraded planner changed placement: layers=%d ring=%d offload=%v",
			s.ExpertSpillLayers, s.ExpertRingBytes, s.CPUOffloadExperts)
	}
	if _, ok := p.ExpertSpillPlacement(); ok {
		t.Fatal("ExpertSpillPlacement reports a plan on a planner that was never graded")
	}
}

func TestSetExpertSpillRefusesAutoWithNoMeasurableBudget(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	p := &InKernelPlanner{m: m} // backend nil => no capacity probe => no budget
	if got := p.expertSpillDeviceBudget(); got != 0 {
		t.Fatalf("expertSpillDeviceBudget with no backend = %d, want 0", got)
	}
	err := p.SetExpertSpill(ExpertSpillAuto, 0)
	if err == nil {
		t.Fatal("auto-fit against an unmeasurable budget was accepted; it would 'fit' by spilling every layer")
	}
	if !strings.Contains(err.Error(), "auto") {
		t.Fatalf("refusal does not name the auto grade: %v", err)
	}
	if _, ok := p.ExpertSpillPlacement(); ok {
		t.Fatal("a refused resolve still installed a placement")
	}

	// An EXPLICIT count is still honored without a measurable budget: the operator stated the
	// split, and only the ring (which needs a byte budget) goes unsized.
	if err := p.SetExpertSpill(3, 0); err != nil {
		t.Fatalf("SetExpertSpill(3) with no device budget: %v", err)
	}
	plan, ok := p.ExpertSpillPlacement()
	if !ok || plan.Fit.SpillLayers != 3 {
		t.Fatalf("explicit spill without a budget: ok=%v layers=%d, want ok=true layers=3", ok, plan.Fit.SpillLayers)
	}
	if plan.RingBytes != 0 {
		t.Fatalf("RingBytes = %d with no measurable budget, want 0 (ring off)", plan.RingBytes)
	}
}

func TestSetExpertSpillRefusesOutOfRangeCount(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	base, perLayer, layers := moeSpillBytes(t, m)
	p := &InKernelPlanner{m: m}
	err := p.SetExpertSpill(layers+1, base+perLayer)
	if err == nil {
		t.Fatalf("N = %d on a %d-MoE-layer model was accepted", layers+1, layers)
	}
	var rangeErr *model.ExpertSpillRangeError
	if !errors.As(err, &rangeErr) {
		t.Fatalf("want a typed *model.ExpertSpillRangeError so a caller can tell a typo from a bad budget, got %T: %v", err, err)
	}
	if rangeErr.N != layers+1 || rangeErr.Layers != layers {
		t.Fatalf("range error = %+v, want N=%d Layers=%d", rangeErr, layers+1, layers)
	}
	if _, ok := p.ExpertSpillPlacement(); ok {
		t.Fatal("an out-of-range resolve still installed a placement")
	}
}

func TestParseExpertSpillGrade(t *testing.T) {
	for _, tc := range []struct {
		in   string
		n    int
		set  bool
		fail bool
	}{
		{in: "", n: 0, set: false},
		{in: "off", n: 0, set: false},
		{in: "none", n: 0, set: false},
		{in: "auto", n: ExpertSpillAuto, set: true},
		{in: "  AUTO  ", n: ExpertSpillAuto, set: true},
		{in: "0", n: 0, set: true},   // size the ring, spill nothing
		{in: "12", n: 12, set: true}, // range is checked against the model, not here
		{in: "atuo", fail: true},     // a typo must refuse, never fall back to off
		{in: "-1", fail: true},       // auto is spelled auto; a negative count is not a count
		{in: "1.5", fail: true},
	} {
		n, set, err := ParseExpertSpillGrade(tc.in)
		if tc.fail {
			if err == nil {
				t.Fatalf("ParseExpertSpillGrade(%q) accepted, want a refusal", tc.in)
			}
			if set {
				t.Fatalf("ParseExpertSpillGrade(%q) refused but still reported set=true", tc.in)
			}
			continue
		}
		if err != nil {
			t.Fatalf("ParseExpertSpillGrade(%q): %v", tc.in, err)
		}
		if n != tc.n || set != tc.set {
			t.Fatalf("ParseExpertSpillGrade(%q) = (%d, %v), want (%d, %v)", tc.in, n, set, tc.n, tc.set)
		}
	}
}

// The env door is what makes the graded spill reachable on a live serve without a rebuild, so it is
// witnessed on the REAL constructor rather than on setExpertSpillFromEnv alone: it must be the
// planner every request decodes through that carries the grade.
func TestExpertSpillEnvGradesTheConstructedPlanner(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())

	t.Run("unset leaves the planner ungraded", func(t *testing.T) {
		t.Setenv(ExpertSpillEnv, "")
		p := NewInKernelPlanner(m, nil, "tiny-moe", false, nil, false)
		if _, ok := p.ExpertSpillPlacement(); ok {
			t.Fatal("an unset env knob still graded the planner")
		}
	})

	t.Run("an explicit count grades it", func(t *testing.T) {
		t.Setenv(ExpertSpillEnv, "3")
		p := NewInKernelPlanner(m, nil, "tiny-moe", false, nil, false)
		plan, ok := p.ExpertSpillPlacement()
		if !ok {
			t.Fatalf("%s=3 did not grade the planner", ExpertSpillEnv)
		}
		if plan.Fit.SpillLayers != 3 {
			t.Fatalf("SpillLayers = %d, want 3", plan.Fit.SpillLayers)
		}
		s := m.NewSession()
		defer s.Close()
		p.applyExpertSpill(s)
		if s.ExpertSpillLayers != 3 || !s.CPUOffloadExperts {
			t.Fatalf("the env grade did not reach the session: layers=%d offload=%v", s.ExpertSpillLayers, s.CPUOffloadExperts)
		}
	})

	t.Run("a refused value serves ungraded rather than taking the serve down", func(t *testing.T) {
		t.Setenv(ExpertSpillEnv, "atuo")
		p := NewInKernelPlanner(m, nil, "tiny-moe", false, nil, false)
		if _, ok := p.ExpertSpillPlacement(); ok {
			t.Fatal("a refused env value still installed a placement")
		}
	})
}

func TestSetExpertSpillRefusesOnlyAnActualSpillOnADenseModel(t *testing.T) {
	m := model.NewSynthetic(tinyCfg()) // dense: no routed experts anywhere
	p := &InKernelPlanner{m: m}

	if err := p.SetExpertSpill(2, 1<<30); err == nil {
		t.Fatal("asking a dense model to spill 2 expert layers was accepted; the operator would believe it happened")
	}

	// Neither of these asked to MOVE anything, so neither is an error — a launcher that always
	// passes `auto` must not fail on a dense model.
	for _, n := range []int{0, ExpertSpillAuto} {
		if err := p.SetExpertSpill(n, 1<<30); err != nil {
			t.Fatalf("SetExpertSpill(%d) on a dense model: %v", n, err)
		}
		if _, ok := p.ExpertSpillPlacement(); ok {
			t.Fatalf("SetExpertSpill(%d) installed a placement on a model with no experts to place", n)
		}
	}
}
