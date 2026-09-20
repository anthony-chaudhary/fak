package agent

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_cross_layer_prefetch_test.go — witnesses for the served per-session next-layer gate-prefetch
// seam (#1297/#1401).
//
// The predictor and its precision/recall ledger are witnessed in internal/model
// (expert_readahead_test.go). What these pin is the part only this package can get wrong: that the
// planner's opt-in actually reaches the session a request decodes on, that the default planner
// leaves the session carrying the model's own zero value byte-for-byte, and that the env door on the
// real constructor grades the planner every request decodes through.

// TestSetCrossLayerGatePrefetchInstallsOnEveryPlannerBuiltSession is the load-bearing seam: the knob
// reaches the session a request would decode on, and the install is NOT first-only — a served planner
// builds one session per request.
func TestSetCrossLayerGatePrefetchInstallsOnEveryPlannerBuiltSession(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	p := &InKernelPlanner{m: m}
	p.SetCrossLayerGatePrefetch(true)
	if !p.CrossLayerGatePrefetchEnabled() {
		t.Fatal("SetCrossLayerGatePrefetch(true) did not set the planner field")
	}

	for i := 0; i < 2; i++ {
		s := m.NewSession()
		p.applyCrossLayerGatePrefetch(s)
		if !s.CrossLayerGatePrefetch {
			t.Fatalf("session %d: CrossLayerGatePrefetch = false after SetCrossLayerGatePrefetch(true)", i)
		}
		s.Close()
	}

	// Explicit off restores the no-op install: a fresh session must carry the model's zero value.
	p.SetCrossLayerGatePrefetch(false)
	s := m.NewSession()
	defer s.Close()
	p.applyCrossLayerGatePrefetch(s)
	if s.CrossLayerGatePrefetch {
		t.Fatal("SetCrossLayerGatePrefetch(false) still installed the knob")
	}
}

// TestDefaultPlannerLeavesCrossLayerGatePrefetchOff pins the byte-for-byte default: a planner that
// was never opted in must build sessions the model itself left at the zero value.
func TestDefaultPlannerLeavesCrossLayerGatePrefetchOff(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	p := &InKernelPlanner{m: m} // no SetCrossLayerGatePrefetch: the default every serve runs today
	if p.CrossLayerGatePrefetchEnabled() {
		t.Fatal("a default planner reports the knob enabled")
	}
	s := m.NewSession()
	defer s.Close()
	p.applyCrossLayerGatePrefetch(s)
	if s.CrossLayerGatePrefetch {
		t.Fatal("default planner install changed the session")
	}
}

// The env door is what makes the knob reachable on a live serve without a rebuild, so it is witnessed
// on the REAL constructor rather than on setCrossLayerGatePrefetchFromEnv alone: it must be the
// planner every request decodes through that carries the setting.
func TestCrossLayerGatePrefetchEnvGradesTheConstructedPlanner(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())

	t.Run("unset leaves the planner off", func(t *testing.T) {
		t.Setenv(CrossLayerGatePrefetchEnv, "")
		p := NewInKernelPlanner(m, nil, "tiny-moe", false, nil, false)
		if p.CrossLayerGatePrefetchEnabled() {
			t.Fatal("an unset env knob still enabled the planner")
		}
		s := m.NewSession()
		defer s.Close()
		p.applyCrossLayerGatePrefetch(s)
		if s.CrossLayerGatePrefetch {
			t.Fatal("an unset env knob still reached the session")
		}
	})

	t.Run("a truthy value enables it", func(t *testing.T) {
		t.Setenv(CrossLayerGatePrefetchEnv, "true")
		p := NewInKernelPlanner(m, nil, "tiny-moe", false, nil, false)
		if !p.CrossLayerGatePrefetchEnabled() {
			t.Fatalf("%s=true did not enable the planner", CrossLayerGatePrefetchEnv)
		}
		s := m.NewSession()
		defer s.Close()
		p.applyCrossLayerGatePrefetch(s)
		if !s.CrossLayerGatePrefetch {
			t.Fatal("the env setting did not reach the session")
		}
	})

	t.Run("a refused value serves with the knob off rather than taking the serve down", func(t *testing.T) {
		t.Setenv(CrossLayerGatePrefetchEnv, "maybe")
		p := NewInKernelPlanner(m, nil, "tiny-moe", false, nil, false)
		if p.CrossLayerGatePrefetchEnabled() {
			t.Fatal("a refused env value still enabled the knob")
		}
	})
}

func TestParseCrossLayerGatePrefetch(t *testing.T) {
	for _, tc := range []struct {
		in   string
		on   bool
		set  bool
		fail bool
	}{
		{in: "", set: false},
		{in: "   ", set: false},
		{in: "on", on: true, set: true},
		{in: "ON", on: true, set: true},
		{in: "off", on: false, set: true},
		{in: "true", on: true, set: true},
		{in: "1", on: true, set: true},
		{in: "0", on: false, set: true},
		{in: "yes", on: true, set: true},
		{in: "no", on: false, set: true},
		{in: "maybe", fail: true},
		{in: "2", fail: true},
		{in: "onoff", fail: true},
	} {
		on, set, err := ParseCrossLayerGatePrefetch(tc.in)
		if tc.fail {
			if err == nil {
				t.Fatalf("ParseCrossLayerGatePrefetch(%q) accepted, want a refusal", tc.in)
			}
			if set {
				t.Fatalf("ParseCrossLayerGatePrefetch(%q) refused but still reported set=true", tc.in)
			}
			continue
		}
		if err != nil {
			t.Fatalf("ParseCrossLayerGatePrefetch(%q): %v", tc.in, err)
		}
		if on != tc.on || set != tc.set {
			t.Fatalf("ParseCrossLayerGatePrefetch(%q) = (%v, %v), want (%v, %v)", tc.in, on, set, tc.on, tc.set)
		}
	}
}

// The refusal names the knob so an operator can tell a typo from a bad grade, matching the strict
// door's posture on the serve flag.
func TestParseCrossLayerGatePrefetchRefusalNamesTheKnob(t *testing.T) {
	_, _, err := ParseCrossLayerGatePrefetch("maybe")
	if err == nil || !strings.Contains(err.Error(), "cross-layer-gate-prefetch") {
		t.Fatalf("refusal %v does not name the knob", err)
	}
}

// The report is the operator surface for the knob (#1401 part C): a session that opted in reports a
// non-nil CrossLayer ledger (the zero ledger until it actually predicts — the honest "enabled, not
// yet measured" reading), while a session with the knob off reports nil so the JSON omits the field
// and the report is byte-for-byte the pre-#1401 shape.
func TestCrossLayerGatePrefetchRidesOutThroughTheResidencyReport(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())

	on := m.NewSession()
	defer on.Close()
	on.CrossLayerGatePrefetch = true
	if rep := on.MoEResidency(model.MoEResidencyOptions{}); rep.CrossLayer == nil {
		t.Fatal("an opted-in session reported a nil CrossLayer ledger; the knob is unobservable")
	} else if rep.CrossLayer.Predicted != 0 || rep.CrossLayer.Actual != 0 || rep.CrossLayer.Hits != 0 {
		t.Fatalf("a fresh opted-in session reported non-zero prediction counters: %+v", *rep.CrossLayer)
	}

	off := m.NewSession()
	defer off.Close()
	if rep := off.MoEResidency(model.MoEResidencyOptions{}); rep.CrossLayer != nil {
		t.Fatalf("a session with the knob off reported a CrossLayer ledger: %+v", *rep.CrossLayer)
	}
}

// The planner install is what makes the report field reachable: a session the planner built with the
// knob on must report it, and one the default planner built must not.
func TestPlannerInstallDrivesTheReportField(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())

	p := &InKernelPlanner{m: m}
	p.SetCrossLayerGatePrefetch(true)
	on := m.NewSession()
	defer on.Close()
	p.applyCrossLayerGatePrefetch(on)
	if rep := on.MoEResidency(model.MoEResidencyOptions{}); rep.CrossLayer == nil {
		t.Fatal("a planner-installed session did not report CrossLayer")
	}

	d := &InKernelPlanner{m: m}
	off := m.NewSession()
	defer off.Close()
	d.applyCrossLayerGatePrefetch(off)
	if rep := off.MoEResidency(model.MoEResidencyOptions{}); rep.CrossLayer != nil {
		t.Fatal("a default-planner session reported CrossLayer")
	}
}
