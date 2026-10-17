package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestV41GroupedOutputBindingLegacyAndNil(t *testing.T) {
	fault := errors.New("legacy grouped failure")
	callback := func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return []float32{7}, v41ProjectionError, fault
	}
	for _, route := range []v41GroupedOutputRoute{callback, v41GroupedOutputFunc(callback)} {
		y, o, err := v41GroupedOutputCallback(route)(0, nil, 1, 1, 1, 1, 1)
		if !reflect.DeepEqual(y, []float32{7}) || o != v41ProjectionError || err != fault {
			t.Fatal("legacy callback changed")
		}
		if _, ok := route.(*v41GroupedOutputBinding); ok {
			t.Fatal("legacy callback acquired native authority")
		}
	}
	var named v41GroupedOutputFunc
	var unnamed func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error)
	var binding *v41GroupedOutputBinding
	for _, route := range []v41GroupedOutputRoute{nil, named, unnamed, binding} {
		if v41GroupedOutputCallback(route) != nil {
			t.Fatalf("nil route %T no longer disables callback", route)
		}
	}
	for _, route := range []v41GroupedOutputRoute{17, "unknown", struct{}{}} {
		y, o, err := v41GroupedOutputCallback(route)(0, nil, 1, 1, 1, 1, 1)
		if y != nil || o != v41ProjectionError || !errors.Is(err, errV41ProjectionResult) {
			t.Fatalf("unsupported route %T did not fail closed", route)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41GroupedOutputBindingOwnerAndRevocation(t *testing.T) {
	m := v41GroupedBF16Fixture(t)
	defer m.CloseWeights()
	b := newV41CohortPanelRecorder()
	s, peer := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
	st := s.v41State()
	binding, ok := st.groupedOutput.(*v41GroupedOutputBinding)
	if !ok || !binding.canonical(s) || binding.canonical(peer) {
		t.Fatal("canonical owner identity missing")
	}
	old := st.groupedOutput
	fault := errors.New("custom projection")
	st.groupedOutput = func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionError, fault
	}
	if _, ok := st.groupedOutput.(*v41GroupedOutputBinding); ok {
		t.Fatal("override inherited canonical authority")
	}
	if _, _, err := v41GroupedOutputCallback(st.groupedOutput)(0, nil, 1, 1, 1, 1, 1); err != fault {
		t.Fatal("override error lost")
	}
	if !binding.canonical(s) {
		t.Fatal("override mutated an in-flight immutable binding")
	}
	st.groupedOutput = nil
	if v41GroupedOutputCallback(st.groupedOutput) != nil {
		t.Fatal("explicit nil clear reactivated native callback")
	}
	st.groupedOutput = old
	s.v41BindDeviceOnlyDeclineRefusals(st)
	if _, ok := st.groupedOutput.(*v41GroupedOutputBinding); ok {
		t.Fatal("device-only ordinary wrapper inherited native authority")
	}
	if v41GroupedOutputCallback(st.groupedOutput) == nil {
		t.Fatal("device-only wrapper disappeared")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41GroupedOutputBindingAccessorAllocations(t *testing.T) {
	ordinary := v41GroupedOutputFunc(func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionDeclined, nil
	})
	route := v41GroupedOutputRoute(&v41GroupedOutputBinding{ordinary: ordinary})
	var resolved v41GroupedOutputFunc
	if n := testing.AllocsPerRun(100, func() { resolved = v41GroupedOutputCallback(route) }); n != 0 {
		t.Fatalf("canonical accessor allocates %g", n)
	}
	if resolved == nil {
		t.Fatal("cached callback missing")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41GroupedOutputBindingConstructionCost(t *testing.T) {
	m := v41GroupedBF16Fixture(t)
	defer m.CloseWeights()
	s := v41DenseTestSession(t, m, newV41CohortPanelRecorder())
	var old v41GroupedOutputFunc
	var bound v41GroupedOutputRoute
	oldCount := testing.AllocsPerRun(100, func() { old = s.v41GroupedOutputFunc() })
	newCount := testing.AllocsPerRun(100, func() { bound = s.v41BindGroupedOutput() })
	if old == nil || v41GroupedOutputCallback(bound) == nil {
		t.Fatal("fixture did not bind native scalar callback")
	}
	t.Logf("once-per-session binding construction: scalar=%g canonical=%g delta=%g allocations; accessor cost separately gated", oldCount, newCount, newCount-oldCount)
}
