package engine

import "testing"

func TestSoftCapControllerSuppressesBoundaryNoise(t *testing.T) {
	controller := newSoftCapControllerForTest(t, 3)

	for i, used := range []int64{90, 101, 99, 102, 98, 101} {
		decision := controller.Observe(used)
		if decision.State != SoftCapNormal || decision.Changed {
			t.Fatalf("sample %d (%d): decision = %+v, want stable normal", i, used, decision)
		}
	}
}

func TestSoftCapControllerRequiresSustainedPressureAndRecovery(t *testing.T) {
	controller := newSoftCapControllerForTest(t, 3)

	for i, used := range []int64{101, 110, 119} {
		decision := controller.Observe(used)
		if i < 2 && (decision.State != SoftCapNormal || decision.Changed) {
			t.Fatalf("pressure sample %d: decision = %+v, want pending normal", i, decision)
		}
		if i == 2 && (!decision.Changed || decision.State != SoftCapPressure || decision.Reason != "sustained") {
			t.Fatalf("pressure sample %d: decision = %+v, want sustained soft pressure", i, decision)
		}
	}

	for i, used := range []int64{99, 80, 70} {
		decision := controller.Observe(used)
		if i < 2 && (decision.State != SoftCapPressure || decision.Changed) {
			t.Fatalf("recovery sample %d: decision = %+v, want pending pressure", i, decision)
		}
		if i == 2 && (!decision.Changed || decision.State != SoftCapNormal || decision.Reason != "sustained") {
			t.Fatalf("recovery sample %d: decision = %+v, want sustained normal", i, decision)
		}
	}
}

func TestSoftCapControllerHardLimitIsImmediate(t *testing.T) {
	controller := newSoftCapControllerForTest(t, 4)

	decision := controller.Observe(200)
	if !decision.Changed || decision.State != SoftCapHard || decision.Reason != "hard-limit" {
		t.Fatalf("hard-limit decision = %+v, want immediate hard pressure", decision)
	}

	for i, used := range []int64{150, 150, 150} {
		decision = controller.Observe(used)
		if decision.State != SoftCapHard || decision.Changed {
			t.Fatalf("hard recovery sample %d: decision = %+v, want pending hard state", i, decision)
		}
	}
	decision = controller.Observe(150)
	if !decision.Changed || decision.State != SoftCapPressure {
		t.Fatalf("hard recovery decision = %+v, want soft pressure after sustained samples", decision)
	}
}

func TestSoftCapControllerAlternatingTargetsResetHysteresis(t *testing.T) {
	controller := newSoftCapControllerForTest(t, 2)
	controller.Observe(200)

	for i, used := range []int64{150, 90, 150, 90} {
		decision := controller.Observe(used)
		if decision.State != SoftCapHard || decision.Changed {
			t.Fatalf("alternating sample %d (%d): decision = %+v, want stable hard state", i, used, decision)
		}
		if decision.Pending != 1 {
			t.Fatalf("alternating sample %d (%d): pending = %d, want reset streak 1", i, used, decision.Pending)
		}
	}
}

func TestNewSoftCapControllerRejectsInvalidConfig(t *testing.T) {
	tests := []SoftCapConfig{
		{SoftLimitBytes: 0, HardLimitBytes: 200, Samples: 2},
		{SoftLimitBytes: 100, HardLimitBytes: 100, Samples: 2},
		{SoftLimitBytes: 100, HardLimitBytes: 200, Samples: 0},
	}
	for _, config := range tests {
		if _, err := NewSoftCapController(config); err == nil {
			t.Fatalf("NewSoftCapController(%+v) succeeded, want error", config)
		}
	}
}

func newSoftCapControllerForTest(t *testing.T, samples int) *SoftCapController {
	t.Helper()
	controller, err := NewSoftCapController(SoftCapConfig{
		SoftLimitBytes: 100,
		HardLimitBytes: 200,
		Samples:        samples,
	})
	if err != nil {
		t.Fatal(err)
	}
	return controller
}
