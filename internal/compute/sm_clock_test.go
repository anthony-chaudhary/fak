package compute

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestSMClockForPrefersObservedReaderOfSameBackend(t *testing.T) {
	t.Setenv(SMClockEnv, "1500")
	unregister := RegisterSMClockReader("cuda", func() (float64, bool) { return 2617, true })
	defer unregister()

	if got, ok := SMClockFor("cuda"); !ok || got != (SMClock{MHz: 2617, Source: SMClockObserved}) {
		t.Fatalf("cuda clock = %+v ok=%v, want observed 2617", got, ok)
	}
	if got, ok := SMClockFor("vulkan"); !ok || got != (SMClock{MHz: 1500, Source: SMClockDeclared}) {
		t.Fatalf("vulkan clock = %+v ok=%v, want declared 1500 (reader belongs to cuda)", got, ok)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSMClockForFallsBackWhenReaderUnreadable(t *testing.T) {
	t.Setenv(SMClockEnv, "1800")
	unregister := RegisterSMClockReader("cuda", func() (float64, bool) { return 0, false })
	defer unregister()

	if got, ok := SMClockFor("cuda"); !ok || got.Source != SMClockDeclared || got.MHz != 1800 {
		t.Fatalf("clock = %+v ok=%v, want declared 1800", got, ok)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSMClockForUnknownIsNotZero(t *testing.T) {
	for _, v := range []string{"", "abc", "0", "-5", "2617000", "NaN"} {
		t.Setenv(SMClockEnv, v)
		if got, ok := SMClockFor("cuda"); ok {
			t.Fatalf("env %q: clock = %+v, want unknown", v, got)
		}
	}
}
