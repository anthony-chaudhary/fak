package fleetsim

import "testing"

// BenchmarkFleetSim exercises synthetic fleet replay and endurance simulation in a loop.
func BenchmarkFleetSim(b *testing.B) {
	fixture := DefaultFixture()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Replay(fixture)
		if rep.TotalSessions == 0 {
			b.Fatal("unexpected empty replay report")
		}
	}
}

// BenchmarkEnduranceSimulation exercises multi-cycle endurance simulation in a loop.
func BenchmarkEnduranceSimulation(b *testing.B) {
	cfg := EnduranceConfig{
		Issues:        50,
		Workers:       8,
		ClosePerCycle: 4,
		MaxCycles:     20,
		RefusalCycles: map[int]bool{2: true},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := ReplayEndurance(cfg)
		if rep.ClosedIssues == 0 {
			b.Fatal("unexpected empty endurance report")
		}
	}
}

func TestBenchmarkFleetSimRuns(t *testing.T) {
	fixture := DefaultFixture()
	rep := Replay(fixture)
	if rep.TotalSessions == 0 {
		t.Fatal("unexpected zero sessions")
	}

	cfg := EnduranceConfig{
		Issues:        20,
		Workers:       4,
		ClosePerCycle: 2,
		MaxCycles:     10,
	}
	endRep := ReplayEndurance(cfg)
	if endRep.ClosedIssues == 0 {
		t.Fatal("unexpected zero closed issues in endurance run")
	}
}
