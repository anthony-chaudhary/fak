package fleetbottleneck

import "testing"

func BenchmarkFleetBottleneck(b *testing.B) {
	snapshot := Snapshot{
		Machines:       10,
		Sessions:       80,
		SeatCapacity:   100,
		ThrottledSeats: 15,
		HealthySeats:   65,
		ResumeBacklog:  5,
		HostLoad:       0.88,
		AuthBlocked:    2,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Rank(snapshot)
	}
}
