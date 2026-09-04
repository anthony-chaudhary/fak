package seatpark

import "testing"

func BenchmarkDecide_FirstEncounter(b *testing.B) {
	in := Input{TaskID: "task-01", Parks: 0, NowUnix: 1000}
	var sink Decision
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Decide(in)
	}
	if sink.Status != StatusReady {
		b.Fatalf("unexpected status: %s", sink.Status)
	}
}

func BenchmarkDecide_ParkedInWindow(b *testing.B) {
	in := Input{TaskID: "task-02", Parks: 2, LastParkUnix: 1000, NowUnix: 1020}
	var sink Decision
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Decide(in)
	}
	if sink.Status != StatusParked {
		b.Fatalf("unexpected status: %s", sink.Status)
	}
}

func BenchmarkDecide_ElapsedWindow(b *testing.B) {
	in := Input{TaskID: "task-03", Parks: 2, LastParkUnix: 1000, NowUnix: 1100}
	var sink Decision
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Decide(in)
	}
	if sink.Status != StatusReady {
		b.Fatalf("unexpected status: %s", sink.Status)
	}
}

func BenchmarkDecide_Exhausted(b *testing.B) {
	in := Input{TaskID: "task-04", Parks: 5, LastParkUnix: 1000, NowUnix: 1010}
	var sink Decision
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Decide(in)
	}
	if sink.Status != StatusExhausted {
		b.Fatalf("unexpected status: %s", sink.Status)
	}
}

func BenchmarkDecide_NoClock(b *testing.B) {
	in := Input{TaskID: "task-05", Parks: 3, LastParkUnix: 1000, NowUnix: 0}
	var sink Decision
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Decide(in)
	}
	if sink.Status != StatusReady {
		b.Fatalf("unexpected status: %s", sink.Status)
	}
}

func BenchmarkDecide_CustomPolicy(b *testing.B) {
	in := Input{
		TaskID:       "task-06",
		Parks:        3,
		LastParkUnix: 1000,
		NowUnix:      1050,
		Policy: Policy{
			MaxParks:    10,
			BaseSeconds: 15,
			Factor:      3,
			CapSeconds:  600,
		},
	}
	var sink Decision
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = Decide(in)
	}
	if sink.Status != StatusParked {
		b.Fatalf("unexpected status: %s", sink.Status)
	}
}

func BenchmarkBackoffSecondsScaling(b *testing.B) {
	p := Policy{}.withDefaults()
	var total int64
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for parks := 0; parks <= 10; parks++ {
			total += backoffSeconds(p, parks)
		}
	}
	if total == 0 {
		b.Fatal("unexpected zero total backoff")
	}
}

func BenchmarkBatchDecide(b *testing.B) {
	tasks := make([]Input, 40)
	for i := 0; i < 40; i++ {
		tasks[i] = Input{
			TaskID:       "task",
			Parks:        i % 6,
			LastParkUnix: 1000,
			NowUnix:      int64(1000 + (i%5)*20),
		}
	}
	var readyCount int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		readyCount = 0
		for j := range tasks {
			d := Decide(tasks[j])
			if d.ShouldAttempt() {
				readyCount++
			}
		}
	}
	if readyCount == 0 {
		b.Fatal("expected non-zero ready tasks")
	}
}

func TestBenchmarkSeatparkSanity(t *testing.T) {
	in := Input{TaskID: "sanity", Parks: 1, LastParkUnix: 1000, NowUnix: 1010}
	d := Decide(in)
	if d.Status != StatusParked || !d.Retryable() || d.ShouldAttempt() {
		t.Fatalf("unexpected sanity decision: %+v", d)
	}
}
