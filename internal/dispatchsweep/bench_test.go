package dispatchsweep

import (
	"testing"
)

func TestBenchmarkDispatchSweep(t *testing.T) {
	cfg := Config{MaxAgents: 5, Live: true}
	tick := func(i int) (TickResult, error) {
		if i < 2 {
			return TickResult{
				Action:           "spawned",
				Verdict:          "SPAWNED",
				OK:               true,
				Issue:            200 + i,
				Lane:             "docs",
				Account:          "acct-a",
				PreflightVerdict: "SPAWN_OK",
			}, nil
		}
		return TickResult{
			Action:           "refused",
			Verdict:          "REFUSE_AT_CAP",
			OK:               false,
			Lane:             "docs",
			PreflightVerdict: "REFUSE_AT_CAP",
		}, nil
	}
	rec := RunSweep(cfg, tick, func() {})
	if !rec.OK || rec.SpawnedCount != 2 {
		t.Fatalf("unexpected sweep result: ok=%v spawned=%d", rec.OK, rec.SpawnedCount)
	}
}

func BenchmarkDispatchSweep(b *testing.B) {
	cfg := Config{MaxAgents: 10, Live: true}
	tick := func(i int) (TickResult, error) {
		if i < 5 {
			return TickResult{
				Action:           "spawned",
				Verdict:          "SPAWNED",
				OK:               true,
				Issue:            100 + i,
				Lane:             "docs",
				Account:          "acct-a",
				PreflightVerdict: "SPAWN_OK",
			}, nil
		}
		return TickResult{
			Action:           "refused",
			Verdict:          "REFUSE_AT_CAP",
			OK:               false,
			Lane:             "docs",
			PreflightVerdict: "REFUSE_AT_CAP",
		}, nil
	}
	settle := func() {}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := RunSweep(cfg, tick, settle)
		if !rec.OK {
			b.Fatalf("unexpected failure: %v", rec.StopVerdict)
		}
	}
}
