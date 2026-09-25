package session

import "testing"

func TestParseResourceBudgetLimitsMemoryCPUProcessesAndDiskRates(t *testing.T) {
	budget, err := ParseResourceBudget("peak-rss=8GiB,cpu-seconds=3600,processes=8,io-read-bps=1GiB,io-write-bps=512MiB")
	if err != nil {
		t.Fatalf("ParseResourceBudget: %v", err)
	}
	if budget.PeakRSSBytes != 8*1024*1024*1024 {
		t.Fatalf("peak RSS = %d", budget.PeakRSSBytes)
	}
	if budget.CPUSeconds != 3600 {
		t.Fatalf("CPU seconds = %v", budget.CPUSeconds)
	}
	if budget.ProcessCount != 8 {
		t.Fatalf("process count = %d", budget.ProcessCount)
	}
	if budget.ReadBytesPerSecond != 1024*1024*1024 {
		t.Fatalf("read bytes/sec = %d", budget.ReadBytesPerSecond)
	}
	if budget.WriteBytesPerSecond != 512*1024*1024 {
		t.Fatalf("write bytes/sec = %d", budget.WriteBytesPerSecond)
	}
}

func TestResourceBudgetDecideUsesPresenceAndDeterministicAxisOrder(t *testing.T) {
	budget := ResourceBudget{ProcessCount: 2, ReadBytesPerSecond: 100, WriteBytesPerSecond: 100}
	usage := ResourceUsage{
		ProcessCount:        2,
		ReadBytesPerSecond:  100,
		WriteBytesPerSecond: 100,
		HaveProcessCount:    true,
		HaveReadRate:        true,
		HaveWriteRate:       true,
	}
	decision := budget.Decide(usage)
	if !decision.Stop || decision.Axis != ResourceAxisProcess {
		t.Fatalf("decision = %+v, want process axis first", decision)
	}
	usage.ProcessCount = 1
	usage.HaveReadRate = false
	usage.HaveWriteRate = false
	if decision := budget.Decide(usage); decision.Stop {
		t.Fatalf("missing I/O should not decide: %+v", decision)
	}
}

func TestResourceBudgetExhaustionDrainsSession(t *testing.T) {
	tbl := NewTable()
	tbl.SetResourceBudget("trace-resource", ResourceBudget{
		PeakRSSBytes:        8 * 1024 * 1024 * 1024,
		CPUSeconds:          2,
		ProcessCount:        8,
		ReadBytesPerSecond:  1024 * 1024 * 1024,
		WriteBytesPerSecond: 512 * 1024 * 1024,
	})
	_, stopped, decision := tbl.ObserveResource("trace-resource", ResourceUsage{
		PeakRSSBytes:        7 * 1024 * 1024 * 1024,
		CPUSeconds:          3,
		ProcessCount:        1,
		ReadBytesPerSecond:  0,
		WriteBytesPerSecond: 0,
		HavePeakRSS:         true,
		HaveCPUSeconds:      true,
		HaveProcessCount:    true,
		HaveReadRate:        true,
		HaveWriteRate:       true,
	})
	if !stopped {
		t.Fatal("resource observation did not stop the session")
	}
	if !decision.Stop || decision.Axis != ResourceAxisCPUSeconds {
		t.Fatalf("decision = %+v, want CPU breach", decision)
	}
	state := tbl.Get("trace-resource")
	if state.Run != Stopped {
		t.Fatalf("run = %s, want stopped", state.Run)
	}
	if state.Reason != ReasonResourceBudgetExhausted {
		t.Fatalf("reason = %q, want %q", state.Reason, ReasonResourceBudgetExhausted)
	}
	if state.Resource.PeakRSSBytes != 8*1024*1024*1024 {
		t.Fatalf("resource peak RSS = %d", state.Resource.PeakRSSBytes)
	}
}
