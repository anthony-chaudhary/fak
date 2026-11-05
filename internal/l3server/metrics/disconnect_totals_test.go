package metrics

import (
	"testing"
	"time"
)

// fak-test:runtime medium est=10ms lane=default
func TestDisconnectConservesGlobalTotals(t *testing.T) {
	r := NewConnRegistry()
	add := func(cm *ConnMetrics, recv, sent int64, requests int) {
		cm.AddBytesRecv(recv)
		cm.AddBytesSent(sent)
		for i := 0; i < requests; i++ {
			cm.IncrRequests()
		}
	}
	assertTotals := func(recv, sent, requests int64) {
		t.Helper()
		if got := [3]int64{r.TotalBytesRecv(), r.TotalBytesSent(), r.TotalRequests()}; got != [3]int64{recv, sent, requests} {
			t.Errorf("global totals = %v, want [%d %d %d]", got, recv, sent, requests)
		}
	}

	retired := r.Register("tcp", "retired", "", "")
	add(retired, 5, 7, 1)
	r.Deregister(retired.ID)
	active := r.Register("tcp", "active", "", "")
	add(active, 11, 13, 2)
	departing := r.Register("tcp", "departing", "", "")
	add(departing, 17, 19, 3)
	assertTotals(33, 39, 6)

	// Hold the existing cleanup lock. Once the active map loses departing,
	// Deregister is blocked in client-stats removal, with no timing guess.
	r.ClientStatsReg = NewClientStatsRegistry()
	r.ClientStatsReg.mu.Lock()
	locked := true
	done := make(chan struct{})
	go func() {
		r.Deregister(departing.ID)
		close(done)
	}()
	t.Cleanup(func() {
		if locked {
			r.ClientStatsReg.mu.Unlock()
		}
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("disconnect did not finish after releasing client-stats cleanup")
		}
	})

	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for r.ActiveCount() != 1 {
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatal("disconnect did not remove the departing connection")
		}
	}
	assertTotals(33, 39, 6)

	// A flush during delayed cleanup must not be undone when cleanup resumes.
	r.ResetAll()
	assertTotals(0, 0, 0)
	r.ClientStatsReg.mu.Unlock()
	locked = false
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("disconnect did not finish")
	}
	assertTotals(0, 0, 0)
	r.Deregister(departing.ID)
	r.Deregister(^uint64(0))
	assertTotals(0, 0, 0)

	add(active, 3, 4, 1)
	assertTotals(3, 4, 1)
	r.Deregister(active.ID)
	r.Deregister(active.ID)
	assertTotals(3, 4, 1)
	if got := r.ActiveCount(); got != 0 {
		t.Errorf("active connections = %d, want 0", got)
	}
}
