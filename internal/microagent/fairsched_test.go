package microagent

import (
	"fmt"
	"testing"
	"time"
)

func TestTenantQueueWeightedServiceAndInteractiveTieBreak(t *testing.T) {
	s, _ := NewTenantQueue([]TenantEnvelope{{Tenant: "interactive", Weight: 3, MaxQueued: 20, MaxConcurrent: 1}, {Tenant: "bulk", Weight: 1, MaxQueued: 20, MaxConcurrent: 1}})
	now := time.Unix(1, 0)
	for i := 0; i < 12; i++ {
		_ = s.Submit(TenantTask{ID: fmt.Sprint("i", i), Tenant: "interactive", Interactive: true, Deadline: now.Add(time.Second), CostMicros: 1})
		_ = s.Submit(TenantTask{ID: fmt.Sprint("b", i), Tenant: "bulk", CostMicros: 1})
	}
	for i := 0; i < 16; i++ {
		if _, ok := s.Next(now); !ok {
			t.Fatal("drain")
		}
	}
	x := s.Snapshot()
	if x.Scheduled["interactive"] != 12 || x.Scheduled["bulk"] != 4 {
		t.Fatalf("snapshot=%+v", x)
	}
	if x.MaxLag["bulk"] > 1 {
		t.Fatalf("lag=%+v", x.MaxLag)
	}
}
func TestTenantQueueEnforcesSpendRateCancellation(t *testing.T) {
	s, _ := NewTenantQueue([]TenantEnvelope{{Tenant: "a", Weight: 1, MaxQueued: 2, MaxConcurrent: 1, MaxSpendMicros: 3, RatePerMinute: 1}})
	now := time.Unix(1, 0)
	_ = s.Submit(TenantTask{ID: "cancel", Tenant: "a", Cancelled: true})
	_ = s.Submit(TenantTask{ID: "one", Tenant: "a", CostMicros: 2})
	if err := s.Submit(TenantTask{ID: "costly", Tenant: "a", CostMicros: 4}); err == nil {
		t.Fatal("spend")
	}
	if _, ok := s.Next(now); !ok {
		t.Fatal("first")
	}
	_ = s.Submit(TenantTask{ID: "two", Tenant: "a", CostMicros: 1})
	if _, ok := s.Next(now); ok {
		t.Fatal("rate")
	}
	if _, ok := s.Next(now.Add(time.Minute + time.Second)); !ok {
		t.Fatal("rate reset")
	}
	if s.Snapshot().Cancelled["a"] != 1 {
		t.Fatal("cancel")
	}
}
