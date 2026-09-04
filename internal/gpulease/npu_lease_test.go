package gpulease

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNPULease_ExclusiveResidencyAndRefusal(t *testing.T) {
	fakeNow := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clock := func() time.Time { return fakeNow }

	mgr := NewNPULeaseManager(
		WithXCLBINSwapCost(250*time.Millisecond),
		WithNowFunc(clock),
	)

	// Initial acquire on cold accelerator
	l1, err := mgr.Acquire("session-alpha", "qwen2.5-1.5b-bfp16", 10*time.Second)
	if err != nil {
		t.Fatalf("first acquire failed: %v", err)
	}
	if !l1.Reloaded {
		t.Errorf("expected cold start to reload xclbin")
	}
	if l1.SwapCost != 250*time.Millisecond {
		t.Errorf("expected swap cost 250ms, got %v", l1.SwapCost)
	}

	// Advance time by 3 seconds
	fakeNow = fakeNow.Add(3 * time.Second)

	// Second session requests residency while first holds it -> ErrNPUBusy refusal
	_, err = mgr.Acquire("session-beta", "smollm2-135m-bfp16", 5*time.Second)
	if err == nil {
		t.Fatalf("expected second session request to be refused, got nil error")
	}

	var busy *ErrNPUBusy
	if !errors.As(err, &busy) {
		t.Fatalf("expected error to be *ErrNPUBusy, got %T (%v)", err, err)
	}
	if busy.HolderID != "session-alpha" {
		t.Errorf("expected HolderID 'session-alpha', got %q", busy.HolderID)
	}
	if busy.Remaining != 7*time.Second {
		t.Errorf("expected Remaining 7s, got %v", busy.Remaining)
	}
	if busy.ResidentModel != "qwen2.5-1.5b-bfp16" {
		t.Errorf("expected ResidentModel 'qwen2.5-1.5b-bfp16', got %q", busy.ResidentModel)
	}
	if !errors.Is(err, ErrNPUBusySentinel) {
		t.Errorf("expected errors.Is(err, ErrNPUBusySentinel) to be true")
	}

	// First session releases lease
	l1.Release()

	// Second session requests residency again; NPU is now free
	l2, err := mgr.Acquire("session-beta", "smollm2-135m-bfp16", 5*time.Second)
	if err != nil {
		t.Fatalf("acquire after release failed: %v", err)
	}
	if !l2.Reloaded {
		t.Errorf("expected reload because model changed from qwen to smollm2")
	}
	if l2.SwapCost != 250*time.Millisecond {
		t.Errorf("expected swap cost 250ms, got %v", l2.SwapCost)
	}

	l2.Release()

	// Third session requests the SAME model that is already resident
	l3, err := mgr.Acquire("session-gamma", "smollm2-135m-bfp16", 5*time.Second)
	if err != nil {
		t.Fatalf("acquire same model failed: %v", err)
	}
	if l3.Reloaded {
		t.Errorf("expected reload to be false for already-resident model")
	}
	if l3.SwapCost != 0 {
		t.Errorf("expected swap cost 0 for already-resident model, got %v", l3.SwapCost)
	}
	l3.Release()
}

func TestNPULease_TTLExpiration(t *testing.T) {
	fakeNow := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clock := func() time.Time { return fakeNow }

	mgr := NewNPULeaseManager(WithNowFunc(clock))

	_, err := mgr.Acquire("session-1", "model-a", 5*time.Second)
	if err != nil {
		t.Fatalf("session-1 acquire failed: %v", err)
	}

	// Advance time past TTL without calling Release()
	fakeNow = fakeNow.Add(6 * time.Second)

	// Session 2 should acquire successfully since lease expired
	l2, err := mgr.Acquire("session-2", "model-b", 5*time.Second)
	if err != nil {
		t.Fatalf("session-2 acquire after TTL expiry failed: %v", err)
	}
	if l2.SessionID != "session-2" {
		t.Errorf("expected session-2 to hold lease, got %s", l2.SessionID)
	}
}

func TestNPULease_SameSessionRenewal(t *testing.T) {
	fakeNow := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clock := func() time.Time { return fakeNow }

	mgr := NewNPULeaseManager(WithNowFunc(clock))

	l1, err := mgr.Acquire("session-1", "model-a", 5*time.Second)
	if err != nil {
		t.Fatalf("initial acquire failed: %v", err)
	}

	fakeNow = fakeNow.Add(2 * time.Second)

	// Same session requests renewal
	l2, err := mgr.Acquire("session-1", "model-a", 10*time.Second)
	if err != nil {
		t.Fatalf("renewal failed: %v", err)
	}
	if l2 != l1 {
		t.Errorf("expected same lease instance on renewal")
	}
	expectedExpiry := fakeNow.Add(10 * time.Second)
	if !l2.ExpiresAt.Equal(expectedExpiry) {
		t.Errorf("expected expiry %v, got %v", expectedExpiry, l2.ExpiresAt)
	}
}

func TestNPULease_ConcurrentContention(t *testing.T) {
	mgr := NewNPULeaseManager(WithXCLBINSwapCost(50 * time.Millisecond))

	const numWorkers = 20
	var successCount int64
	var busyCount int64
	var wg sync.WaitGroup

	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go func(id int) {
			defer wg.Done()
			sess := "session-" + string(rune('A'+id))
			l, err := mgr.Acquire(sess, "shared-model", 200*time.Millisecond)
			if err == nil {
				atomic.AddInt64(&successCount, 1)
				time.Sleep(10 * time.Millisecond)
				l.Release()
			} else {
				var busy *ErrNPUBusy
				if errors.As(err, &busy) {
					atomic.AddInt64(&busyCount, 1)
				}
			}
		}(i)
	}

	wg.Wait()

	if successCount == 0 {
		t.Fatalf("expected at least 1 successful acquisition")
	}
	if successCount+busyCount != numWorkers {
		t.Fatalf("expected total outcomes to equal worker count (%d), got success=%d busy=%d",
			numWorkers, successCount, busyCount)
	}
}

func TestNPULease_SameSessionModelSwap(t *testing.T) {
	fakeNow := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clock := func() time.Time { return fakeNow }

	mgr := NewNPULeaseManager(
		WithXCLBINSwapCost(300*time.Millisecond),
		WithNowFunc(clock),
	)

	// Acquire model-a
	l1, err := mgr.Acquire("session-1", "model-a", 10*time.Second)
	if err != nil {
		t.Fatalf("acquire model-a failed: %v", err)
	}
	if !l1.Reloaded || l1.SwapCost != 300*time.Millisecond {
		t.Errorf("expected cold reload for model-a")
	}

	// Advance 2s
	fakeNow = fakeNow.Add(2 * time.Second)

	// Same session requests model-b while holding lease -> swaps model with swap cost
	l2, err := mgr.Acquire("session-1", "model-b", 5*time.Second)
	if err != nil {
		t.Fatalf("swap to model-b failed: %v", err)
	}
	if l2.ModelID != "model-b" {
		t.Errorf("expected model-b, got %s", l2.ModelID)
	}
	if !l2.Reloaded {
		t.Errorf("expected reloaded=true on model swap")
	}
	if l2.SwapCost != 300*time.Millisecond {
		t.Errorf("expected swap cost 300ms, got %v", l2.SwapCost)
	}

	resModel, holder, remaining := mgr.CurrentResident()
	if resModel != "model-b" {
		t.Errorf("expected resident model-b, got %s", resModel)
	}
	if holder != "session-1" {
		t.Errorf("expected holder session-1, got %s", holder)
	}
	if remaining != 5*time.Second {
		t.Errorf("expected 5s remaining, got %v", remaining)
	}
}

func TestNPULease_CurrentResidentReporting(t *testing.T) {
	fakeNow := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clock := func() time.Time { return fakeNow }

	mgr := NewNPULeaseManager(WithNowFunc(clock))

	// Cold initial state
	model, holder, rem := mgr.CurrentResident()
	if model != "" || holder != "" || rem != 0 {
		t.Errorf("expected empty cold state, got model=%q holder=%q rem=%v", model, holder, rem)
	}

	// Acquire lease
	l, err := mgr.Acquire("sess-1", "resident-model", 8*time.Second)
	if err != nil {
		t.Fatal(err)
	}

	model, holder, rem = mgr.CurrentResident()
	if model != "resident-model" || holder != "sess-1" || rem != 8*time.Second {
		t.Errorf("expected active lease state, got model=%q holder=%q rem=%v", model, holder, rem)
	}

	// Release lease: model remains resident, but holder is empty and rem=0
	l.Release()
	model, holder, rem = mgr.CurrentResident()
	if model != "resident-model" || holder != "" || rem != 0 {
		t.Errorf("expected released resident state, got model=%q holder=%q rem=%v", model, holder, rem)
	}

	// Double release is a safe no-op
	l.Release()

	// Nil lease release is safe
	var nilLease *NPULease
	nilLease.Release()
}

func TestNPULease_InvalidArguments(t *testing.T) {
	mgr := NewNPULeaseManager()

	if _, err := mgr.Acquire("", "model-1", time.Second); err == nil {
		t.Errorf("expected error on empty session ID")
	}
	if _, err := mgr.Acquire("sess-1", "", time.Second); err == nil {
		t.Errorf("expected error on empty model ID")
	}
	if _, err := mgr.Acquire("sess-1", "model-1", 0); err == nil {
		t.Errorf("expected error on zero TTL")
	}
	if _, err := mgr.Acquire("sess-1", "model-1", -time.Second); err == nil {
		t.Errorf("expected error on negative TTL")
	}

	var nilMgr *NPULeaseManager
	if _, err := nilMgr.Acquire("sess-1", "model-1", time.Second); err == nil {
		t.Errorf("expected error on nil manager")
	}
	if m, h, r := nilMgr.CurrentResident(); m != "" || h != "" || r != 0 {
		t.Errorf("expected zero values on nil manager CurrentResident")
	}
}

func TestNPULease_ErrNPUBusyMethods(t *testing.T) {
	var nilErr *ErrNPUBusy
	if nilErr.Error() != "gpulease: NPU is busy" {
		t.Errorf("unexpected error string for nil ErrNPUBusy: %s", nilErr.Error())
	}
	if nilErr.Is(errors.New("other")) {
		t.Errorf("nil ErrNPUBusy should not match non-sentinel error")
	}

	err := &ErrNPUBusy{
		HolderID:       "sess-x",
		Remaining:      3 * time.Second,
		ResidentModel:  "model-x",
		RequestedModel: "model-y",
	}

	if !errors.Is(err, ErrNPUBusySentinel) {
		t.Errorf("expected errors.Is with ErrNPUBusySentinel to return true")
	}
	if !errors.Is(err, &ErrNPUBusy{HolderID: "sess-x"}) {
		t.Errorf("expected errors.Is matching holder ID")
	}
	if errors.Is(err, &ErrNPUBusy{HolderID: "sess-diff"}) {
		t.Errorf("expected errors.Is mismatch on different holder ID")
	}
	if err.Unwrap() != ErrNPUBusySentinel {
		t.Errorf("Unwrap did not return ErrNPUBusySentinel")
	}
}
