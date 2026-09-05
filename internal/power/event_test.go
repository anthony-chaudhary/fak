package power

import (
	"context"
	"sync"
	"testing"
	"time"
)

type testObserver struct {
	mu           sync.Mutex
	suspends     []PowerEvent
	resumes      []PowerEvent
	suspendNotif chan struct{}
	resumeNotif  chan struct{}
}

func newTestObserver() *testObserver {
	return &testObserver{
		suspendNotif: make(chan struct{}, 10),
		resumeNotif:  make(chan struct{}, 10),
	}
}

func (o *testObserver) OnSuspend(event PowerEvent) {
	o.mu.Lock()
	o.suspends = append(o.suspends, event)
	o.mu.Unlock()
	select {
	case o.suspendNotif <- struct{}{}:
	default:
	}
}

func (o *testObserver) OnResume(event PowerEvent) {
	o.mu.Lock()
	o.resumes = append(o.resumes, event)
	o.mu.Unlock()
	select {
	case o.resumeNotif <- struct{}{}:
	default:
	}
}

func (o *testObserver) SuspendCount() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return len(o.suspends)
}

func (o *testObserver) ResumeCount() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return len(o.resumes)
}

func TestBroadcasterAndObservers(t *testing.T) {
	b := NewPowerBroadcaster()

	obs1 := newTestObserver()
	obs2 := newTestObserver()

	var funcCalls []PowerEvent
	var funcMu sync.Mutex

	cancel1 := b.RegisterObserver(obs1)
	cancel2 := b.RegisterObserver(obs2)
	cancelFunc := b.RegisterFunc(func(e PowerEvent) {
		funcMu.Lock()
		funcCalls = append(funcCalls, e)
		funcMu.Unlock()
	})

	if count := b.ObserverCount(); count != 3 {
		t.Fatalf("expected 3 observers/funcs, got %d", count)
	}

	sleepEvent := PowerEvent{
		Type:      EventSleep,
		Timestamp: time.Now(),
		Source:    "test",
		Details:   "system going down",
	}

	b.Broadcast(sleepEvent)

	if obs1.SuspendCount() != 1 || obs2.SuspendCount() != 1 {
		t.Fatalf("expected 1 suspend each, got %d and %d", obs1.SuspendCount(), obs2.SuspendCount())
	}
	funcMu.Lock()
	if len(funcCalls) != 1 || funcCalls[0].Type != EventSleep {
		t.Fatalf("expected 1 sleep func call, got %d", len(funcCalls))
	}
	funcMu.Unlock()

	// Cancel obs1 and func, then broadcast Wake
	cancel1()
	cancelFunc()

	wakeEvent := PowerEvent{
		Type:      EventWake,
		Timestamp: time.Now(),
		Source:    "test",
		Details:   "system resumed",
	}
	b.Broadcast(wakeEvent)

	if obs1.ResumeCount() != 0 {
		t.Fatalf("obs1 was cancelled, expected 0 resumes, got %d", obs1.ResumeCount())
	}
	if obs2.ResumeCount() != 1 {
		t.Fatalf("obs2 active, expected 1 resume, got %d", obs2.ResumeCount())
	}

	funcMu.Lock()
	if len(funcCalls) != 1 {
		t.Fatalf("func was cancelled, expected still 1 call, got %d", len(funcCalls))
	}
	funcMu.Unlock()

	cancel2()
	if count := b.ObserverCount(); count != 0 {
		t.Fatalf("expected 0 observers after all cancels, got %d", count)
	}
}

func TestLeaseFreezeCoordinatorTransitions(t *testing.T) {
	b := NewPowerBroadcaster()
	coord := NewLeaseFreezeCoordinator(b)
	defer coord.Close()

	if coord.State() != StateActive {
		t.Fatalf("expected StateActive initially, got %s", coord.State())
	}
	if coord.IsFrozen() {
		t.Fatal("expected IsFrozen() to be false initially")
	}
	if coord.NeedsReadjudication() {
		t.Fatal("expected NeedsReadjudication() to be false initially")
	}

	var freezeHooks []PowerEvent
	var thawHooks []PowerEvent
	var hookMu sync.Mutex

	coord.OnFreeze(func(e PowerEvent) {
		hookMu.Lock()
		freezeHooks = append(freezeHooks, e)
		hookMu.Unlock()
	})
	coord.OnThaw(func(e PowerEvent) {
		hookMu.Lock()
		thawHooks = append(thawHooks, e)
		hookMu.Unlock()
	})

	sleepTime := time.Now().Add(-100 * time.Millisecond)
	b.Broadcast(PowerEvent{
		Type:      EventSleep,
		Timestamp: sleepTime,
		Source:    "test-sleep",
		Details:   "going to sleep",
	})

	if coord.State() != StateSuspendedFreeze {
		t.Fatalf("expected StateSuspendedFreeze after sleep, got %s", coord.State())
	}
	if !coord.IsFrozen() {
		t.Fatal("expected IsFrozen() to be true")
	}
	if coord.NeedsReadjudication() {
		t.Fatal("expected NeedsReadjudication() to be false while sleeping")
	}
	if coord.SuspendCount() != 1 {
		t.Fatalf("expected SuspendCount=1, got %d", coord.SuspendCount())
	}
	if dur := coord.FrozenDuration(); dur < 50*time.Millisecond {
		t.Fatalf("expected FrozenDuration >= 50ms, got %v", dur)
	}

	hookMu.Lock()
	if len(freezeHooks) != 1 || freezeHooks[0].Source != "test-sleep" {
		t.Fatalf("expected 1 freeze hook call, got %d", len(freezeHooks))
	}
	if len(thawHooks) != 0 {
		t.Fatalf("expected 0 thaw hook calls during freeze, got %d", len(thawHooks))
	}
	hookMu.Unlock()

	wakeTime := time.Now()
	b.Broadcast(PowerEvent{
		Type:      EventWake,
		Timestamp: wakeTime,
		Source:    "test-wake",
		Details:   "woke up",
	})

	if coord.State() != StateNeedsReadjudication {
		t.Fatalf("expected StateNeedsReadjudication after wake, got %s", coord.State())
	}
	if coord.IsFrozen() {
		t.Fatal("expected IsFrozen() to be false after wake")
	}
	if !coord.NeedsReadjudication() {
		t.Fatal("expected NeedsReadjudication() to be true after wake")
	}
	if coord.ResumeCount() != 1 {
		t.Fatalf("expected ResumeCount=1, got %d", coord.ResumeCount())
	}

	hookMu.Lock()
	if len(thawHooks) != 1 || thawHooks[0].Source != "test-wake" {
		t.Fatalf("expected 1 thaw hook call, got %d", len(thawHooks))
	}
	hookMu.Unlock()

	// Finish re-adjudication
	coord.MarkAdjudicated()

	if coord.State() != StateActive {
		t.Fatalf("expected StateActive after MarkAdjudicated, got %s", coord.State())
	}
	if coord.NeedsReadjudication() {
		t.Fatal("expected NeedsReadjudication() to be false after MarkAdjudicated")
	}
}

func TestObserverFunc(t *testing.T) {
	b := NewPowerBroadcaster()
	var suspended, resumed bool

	obs := ObserverFunc{
		SuspendFn: func(e PowerEvent) { suspended = true },
		ResumeFn:  func(e PowerEvent) { resumed = true },
	}

	cancel := b.RegisterObserver(obs)
	defer cancel()

	b.Broadcast(PowerEvent{Type: EventSleep})
	if !suspended {
		t.Fatal("expected suspend func to be invoked")
	}

	b.Broadcast(PowerEvent{Type: EventWake})
	if !resumed {
		t.Fatal("expected resume func to be invoked")
	}
}

func TestGlobalRegistrationAndBroadcast(t *testing.T) {
	var received []PowerEvent
	var mu sync.Mutex

	cancel := RegisterSleepFunc(func(e PowerEvent) {
		mu.Lock()
		received = append(received, e)
		mu.Unlock()
	})
	defer cancel()

	BroadcastEvent(PowerEvent{
		Type:    EventSleep,
		Source:  "global-test",
		Details: "broadcasting to global default",
	})

	mu.Lock()
	defer mu.Unlock()
	if len(received) != 1 || received[0].Source != "global-test" {
		t.Fatalf("expected 1 event with source 'global-test', got %v", received)
	}
}

func TestNoOpSleepListener(t *testing.T) {
	listener := newNoOpSleepListener()
	ctx := context.Background()

	if err := listener.Start(ctx); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	if err := listener.Stop(); err != nil {
		t.Fatalf("Stop failed: %v", err)
	}
}
