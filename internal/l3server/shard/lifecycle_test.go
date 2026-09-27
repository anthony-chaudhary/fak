package shard

import (
	"testing"
	"time"
)

// Shard and Manager Start/Stop must be idempotent (#13518). Before the fix a
// second Stop closed an already-closed quit channel, a second Start spawned a
// second run loop that closed done twice, and a Stop on a never-started shard
// left Done open forever. Stop is terminal: it releases the allocator, so a
// Start after Stop must not revive the shard.

func noPanic(t *testing.T, what string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s panicked: %v", what, r)
		}
	}()
	fn()
}

func waitDone(t *testing.T, what string, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("%s: shard goroutine did not exit", what)
	}
}

func newLifecycleShard(t *testing.T) *Shard {
	t.Helper()
	s, err := New(ShardConfig{
		ID:                0,
		IndexCapacity:     1024,
		MaxMemoryBytes:    8 << 20,
		EvictionPolicy:    "wtinylfu",
		DispatchTimeoutMs: 200,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func setOp(key string, hash uint64) ShardOp {
	return ShardOp{Type: OpSet, Key: []byte(key), KeyHash: hash, Value: []byte("v")}
}

func TestShardStopTwiceIsIdempotent(t *testing.T) {
	s := newLifecycleShard(t)
	s.Start()
	noPanic(t, "first Stop", s.Stop)
	waitDone(t, "first Stop", s.Done())
	noPanic(t, "second Stop", s.Stop)
	waitDone(t, "second Stop", s.Done())
}

func TestShardDuplicateStartRunsOneLoop(t *testing.T) {
	s := newLifecycleShard(t)
	noPanic(t, "Start", s.Start)
	noPanic(t, "duplicate Start", s.Start)
	if r := s.Submit(setOp("dup-start", 1)); r.Err != nil {
		t.Fatalf("Set on a started shard: %v", r.Err)
	}
	noPanic(t, "Stop", s.Stop)
	waitDone(t, "Stop", s.Done())
	// A second run loop would also observe quit and close done again, which
	// panics the process; give it time to reach that close.
	time.Sleep(200 * time.Millisecond)
}

func TestShardStartAfterStopIsTerminal(t *testing.T) {
	s := newLifecycleShard(t)
	s.Start()
	if r := s.Submit(setOp("before-stop", 1)); r.Err != nil {
		t.Fatalf("Set before Stop: %v", r.Err)
	}
	noPanic(t, "Stop", s.Stop)
	waitDone(t, "Stop", s.Done())

	// The allocator is released; a revived run loop would serve from it.
	noPanic(t, "Start after Stop", s.Start)
	if r := s.Submit(setOp("after-stop", 2)); r.Err == nil {
		t.Fatalf("Set after Stop succeeded; a stopped shard must not serve ops")
	}
	noPanic(t, "Stop after Start-after-Stop", s.Stop)
	waitDone(t, "Stop after Start-after-Stop", s.Done())
}

func TestShardStopBeforeStartReleases(t *testing.T) {
	s := newLifecycleShard(t)
	noPanic(t, "Stop before Start", s.Stop)
	waitDone(t, "Stop before Start", s.Done())
	noPanic(t, "Start after early Stop", s.Start)
	noPanic(t, "second Stop", s.Stop)
	waitDone(t, "second Stop", s.Done())
}

func newLifecycleManager(t *testing.T) *Manager {
	t.Helper()
	mgr, err := NewManager(ManagerConfig{
		NumShards:      2,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		IndexCapacity:  1024,
		Vacuum:         VacuumConfig{Enabled: true, IntervalSeconds: 3600},
	})
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	return mgr
}

func TestManagerStartStopIsIdempotent(t *testing.T) {
	mgr := newLifecycleManager(t)
	noPanic(t, "Start", mgr.Start)
	noPanic(t, "duplicate Start", mgr.Start)
	key := []byte("mgr-lifecycle")
	mgr.Set(key, []byte("v"), 0)
	if _, ok := mgr.Get(key); !ok {
		t.Fatal("Get after Start missed")
	}
	noPanic(t, "Stop", mgr.Stop)
	noPanic(t, "second Stop", mgr.Stop)
	noPanic(t, "Start after Stop", mgr.Start)
	noPanic(t, "Stop after Start-after-Stop", mgr.Stop)
}

func TestManagerStopWithoutStartCompletes(t *testing.T) {
	mgr := newLifecycleManager(t)
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		mgr.Stop()
	}()
	waitDone(t, "Manager.Stop without Start", stopped)
}
