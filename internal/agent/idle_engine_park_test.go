package agent

// Reproduction/regression pin for the idle-engine park lifecycle adapted from
// ashhart/TensorFold zig/src/server/idle_unload.zig @9e8c368b0b4c2c7892a0c9f0f302d7c164016209:
// parks only at zero active; concurrent wakes coalesce into one reload; failed
// opens stay parked and retryable; multi-rank GLM CUDA is refused. All engine
// I/O is faked; no hardware or model is involved.

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms lane=default
func TestIdleEngineUnloadCompletionWakesAndPreservesStop(t *testing.T) {
	for _, stop := range []bool{false, true} {
		for _, fail := range []bool{false, true} {
			t.Run(fmt.Sprintf("stop=%v/fail=%v", stop, fail), func(t *testing.T) {
				started, release := make(chan struct{}), make(chan struct{})
				var once sync.Once
				unblock := func() { once.Do(func() { close(release) }) }
				closeFailure := errors.New("injected close failure")
				opens := 0
				now := time.Unix(0, 0)
				m := NewIdleEngineManager(now, func() error { opens++; return nil }, func() error {
					close(started)
					<-release
					if fail {
						return closeFailure
					}
					return nil
				})
				t.Cleanup(func() { m.Stop(); unblock() })
				tickDone, wakeDone := make(chan error, 1), make(chan error, 1)
				go func() { tickDone <- m.Tick(now.Add(time.Second), time.Second) }()
				select {
				case <-started:
				case <-time.After(time.Second):
					t.Fatal("close did not start")
				}
				go func() { wakeDone <- m.Wake(now.Add(time.Second)) }()
				deadline := time.Now().Add(time.Second)
				for m.Waiting() != 1 && time.Now().Before(deadline) {
					time.Sleep(time.Millisecond)
				}
				if m.Waiting() != 1 {
					t.Fatal("wake did not join unloading")
				}
				if stop {
					m.Stop()
				}
				unblock()
				select {
				case err := <-tickDone:
					if (fail && !errors.Is(err, closeFailure)) || (!fail && err != nil) {
						t.Fatalf("close error changed: %v", err)
					}
				case <-time.After(time.Second):
					t.Fatal("unload did not finish")
				}
				select {
				case err := <-wakeDone:
					if (stop && !errors.Is(err, ErrIdleEngineStopped)) || (!stop && err != nil) {
						t.Fatalf("wake result: %v", err)
					}
				case <-time.After(time.Second):
					t.Fatal("unload completion left wake blocked")
				}
				if stop {
					if m.State() != IdleEngineStopped || m.Active() != 0 || opens != 0 {
						t.Fatal("unload completion resurrected stopped engine")
					}
				} else {
					wantOpens := 1
					if fail {
						wantOpens = 0
					}
					if m.State() != IdleEngineResident || m.Active() != 1 || opens != wantOpens {
						t.Fatal("wake did not acquire the correct resident engine")
					}
					m.Release(now.Add(time.Second))
				}
			})
		}
	}
}

// fakeEngine counts open/shut calls and can fail or block its open so the
// lifecycle's failure and coalescing paths are deterministic.
type fakeEngine struct {
	mu         sync.Mutex
	opens      int
	shuts      int
	openErr    error
	gate       chan struct{} // when non-nil, open blocks until it closes
	openSignal chan struct{} // when non-nil, closed once when an open starts
}

func (f *fakeEngine) open() error {
	f.mu.Lock()
	f.opens++
	err := f.openErr
	gate := f.gate
	sig := f.openSignal
	f.mu.Unlock()
	if sig != nil {
		close(sig)
	}
	if gate != nil {
		<-gate
	}
	return err
}

func (f *fakeEngine) shut() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.shuts++
	return nil
}

func (f *fakeEngine) counts() (opens, shuts int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.opens, f.shuts
}

const parkIdle = 30 * time.Second

var errFakeOpen = errors.New("fake open failed")

func TestIdleEngineParkLifecycle(t *testing.T) {
	t0 := time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC)

	t.Run("parks only at zero active", func(t *testing.T) {
		fake := &fakeEngine{}
		m := NewIdleEngineManager(t0, fake.open, fake.shut)
		if got := m.State(); got != IdleEngineResident {
			t.Fatalf("new manager state = %v, want resident", got)
		}

		// Active work prevents the park.
		if err := m.Wake(t0); err != nil {
			t.Fatalf("Wake(resident) = %v, want nil", err)
		}
		if got := m.Active(); got != 1 {
			t.Fatalf("Active after wake = %d, want 1", got)
		}
		if err := m.Tick(t0.Add(2*parkIdle), parkIdle); err != nil {
			t.Fatalf("Tick with active work = %v, want nil", err)
		}
		if got := m.State(); got != IdleEngineResident {
			t.Fatalf("state under active work = %v, want resident", got)
		}

		// Its end restarts the window: half a quiet span after the release
		// does not park yet...
		m.Release(t0.Add(parkIdle))
		if err := m.Tick(t0.Add(parkIdle+parkIdle/2), parkIdle); err != nil {
			t.Fatalf("Tick within restarted window = %v, want nil", err)
		}
		if got := m.State(); got != IdleEngineResident {
			t.Fatalf("state within restarted window = %v, want resident", got)
		}
		// ...but a full quiet span after the release parks.
		if err := m.Tick(t0.Add(2*parkIdle), parkIdle); err != nil {
			t.Fatalf("Tick past idle deadline = %v, want nil", err)
		}
		if got := m.State(); got != IdleEngineUnloaded {
			t.Fatalf("state after park = %v, want unloaded", got)
		}
		if opens, shuts := fake.counts(); opens != 0 || shuts != 1 {
			t.Fatalf("counts after park = (%d opens, %d shuts), want (0, 1)", opens, shuts)
		}
	})

	t.Run("concurrent wakes coalesce into one reload", func(t *testing.T) {
		fake := &fakeEngine{}
		m := NewIdleEngineManager(t0, fake.open, fake.shut)
		if err := m.Tick(t0.Add(parkIdle), parkIdle); err != nil {
			t.Fatalf("park setup = %v", err)
		}

		const wakers = 4
		start := make(chan struct{})
		errs := make([]error, wakers)
		var wg sync.WaitGroup
		for i := range errs {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				<-start
				errs[i] = m.Wake(t0.Add(2 * parkIdle))
			}(i)
		}
		close(start)
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Fatalf("waker %d Wake = %v, want nil", i, err)
			}
		}
		if got := m.State(); got != IdleEngineResident {
			t.Fatalf("state after coalesced reload = %v, want resident", got)
		}
		if got := m.Active(); got != wakers {
			t.Fatalf("Active after coalesced reload = %d, want %d", got, wakers)
		}
		if opens, _ := fake.counts(); opens != 1 {
			t.Fatalf("opens after coalesced reload = %d, want 1", opens)
		}
		for range errs {
			m.Release(t0.Add(2 * parkIdle))
		}
	})

	t.Run("failed open stays parked and retryable", func(t *testing.T) {
		fake := &fakeEngine{openErr: errFakeOpen}
		m := NewIdleEngineManager(t0, fake.open, fake.shut)
		if err := m.Tick(t0.Add(parkIdle), parkIdle); err != nil {
			t.Fatalf("park setup = %v", err)
		}

		// A blocked failing open: the joining waker gets the same failure and
		// triggers no extra open. The loader parks on the gate; the joiner is
		// admitted only once observed blocked behind the load, so exactly one
		// open happens regardless of scheduling.
		gate := make(chan struct{})
		started := make(chan struct{})
		fake.mu.Lock()
		fake.gate = gate
		fake.openSignal = started
		fake.mu.Unlock()
		errs := make([]error, 2)
		var wg sync.WaitGroup
		for i := range errs {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				errs[i] = m.Wake(t0.Add(2 * parkIdle))
			}(i)
		}
		<-started // the loader is inside the blocked open
		deadline := time.Now().Add(5 * time.Second)
		for m.Waiting() == 0 && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond) // wait for the second waker to join
		}
		if m.Waiting() != 1 {
			t.Fatalf("Waiting = %d, want 1: the second waker must join the in-flight load", m.Waiting())
		}
		close(gate)
		wg.Wait()
		for i, err := range errs {
			if !errors.Is(err, ErrIdleEngineOpenFailed) || !errors.Is(err, errFakeOpen) {
				t.Fatalf("waker %d error = %v, want ErrIdleEngineOpenFailed wrapping errFakeOpen", i, err)
			}
		}
		if got := m.State(); got != IdleEngineUnloaded {
			t.Fatalf("state after failed open = %v, want unloaded (still parked)", got)
		}
		if opens, _ := fake.counts(); opens != 1 {
			t.Fatalf("opens across coalesced failure = %d, want 1", opens)
		}

		// The next fresh wake retries and succeeds: parked servers stay
		// retryable.
		fake.mu.Lock()
		fake.openErr = nil
		fake.gate = nil
		fake.openSignal = nil
		fake.mu.Unlock()
		if err := m.Wake(t0.Add(3 * parkIdle)); err != nil {
			t.Fatalf("retry Wake = %v, want nil", err)
		}
		if got := m.State(); got != IdleEngineResident {
			t.Fatalf("state after retry = %v, want resident", got)
		}
		if opens, _ := fake.counts(); opens != 2 {
			t.Fatalf("opens after retry = %d, want 2", opens)
		}
		m.Release(t0.Add(3 * parkIdle))
	})

	t.Run("stop is terminal", func(t *testing.T) {
		fake := &fakeEngine{}
		m := NewIdleEngineManager(t0, fake.open, fake.shut)
		if err := m.Tick(t0.Add(parkIdle), parkIdle); err != nil {
			t.Fatalf("park setup = %v", err)
		}
		m.Stop()
		if got := m.State(); got != IdleEngineStopped {
			t.Fatalf("state after Stop = %v, want stopped", got)
		}
		if err := m.Wake(t0.Add(2 * parkIdle)); !errors.Is(err, ErrIdleEngineStopped) {
			t.Fatalf("Wake after Stop = %v, want ErrIdleEngineStopped", err)
		}
		if opens, _ := fake.counts(); opens != 0 {
			t.Fatalf("opens after Stop = %d, want 0", opens)
		}
	})

	t.Run("stop wins over an in-flight load", func(t *testing.T) {
		fake := &fakeEngine{}
		m := NewIdleEngineManager(t0, fake.open, fake.shut)
		if err := m.Tick(t0.Add(parkIdle), parkIdle); err != nil {
			t.Fatalf("park setup = %v", err)
		}
		gate := make(chan struct{})
		started := make(chan struct{})
		fake.mu.Lock()
		fake.gate = gate
		fake.openSignal = started
		fake.mu.Unlock()
		done := make(chan error, 1)
		go func() { done <- m.Wake(t0.Add(2 * parkIdle)) }()
		<-started // the loader is inside the blocked open
		m.Stop()
		close(gate)
		if err := <-done; !errors.Is(err, ErrIdleEngineStopped) {
			t.Fatalf("Wake racing Stop = %v, want ErrIdleEngineStopped", err)
		}
		if got := m.State(); got != IdleEngineStopped {
			t.Fatalf("state after racing Stop = %v, want stopped", got)
		}
		if opens, shuts := fake.counts(); opens != 1 || shuts != 2 {
			t.Fatalf("counts after racing Stop = (%d opens, %d shuts), want (1, 2): the loaded engine is dropped", opens, shuts)
		}
	})

	t.Run("failed close keeps the engine resident", func(t *testing.T) {
		fake := &fakeEngine{}
		shutErr := errors.New("fake shut failed")
		shut := func() error { return shutErr }
		m := NewIdleEngineManager(t0, fake.open, shut)
		if err := m.Wake(t0); err != nil {
			t.Fatalf("Wake = %v", err)
		}
		m.Release(t0.Add(parkIdle))
		if err := m.Tick(t0.Add(2*parkIdle), parkIdle); !errors.Is(err, shutErr) {
			t.Fatalf("Tick with failing close = %v, want the close error", err)
		}
		if got := m.State(); got != IdleEngineResident {
			t.Fatalf("state after failed close = %v, want resident", got)
		}
		if err := m.Wake(t0.Add(2 * parkIdle)); err != nil {
			t.Fatalf("Wake after failed close = %v, want nil", err)
		}
		m.Release(t0.Add(2 * parkIdle))
	})
}

func TestIdleUnloadRefusedPolicy(t *testing.T) {
	t.Run("multi-rank GLM on CUDA is refused", func(t *testing.T) {
		err := IdleUnloadRefused([]string{"cuda"}, "GLM", 4)
		if !errors.Is(err, ErrIdleUnloadRefused) {
			t.Fatalf("IdleUnloadRefused(cuda, GLM, 4) = %v, want ErrIdleUnloadRefused", err)
		}
	})

	cases := []struct {
		name     string
		backends []string
		model    string
		world    int
	}{
		{"single rank", []string{"cuda"}, "glm", 1},
		{"non-GLM multi-rank", []string{"cuda"}, "qwen", 4},
		{"GLM without CUDA", []string{"metal"}, "glm", 4},
		{"no backends", nil, "glm", 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := IdleUnloadRefused(tc.backends, tc.model, tc.world); err != nil {
				t.Fatalf("IdleUnloadRefused(%v, %q, %d) = %v, want nil", tc.backends, tc.model, tc.world, err)
			}
		})
	}
}
