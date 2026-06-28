package bgloop

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Default backoff bounds for a failing loop: the first retry waits backoffMin, then
// doubles each consecutive failure up to backoffMax. A clean tick resets it.
const (
	defaultBackoffMin = time.Second
	defaultBackoffMax = time.Minute
)

// Supervisor runs a set of Loops on the kernel's lifecycle. Construct it with New,
// Register every loop BEFORE Start, then Start(ctx) with the serve lifecycle context.
// It spawns one goroutine per loop, supervises panics and errors with backoff, and
// exposes the live progress as a Snapshot. Shutdown cancels every loop and joins the
// goroutines within a deadline. All methods are safe for concurrent use.
type Supervisor struct {
	backoffMin time.Duration
	backoffMax time.Duration
	admit      func(name string) (ok bool, reason string)
	observer   func(Status)

	mu      sync.Mutex
	loops   []*loopState
	byName  map[string]*loopState
	started bool
	cancel  context.CancelFunc
	wg      sync.WaitGroup
}

// Option configures a Supervisor at construction.
type Option func(*Supervisor)

// WithBackoff sets the failure backoff bounds (first retry waits min, doubling up to
// max). Non-positive or inverted bounds are ignored in favor of the defaults.
func WithBackoff(min, max time.Duration) Option {
	return func(s *Supervisor) {
		if min > 0 && max >= min {
			s.backoffMin, s.backoffMax = min, max
		}
	}
}

// WithObserver installs a callback invoked with a loop's Status after each completed
// tick (and each refused fire). It is the PUSH seam a host uses to fold in-kernel
// loop activity into an external surface such as the loopmgr ledger. It must not
// block; it runs on the loop's own goroutine. Metrics do not need it — read Snapshot
// at scrape time instead.
func WithObserver(fn func(Status)) Option {
	return func(s *Supervisor) { s.observer = fn }
}

// WithAdmit installs an admission gate consulted before every fire. Returning false
// holds the fire (StatePaused) with the given reason and re-checks on the next
// interval — the BACKPRESSURE seam a host wires to loopmgr.Governor.Admit so an
// operator can pause, disable, or rate-floor a loop the kernel runs.
func WithAdmit(fn func(name string) (ok bool, reason string)) Option {
	return func(s *Supervisor) { s.admit = fn }
}

// New returns an empty Supervisor with default backoff bounds.
func New(opts ...Option) *Supervisor {
	s := &Supervisor{
		backoffMin: defaultBackoffMin,
		backoffMax: defaultBackoffMax,
		byName:     map[string]*loopState{},
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Register adds a loop. It must be called before Start; a name must be non-empty and
// unique, and Tick must be non-nil. Returns an error otherwise (the loop is rejected,
// the supervisor is unchanged).
func (s *Supervisor) Register(l Loop) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return errors.New("bgloop: Register called after Start")
	}
	name := strings.TrimSpace(l.Name)
	if name == "" {
		return errors.New("bgloop: loop name is required")
	}
	if l.Tick == nil {
		return fmt.Errorf("bgloop: loop %q has a nil Tick", name)
	}
	if _, dup := s.byName[name]; dup {
		return fmt.Errorf("bgloop: duplicate loop name %q", name)
	}
	ls := &loopState{name: name, interval: l.Interval, tick: l.Tick, state: StateIdle}
	s.byName[name] = ls
	s.loops = append(s.loops, ls)
	return nil
}

// Start launches every registered loop on a context derived from ctx, then returns
// immediately (it does not block). It is idempotent — a second call is a no-op. After
// Start, Register is refused. Cancelling ctx, or calling Shutdown, stops every loop.
func (s *Supervisor) Start(ctx context.Context) {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return
	}
	s.started = true
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	loops := append([]*loopState(nil), s.loops...)
	s.mu.Unlock()

	for _, ls := range loops {
		s.wg.Add(1)
		go s.run(runCtx, ls)
	}
}

// Shutdown cancels every loop and waits for the goroutines to exit, up to ctx's
// deadline. It returns nil once all loops are joined, or a timeout error naming the
// loops still running if a Tick ignored cancellation past the deadline. Safe to call
// before Start (a no-op) and more than once.
func (s *Supervisor) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	cancel := s.cancel
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}

	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		stuck := s.runningLoops()
		if len(stuck) == 0 {
			return nil
		}
		return fmt.Errorf("bgloop: shutdown timed out, loop(s) still running: %s: %w",
			strings.Join(stuck, ", "), ctx.Err())
	}
}

// Snapshot returns the live Status of every loop, sorted by name. Safe to call any
// time (before Start it reports the registered loops as idle with zero counters).
func (s *Supervisor) Snapshot() []Status {
	s.mu.Lock()
	loops := append([]*loopState(nil), s.loops...)
	s.mu.Unlock()
	out := make([]Status, 0, len(loops))
	for _, ls := range loops {
		out = append(out, ls.snapshot())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Get returns one loop's Status by name.
func (s *Supervisor) Get(name string) (Status, bool) {
	s.mu.Lock()
	ls := s.byName[name]
	s.mu.Unlock()
	if ls == nil {
		return Status{}, false
	}
	return ls.snapshot(), true
}

// Len is the number of registered loops.
func (s *Supervisor) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.loops)
}

// run is one loop's supervised goroutine. It alternates tick / wait, recovering a
// panic or error into backoff, until the lifecycle context is cancelled.
func (s *Supervisor) run(ctx context.Context, ls *loopState) {
	defer s.wg.Done()
	ls.begin()
	backoff := s.backoffMin

	for {
		if ctx.Err() != nil {
			ls.setState(StateStopped)
			return
		}

		// Operator backpressure: a refused fire holds and re-checks next interval.
		if s.admit != nil {
			if ok, _ := s.admit(ls.name); !ok {
				ls.markPaused()
				s.fire(ls)
				if !sleepCtx(ctx, pollWait(ls.interval, s.backoffMin)) {
					ls.setState(StateStopped)
					return
				}
				continue
			}
		}

		ls.markRunning()
		start := time.Now()
		err, panicked := safeTick(ctx, ls.tick)
		ls.recordTick(err, panicked, time.Since(start))
		s.fire(ls)

		// Cancellation during the tick wins over backoff: stop cleanly rather than
		// scheduling a retry the kernel is about to tear down.
		if ctx.Err() != nil {
			ls.setState(StateStopped)
			return
		}

		if err != nil || panicked {
			wait := backoff
			ls.markBackoff(time.Now().Add(wait))
			backoff = nextBackoff(backoff, s.backoffMax)
			if !sleepCtx(ctx, wait) {
				ls.setState(StateStopped)
				return
			}
			continue
		}

		// Clean tick: reset backoff and wait out the interval.
		backoff = s.backoffMin
		if ls.interval <= 0 {
			ls.markIdle(time.Time{})
			continue // continuous: the Tick paces itself
		}
		ls.markIdle(time.Now().Add(ls.interval))
		if !sleepCtx(ctx, ls.interval) {
			ls.setState(StateStopped)
			return
		}
	}
}

// fire pushes the loop's current Status to the observer, if one is installed.
func (s *Supervisor) fire(ls *loopState) {
	if s.observer == nil {
		return
	}
	s.observer(ls.snapshot())
}

// runningLoops returns the names of loops not yet in StateStopped — the diagnostic
// payload of a Shutdown timeout.
func (s *Supervisor) runningLoops() []string {
	s.mu.Lock()
	loops := append([]*loopState(nil), s.loops...)
	s.mu.Unlock()
	var out []string
	for _, ls := range loops {
		if st := ls.snapshot(); st.State != StateStopped {
			out = append(out, st.Name)
		}
	}
	sort.Strings(out)
	return out
}

// safeTick runs one tick, converting a panic into an error plus a panicked flag so a
// crashing loop is contained, never fatal to the kernel.
func safeTick(ctx context.Context, tick func(context.Context) error) (err error, panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	return tick(ctx), false
}

// sleepCtx waits for d or until ctx is cancelled. It returns true if the full
// duration elapsed, false if ctx was cancelled first (the loop should stop).
func sleepCtx(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// nextBackoff doubles cur, capped at max.
func nextBackoff(cur, max time.Duration) time.Duration {
	next := cur * 2
	if next > max {
		return max
	}
	return next
}

// pollWait is how long a paused loop holds before re-checking the admit gate: its own
// interval when set, else the backoff floor.
func pollWait(interval, floor time.Duration) time.Duration {
	if interval > 0 {
		return interval
	}
	return floor
}
