package bgloop

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Clock is the time boundary used by durable wake scheduling. Tests can advance
// it without sleeping; production uses WallClock.
type Clock interface {
	Now() time.Time
	After(time.Duration) <-chan time.Time
}

// WallClock is the production clock.
type WallClock struct{}

func (WallClock) Now() time.Time                         { return time.Now() }
func (WallClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// WakeFunc reconstitutes one persisted loop when its durable alarm becomes due.
type WakeFunc func(context.Context, Job) error

// DurableScheduler stores alarms in loopmgr's registry rather than holding a
// sleeping goroutine per loop. A new scheduler over the same registry re-arms
// every alarm after process restart.
type DurableScheduler struct {
	path  string
	clock Clock
	wake  WakeFunc
	store RegistryStore
	mu    sync.Mutex
}

func NewDurableScheduler(path string, clock Clock, store RegistryStore, wake WakeFunc) (*DurableScheduler, error) {
	if clock == nil {
		clock = WallClock{}
	}
	if store == nil {
		return nil, errors.New("durable registry store is required")
	}
	if wake == nil {
		return nil, errors.New("durable wake callback is required")
	}
	if err := store.Validate(path); err != nil {
		return nil, err
	}
	return &DurableScheduler{path: path, clock: clock, wake: wake, store: store}, nil
}

// SleepUntil atomically records an absolute wake deadline on an existing loop.
func (s *DurableScheduler) SleepUntil(jobID string, at time.Time, duty DutyCycle) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.store.SetWake(s.path, jobID, at, duty, s.clock.Now())
}

// Poll re-arms persisted alarms, firing every due alarm at most once. A due
// alarm outside its duty window is moved to the next on-window without firing.
func (s *DurableScheduler) Poll(ctx context.Context) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	wakes, err := s.store.ArmedWakes(s.path)
	if err != nil {
		return 0, err
	}
	now := s.clock.Now().UTC()
	fired := 0
	for _, wake := range wakes {
		if wake.WakeAt.IsZero() || now.Before(wake.WakeAt) {
			continue
		}
		if wake.Duty != nil && !wake.Duty.Active(now) {
			next, err := wake.Duty.NextOn(now)
			if err != nil {
				return fired, fmt.Errorf("job %q duty cycle: %w", wake.Job.JobID(), err)
			}
			if err := s.store.MoveWake(s.path, wake.Job.JobID(), next, now); err != nil {
				return fired, err
			}
			continue
		}
		if err := s.wake(ctx, wake.Job); err != nil {
			return fired, fmt.Errorf("wake job %q: %w", wake.Job.JobID(), err)
		}
		if err := s.store.ClearWake(s.path, wake.Job.JobID(), now); err != nil {
			return fired, err
		}
		fired++
	}
	return fired, nil
}

// Run waits only until the nearest persisted deadline, then reloads the
// registry. Registering alarms in another process therefore survives death and
// is observed on the next startup/poll rather than requiring a held sleeper.
func (s *DurableScheduler) Run(ctx context.Context) error {
	for {
		if _, err := s.Poll(ctx); err != nil {
			return err
		}
		wakes, err := s.store.ArmedWakes(s.path)
		if err != nil {
			return err
		}
		var next time.Time
		for _, wake := range wakes {
			if wake.WakeAt.IsZero() {
				continue
			}
			if next.IsZero() || wake.WakeAt.Before(next) {
				next = wake.WakeAt
			}
		}
		if next.IsZero() {
			return nil
		}
		d := next.Sub(s.clock.Now())
		if d < 0 {
			d = 0
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-s.clock.After(d):
		}
	}
}
