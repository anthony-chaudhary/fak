package bgloop

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
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
type WakeFunc func(context.Context, loopmgr.Job) error

// DurableScheduler stores alarms in loopmgr's registry rather than holding a
// sleeping goroutine per loop. A new scheduler over the same registry re-arms
// every alarm after process restart.
type DurableScheduler struct {
	path  string
	clock Clock
	wake  WakeFunc
	mu    sync.Mutex
}

func NewDurableScheduler(path string, clock Clock, wake WakeFunc) (*DurableScheduler, error) {
	if clock == nil {
		clock = WallClock{}
	}
	if wake == nil {
		return nil, errors.New("durable wake callback is required")
	}
	if _, err := loopmgr.LoadRegistry(path); err != nil {
		return nil, err
	}
	return &DurableScheduler{path: path, clock: clock, wake: wake}, nil
}

// SleepUntil atomically records an absolute wake deadline on an existing loop.
func (s *DurableScheduler) SleepUntil(jobID string, at time.Time, duty *loopmgr.DutyCycle) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := loopmgr.LoadRegistry(s.path)
	if err != nil {
		return err
	}
	job, ok := r.Get(jobID)
	if !ok {
		return fmt.Errorf("loop registry has no job %q", jobID)
	}
	job.WakeAtUnixNano = at.UTC().UnixNano()
	job.Duty = duty
	if err := r.Put(job, s.clock.Now()); err != nil {
		return err
	}
	return loopmgr.SaveRegistry(s.path, r)
}

// Poll re-arms persisted alarms, firing every due alarm at most once. A due
// alarm outside its duty window is moved to the next on-window without firing.
func (s *DurableScheduler) Poll(ctx context.Context) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := loopmgr.LoadRegistry(s.path)
	if err != nil {
		return 0, err
	}
	now := s.clock.Now().UTC()
	fired := 0
	changed := false
	for _, job := range r.ArmedJobs() {
		if job.WakeAtUnixNano == 0 || now.Before(time.Unix(0, job.WakeAtUnixNano)) {
			continue
		}
		if job.Duty != nil && !job.Duty.Active(now) {
			next, err := job.Duty.NextOn(now)
			if err != nil {
				return fired, fmt.Errorf("job %q duty cycle: %w", job.JobID(), err)
			}
			job.WakeAtUnixNano = next.UnixNano()
			if err := r.Put(job, now); err != nil {
				return fired, err
			}
			changed = true
			continue
		}
		if err := s.wake(ctx, job); err != nil {
			return fired, fmt.Errorf("wake job %q: %w", job.JobID(), err)
		}
		job.WakeAtUnixNano = 0
		if err := r.Put(job, now); err != nil {
			return fired, err
		}
		fired++
		changed = true
	}
	if changed {
		if err := loopmgr.SaveRegistry(s.path, r); err != nil {
			return fired, err
		}
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
		r, err := loopmgr.LoadRegistry(s.path)
		if err != nil {
			return err
		}
		var next time.Time
		for _, job := range r.ArmedJobs() {
			if job.WakeAtUnixNano == 0 {
				continue
			}
			at := time.Unix(0, job.WakeAtUnixNano)
			if next.IsZero() || at.Before(next) {
				next = at
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
