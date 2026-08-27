package bgloop_test

import (
	"context"
	"sync"
	"testing"

	. "github.com/anthony-chaudhary/fak/internal/bgloop"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

type manualClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *manualClock) Now() time.Time                       { c.mu.Lock(); defer c.mu.Unlock(); return c.now }
func (c *manualClock) After(time.Duration) <-chan time.Time { ch := make(chan time.Time); return ch }
func (c *manualClock) advance(d time.Duration)              { c.mu.Lock(); c.now = c.now.Add(d); c.mu.Unlock() }

func seedJob(t *testing.T, path string, now time.Time, id string) {
	t.Helper()
	r := loopmgr.Registry{Jobs: map[string]loopmgr.Job{}}
	job := loopmgr.Job{Schedule: loopmgr.Schedule{JobID: id, IntervalSeconds: 60, MissedRun: loopmgr.MissedSkip}, State: loopmgr.JobArmed}
	if err := r.Put(job, now); err != nil {
		t.Fatal(err)
	}
	if err := loopmgr.SaveRegistry(path, r); err != nil {
		t.Fatal(err)
	}
}

func TestDurableWakeSurvivesRestartAndFiresOnce(t *testing.T) {
	now := time.Date(2026, 8, 27, 10, 0, 0, 0, time.UTC)
	clock := &manualClock{now: now}
	path := t.TempDir() + "/loops.json"
	seedJob(t, path, now, "nightly")
	first, err := NewDurableScheduler(path, clock, loopmgr.BGLoopStore{}, func(context.Context, Job) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if err := first.SleepUntil("nightly", now.Add(time.Hour), nil); err != nil {
		t.Fatal(err)
	}
	// Simulate process death: discard the scheduler and construct a new one from disk.
	fired := 0
	restarted, err := NewDurableScheduler(path, clock, loopmgr.BGLoopStore{}, func(_ context.Context, j Job) error { fired++; return nil })
	if err != nil {
		t.Fatal(err)
	}
	clock.advance(time.Hour)
	if n, err := restarted.Poll(context.Background()); err != nil || n != 1 {
		t.Fatalf("first poll = %d, %v; want 1, nil", n, err)
	}
	if n, err := restarted.Poll(context.Background()); err != nil || n != 0 {
		t.Fatalf("second poll = %d, %v; want 0, nil", n, err)
	}
	if fired != 1 {
		t.Fatalf("wake calls = %d, want 1", fired)
	}
}

func TestDurableDutyCycleFiresOnlyInsideOnWindow(t *testing.T) {
	// Monday 08:00 UTC, one hour before the weekday on-window.
	now := time.Date(2026, 8, 24, 8, 0, 0, 0, time.UTC)
	clock := &manualClock{now: now}
	path := t.TempDir() + "/loops.json"
	seedJob(t, path, now, "weekday")
	fired := 0
	s, err := NewDurableScheduler(path, clock, loopmgr.BGLoopStore{}, func(context.Context, Job) error { fired++; return nil })
	if err != nil {
		t.Fatal(err)
	}
	duty := &loopmgr.DutyCycle{Location: "UTC", Weekdays: []int{1, 2, 3, 4, 5}, OnMinute: 9 * 60, OffMinute: 17 * 60}
	if err := s.SleepUntil("weekday", now, duty); err != nil {
		t.Fatal(err)
	}
	if n, err := s.Poll(context.Background()); err != nil || n != 0 {
		t.Fatalf("off-window poll = %d, %v", n, err)
	}
	r, err := loopmgr.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	job, _ := r.Get("weekday")
	want := now.Add(time.Hour).UnixNano()
	if job.WakeAtUnixNano != want {
		t.Fatalf("rearmed at %s, want %s", time.Unix(0, job.WakeAtUnixNano), time.Unix(0, want))
	}
	clock.advance(time.Hour)
	if n, err := s.Poll(context.Background()); err != nil || n != 1 {
		t.Fatalf("on-window poll = %d, %v", n, err)
	}
	if fired != 1 {
		t.Fatalf("wake calls = %d, want 1", fired)
	}
}
