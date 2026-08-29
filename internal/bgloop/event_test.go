package bgloop_test

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"testing"

	. "github.com/anthony-chaudhary/fak/internal/bgloop"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

func TestScheduleWakeCohortSpreadsHundredWithoutRefusal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.json")
	now := time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC)
	registry := loopmgr.Registry{Schema: loopmgr.SchemaRegistry, Jobs: map[string]loopmgr.Job{}}
	ids := make([]string, 100)
	for i := range ids {
		ids[i] = fmt.Sprintf("loop-%03d", i)
		registry.Jobs[ids[i]] = armedTestJob(ids[i])
	}
	if err := loopmgr.SaveRegistry(path, registry); err != nil {
		t.Fatal(err)
	}
	window := 10 * time.Second
	scheduler := EventScheduler{RegistryPath: path, Store: loopmgr.BGLoopStore{}, Clock: eventFakeClock{now: now}, JitterWindow: window}
	deadlines, err := scheduler.ScheduleWakeCohort(ids, now)
	if err != nil {
		t.Fatal(err)
	}
	if len(deadlines) != 100 { //boundarylint:ignore CHANGE_DETECTOR_TEST the test intentionally samples exactly 100 deadlines to prove deterministic sequence and uniqueness
		t.Fatalf("deadlines = %d, want 100", len(deadlines))
	}
	if !deadlines[0].Equal(now) || !deadlines[len(deadlines)-1].Equal(now.Add(window)) {
		t.Fatalf("range = %s..%s, want %s..%s", deadlines[0], deadlines[len(deadlines)-1], now, now.Add(window))
	}
	loaded, err := loopmgr.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	distinct := map[int64]bool{}
	for _, id := range ids {
		job, ok := loaded.Get(id)
		if !ok || job.WakeAtUnixNano == 0 {
			t.Fatalf("%s not durably scheduled", id)
		}
		distinct[job.WakeAtUnixNano] = true
	}
	if len(distinct) != 100 { //boundarylint:ignore CHANGE_DETECTOR_TEST the test intentionally samples exactly 100 deadlines to prove deterministic sequence and uniqueness
		t.Fatalf("distinct wake instants = %d, want 100", len(distinct))
	}
}

func TestAuthenticatedEventReconstitutesDormantLoopThroughGate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.json")
	registry := loopmgr.Registry{Schema: loopmgr.SchemaRegistry, Jobs: map[string]loopmgr.Job{}}
	registry.Jobs["dormant"] = armedTestJob("dormant")
	if err := loopmgr.SaveRegistry(path, registry); err != nil {
		t.Fatal(err)
	}
	ranGate, admittedRuntime, fired := false, false, false
	gate := rehydrate.NewGate(rehydrate.NewRungAt(rehydrate.StalePlan, dormancy.Cold, func(context.Context) rehydrate.Verdict { ranGate = true; return rehydrate.Clear() }))
	scheduler := EventScheduler{RegistryPath: path, Gate: rehydrate.BGLoopGate{Gate: gate}, Store: loopmgr.BGLoopStore{}, Clock: eventFakeClock{now: time.Now()}, Admit: func(_ context.Context, req WakeRequest) (func(), error) {
		admittedRuntime = req.Cause == "authenticated_event" && req.Principal == "event:source"
		return nil, nil
	}, Fire: func(_ context.Context, job Job) error { fired = job.JobID() == "dormant"; return nil }}
	admission, err := scheduler.DeliverEvent(context.Background(), Event{JobID: "dormant", Principal: "event:source", SignatureVerified: true, Horizon: dormancy.Cold})
	if err != nil {
		t.Fatal(err)
	}
	if !admission.Admitted || !ranGate || !admittedRuntime || !fired {
		t.Fatalf("admission=%+v gate=%v runtime=%v fired=%v", admission, ranGate, admittedRuntime, fired)
	}
}

func TestEventWakeRejectsUnauthenticatedOrRehydrationRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.json")
	registry := loopmgr.Registry{Schema: loopmgr.SchemaRegistry, Jobs: map[string]loopmgr.Job{}}
	registry.Jobs["dormant"] = armedTestJob("dormant")
	if err := loopmgr.SaveRegistry(path, registry); err != nil {
		t.Fatal(err)
	}
	fired := false
	scheduler := EventScheduler{RegistryPath: path, Store: loopmgr.BGLoopStore{}, Fire: func(context.Context, Job) error { fired = true; return nil }}
	if _, err := scheduler.DeliverEvent(context.Background(), Event{JobID: "dormant"}); !errors.Is(err, ErrUnauthenticatedEvent) {
		t.Fatalf("error=%v, want unauthenticated", err)
	}
	scheduler.Gate = rehydrate.BGLoopGate{Gate: rehydrate.NewGate(rehydrate.NewRungAt(rehydrate.StalePlan, dormancy.Cold, func(context.Context) rehydrate.Verdict {
		return rehydrate.Refuse(rehydrate.StalePlan, "policy drift")
	}))}
	admission, err := scheduler.DeliverEvent(context.Background(), Event{JobID: "dormant", Principal: "event:source", SignatureVerified: true, Horizon: dormancy.Cold})
	if err == nil || admission.Admitted || admission.RefusedBy != string(rehydrate.StalePlan) || fired {
		t.Fatalf("admission=%+v err=%v fired=%v", admission, err, fired)
	}
}

func armedTestJob(id string) loopmgr.Job {
	return loopmgr.Job{Schedule: loopmgr.Schedule{JobID: id, IntervalSeconds: 3600, MissedRun: loopmgr.MissedSkip}, State: loopmgr.JobArmed}
}

type eventFakeClock struct{ now time.Time }

func (c eventFakeClock) Now() time.Time                       { return c.now }
func (c eventFakeClock) After(time.Duration) <-chan time.Time { return make(chan time.Time) }
