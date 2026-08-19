package main

import (
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func TestCrashJournalPulseWritesBootStampedOpenBeatClose(t *testing.T) {
	oldAppend, oldNow, oldInterval := crashJournalPulseAppend, crashJournalPulseNow, crashJournalPulseInterval
	defer func() {
		crashJournalPulseAppend, crashJournalPulseNow, crashJournalPulseInterval = oldAppend, oldNow, oldInterval
	}()
	crashJournalPulseInterval = time.Millisecond
	crashJournalPulseNow = func() time.Time { return time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC) }
	events := make(chan sessionjournal.Event, 16)
	crashJournalPulseAppend = func(_ string, ev sessionjournal.Event) error {
		events <- ev
		return nil
	}

	lifecycle := startCrashJournalPulse("trace-lifecycle", 42)
	if lifecycle == nil {
		t.Fatal("lifecycle not started")
	}
	deadline := time.After(time.Second)
	for len(events) < 2 {
		select {
		case <-deadline:
			t.Fatal("heartbeat was not written within one interval")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	lifecycle.finish(true)

	close(events)
	var got []sessionjournal.Event
	for ev := range events {
		got = append(got, ev)
	}
	if len(got) < 3 || got[0].Kind != sessionjournal.KindOpen || got[len(got)-1].Kind != sessionjournal.KindClose {
		t.Fatalf("events=%+v", got)
	}
	beat := false
	for _, ev := range got {
		if ev.Kind == sessionjournal.KindBeat {
			beat = true
		}
		if ev.Schema != sessionjournal.Schema || ev.ID != "trace-lifecycle" || ev.PID != 42 || ev.Boot == "" || ev.TS == "" {
			t.Fatalf("event missing lifecycle identity: %+v", ev)
		}
	}
	if !beat {
		t.Fatalf("events contain no heartbeat: %+v", got)
	}
}

func TestCrashJournalPulseCloseClassifiesClosed(t *testing.T) {
	oldAppend, oldNow, oldInterval := crashJournalPulseAppend, crashJournalPulseNow, crashJournalPulseInterval
	defer func() {
		crashJournalPulseAppend, crashJournalPulseNow, crashJournalPulseInterval = oldAppend, oldNow, oldInterval
	}()
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	crashJournalPulseNow = func() time.Time { return now }
	crashJournalPulseInterval = time.Hour
	var events []sessionjournal.Event
	crashJournalPulseAppend = func(_ string, ev sessionjournal.Event) error {
		events = append(events, ev)
		return nil
	}
	lifecycle := startCrashJournalPulse("trace-clean", os.Getpid())
	lifecycle.finish(true)

	classified := sessionjournal.Classify(sessionjournal.FoldEvents(events), sessionjournal.ClassifyConfig{
		Now:      now.Add(time.Hour),
		BootTime: now.Add(-time.Hour),
		PIDAlive: func(int) bool { return false },
	})
	if len(classified) != 1 || classified[0].Status != sessionjournal.StatusClosed {
		t.Fatalf("classified=%+v events=%+v", classified, events)
	}
}

func TestCrashJournalPulseCrashRemainsCrashed(t *testing.T) {
	oldAppend, oldNow, oldInterval := crashJournalPulseAppend, crashJournalPulseNow, crashJournalPulseInterval
	defer func() {
		crashJournalPulseAppend, crashJournalPulseNow, crashJournalPulseInterval = oldAppend, oldNow, oldInterval
	}()
	now := time.Date(2026, 8, 18, 12, 0, 0, 0, time.UTC)
	crashJournalPulseNow = func() time.Time { return now }
	crashJournalPulseInterval = time.Hour
	var events []sessionjournal.Event
	crashJournalPulseAppend = func(_ string, ev sessionjournal.Event) error {
		events = append(events, ev)
		return nil
	}
	lifecycle := startCrashJournalPulse("trace-crash", 999999)
	lifecycle.finish(false)

	classified := sessionjournal.Classify(sessionjournal.FoldEvents(events), sessionjournal.ClassifyConfig{
		Now:      now.Add(time.Minute),
		BootTime: now.Add(-time.Hour),
		PIDAlive: func(int) bool { return false },
	})
	if len(classified) != 1 || classified[0].Status != sessionjournal.StatusCrashed || classified[0].Reason != "PID_DEAD" {
		t.Fatalf("classified=%+v events=%+v", classified, events)
	}
	for _, ev := range events {
		if ev.Kind == sessionjournal.KindClose {
			t.Fatalf("crash wrote clean close: %+v", events)
		}
	}
}
