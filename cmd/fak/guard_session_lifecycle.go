package main

import (
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

const crashJournalPulseEvery = 30 * time.Second

var crashJournalPulseInterval = crashJournalPulseEvery
var crashJournalPulseAppend = sessionjournal.Append
var crashJournalPulseNow = time.Now

// crashJournalPulse keeps the crash journal current while a guarded child is alive.
// Each tick is one append-only write; loss is bounded to one heartbeat interval.
type crashJournalPulse struct {
	id   string
	pid  int
	cwd  string
	stop chan struct{}
	done chan struct{}
	once sync.Once
}

func startCrashJournalPulse(id string, pid int) *crashJournalPulse {
	id = guardSessionJournalID(id)
	if id == "" {
		return nil
	}
	cwd, _ := os.Getwd()
	l := &crashJournalPulse{id: id, pid: pid, cwd: cwd, stop: make(chan struct{}), done: make(chan struct{})}
	l.append(sessionjournal.KindOpen)
	go l.run(crashJournalPulseInterval)
	return l
}

func (l *crashJournalPulse) run(interval time.Duration) {
	defer close(l.done)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			l.append(sessionjournal.KindBeat)
		case <-l.stop:
			return
		}
	}
}

func (l *crashJournalPulse) finish(clean bool) {
	if l == nil {
		return
	}
	l.once.Do(func() {
		close(l.stop)
		<-l.done
		if clean {
			l.append(sessionjournal.KindClose)
		}
	})
}

func (l *crashJournalPulse) append(kind sessionjournal.Kind) {
	now := crashJournalPulseNow().UTC()
	bootTime, _ := sessionjournal.BootTime(now)
	_ = crashJournalPulseAppend("", sessionjournal.Event{
		Schema: sessionjournal.Schema,
		Kind:   kind,
		ID:     l.id,
		PID:    l.pid,
		CWD:    l.cwd,
		TS:     now.Format(time.RFC3339Nano),
		Boot:   sessionjournal.BootID(bootTime),
	})
}
