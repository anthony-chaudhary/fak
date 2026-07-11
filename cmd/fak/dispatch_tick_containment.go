package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// Blast-radius containment, live half (#2170 enforcement). The console-fault
// supervisor already scopes a terminal crash to the DEAD call; the `contain`
// CLI lets an operator ask "should the next spawn proceed?". This file wires
// that same fold into the LIVE dispatch admission path, so the answer actually
// refuses a launch instead of only being reportable.
//
// It is consulted BEFORE the capability broker in defaultLaunchSpawnBroker: a
// capability-valid spawn is still HELD when the recorded console-fault history
// says the host is in a cross-session storm (BREAKER_OPEN) or the target
// surface is in a re-crash loop (QUARANTINE_SURFACE). The breaker is the
// load-bearing live protection today — it is surface-agnostic, so it bites
// regardless of how a dispatch surface name maps onto a console-surface name;
// surface-quarantine and the co-location cap sharpen as that placement
// vocabulary and a live per-surface fan-in are fed in (follow-on).

// defaultConsoleFaultJournalPath is the durable console-fault snapshot the
// `console-faults --ingest` producer writes and both `contain` and this live
// gate fold. One path, one source of truth for recorded terminal crashes.
func defaultConsoleFaultJournalPath() string {
	return filepath.Join(".fak", "toolproc", "console-faults.jsonl")
}

// launchContainmentGate adjudicates whether a proposed child spawn is safe to
// place on its console surface given the recorded fault history. Swappable so a
// test can drive a deterministic verdict; the default folds the durable journal
// through DecideContainment under the wired DefaultContainmentPolicy.
var launchContainmentGate = defaultLaunchContainmentGate

func defaultLaunchContainmentGate(surface string, live int) toolprocgate.ContainmentDecision {
	events := readConsoleFaultJournalForContainment(defaultConsoleFaultJournalPath())
	return toolprocgate.DecideContainment(
		toolprocgate.DefaultContainmentPolicy(),
		events,
		toolprocgate.ContainmentRequest{
			Surface:       strings.TrimSpace(surface),
			LiveOnSurface: live,
			NowMS:         time.Now().UnixMilli(),
		},
	)
}

// readConsoleFaultJournalForContainment reads the durable console-fault journal
// for the LIVE gate. Unlike the `contain` CLI, the live path is a protective
// OVERLAY: it may only ADD a refusal off real recorded history, never wedge a
// launch because the journal is absent or unreadable. So every read problem
// (missing file, IO error, drifted row) folds to an empty history — the gate
// then admits, and the capability broker decides as before.
func readConsoleFaultJournalForContainment(path string) []toolprocgate.ConsoleFaultEvent {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	return parseConsoleFaultJournalLenient(f)
}

func parseConsoleFaultJournalLenient(r io.Reader) []toolprocgate.ConsoleFaultEvent {
	events, err := toolprocgate.ParseConsoleFaultEvents(r)
	if err != nil {
		return nil
	}
	return events
}
