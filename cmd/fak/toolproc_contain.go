package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// runToolprocContain is the offline blast-radius containment GATE for #2170: the
// enforcement seam that makes DecideContainment bite from a launcher or an
// operator. It folds the durable console-fault journal (the same rows
// `console-faults` renders) into a closed containment verdict for a PROPOSED
// spawn on --surface, then exits 0 to ADMIT or 3 to refuse — so a shell/agent
// launcher can consult it before starting a child, and a human can ask "is it
// safe to place another agent here right now?".
//
// The console-fault OBSERVABILITY answers "did a terminal crash?"; this answers
// the next question the observability alone never closed: "given the crashes we
// recorded, should the NEXT spawn proceed?". It is a pure fold over recorded
// history plus the caller's placement facts, so the same journal + request
// always yields the same verdict — offline-provable and identical to the fold
// Supervisor.AdmitSpawn runs live.
//
// A MISSING journal is not an error here: absence of recorded faults is absence
// of evidence of instability, so the gate fail-opens to ADMIT (a protective
// overlay must not wedge every launch just because ingest never ran). An
// unreadable or drifted journal, by contrast, is a hard refusal to guess.
func runToolprocContain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc contain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventsPath := fs.String("events", filepath.Join(".fak", "toolproc", "console-faults.jsonl"),
		"JSONL journal of console-fault events to fold ('-' reads stdin; a missing file fail-opens to ADMIT)")
	surface := fs.String("surface", "", "the console surface (stdout|stderr|pty|renderer|...) the proposed spawn would run on")
	live := fs.Int("live", 0, "how many agents are already live on --surface (the placement fan-in the caller knows)")
	nowMS := fs.Int64("now-ms", 0, "clock anchor for the fault window in epoch ms (default: now)")
	asJSON := fs.Bool("json", false, "emit the containment decision as JSON")
	// Policy knobs default to the wired DefaultContainmentPolicy; an embedder
	// tunes them to its own fleet shape (the same doctrine as the supervisor's
	// tick cadence). See ContainmentPolicy for the meaning of each.
	def := toolprocgate.DefaultContainmentPolicy()
	windowMS := fs.Int64("window-ms", def.WindowMS, "fault lookback window in ms (<=0 counts all faults)")
	maxPerSurface := fs.Int("max-per-surface", def.MaxAgentsPerSurface, "co-location cap: max live agents on one surface (0 disables)")
	quarantineFaults := fs.Int("quarantine-faults", def.SurfaceQuarantineFaults, "faults on one surface in-window that quarantine it (0 disables)")
	breakerFaults := fs.Int("breaker-faults", def.BreakerFaults, "total in-window faults that can open the fleet breaker (0 disables)")
	breakerSessions := fs.Int("breaker-sessions", def.BreakerSessions, "distinct faulted sessions required for the breaker storm (<=1 = any)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak toolproc contain: takes no positional args")
		return 2
	}

	events, ok := readConsoleFaultJournalForGate(stdout, stderr, *eventsPath)
	if !ok {
		return 1
	}

	now := *nowMS
	if now <= 0 {
		now = time.Now().UnixMilli()
	}
	pol := toolprocgate.ContainmentPolicy{
		WindowMS:                *windowMS,
		MaxAgentsPerSurface:     *maxPerSurface,
		SurfaceQuarantineFaults: *quarantineFaults,
		BreakerFaults:           *breakerFaults,
		BreakerSessions:         *breakerSessions,
	}
	dec := toolprocgate.DecideContainment(pol, events, toolprocgate.ContainmentRequest{
		Surface:       strings.TrimSpace(*surface),
		LiveOnSurface: *live,
		NowMS:         now,
	})

	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, dec, "fak toolproc contain"); code != 0 {
			return code
		}
	} else {
		renderContainmentDecision(stdout, strings.TrimSpace(*surface), dec)
	}
	if dec.Admit {
		return 0
	}
	return 3
}

// readConsoleFaultJournalForGate reads and fail-closed-parses the console-fault
// journal for the containment gate. Returns (events, true) on success — a
// MISSING file yields an empty history (fail-open to ADMIT), while an unreadable
// or drifted file yields ok=false so the gate refuses to guess.
func readConsoleFaultJournalForGate(stdout, stderr io.Writer, path string) ([]toolprocgate.ConsoleFaultEvent, bool) {
	var in io.Reader = os.Stdin
	if path != "-" {
		f, err := os.Open(path)
		if err != nil {
			if os.IsNotExist(err) {
				fmt.Fprintf(stderr, "fak toolproc contain: no fault journal at %s; admitting (no recorded faults)\n", path)
				return nil, true
			}
			fmt.Fprintf(stderr, "fak toolproc contain: %v\n", err)
			return nil, false
		}
		defer f.Close()
		in = f
	}
	events, err := toolprocgate.ParseConsoleFaultEvents(in)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc contain: %v\n", err)
		return nil, false
	}
	return events, true
}

// renderContainmentDecision prints the human view of a containment verdict: the
// one gate line first (ADMIT/refusal), then the evidence that produced it, so a
// refusal is auditable rather than opaque.
func renderContainmentDecision(w io.Writer, surface string, dec toolprocgate.ContainmentDecision) {
	tag := "ADMIT"
	if !dec.Admit {
		tag = "REFUSE"
	}
	where := surface
	if where == "" {
		where = "(unspecified surface)"
	}
	fmt.Fprintf(w, "%s %s -> %s: %s\n", tag, where, dec.Verdict, dec.Reason)
	if dec.Advice != "" {
		fmt.Fprintf(w, "  advice: %s\n", dec.Advice)
	}
	fmt.Fprintf(w, "  evidence: surface_faults=%d window_faults=%d window_sessions=%d live_on_surface=%d\n",
		dec.SurfaceFaults, dec.WindowFaults, dec.WindowSessions, dec.LiveOnSurface)
}
