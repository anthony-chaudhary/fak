package main

// dispatch_doa.go — the runs-directory half of the DOA-spawn detector (#5868). The
// decision is pure and lives in internal/dispatchdoa; this file is only the wire: walk
// .dispatch-runs for finished resolve-*.log records inside a window, classify each, and
// hand the fold to `fak dispatch status`.
//
// WHY IT LANDS ON `fak dispatch status` rather than in the no-commit classifier. The
// 2026-07-28..08-03 outage killed 350 of 382 spawned workers at flag parse and ran six
// extra days because nothing SAID SO. Naming the class in the witness sweep would have
// added a row to a breakdown an operator has to go read; `fak dispatch status` is the
// card an operator already reads to answer "is the fleet working?", and throughout the
// outage it printed "0 live worker(s)" — indistinguishable from an idle fleet, because
// every worker died in well under a second and none was ever live to be counted. The
// alarm has to be on the card that was already being read.
//
// COST. One stat per run log; bytes are read ONLY for logs already known to be at/under
// the stub floor. On the healthy 08-04+ corpus that is 287 stats and zero reads.

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchdoa"
)

// dispatchDOAWindow is how far back the spawn-health fold looks. A day is short enough
// that a resolved drift stops being reported the next day, and long enough that a
// low-cadence fleet still has a denominator worth a rate.
const dispatchDOAWindow = 24 * time.Hour

// dispatchDOASettle is the age below which a run is NOT graded. This is the live-worker
// guard, and it is the second reason a healthy run can never be called DOA: a worker
// spawned moments ago has flushed only the `# fak-spawn` header and has not yet printed
// the guard's launch banner, so for a fraction of a second it is SHAPED like a DOA
// record. Its log mtime is by definition fresh, so this floor excludes it outright,
// while a DOA record's mtime froze at spawn and is never fresh again. Two minutes is
// ~100x the observed spawn-to-banner gap.
const dispatchDOASettle = 2 * time.Minute

// scanDispatchDOA folds the finished resolve-*.log records in runsDir into the
// spawn-health report. Pure over (runsDir, now): same directory + same clock, same
// report. Best-effort — an unreadable entry is skipped rather than counted, so a
// transient sharing lock can never fabricate a DOA.
func scanDispatchDOA(runsDir string, window, settle time.Duration, now time.Time) dispatchdoa.Report {
	paths, err := fsGlob(filepath.Join(runsDir, "resolve-*.log"))
	if err != nil {
		return dispatchdoa.Fold(nil)
	}
	oldest := now.Add(-window)
	newest := now.Add(-settle)
	runs := make([]dispatchdoa.Run, 0, len(paths))
	for _, path := range paths {
		st, serr := fsStat(path)
		if serr != nil || st.IsDir() {
			continue
		}
		mod := st.ModTime()
		if mod.Before(oldest) || mod.After(newest) {
			continue // outside the window, or still settling / live
		}
		run := dispatchdoa.Run{Log: filepath.Base(path)}
		if st.Size() <= dispatchdoa.StubMaxBytes {
			// Only a stub is ever read, and only its head. A launched worker's log is
			// never opened by this check.
			run.Verdict = dispatchdoa.Classify(readDispatchDOAHead(path), st.Size())
		}
		runs = append(runs, run)
	}
	return dispatchdoa.Fold(runs)
}

// readDispatchDOAHead reads at most dispatchdoa.HeadBytes from path through the runs-dir
// I/O seam. An unreadable log yields "", which fails open to a non-DOA verdict.
func readDispatchDOAHead(path string) string {
	f, err := fsOpen(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	buf := make([]byte, dispatchdoa.HeadBytes)
	n, rerr := io.ReadFull(f, buf)
	if n == 0 && rerr != nil {
		return ""
	}
	return string(buf[:n])
}

// dispatchDOALine renders the spawn-health fold as the operator line. A clear window
// gets one quiet confirmation that the check ran (so silence is never mistaken for the
// check being absent); a warn names the count; an alarm is unmissable and says what
// drifted and where the evidence is.
func dispatchDOALine(rep dispatchdoa.Report, window time.Duration) string {
	if rep.Runs == 0 {
		return ""
	}
	switch rep.Status {
	case dispatchdoa.StatusClear:
		return fmt.Sprintf("spawn health: %d finished spawn(s) in the last %s, 0 dead on arrival", rep.Runs, dispatchDOAWindowLabel(window))
	case dispatchdoa.StatusAlarm:
		return fmt.Sprintf("spawn health: ALARM — %d of %d spawn(s) in the last %s DIED ON ARRIVAL (%.0f%%): %s. "+
			"The dispatcher's argv and the worker binary have drifted; throughput is zero by construction. Evidence: %s",
			rep.DOA, rep.Runs, dispatchDOAWindowLabel(window), rep.Rate*100,
			strings.Join(rep.TopCauses(), " "), dispatchDOASample(rep))
	default:
		return fmt.Sprintf("spawn health: WARN — %d of %d spawn(s) in the last %s died on arrival (%.0f%%): %s. Evidence: %s",
			rep.DOA, rep.Runs, dispatchDOAWindowLabel(window), rep.Rate*100,
			strings.Join(rep.TopCauses(), " "), dispatchDOASample(rep))
	}
}

func dispatchDOASample(rep dispatchdoa.Report) string {
	if len(rep.Sample) == 0 {
		return "(no named log)"
	}
	s := strings.Join(rep.Sample, ", ")
	if rep.DOA > len(rep.Sample) {
		s += fmt.Sprintf(" (+%d more)", rep.DOA-len(rep.Sample))
	}
	return s
}

func dispatchDOAWindowLabel(window time.Duration) string {
	if window >= time.Hour {
		return fmt.Sprintf("%.0fh", window.Hours())
	}
	return window.String()
}
