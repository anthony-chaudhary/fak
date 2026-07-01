package main

import (
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// usagelog_record.go wires internal/usagelog into main()'s dispatch: one row
// for an invocation that returns normally from the verb switch (exit 0), one
// for the two usage-error os.Exit(2) sites that live directly in main(), and
// one for a recovered panic (exit 2, then re-panicked so the process's own
// exit code and crash output are unchanged).
//
// COVERAGE GAP — read before trusting a usage roll-up's error counts. cmd/fak
// has ~300 os.Exit(N) call sites scattered inside individual verb handlers
// (a `cmdXxx` function calling os.Exit(1) deep inside its own error path).
// os.Exit terminates the process immediately WITHOUT running deferred
// functions, so an invocation that errors via one of those direct calls is
// NOT recorded here — only the three sites above are. Closing that gap fully
// means routing every verb handler's exit through a shared helper, a large
// mechanical change across ~130 files on a shared trunk; tracked as a
// follow-up rather than done silently. `fak usage`'s exit-code distribution
// under-counts non-zero exits accordingly — it is still an honest signal for
// verb popularity and timing, which is the part it is used for.

// usageLogExcluded holds verbs deliberately never recorded because logging
// them would corrupt what they measure or overwhelm them with I/O. "hook" is
// the spawned-hook decide transport internal/bench.MeasureSpawnedBaseline
// spawns in a tight per-sample timing loop (internal/bench/bench.go); a
// usage-log open+fsync on every spawn would both inflate the very
// spawned-hook p50/p99 the benchmark exists to measure and turn an n=1000
// bench run into 1000 log appends for a process nobody cares to audit.
var usageLogExcluded = map[string]bool{
	"hook": true,
}

// usageLogPath is the file the recorder appends to: usagelog.DefaultPath(),
// overridable with FAK_USAGE_LOG_PATH (mirrors FAK_AUDIT_JOURNAL's role for
// the decision journal — see guardDefaultAuditPath in cmd/fak/guard.go).
func usageLogPath() string {
	if p := os.Getenv("FAK_USAGE_LOG_PATH"); p != "" {
		return p
	}
	return usagelog.DefaultPath()
}

// recordUsage appends one best-effort usage row. It never changes the
// invocation's own outcome: usagelog.Enabled()==false, an excluded verb, or
// any I/O failure (salt, open, append) is swallowed silently rather than
// surfaced — a meta-observability write must not be able to break or mask a
// real verb's exit code.
func recordUsage(verb string, argv []string, exitCode int, start time.Time) {
	if usageLogExcluded[verb] || !usagelog.Enabled() {
		return
	}
	salt, err := usagelog.LoadOrCreateSalt(usagelog.DefaultSaltPath())
	if err != nil {
		return
	}
	lg, err := usagelog.Open(usageLogPath())
	if err != nil {
		return
	}
	defer lg.Close()
	host, _ := os.Hostname()
	_, _ = lg.Append(usagelog.Row{
		Verb:       verb,
		Argc:       len(argv),
		ArgsDigest: usagelog.Digest(salt, argv),
		ExitCode:   exitCode,
		DurationMS: time.Since(start).Milliseconds(),
		FakVersion: appversion.Current(),
		Host:       host,
		PID:        os.Getpid(),
	})
}
