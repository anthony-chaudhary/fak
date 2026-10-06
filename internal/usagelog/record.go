package usagelog

import (
	"os"
	"time"
)

// record.go is the shared BEST-EFFORT append helper for a top-level CLI process
// that wants to write one usage row at exit without re-implementing the
// salt/open/append dance. cmd/fak has its own richer recorder (usagelog_record.go)
// with verb exclusions and in-process instrumentation; this helper is the plain
// path a SECOND binary (cmd/fak-dev) uses to join the SAME journal and chain.
//
// Best-effort is the contract: Record never returns an error and never changes the
// caller's exit code. A disabled journal, a salt/open/append failure, or a busy
// cross-process lock is swallowed silently so a meta-observability write can never
// break a real verb.

// Record appends one best-effort usage row for verb to the journal at path. An
// empty path resolves via FAK_USAGE_LOG_PATH (the same override cmd/fak honors),
// then DefaultPath. It is a no-op when Enabled() is false. start is the process
// start time; a zero start writes duration 0 rather than a negative value.
func Record(path, verb string, argv []string, exitCode int, start time.Time) {
	if !Enabled() {
		return
	}
	if path == "" {
		if p := os.Getenv("FAK_USAGE_LOG_PATH"); p != "" {
			path = p
		} else {
			path = DefaultPath()
		}
	}
	durationMS := time.Since(start).Milliseconds()
	if start.IsZero() || durationMS < 0 {
		durationMS = 0
	}
	// Use the shared per-user salt (DefaultSaltPath) so a fak row and a fak-dev
	// row for the same argv hash identically; an FAK_USAGE_LOG_PATH override
	// relocates only the journal, not the redaction domain.
	salt, err := LoadOrCreateSalt(DefaultSaltPath())
	if err != nil {
		return
	}
	lg, err := Open(path)
	if err != nil {
		return
	}
	defer lg.Close()
	host, _ := os.Hostname()
	_, _ = lg.Append(Row{
		Verb:       verb,
		Argc:       len(argv),
		ArgsDigest: Digest(salt, argv),
		ExitCode:   exitCode,
		DurationMS: durationMS,
		Host:       host,
		PID:        os.Getpid(),
	})
}
