package main

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// The fak_cli_* fold reads the whole byte-bounded journal TAIL
// (usagelog.DefaultMaxReadBytes) with NO time cutoff. fak_cli_invocations_total and
// fak_cli_invocation_duration_ms_total are TYPEd counter and the board takes rate()
// of them, so they must only grow: a sliding time window would shrink them as rows
// age out and rate() would read every aged-out row as a counter reset. They drop
// only when the bounded tail rolls past old rows, which rate() treats as one reset.

// renderFleetCLIUsageExposition folds the local CLI-invocation usage journal (the
// one cmd/fak and cmd/fak-dev append to: FAK_USAGE_LOG_PATH, else
// usagelog.DefaultPath) into the fak_cli_* families the "FAK CLI Invocation
// Telemetry" board (uid fak-cli-invocation-telemetry) queries on job=fak_fleet.
//
// A missing journal is an honest empty fold (readable=1, rows=0). A read error
// flips fak_cli_usage_log_readable to 0 and emits no other fak_cli_* family, so
// an unreadable journal never reads as "no recent invocations".
func renderFleetCLIUsageExposition(path string, now time.Time) string {
	var b strings.Builder
	b.WriteString("# HELP fak_cli_usage_log_readable 1 when the local CLI-invocation usage journal was read (a missing journal is an honest empty fold); 0 means it could not be read, not that nothing ran.\n")
	b.WriteString("# TYPE fak_cli_usage_log_readable gauge\n")
	proj, err := usagelog.FoldPath(path, usagelog.MetricOptions{})
	if err != nil {
		b.WriteString("fak_cli_usage_log_readable 0\n")
		return b.String()
	}
	b.WriteString("fak_cli_usage_log_readable 1\n")
	b.WriteString(proj.RenderPrometheus(now))
	return b.String()
}
