package main

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// fakDevStart is the process start time, captured at package init so the
// recorded duration covers the whole invocation.
var fakDevStart = time.Now()

// recordFakDevUsage appends one best-effort row for this fak-dev invocation to
// the SAME usagelog journal cmd/fak writes (FAK_USAGE_LOG_PATH, else
// usagelog.DefaultPath), so the fak_cli_* families count developer tooling
// alongside runtime verbs. The verb is argv[0]; Argc counts the operands after
// it. The internal strix known-hosts broker child is a transport, not an
// operator verb, and is never recorded. It never fails and never alters the
// caller's exit code.
func recordFakDevUsage(argv []string, exitCode int) {
	if len(argv) == 0 || argv[0] == amdgpu.StrixKnownHostsOperand {
		return
	}
	usagelog.Record("", argv[0], argv[1:], exitCode, fakDevStart)
}
