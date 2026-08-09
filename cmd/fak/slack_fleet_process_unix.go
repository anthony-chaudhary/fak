//go:build !windows

package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// collectBackgroundProcesses reads the host's pid/ppid/cmdline census for
// `fak slack fleet-status`.
//
// It goes through procguard.CollectRelations rather than shelling out here, because this
// call site carried both halves of the #5385 defect verbatim (#5537):
//
//   - It ran one hard-coded `ps -eo pid=,ppid=,etimes=,args=` on EVERY POSIX host. `etimes`
//     is a procps-ng extension; the BSD dialect has `etime` instead and rejects the whole
//     invocation on an unknown keyword rather than dropping that one column. So on macOS the
//     census did not lose an age column, it returned nothing at all.
//   - It called .Output() and answered `nil, err`, throwing away any table `ps` had already
//     printed. "The tool failed" and "this host is running nothing in the background" then
//     became the same value, and runSlackFleetStatus renders the second one — an empty fleet
//     — while the warning it prints alongside says only that a command failed.
//
// procguard already fixed both, per GOOS and once: psRelationSpec picks the argv (`etime` on
// darwin, `etimes` on procps) and runTool keeps stdout when the tool exits non-zero, with
// censusError deciding that rows-and-an-error is a census and no-rows-and-an-error is a
// failure. Re-deriving that here is what produced this bug the first time, so this function
// is now only the projection onto loopfleet.Process.
func collectBackgroundProcesses() ([]loopfleet.Process, error) {
	procs, collectErr := procguard.CollectRelations()
	if collectErr != "" {
		return nil, fmt.Errorf("process inventory: %s", collectErr)
	}
	got := make([]loopfleet.Process, 0, len(procs))
	for _, p := range procs {
		row := loopfleet.Process{PID: p.PID, Command: p.Cmdline}
		if p.PPID != nil {
			row.ParentPID = *p.PPID
		}
		got = append(got, row)
	}
	return got, nil
}
