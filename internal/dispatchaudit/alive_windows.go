//go:build windows

package dispatchaudit

import "github.com/anthony-chaudhary/fak/internal/processalive"

// ProcessAlive reports whether a process with the given pid is currently running,
// WITHOUT spawning anything. OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) succeeds
// for a live pid and fails once it has exited; a still-open handle to a not-yet-reaped
// zombie reports its exit code, so GetExitCodeProcess is checked too and only
// STILL_ACTIVE (259) counts as running. An unreadable exit code conservatively reads as
// alive — freeing a slot we are unsure about is the worse error.
//
// The no-spawn property is load-bearing: the dispatch tick calls this 50-80 times per
// tick across its scan loops, and the `tasklist` fan-out it replaced was a process-spawn
// storm that stalled the box while CPU and RAM read low.
//
// Exported because liveness is ONE question with one answer: the dispatch tick's own pid
// probe (cmd/fak) carried a byte-identical private copy on both platforms.
func ProcessAlive(pid int) bool { return processalive.Check(pid) }
