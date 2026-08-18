//go:build windows

package safecommit

import "github.com/anthony-chaudhary/fak/internal/processalive"

// processAlive reports whether a process with the given pid is currently running.
//
// On Windows, OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) succeeds for a live
// process and fails for one that has exited. A still-open handle to a zombie (exited
// but not yet reaped) reports a known exit code, so we additionally check
// GetExitCodeProcess: STILL_ACTIVE (259) means running, anything else means the holder
// is gone. Any error resolving the pid is treated as "not alive" — a pid we cannot
// confirm is live must not keep a stale commit lock wedged.
func processAlive(pid int) bool { return processalive.Check(pid) }
