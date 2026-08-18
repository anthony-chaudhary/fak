//go:build !windows

package dispatchaudit

import "github.com/anthony-chaudhary/fak/internal/processalive"

// ProcessAlive reports whether a process with the given pid is currently running.
// Signal 0 probes without delivering anything: nil means the process exists, ESRCH
// means it is gone, and EPERM (a live process this user does not own) still confirms
// existence. A non-positive pid is never alive — "no pid recorded" must not read as
// "running" and keep a slot or lock wedged.
//
// Exported because liveness is ONE question with one answer: the dispatch tick's own
// pid probe (cmd/fak) carried a byte-identical private copy on both platforms, and a
// reaper that judged a pid dead while its peer judged it live would free a slot out
// from under a running worker.
func ProcessAlive(pid int) bool { return processalive.Check(pid) }
