//go:build windows

package nightrun

import "os/exec"

// configureProcGroup is a no-op on Windows: there is no POSIX process group to
// signal. The context-cancel default (kill the child) plus cmd.WaitDelay (force
// the pipes closed) still bound the attempt so the loop moves on. Windows is not a
// collection host in practice (native `go test` is blocked here too), so the
// belt-and-suspenders group reap lives only in the unix build.
func configureProcGroup(cmd *exec.Cmd) {}
