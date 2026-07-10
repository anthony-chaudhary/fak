//go:build !windows

package webbench

import "syscall"

// detachSysProcAttr puts the spawned grandchild in its own process group
// (Setpgid) so the reap witness exercises a detached child. A new-group child
// is NOT reached by the harness's process-group SIGKILL (kill(-pid) targets the
// harness's own group), so this case proves the descendant-walk fallback in
// procguard.killSignal — the ps PPID walk — is what actually reaps it. The
// grandchild's PPID still points at the (live) harness, so the walk reaches it;
// only a fully reparented/daemonized child (PPID->1) would escape.
func detachSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}
