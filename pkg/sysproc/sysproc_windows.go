//go:build windows

package sysproc

import (
	"os/exec"
	"syscall"
)

// Windows process creation flags for console suppression and process groups.
const (
	CreateNoWindow        = 0x08000000
	CreateNewProcessGroup = 0x00000200
	DetachedProcess       = 0x00000008
)

// ConfigureBackground runs the child in the background without a visible console window.
func ConfigureBackground(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags |= CreateNoWindow
}

// ConfigureDetached runs the child detached from any console: DETACHED_PROCESS
// allocates no console at all (the per-worker console COST win measured by
// #3597/#2340/#3405). It clears CreateNoWindow because Windows ignores
// CREATE_NO_WINDOW when DETACHED_PROCESS is set.
//
// LIFETIME LIMITATION: the child is not a process-group root, so it does not
// survive the exit of a launcher that owns the console and redirected logs end
// up empty (reproduced in fak#13468: a detached spawn reported a PID and left
// 0-byte stdout/stderr logs). Use it for short-lived work whose output the
// launcher already redirects; for a child that must outlive the launcher, use
// ConfigureDurableChild.
func ConfigureDetached(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags &^= CreateNoWindow
	cmd.SysProcAttr.CreationFlags |= DetachedProcess
}

// ConfigureDurableChild makes the child outlive its launcher without a visible
// console: CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP with HideWindow,
// deliberately not DETACHED_PROCESS (the split requested in fak#13468). The
// child still allocates a console hosted by conhost.exe/OpenConsole.exe, so it
// pays the per-console cost that DETACHED_PROCESS avoids; callers that need
// durability accept that cost.
func ConfigureDurableChild(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags &^= DetachedProcess
	cmd.SysProcAttr.CreationFlags |= CreateNoWindow
	cmd.SysProcAttr.CreationFlags |= CreateNewProcessGroup
}

// ConfigureProcessGroup runs the child in the background as the root of a new process group.
func ConfigureProcessGroup(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	ConfigureBackground(cmd)
	cmd.SysProcAttr.CreationFlags |= CreateNewProcessGroup
}
