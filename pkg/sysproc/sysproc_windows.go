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

// ConfigureBackground configures cmd to run in the background without creating
// a visible console window on Windows.
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

// ConfigureDetached configures cmd to run as a detached process without any console
// attached. CreateNoWindow is cleared because Windows ignores CREATE_NO_WINDOW
// when DETACHED_PROCESS is set.
//
// LIFETIME LIMITATION: a DETACHED_PROCESS child does NOT survive the exit of the
// launcher process when the launcher owns the console. DETACHED_PROCESS allocates
// no console at all (the per-worker console COST win measured by #3597/#2340/#3405),
// but because the child is not a process-group root it is torn down with the
// launcher. Use ConfigureDetached for short-lived work whose output the launcher
// already redirects; use ConfigureDurableChild for a headless worker that must
// outlive the launcher (e.g. `fak-flow spawn`, worktree adoption, service-plane
// launch). Reproduced in fak#13468: a detached spawn reported a PID and left
// 0-byte stdout/stderr logs.
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

// ConfigureDurableChild configures cmd so the child OUTLIVES its launcher on
// Windows while still allocating no visible console window: the flag set is
// CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP with HideWindow, deliberately NOT
// DETACHED_PROCESS.
//
// This separates the two intents ConfigureDetached conflated (fak#13468):
//
//  1. ConfigureDetached is the no-console COST-optimizing primitive (#3597); its
//     child dies with the launcher.
//  2. ConfigureDurableChild is the DURABILITY primitive for a headless worker
//     whose stdout/stderr the launcher redirects to files: the child keeps
//     running after the launcher exits, so its logs are non-empty.
//
// Because CREATE_NEW_PROCESS_GROUP still allocates a console (hosted by
// conhost.exe/OpenConsole.exe), a durable child pays the per-console cost that
// DETACHED_PROCESS avoids. Callers that need durability accept that cost; that is
// the explicit split #13468 asks for. Off Windows, ConfigureDurableChild is the
// POSIX setsid equivalent and carries no such cost.
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

// ConfigureProcessGroup configures cmd to run in the background as the root of a new
// process group on Windows.
func ConfigureProcessGroup(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	ConfigureBackground(cmd)
	cmd.SysProcAttr.CreationFlags |= CreateNewProcessGroup
}
