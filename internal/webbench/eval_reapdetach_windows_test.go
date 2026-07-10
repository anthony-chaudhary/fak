//go:build windows

package webbench

import "syscall"

// createNewProcessGroup is the Windows CreateProcess flag that launches the
// child as the root of a brand-new process group (its own Ctrl+C/Ctrl+Break
// target), detaching it from the launcher's group. It does NOT rewrite the
// child's ParentProcessID, so the Toolhelp32 PPID walk killTreeWindowsNative
// uses still reaches the grandchild — which is exactly what the #3914 detached
// witness pins.
const createNewProcessGroup = 0x00000200

// detachSysProcAttr puts the spawned grandchild in its own process group so the
// reap witness exercises a detached child, not an in-group one.
func detachSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: createNewProcessGroup}
}
