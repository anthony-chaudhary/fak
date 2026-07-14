//go:build windows

package main

import (
	"os"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

const (
	ctrlCloseEvent    uint32 = 2
	ctrlLogoffEvent   uint32 = 5
	ctrlShutdownEvent uint32 = 6
)

var (
	kernel32Termination         = syscall.NewLazyDLL("kernel32.dll")
	setConsoleCtrlHandler       = kernel32Termination.NewProc("SetConsoleCtrlHandler")
	processIDToSessionID        = kernel32Termination.NewProc("ProcessIdToSessionId")
	terminationHandlerKeepalive uintptr
	terminationOnce             sync.Once
)

func installGuardHostTerminationObserver(path string) error {
	var installErr error
	terminationOnce.Do(func() {
		cb := syscall.NewCallback(func(control uint32) uintptr {
			name := controlTypeName(control)
			if name == "" {
				return 0
			}
			pid := os.Getpid()
			var session uint32
			processIDToSessionID.Call(uintptr(uint32(pid)), uintptr(unsafe.Pointer(&session)))
			_ = hostfault.AppendHostTermination(path, hostfault.HostTerminationMarker{
				Schema: hostfault.HostTerminationSchema, ControlType: name,
				GuardPID: pid, ConsoleSession: session,
				ObservedAt: time.Now().UTC().Format(time.RFC3339Nano),
			})
			return 0 // record, then preserve the normal Windows teardown path
		})
		terminationHandlerKeepalive = cb
		ok, _, err := setConsoleCtrlHandler.Call(cb, 1)
		if ok == 0 {
			installErr = err
		}
	})
	return installErr
}

func controlTypeName(control uint32) string {
	switch control {
	case ctrlCloseEvent:
		return "CTRL_CLOSE_EVENT"
	case ctrlLogoffEvent:
		return "CTRL_LOGOFF_EVENT"
	case ctrlShutdownEvent:
		return "CTRL_SHUTDOWN_EVENT"
	default:
		return ""
	}
}
