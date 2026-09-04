//go:build windows

package power

import (
	"sync"
	"syscall"
	"unsafe"
)

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	procSetThreadExecutionState = kernel32.NewProc("SetThreadExecutionState")
	procPowerCreateRequest      = kernel32.NewProc("PowerCreateRequest")
	procPowerSetRequest         = kernel32.NewProc("PowerSetRequest")
	procPowerClearRequest       = kernel32.NewProc("PowerClearRequest")
	procCloseHandle             = kernel32.NewProc("CloseHandle")
)

const (
	esContinuous       = 0x80000000
	esSystemRequired   = 0x00000001
	esDisplayRequired  = 0x00000002
	esAwayModeRequired = 0x00000040

	powerRequestContextVersion      = 0
	powerRequestContextSimpleString = 1

	powerRequestDisplayRequired = 0
	powerRequestSystemRequired  = 1
)

type reasonContext struct {
	Version uint32
	Flags   uint32
	Reason  uintptr
}

type windowsLock struct {
	mu          sync.Mutex
	powerHandle uintptr
	flags       WakeFlags
	closed      bool
}

func platformAcquire(reason string, flags WakeFlags) (platformLock, error) {
	// 1. SetThreadExecutionState
	esFlags := uintptr(esContinuous | esSystemRequired)
	if flags&PreventDisplaySleep != 0 {
		esFlags |= esDisplayRequired
	}
	procSetThreadExecutionState.Call(esFlags)

	// 2. PowerCreateRequest / PowerSetRequest (Windows 7+)
	var powerHandle uintptr
	if err := procPowerCreateRequest.Find(); err == nil {
		reasonUTF16, err := syscall.UTF16PtrFromString(reason)
		if err == nil {
			rctx := reasonContext{
				Version: powerRequestContextVersion,
				Flags:   powerRequestContextSimpleString,
				Reason:  uintptr(unsafe.Pointer(reasonUTF16)),
			}
			h, _, _ := procPowerCreateRequest.Call(uintptr(unsafe.Pointer(&rctx)))
			if h != 0 && h != uintptr(syscall.InvalidHandle) {
				powerHandle = h
				procPowerSetRequest.Call(powerHandle, uintptr(powerRequestSystemRequired))
				if flags&PreventDisplaySleep != 0 {
					procPowerSetRequest.Call(powerHandle, uintptr(powerRequestDisplayRequired))
				}
			}
		}
	}

	return &windowsLock{
		powerHandle: powerHandle,
		flags:       flags,
	}, nil
}

func (w *windowsLock) Release() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true

	if w.powerHandle != 0 && w.powerHandle != uintptr(syscall.InvalidHandle) {
		procPowerClearRequest.Call(w.powerHandle, uintptr(powerRequestSystemRequired))
		if w.flags&PreventDisplaySleep != 0 {
			procPowerClearRequest.Call(w.powerHandle, uintptr(powerRequestDisplayRequired))
		}
		procCloseHandle.Call(w.powerHandle)
		w.powerHandle = 0
	}

	// Restore continuous thread execution state
	procSetThreadExecutionState.Call(uintptr(esContinuous))
	return nil
}
