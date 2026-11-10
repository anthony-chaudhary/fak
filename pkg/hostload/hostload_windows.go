//go:build windows

package hostload

import (
	"runtime"
	"strconv"
	"syscall"
	"time"
	"unsafe"
)

var procGetSystemTimes = syscall.NewLazyDLL("kernel32.dll").NewProc("GetSystemTimes")

const (
	windowsQueueSamples  = 5
	windowsSampleSpacing = 100 * time.Millisecond
	hkeyPerformanceData  = syscall.Handle(0x80000004)
	perfDataBufferBytes  = 1 << 18
)

type filetime struct{ Low, High uint32 }

func (f filetime) ticks() uint64 { return uint64(f.High)<<32 | uint64(f.Low) }

func readWindows() (float64, bool) {
	idle0, total0, ok := systemTimes()
	if !ok {
		return 0, false
	}
	queue, ok := processorQueueLength()
	if !ok {
		return 0, false
	}
	idle1, total1, ok := systemTimes()
	if !ok {
		return 0, false
	}
	return windowsLoad(idle0, total0, idle1, total1, runtime.NumCPU(), queue)
}

func systemTimes() (idle, total uint64, ok bool) {
	var i, k, u filetime
	r, _, _ := procGetSystemTimes.Call(uintptr(unsafe.Pointer(&i)), uintptr(unsafe.Pointer(&k)), uintptr(unsafe.Pointer(&u)))
	if r == 0 {
		return 0, 0, false
	}
	return i.ticks(), k.ticks() + u.ticks(), true
}

// processorQueueLength averages the instantaneous system-wide ready queue over
// windowsQueueSamples registry reads of the System object, which also spaces
// the GetSystemTimes pair. The registry read costs milliseconds where the PDH
// counter-name lookup costs seconds on a loaded host.
func processorQueueLength() (float64, bool) {
	name, err := syscall.UTF16PtrFromString(strconv.Itoa(perfObjectSystem))
	if err != nil {
		return 0, false
	}
	defer syscall.RegCloseKey(hkeyPerformanceData)
	buf := make([]byte, perfDataBufferBytes)
	sum, n := 0.0, 0
	for i := 0; i < windowsQueueSamples; i++ {
		if i > 0 {
			time.Sleep(windowsSampleSpacing)
		}
		size := uint32(len(buf))
		if syscall.RegQueryValueEx(hkeyPerformanceData, name, nil, nil, &buf[0], &size) != nil {
			continue
		}
		if v, ok := perfDataCounter(buf[:size], perfObjectSystem, perfCounterProcessorQueueLength); ok {
			sum += float64(v)
			n++
		}
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n), true
}
