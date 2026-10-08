//go:build windows

package model

import (
	"math/bits"
	"syscall"
	"time"
	"unsafe"
)

var (
	v41ClockDLL       = syscall.NewLazyDLL("kernel32.dll")
	v41ClockCounter   = v41ClockDLL.NewProc("QueryPerformanceCounter")
	v41ClockFrequency = v41ClockDLL.NewProc("QueryPerformanceFrequency")
	v41ClockOrigin    = initV41ExpertClock()
)

type v41ExpertClockOrigin struct {
	counter   int64
	frequency int64
	ok        bool
}

func queryV41ExpertClock(proc *syscall.LazyProc) (int64, bool) {
	if proc == nil || proc.Find() != nil {
		return 0, false
	}
	var value int64
	ok, _, _ := proc.Call(uintptr(unsafe.Pointer(&value)))
	return value, ok != 0
}

func initV41ExpertClock() v41ExpertClockOrigin {
	frequency, freqOK := queryV41ExpertClock(v41ClockFrequency)
	counter, counterOK := queryV41ExpertClock(v41ClockCounter)
	if !freqOK || !counterOK || frequency <= 0 || counter < 0 {
		return v41ExpertClockOrigin{}
	}
	return v41ExpertClockOrigin{counter: counter, frequency: frequency, ok: true}
}

func defaultV41ExpertClockNanos() int64 {
	if !v41ClockOrigin.ok {
		return 0
	}
	now, ok := queryV41ExpertClock(v41ClockCounter)
	if !ok || now < v41ClockOrigin.counter {
		return 0
	}
	ticks := now - v41ClockOrigin.counter
	seconds, remainder := ticks/v41ClockOrigin.frequency, ticks%v41ClockOrigin.frequency
	const maxInt64 = int64(^uint64(0) >> 1)
	if seconds > maxInt64/int64(time.Second) {
		return 0
	}
	hi, lo := bits.Mul64(uint64(remainder), uint64(time.Second))
	fraction, _ := bits.Div64(hi, lo, uint64(v41ClockOrigin.frequency))
	base := seconds * int64(time.Second)
	if fraction > uint64(maxInt64-base-1) {
		return 0
	}
	return base + int64(fraction) + 1
}
