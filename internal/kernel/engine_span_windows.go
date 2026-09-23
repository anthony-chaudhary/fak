package kernel

import (
	"syscall"
	"time"
	"unsafe"
)

var (
	engineClockDLL  = syscall.NewLazyDLL("kernel32.dll")
	engineClockNow  = engineClockDLL.NewProc("QueryPerformanceCounter")
	engineClockFreq = engineClockDLL.NewProc("QueryPerformanceFrequency")
)

type measuredSpanClock struct {
	started time.Time
	counter int64
}

func engineCounter(proc *syscall.LazyProc) (int64, bool) {
	var value int64
	ok, _, _ := proc.Call(uintptr(unsafe.Pointer(&value)))
	return value, ok != 0
}

func startMeasuredSpan() measuredSpanClock {
	clock := measuredSpanClock{started: time.Now()}
	clock.counter, _ = engineCounter(engineClockNow)
	return clock
}

func (c measuredSpanClock) elapsedNanos() int64 {
	end, ok := engineCounter(engineClockNow)
	frequency, freqOK := engineCounter(engineClockFreq)
	if ok && freqOK && c.counter > 0 && end > c.counter && frequency > 0 {
		ticks := end - c.counter
		seconds, remainder := ticks/frequency, ticks%frequency
		if seconds <= int64(^uint64(0)>>1)/int64(time.Second) {
			return seconds*int64(time.Second) + remainder*int64(time.Second)/frequency
		}
	}
	return time.Since(c.started).Nanoseconds()
}
