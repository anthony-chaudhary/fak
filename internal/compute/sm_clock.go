package compute

import (
	"math"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
)

// SM clock provenance. A cycle count is GPU time times a clock rate, so it is
// only as trustworthy as the clock; every consumer carries the source along.
const (
	// SMClockObserved is a rate a backend read from the device.
	SMClockObserved = "observed"
	// SMClockDeclared is a rate the operator declared in SMClockEnv, typically
	// after locking clocks (nvidia-smi -lgc); nothing checks the device holds it.
	SMClockDeclared = "declared"
)

// SMClockEnv declares the SM clock in MHz when no backend reader observes one.
const SMClockEnv = "FAK_GPU_SM_CLOCK_MHZ"

// SMClock is the shader-core clock rate that converts GPU time into cycles.
type SMClock struct {
	MHz    float64
	Source string
}

type smClockReader struct {
	backend string
	read    func() (float64, bool)
}

var smClockReaderSlot atomic.Pointer[smClockReader]

// RegisterSMClockReader installs a backend's SM clock reader. read returns the
// clock in MHz averaged over the most recent work, or false when unreadable.
func RegisterSMClockReader(backend string, read func() (float64, bool)) (unregister func()) {
	if backend == "" || read == nil {
		return func() {}
	}
	r := &smClockReader{backend: backend, read: read}
	smClockReaderSlot.Store(r)
	return func() { smClockReaderSlot.CompareAndSwap(r, nil) }
}

// SMClockFor returns the clock for converting backend's GPU time to cycles: the
// registered reader's observation when it belongs to that backend, else the
// SMClockEnv declaration. False means the clock is unknown, and a cycle count
// must be omitted rather than reported as zero.
func SMClockFor(backend string) (SMClock, bool) {
	if r := smClockReaderSlot.Load(); r != nil && r.backend == backend {
		if mhz, ok := r.read(); ok && validClockMHz(mhz) {
			return SMClock{MHz: mhz, Source: SMClockObserved}, true
		}
	}
	if v := strings.TrimSpace(os.Getenv(SMClockEnv)); v != "" {
		if mhz, err := strconv.ParseFloat(v, 64); err == nil && validClockMHz(mhz) {
			return SMClock{MHz: mhz, Source: SMClockDeclared}, true
		}
	}
	return SMClock{}, false
}

// validClockMHz rejects rates no shipping GPU runs at, so a kHz or Hz value
// passed by mistake is refused instead of scaling every cycle count by 1000.
func validClockMHz(mhz float64) bool {
	return mhz >= 50 && mhz <= 10000 && !math.IsNaN(mhz) && !math.IsInf(mhz, 0)
}
