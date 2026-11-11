// Clock offset tracking for cross-clock latency comparison. A client and the
// serving host keep different clocks, so a raw server stamp cannot be placed
// on the client's timeline without correcting the skew between them: each
// probe exchange yields one sample (the client's send/receive stamps and the
// server's stamp carried in the response) and a raw offset estimate,
// server - (send+recv)/2, valid under the symmetric-delay assumption. The
// exchange's RTT bounds how badly that assumption can be bent, so the estimate
// is taken from the surviving sample with the MINIMUM RTT — the
// least-uncertainty pick, the NTP best-sample rule — after a median/MAD gate
// has rejected probe outliers (clock steps, retries, responses queued on the
// server side).
//
// Port of ai-dynamo/aiperf workers/clock_offset_tracker.py at ab610add4e8876fca223197596c728e17365f381
// (Apache-2.0), re-expressed in idiomatic Go; no upstream code copied.
package perfledger

import (
	"math"
	"sort"
)

// DefaultClockWindowCap is the rolling probe window size a tracker keeps when
// constructed without one; DefaultClockMADK is the MAD gate multiplier applied
// before the estimate is taken.
const (
	DefaultClockWindowCap = 32
	DefaultClockMADK      = 3.5
)

// ClockOffsetSample is one client<->server clock probe exchange: the client
// records its own send/receive stamps and the server's stamp carried in the
// response. All values are Unix milliseconds in the respective holder's clock.
type ClockOffsetSample struct {
	ClientSendUnixMS int64
	ClientRecvUnixMS int64
	ServerUnixMS     int64
}

// RTTMS is the exchange's round-trip time in the client's clock: the client's
// own receive-minus-send stamp difference, which needs no server cooperation.
func (s ClockOffsetSample) RTTMS() float64 {
	return float64(s.ClientRecvUnixMS - s.ClientSendUnixMS)
}

// OffsetMS is the raw clock offset the exchange implies: the server stamp
// minus the midpoint of the client's send/receive stamps, positive when the
// server's clock runs ahead of the client's. The midpoint is computed in
// float so the halving cannot truncate.
func (s ClockOffsetSample) OffsetMS() float64 {
	return float64(s.ServerUnixMS) - float64(s.ClientSendUnixMS+s.ClientRecvUnixMS)/2
}

// ClockOffsetTracker keeps the last windowCap admitted probe samples in
// insertion order and derives one clock offset estimate from them. The zero
// value is ready to use and behaves like a tracker built from the defaults.
type ClockOffsetTracker struct {
	window    []ClockOffsetSample
	windowCap int
	madK      float64
}

// NewClockOffsetTracker builds a tracker that keeps at most windowCap admitted
// samples and gates survivors at madK robust deviations (median + k*MAD). A
// non-positive windowCap falls back to DefaultClockWindowCap; a non-positive
// or non-finite madK falls back to DefaultClockMADK.
func NewClockOffsetTracker(windowCap int, madK float64) *ClockOffsetTracker {
	if windowCap <= 0 {
		windowCap = DefaultClockWindowCap
	}
	if madK <= 0 || math.IsNaN(madK) || math.IsInf(madK, 0) {
		madK = DefaultClockMADK
	}
	return &ClockOffsetTracker{windowCap: windowCap, madK: madK}
}

// effectiveCap reports the window capacity in force: the configured cap, or
// the default for a non-positive one (which includes the zero value's unset
// cap).
func (t *ClockOffsetTracker) effectiveCap() int {
	if t.windowCap <= 0 {
		return DefaultClockWindowCap
	}
	return t.windowCap
}

// effectiveK reports the MAD gate multiplier in force: the configured k, or
// the default for a non-positive or non-finite one (which includes the zero
// value's unset k).
func (t *ClockOffsetTracker) effectiveK() float64 {
	if t.madK <= 0 || math.IsNaN(t.madK) || math.IsInf(t.madK, 0) {
		return DefaultClockMADK
	}
	return t.madK
}

// Add admits one probe sample into the rolling window and reports whether it
// was admitted. Samples whose RTT is negative, NaN, or +Inf are refused with
// the window unchanged — such an exchange cannot locate the midpoint within
// its own round trip — while a zero RTT is admissible. An admitted sample
// evicts the oldest one once the window is full (FIFO).
func (t *ClockOffsetTracker) Add(s ClockOffsetSample) bool {
	rtt := s.RTTMS()
	if rtt < 0 || math.IsNaN(rtt) || math.IsInf(rtt, 1) {
		return false
	}
	if len(t.window) >= t.effectiveCap() {
		copy(t.window, t.window[1:])
		t.window[len(t.window)-1] = s
		return true
	}
	t.window = append(t.window, s)
	return true
}

// Offset returns the clock offset estimate in milliseconds and whether one
// could be derived from the window. The estimate is the MAD-gated survivor
// with the minimum RTT — the exchange whose symmetric-delay assumption is
// least strained — with an RTT tie going to the earliest-added survivor, so
// repeated calls on an unchanged window agree. An empty window returns
// (0, false).
func (t *ClockOffsetTracker) Offset() (float64, bool) {
	if len(t.window) == 0 {
		return 0, false
	}
	offsets := make([]float64, len(t.window))
	for i, s := range t.window {
		offsets[i] = s.OffsetMS()
	}
	median := clockMedian(offsets)
	absdevs := make([]float64, len(offsets))
	for i, o := range offsets {
		absdevs[i] = math.Abs(o - median)
	}
	mad := clockMedian(absdevs)
	threshold := t.effectiveK() * mad
	best := -1
	bestRTT := 0.0
	for i, s := range t.window {
		if mad == 0 {
			if offsets[i] != median {
				continue
			}
		} else if absdevs[i] > threshold {
			continue
		}
		rtt := s.RTTMS()
		if best == -1 || rtt < bestRTT {
			best = i
			bestRTT = rtt
		}
	}
	if best == -1 {
		return 0, false
	}
	return t.window[best].OffsetMS(), true
}

// Len reports the current window size, between 0 and the window capacity.
func (t *ClockOffsetTracker) Len() int {
	return len(t.window)
}

// clockMedian is the median of vals: the middle order statistic for an odd
// count, the mean of the two middle order statistics for an even one. It does
// not mutate vals.
func clockMedian(vals []float64) float64 {
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}
