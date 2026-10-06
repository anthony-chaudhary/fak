package gateway

import (
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// firstDeltaClock stamps the arrival of a streamed turn's first non-empty content
// fragment, straight off the planner's sink and ahead of the lift-guard and the
// UTF-8 joiner, so it is the model's first token rather than the first byte the
// guard released. A proxied provider stream reports no Timings, so without this
// stamp every streamed proxy turn reached the perf ledger and the TTFT histogram
// as "ttft not measured" even though the gateway watched the first token arrive.
type firstDeltaClock struct {
	atNS atomic.Int64
}

// wrap returns a sink that stamps the first non-empty fragment and forwards every
// fragment unchanged.
func (c *firstDeltaClock) wrap(sink agent.StreamSink) agent.StreamSink {
	return func(delta string) error {
		if delta != "" && c.atNS.Load() == 0 {
			c.atNS.CompareAndSwap(0, time.Now().UnixNano())
		}
		return sink(delta)
	}
}

// ttft is the first-fragment latency from began, or 0 when no fragment arrived
// (an empty or tool-call-only turn), which every consumer reads as "not measured".
func (c *firstDeltaClock) ttft(began time.Time) time.Duration {
	if c == nil {
		return 0
	}
	at := c.atNS.Load()
	if at == 0 {
		return 0
	}
	if d := time.Unix(0, at).Sub(began); d > 0 {
		return d
	}
	return 0
}
