package gateway

// preemption_live.go wires the #31 KV preemptor's serving-metrics fragment onto a
// Server without making it part of the default gateway state. Hosts that own a live
// native scheduler attach their process-local preemptor; ordinary gateways stay inert.

import (
	"strings"
)

// KVPreemptionMetricWriter is the narrow serving-metrics seam for #31. The gateway keeps
// the writer structural so modelengine can expose the live native scheduler counters without
// creating a package import cycle.
type KVPreemptionMetricWriter interface {
	WriteKVPreemptionMetrics(*strings.Builder)
}

// SetKVPreemptor wires the native KV preemption gate (#31) onto the Server so its
// fak_sched_preempt_* metrics render into /metrics. Passing nil detaches it, keeping the
// surface inert by default for servers that do not own a paged native scheduler.
func (s *Server) SetKVPreemptor(c *KVPreemptor) {
	s.SetKVPreemptionMetrics(c)
}

// SetKVPreemptionMetrics wires a provider of #31 preemption counters onto the live /metrics
// render path. It accepts the older gateway KVPreemptor and the modelengine NativeScheduler.
func (s *Server) SetKVPreemptionMetrics(c KVPreemptionMetricWriter) {
	if s == nil {
		return
	}
	s.preemptionMu.Lock()
	s.preemptionMetrics = c
	s.preemptionMu.Unlock()
}

// WriteKVPreemptionMetrics lets the standalone gateway KVPreemptor satisfy the generic live
// metrics seam above.
func (c *KVPreemptor) WriteKVPreemptionMetrics(b *strings.Builder) {
	if c == nil || b == nil {
		return
	}
	c.WriteMetrics(b)
}

func (s *Server) writePreemptionMetrics(b *strings.Builder) {
	if s == nil || b == nil {
		return
	}
	s.preemptionMu.RLock()
	c := s.preemptionMetrics
	s.preemptionMu.RUnlock()
	if c == nil {
		return
	}
	c.WriteKVPreemptionMetrics(b)
}
