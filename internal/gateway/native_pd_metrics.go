package gateway

// native_pd_metrics.go wires the #28 native prefill/decode role-split telemetry onto
// the live gateway /metrics surface without making P/D serving part of default gateway
// state. Hosts that own a NativePDCluster attach it; ordinary gateways stay inert.

import "strings"

// NativePDMetricsProvider is the narrow serving-metrics seam for #28. modelengine keeps
// the native P/D implementation and already exposes a Metrics string; gateway only knows
// how to splice that fragment into the shared scrape surface.
type NativePDMetricsProvider interface {
	Metrics() string
}

// SetNativePDMetrics wires native P/D role-split metrics onto the Server so the
// fak_native_pd_* family renders into /metrics. Passing nil detaches it, keeping the
// surface free of phantom prefill/decode workers.
func (s *Server) SetNativePDMetrics(p NativePDMetricsProvider) {
	if s == nil {
		return
	}
	s.nativePDMu.Lock()
	s.nativePDMetrics = p
	s.nativePDMu.Unlock()
}

func (s *Server) writeNativePDMetrics(b *strings.Builder) {
	if s == nil || b == nil {
		return
	}
	s.nativePDMu.RLock()
	p := s.nativePDMetrics
	s.nativePDMu.RUnlock()
	if p == nil {
		return
	}
	text := p.Metrics()
	if text == "" {
		return
	}
	b.WriteString(text)
	if !strings.HasSuffix(text, "\n") {
		b.WriteByte('\n')
	}
}
