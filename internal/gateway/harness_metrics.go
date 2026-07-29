package gateway

import "strings"

// SetHarnessMetricsProvider installs a pull source for the fak_harness_* Prometheus
// family (epic #2044). fak guard wires this to its live harness resource sampler
// (internal/harnessres) so a running guarded session's own CPU / memory / disk-I/O is
// scrapeable off /metrics, not only printed in the exit summary. The provider is
// called on each scrape and returns pre-rendered Prometheus text (the sampler already
// owns the schema via Snapshot.PrometheusText). Passing nil detaches it; the default
// `fak serve` path never sets it and renders nothing. Safe on a nil Server.
func (s *Server) SetHarnessMetricsProvider(fn func() string) {
	s.withMetricsLocked(func(m *gatewayMetrics) { m.harnessProvider = fn })
}

// writeHarnessMetrics appends the host-injected fak_harness_* family, if a provider is
// set. It renders whatever the provider returns verbatim (the sampler emits HELP/TYPE
// headers itself), so an empty string — a provider that has nothing to report — adds
// nothing rather than an empty family block.
func (m *gatewayMetrics) writeHarnessMetrics(b *strings.Builder) {
	m.writeProvidedFamily(b, func(m *gatewayMetrics) func() string { return m.harnessProvider })
}
