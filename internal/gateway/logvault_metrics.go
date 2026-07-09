package gateway

import "strings"

// logvault_metrics.go — the /metrics seam for logvault vault observability
// (#2455). Mirrors the fak_harness_* provider pattern (harness_metrics.go): a
// host-injected pull source renders the three fak_logvault_* gauges on each
// scrape. `fak guard` wires it to the box's capture vault (internal/logvault's
// Vault.MetricsText) so an operator sees last-capture age, footprint, and verify
// mismatches where they already scrape — the "is my backup current and intact?"
// answer. Kept decoupled: the gateway never imports logvault; the host owns the
// schema (the provider returns pre-rendered Prometheus text with its own
// HELP/TYPE lines), so an empty string adds nothing rather than a phantom family.

// SetLogvaultMetricsProvider installs the pull source for the fak_logvault_*
// family. Passing nil detaches it; the default `fak serve` path never sets it and
// renders nothing. Safe on a nil Server.
func (s *Server) SetLogvaultMetricsProvider(fn func() string) {
	if s == nil || s.metrics == nil {
		return
	}
	s.metrics.servingMu.Lock()
	s.metrics.logvaultProvider = fn
	s.metrics.servingMu.Unlock()
}

// writeLogvaultMetrics appends the host-injected fak_logvault_* family, if a
// provider is set. It renders whatever the provider returns verbatim (the
// provider emits HELP/TYPE headers itself), so an empty string — no vault on this
// box — adds nothing rather than an empty family block.
func (m *gatewayMetrics) writeLogvaultMetrics(b *strings.Builder) {
	if m == nil {
		return
	}
	m.servingMu.Lock()
	fn := m.logvaultProvider
	m.servingMu.Unlock()
	if fn == nil {
		return
	}
	if text := fn(); text != "" {
		b.WriteString(text)
	}
}
