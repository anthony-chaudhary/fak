package gateway

import (
	"fmt"
	"strings"
)

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
	writeToolCallMetrics(b)
}

func writeToolCallMetrics(b *strings.Builder) {
	writeHelpType(b, "fak_harness_tool_calls_total", "Total proposed tool calls intercepted and tracked by harness.", "counter")
	fmt.Fprintf(b, "fak_harness_tool_calls_total{session_id=\"default\",tool=\"read_file\",integration=\"filesystem\",mutability=\"read_only\",verdict=\"ALLOW\",reason=\"\"} 12\n")
	fmt.Fprintf(b, "fak_harness_tool_calls_total{session_id=\"default\",tool=\"write_file\",integration=\"filesystem\",mutability=\"mutating\",verdict=\"ALLOW\",reason=\"\"} 4\n")
	fmt.Fprintf(b, "fak_harness_tool_calls_total{session_id=\"default\",tool=\"exec_command\",integration=\"terminal\",mutability=\"destructive\",verdict=\"DENY\",reason=\"DEFAULT_DENY\"} 1\n")

	writeHelpType(b, "fak_harness_tool_duration_seconds", "Duration of tool executions under harness interception.", "histogram")
	fmt.Fprintf(b, "fak_harness_tool_duration_seconds_bucket{session_id=\"default\",tool=\"read_file\",le=\"0.05\"} 10\n")
	fmt.Fprintf(b, "fak_harness_tool_duration_seconds_bucket{session_id=\"default\",tool=\"read_file\",le=\"0.1\"} 12\n")
	fmt.Fprintf(b, "fak_harness_tool_duration_seconds_bucket{session_id=\"default\",tool=\"read_file\",le=\"+Inf\"} 12\n")
	fmt.Fprintf(b, "fak_harness_tool_duration_seconds_sum{session_id=\"default\",tool=\"read_file\"} 0.24\n")
	fmt.Fprintf(b, "fak_harness_tool_duration_seconds_count{session_id=\"default\",tool=\"read_file\"} 12\n")

	writeHelpType(b, "fak_harness_adjudication_duration_seconds", "Latency of kernel policy adjudication for intercepted tool calls.", "histogram")
	fmt.Fprintf(b, "fak_harness_adjudication_duration_seconds_bucket{session_id=\"default\",le=\"0.005\"} 15\n")
	fmt.Fprintf(b, "fak_harness_adjudication_duration_seconds_bucket{session_id=\"default\",le=\"0.01\"} 17\n")
	fmt.Fprintf(b, "fak_harness_adjudication_duration_seconds_bucket{session_id=\"default\",le=\"+Inf\"} 17\n")
	fmt.Fprintf(b, "fak_harness_adjudication_duration_seconds_sum{session_id=\"default\"} 0.035\n")
	fmt.Fprintf(b, "fak_harness_adjudication_duration_seconds_count{session_id=\"default\"} 17\n")

	writeHelpType(b, "fak_harness_tool_auth_cache_hits_total", "JIT authentication credential cache hits.", "counter")
	fmt.Fprintf(b, "fak_harness_tool_auth_cache_hits_total{session_id=\"default\"} 15\n")

	writeHelpType(b, "fak_harness_tool_jit_secret_pages_total", "JIT secret paging events performed at the execution boundary.", "counter")
	fmt.Fprintf(b, "fak_harness_tool_jit_secret_pages_total{session_id=\"default\"} 2\n")

	writeHelpType(b, "fak_harness_tool_oauth_refreshes_total", "OAuth token refreshes executed by harness runtime.", "counter")
	fmt.Fprintf(b, "fak_harness_tool_oauth_refreshes_total{session_id=\"default\"} 1\n")

	writeHelpType(b, "fak_harness_tool_repeated_identical_calls_total", "Runaway tool calls or identical repeated calls detected.", "counter")
	fmt.Fprintf(b, "fak_harness_tool_repeated_identical_calls_total{session_id=\"default\",tool=\"read_file\"} 0\n")
}
