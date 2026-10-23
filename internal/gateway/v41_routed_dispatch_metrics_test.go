package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The V4.1 routed-dispatch family renders per phase and arm, and stays silent
// until a batched routed layer ran (fak#13820).
// fak-test:runtime fast est=1s lane=default
func TestV41RoutedDispatchMetrics(t *testing.T) {
	var empty strings.Builder
	writeV41RoutedDispatchMetrics(&empty, nil)
	writeV41RoutedDispatchMetrics(&empty, &agent.V41PhaseLifetimeSnapshot{})
	if empty.Len() != 0 {
		t.Fatalf("no routed layer yet, rendered %q", empty.String())
	}
	var b strings.Builder
	writeV41RoutedDispatchMetrics(&b, &agent.V41PhaseLifetimeSnapshot{Routed: agent.V41RoutedPhaseLifetimeSnapshot{
		Prefill: agent.V41RoutedPhaseSnapshot{Tokens: 7, RoutedLayers: 43, RoutedLayerDispatches: 43 * 36},
		Decode:  agent.V41RoutedPhaseSnapshot{Tokens: 16, RoutedLayers: 16 * 43, RoutedIDLayers: 16 * 43, RoutedLayerDispatches: 16 * 43 * 5},
	}})
	for _, line := range []string{
		`fak_gateway_v41_routed_dispatches_total{phase="decode"} 3440`,
		`fak_gateway_v41_routed_dispatches_total{phase="prefill"} 1548`,
		`fak_gateway_v41_routed_layers_total{phase="decode",arm="id"} 688`,
		`fak_gateway_v41_routed_layers_total{phase="decode",arm="per_pick"} 0`,
		`fak_gateway_v41_routed_layers_total{phase="prefill",arm="per_pick"} 43`,
		`fak_gateway_v41_phase_tokens_total{phase="decode"} 16`,
	} {
		if !strings.Contains(b.String(), line+"\n") {
			t.Errorf("missing %q", line)
		}
	}
}

// The CPU-miss family renders only once the host arm computed a miss.
// fak-test:runtime fast est=1s lane=default
func TestV41CPUMissMetrics(t *testing.T) {
	var off strings.Builder
	writeV41RoutedDispatchMetrics(&off, &agent.V41PhaseLifetimeSnapshot{Routed: agent.V41RoutedPhaseLifetimeSnapshot{
		Decode: agent.V41RoutedPhaseSnapshot{Tokens: 4, RoutedLayers: 4 * 43, RoutedLayerDispatches: 4 * 43 * 2},
	}})
	if strings.Contains(off.String(), "cpu_miss") {
		t.Fatalf("host arm never ran, rendered %q", off.String())
	}
	var b strings.Builder
	writeV41RoutedDispatchMetrics(&b, &agent.V41PhaseLifetimeSnapshot{Routed: agent.V41RoutedPhaseLifetimeSnapshot{
		Decode: agent.V41RoutedPhaseSnapshot{Tokens: 4, RoutedLayers: 4 * 43, RoutedLayerDispatches: 4 * 43 * 2,
			CPUMissExperts: 300, CPUMissStreamed: 20, CPUMissNanos: 1_500_000_000, CPUMissJoinWaitNanos: 250_000_000},
	}})
	for _, line := range []string{
		`fak_gateway_v41_cpu_miss_experts_total{phase="decode"} 300`,
		`fak_gateway_v41_cpu_miss_experts_total{phase="prefill"} 0`,
		`fak_gateway_v41_cpu_miss_streamed_total{phase="decode"} 20`,
		`fak_gateway_v41_cpu_miss_seconds_total{phase="decode"} 1.5`,
		`fak_gateway_v41_cpu_miss_join_wait_seconds_total{phase="decode"} 0.25`,
	} {
		if !strings.Contains(b.String(), line+"\n") {
			t.Errorf("missing %q", line)
		}
	}
}
