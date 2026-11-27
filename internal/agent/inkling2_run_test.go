// fak-test:runtime fast est=1s lane=default
package agent

import (
	"testing"
)

func TestDefaultClientWireReplayTemperatureAndDeviceGreedyGate(t *testing.T) {
	t.Run("omitted_client_temperature_resolves_non_greedy", func(t *testing.T) {
		sp := applySampleOpts(WithClientTemperature(nil))
		if !sp.InheritUpstreamTemperature {
			t.Fatalf("expected InheritUpstreamTemperature=true when client omits temperature")
		}
		if got := resolveInKernelTemperature(0, sp); got != defaultInheritedTemperature || got <= 0 {
			t.Fatalf("InKernelPlanner resolved temp=%v, want non-greedy %v", got, defaultInheritedTemperature)
		}
		if got := inkling2RunEffectiveTemperature(sp); got != defaultInheritedTemperature || got <= 0 {
			t.Fatalf("inkling2RunPlanner resolved temp=%v, want non-greedy %v", got, defaultInheritedTemperature)
		}
		if inkling2RunUseDeviceGreedy(true, inkling2RunEffectiveTemperature(sp), 0, 0, sp) {
			t.Fatalf("omitted client temperature must not route to deviceGreedy")
		}
	})

	t.Run("explicit_zero_temperature_with_penalties_or_bias_skips_device_greedy", func(t *testing.T) {
		zero := 0.0
		pres := 0.5
		spPenalty := applySampleOpts(WithClientTemperature(&zero), WithPresencePenalty(&pres))
		if spPenalty.InheritUpstreamTemperature {
			t.Fatalf("explicit temperature=0 must not set InheritUpstreamTemperature")
		}
		if got := inkling2RunEffectiveTemperature(spPenalty); got != 0 {
			t.Fatalf("explicit temperature=0 resolved temp=%v, want 0", got)
		}
		if inkling2RunUseDeviceGreedy(true, 0, 0, pres, spPenalty) {
			t.Fatalf("explicit temperature=0 + presence_penalty must route through logits sampling, not deviceGreedy")
		}

		spBias := applySampleOpts(WithClientTemperature(&zero), WithLogitBias(map[int]float64{42: -5.0}))
		if inkling2RunUseDeviceGreedy(true, 0, 0, 0, spBias) {
			t.Fatalf("explicit temperature=0 + logit_bias must route through logits sampling, not deviceGreedy")
		}

		spPlainZero := applySampleOpts(WithClientTemperature(&zero))
		if !inkling2RunUseDeviceGreedy(true, 0, 0, 0, spPlainZero) {
			t.Fatalf("explicit temperature=0 without penalties or logit_bias should allow deviceGreedy")
		}
	})
}
