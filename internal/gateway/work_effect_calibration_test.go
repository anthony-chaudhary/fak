package gateway

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestCalibratedWorkEffectsUnavailableDoesNotConvertDecisionsToSavings(t *testing.T) {
	t.Setenv("FAK_WORK_EFFECT_CALIBRATION_JSON", "")
	route, deferred := calibratedWorkEffects("fak-route/v1", 3, 5)
	if route.Fired != 3 || deferred.Fired != 5 || route.Evidence != "unavailable" || deferred.Evidence != "unavailable" {
		t.Fatalf("route=%+v deferred=%+v", route, deferred)
	}
	if route.ModeledTokens != 0 || route.ModeledCalls != 0 || route.ModeledSeconds != 0 {
		t.Fatalf("decision count became savings: %+v", route)
	}
}

func TestCalibratedWorkEffectsPairedAndCompatibilityGated(t *testing.T) {
	t.Setenv("FAK_WORK_EFFECT_CALIBRATION_JSON", `{"schema":"fak.work-effect-calibration/1","route_manifest_version":"fak-route/v1","sample_pairs":20,"routing":{"input_tokens_delta_per_decision":10,"model_calls_delta_per_decision":-0.1,"latency_seconds_delta_per_decision":0.5},"defer":{"input_tokens_delta_per_decision":4,"model_calls_delta_per_decision":0,"latency_seconds_delta_per_decision":0.2}}`)
	route, deferred := calibratedWorkEffects("fak-route/v1", 3, 5)
	if route.Evidence != "modeled_calibrated" || route.ModeledTokens != 30 || (route.ModeledCalls < -0.301 || route.ModeledCalls > -0.299) || route.ModeledSeconds != 1.5 || route.Fingerprint == "" {
		t.Fatalf("route=%+v", route)
	}
	if deferred.ModeledTokens != 20 || deferred.ModeledSeconds != 1 || deferred.Baseline != "eager_full_tool_catalog/v1" {
		t.Fatalf("deferred=%+v", deferred)
	}
	incompatible, _ := calibratedWorkEffects("route/v2", 3, 5)
	if incompatible.Evidence != "unavailable" || incompatible.Unavailable != "route_manifest_incompatible" || incompatible.ModeledTokens != 0 {
		t.Fatalf("incompatible=%+v", incompatible)
	}
}

func TestTokenSavingsDebugSeparatesRoutingDecisionFromModeledEffects(t *testing.T) {
	m := &modelroute.Manifest{Version: "fak-route/v1", Default: modelroute.Plan{Members: []modelroute.Member{{Model: "test"}}}}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	s := routeServer(t, m)
	if _, err := s.buildCall(context.Background(), "read", `{}`, true, "", ""); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_WORK_EFFECT_CALIBRATION_JSON", "")
	got := s.debugVars(time.Now()).TokenSavings.ModelRouting
	if got.Fired != 1 || got.Evidence != "unavailable" || got.ModeledTokens != 0 || got.ModeledCalls != 0 || got.ModeledSeconds != 0 {
		t.Fatalf("uncalibrated=%+v", got)
	}
	t.Setenv("FAK_WORK_EFFECT_CALIBRATION_JSON", `{"schema":"fak.work-effect-calibration/1","route_manifest_version":"fak-route/v1","sample_pairs":8,"routing":{"input_tokens_delta_per_decision":12,"latency_seconds_delta_per_decision":0.4}}`)
	got = s.debugVars(time.Now()).TokenSavings.ModelRouting
	if got.Fired != 1 || got.Evidence != "modeled_calibrated" || got.ModeledTokens != 12 || got.ModeledSeconds != 0.4 || got.Fingerprint == "" {
		t.Fatalf("calibrated=%+v", got)
	}
}
