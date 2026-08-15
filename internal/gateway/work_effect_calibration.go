package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

const workEffectCalibrationSchema = "fak.work-effect-calibration/1"

type workEffectCalibration struct {
	Schema               string           `json:"schema"`
	RouteManifestVersion string           `json:"route_manifest_version"`
	SamplePairs          uint64           `json:"sample_pairs"`
	Routing              calibratedEffect `json:"routing"`
	Defer                calibratedEffect `json:"defer"`
}
type calibratedEffect struct {
	InputTokensDeltaPerDecision    float64 `json:"input_tokens_delta_per_decision"`
	ModelCallsDeltaPerDecision     float64 `json:"model_calls_delta_per_decision"`
	LatencySecondsDeltaPerDecision float64 `json:"latency_seconds_delta_per_decision"`
}

func calibratedWorkEffects(routeVersion string, routeDecisions, deferDecisions uint64) (guardvars.TokenSavingLever, guardvars.TokenSavingLever) {
	route := unavailableEffect(routeDecisions, "direct_default_route/v1")
	deferred := unavailableEffect(deferDecisions, "eager_full_tool_catalog/v1")
	raw := os.Getenv("FAK_WORK_EFFECT_CALIBRATION_JSON")
	if raw == "" {
		return route, deferred
	}
	var c workEffectCalibration
	if json.Unmarshal([]byte(raw), &c) != nil || c.Schema != workEffectCalibrationSchema || c.SamplePairs == 0 {
		route.Unavailable, deferred.Unavailable = "invalid_calibration", "invalid_calibration"
		return route, deferred
	}
	sum := sha256.Sum256([]byte(raw))
	fp := "sha256:" + hex.EncodeToString(sum[:])
	if c.RouteManifestVersion != routeVersion {
		route.Unavailable = "route_manifest_incompatible"
	} else {
		route = modeledEffect(routeDecisions, route.Baseline, fp, c.Routing)
	}
	deferred = modeledEffect(deferDecisions, deferred.Baseline, fp, c.Defer)
	return route, deferred
}
func unavailableEffect(n uint64, baseline string) guardvars.TokenSavingLever {
	return guardvars.TokenSavingLever{Fired: n, Units: n, Evidence: "unavailable", Unavailable: "paired_calibration_unavailable", Baseline: baseline}
}
func modeledEffect(n uint64, baseline, fp string, e calibratedEffect) guardvars.TokenSavingLever {
	return guardvars.TokenSavingLever{Fired: n, Units: n, Evidence: "modeled_calibrated", Baseline: baseline, Fingerprint: fp,
		ModeledTokens: float64(n) * e.InputTokensDeltaPerDecision, ModeledCalls: float64(n) * e.ModelCallsDeltaPerDecision, ModeledSeconds: float64(n) * e.LatencySecondsDeltaPerDecision}
}
