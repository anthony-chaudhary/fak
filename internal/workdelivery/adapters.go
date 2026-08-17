package workdelivery

import (
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/deliverystages"
)

// AdapterKind names an existing delivery seam without collapsing its state into readiness.
type AdapterKind string

const (
	AdapterCommit      AdapterKind = "commit"
	AdapterBuild       AdapterKind = "build"
	AdapterDispatch    AdapterKind = "dispatch"
	AdapterFleet       AdapterKind = "fleet"
	AdapterIntegration AdapterKind = "integration"
	AdapterRelease     AdapterKind = "release"
)

// AdapterObservation is the stable cross-seam envelope. Stage and bottleneck use the
// canonical delivery registry; Receipt changes exactly one work-delivery axis.
type AdapterObservation struct {
	Schema     string                      `json:"schema"`
	Adapter    AdapterKind                 `json:"adapter"`
	UnitID     string                      `json:"unit_id"`
	Stage      deliverystages.StageID      `json:"stage"`
	Bottleneck deliverystages.BottleneckID `json:"bottleneck,omitempty"`
	Receipt    *Receipt                    `json:"receipt,omitempty"`
	Evidence   []Evidence                  `json:"evidence,omitempty"`
	NextAction string                      `json:"next_action,omitempty"`
}

const AdapterSchema = "fak.work-delivery-adapter/v1"

func RecordingObservation(unit WorkUnit, commitSHA, owner string, at time.Time) (AdapterObservation, error) {
	receipt := Receipt{Schema: Schema, UnitID: unit.ID, Transition: Transition{Axis: AxisAuthoring, From: string(unit.Axes.Authoring), To: string(AuthoringRecorded)}, Gate: "commit", Owner: owner, ObservedAt: at.UTC(), Evidence: []Evidence{{Kind: "commit", Reference: commitSHA, Witnessed: true}}}
	if _, err := Apply(unit, receipt); err != nil {
		return AdapterObservation{}, err
	}
	return observation(AdapterCommit, unit.ID, "recording", "", &receipt, receipt.Evidence, "declare compile admission separately"), nil
}

func VerificationObservation(unit WorkUnit, passed bool, gate, evidenceRef, owner string, at time.Time) (AdapterObservation, error) {
	to := VerificationFailed
	bottleneck := deliverystages.BottleneckID("verification-failure")
	next := "inspect the exact failing check and recursively split its scope"
	if passed {
		to = VerificationPassed
		bottleneck = ""
		next = "record integration independently"
	}
	receipt := Receipt{Schema: Schema, UnitID: unit.ID, Transition: Transition{Axis: AxisVerification, From: string(unit.Axes.Verification), To: string(to)}, Gate: gate, Owner: owner, ObservedAt: at.UTC(), Evidence: []Evidence{{Kind: "verification", Reference: evidenceRef, Witnessed: true}}}
	if _, err := Apply(unit, receipt); err != nil {
		return AdapterObservation{}, err
	}
	return observation(AdapterBuild, unit.ID, "build", bottleneck, &receipt, receipt.Evidence, next), nil
}

func IntegrationObservation(unit WorkUnit, pushed bool, ref, owner string, at time.Time) (AdapterObservation, error) {
	if !pushed {
		return observation(AdapterIntegration, unit.ID, "integration", "integration-conflict", nil, []Evidence{{Kind: "integration", Reference: ref, Witnessed: true}}, "resolve the integration blocker and retry push"), nil
	}
	receipt := Receipt{Schema: Schema, UnitID: unit.ID, Transition: Transition{Axis: AxisIntegration, From: string(unit.Axes.Integration), To: string(IntegrationIntegrated)}, Gate: "push", Owner: owner, ObservedAt: at.UTC(), Evidence: []Evidence{{Kind: "integration", Reference: ref, Witnessed: true}}}
	if _, err := Apply(unit, receipt); err != nil {
		return AdapterObservation{}, err
	}
	return observation(AdapterIntegration, unit.ID, "integration", "", &receipt, receipt.Evidence, "evaluate release admission independently"), nil
}

// ReleaseReadinessObservation creates the explicit evidence required by release admission.
func ReleaseReadinessObservation(unit WorkUnit, evidenceRef, owner string, at time.Time) (AdapterObservation, error) {
	receipt := Receipt{Schema: Schema, UnitID: unit.ID, Transition: Transition{Axis: AxisRelease, From: string(unit.Axes.Release), To: string(ReleaseReady)}, Gate: "release-admission", Owner: owner, ObservedAt: at.UTC(), Evidence: []Evidence{{Kind: "release-readiness", Reference: evidenceRef, Witnessed: true}}}
	if _, err := Apply(unit, receipt); err != nil {
		return AdapterObservation{}, err
	}
	return observation(AdapterRelease, unit.ID, "release-admission", "", &receipt, receipt.Evidence, "release publication may proceed"), nil
}

// RequireReleaseReady refuses the historic inference that committed, green, or pushed means releasable.
// The matching witnessed receipt must be supplied because WorkUnit intentionally stores state, not proof.
func RequireReleaseReady(unit WorkUnit, readiness *Receipt) (AdapterObservation, error) {
	refuse := func(detail string) (AdapterObservation, error) {
		obs := observation(AdapterRelease, unit.ID, "release-admission", "release-blocked", nil, nil, "attach an explicit witnessed release-readiness receipt")
		return obs, fmt.Errorf("RELEASE_READINESS_REQUIRED: %s", detail)
	}
	if unit.Axes.Release != ReleaseReady {
		return refuse(fmt.Sprintf("unit %s is %s, not %s", unit.ID, unit.Axes.Release, ReleaseReady))
	}
	if readiness == nil || readiness.UnitID != unit.ID || readiness.Transition.Axis != AxisRelease || readiness.Transition.To != string(ReleaseReady) {
		return refuse(fmt.Sprintf("unit %s has no matching readiness receipt", unit.ID))
	}
	if len(readiness.Evidence) == 0 {
		return refuse(fmt.Sprintf("unit %s readiness receipt has no evidence", unit.ID))
	}
	for _, item := range readiness.Evidence {
		if item.Reference == "" || !item.Witnessed {
			return refuse(fmt.Sprintf("unit %s readiness evidence is not witnessed", unit.ID))
		}
	}
	return observation(AdapterRelease, unit.ID, "release-admission", "", readiness, readiness.Evidence, "release publication may proceed"), nil
}

// BlockedObservation gives dispatch and fleet one exact unit/stage/bottleneck vocabulary.
func BlockedObservation(adapter AdapterKind, unitID string, stage deliverystages.StageID, bottleneck deliverystages.BottleneckID, evidence []Evidence, next string) (AdapterObservation, error) {
	if adapter != AdapterDispatch && adapter != AdapterFleet {
		return AdapterObservation{}, fmt.Errorf("blocked observation adapter must be dispatch or fleet, got %q", adapter)
	}
	registry := deliverystages.Default()
	if _, ok := registry.Stage(stage); !ok {
		return AdapterObservation{}, fmt.Errorf("unknown delivery stage %q", stage)
	}
	found := false
	for _, candidate := range registry.Bottlenecks {
		if candidate.ID == bottleneck {
			found = true
			break
		}
	}
	if !found {
		return AdapterObservation{}, fmt.Errorf("unknown delivery bottleneck %q", bottleneck)
	}
	if unitID == "" {
		return AdapterObservation{}, fmt.Errorf("blocked observation requires exact unit_id")
	}
	return observation(adapter, unitID, stage, bottleneck, nil, evidence, next), nil
}

func observation(adapter AdapterKind, unitID string, stage deliverystages.StageID, bottleneck deliverystages.BottleneckID, receipt *Receipt, evidence []Evidence, next string) AdapterObservation {
	return AdapterObservation{Schema: AdapterSchema, Adapter: adapter, UnitID: unitID, Stage: stage, Bottleneck: bottleneck, Receipt: receipt, Evidence: evidence, NextAction: next}
}
