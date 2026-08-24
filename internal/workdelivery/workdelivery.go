// Package workdelivery defines the versioned contract for independently tracking
// authored work, compile admission, verification, integration, release, activation, and operator acceptance.
package workdelivery

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

const Schema = "fak.work-delivery/v1"

type Axis string

const (
	AxisAuthoring    Axis = "authoring"
	AxisAdmission    Axis = "compile_admission"
	AxisVerification Axis = "verification"
	AxisIntegration  Axis = "integration"
	AxisRelease      Axis = "release"
	AxisActivation   Axis = "activation"
	AxisAcceptance   Axis = "operator_acceptance"
)

type AuthoringState string
type AdmissionState string
type VerificationState string
type IntegrationState string
type ReleaseState string
type ActivationState string
type AcceptanceState string

const (
	AuthoringDraft    AuthoringState = "draft"
	AuthoringRecorded AuthoringState = "recorded"

	AdmissionUndeclared AdmissionState = "undeclared"
	AdmissionExcluded   AdmissionState = "excluded"
	AdmissionAdmitted   AdmissionState = "admitted"

	VerificationUnverified VerificationState = "unverified"
	VerificationPassed     VerificationState = "passed"
	VerificationFailed     VerificationState = "failed"

	IntegrationUnintegrated IntegrationState = "unintegrated"
	IntegrationIntegrated   IntegrationState = "integrated"

	ReleaseNotReady ReleaseState = "not_ready"
	ReleaseReady    ReleaseState = "ready"
	ReleaseReleased ReleaseState = "released"

	ActivationInactive  ActivationState = "inactive"
	ActivationActivated ActivationState = "activated"

	AcceptanceUnaccepted AcceptanceState = "unaccepted"
	AcceptanceAccepted   AcceptanceState = "accepted"
)

type Axes struct {
	Authoring    AuthoringState    `json:"authoring"`
	Admission    AdmissionState    `json:"compile_admission"`
	Verification VerificationState `json:"verification"`
	Integration  IntegrationState  `json:"integration"`
	Release      ReleaseState      `json:"release"`
	Activation   ActivationState   `json:"activation,omitempty"`
	Acceptance   AcceptanceState   `json:"operator_acceptance,omitempty"`
}

func InitialAxes() Axes {
	return Axes{Authoring: AuthoringDraft, Admission: AdmissionUndeclared, Verification: VerificationUnverified, Integration: IntegrationUnintegrated, Release: ReleaseNotReady, Activation: ActivationInactive, Acceptance: AcceptanceUnaccepted}
}

type Artifact struct {
	Path   string `json:"path"`
	Digest string `json:"digest,omitempty"`
	Kind   string `json:"kind,omitempty"`
}

type RuntimeIdentity struct {
	Revision     string `json:"revision"`
	BuildDigest  string `json:"build_digest"`
	ConfigDigest string `json:"config_digest"`
}

type OperatorAcceptance struct {
	RuntimeIdentity
	Journey string `json:"journey"`
}

type WorkUnit struct {
	Schema    string              `json:"schema"`
	ID        string              `json:"id"`
	Revision  string              `json:"revision,omitempty"`
	Artifacts []Artifact          `json:"artifacts,omitempty"`
	Axes      Axes                `json:"axes"`
	Activated *RuntimeIdentity    `json:"activated,omitempty"`
	Accepted  *OperatorAcceptance `json:"accepted,omitempty"`
}

type Evidence struct {
	Kind      string `json:"kind"`
	Reference string `json:"reference"`
	Digest    string `json:"digest,omitempty"`
	Witnessed bool   `json:"witnessed"`
}

type Blocker struct {
	Code                 string `json:"code"`
	Gate                 string `json:"gate"`
	Detail               string `json:"detail,omitempty"`
	MissingDiscriminator string `json:"missing_discriminator,omitempty"`
	NextAction           string `json:"next_action,omitempty"`
}

type Transition struct {
	Axis Axis   `json:"axis"`
	From string `json:"from"`
	To   string `json:"to"`
}

type Receipt struct {
	Schema          string           `json:"schema"`
	UnitID          string           `json:"unit_id"`
	Transition      Transition       `json:"transition"`
	Gate            string           `json:"gate"`
	Evidence        []Evidence       `json:"evidence,omitempty"`
	Owner           string           `json:"owner,omitempty"`
	Lease           string           `json:"lease,omitempty"`
	ObservedAt      time.Time        `json:"observed_at"`
	Blocker         *Blocker         `json:"blocker,omitempty"`
	RuntimeIdentity *RuntimeIdentity `json:"runtime_identity,omitempty"`
	Journey         string           `json:"journey,omitempty"`
}

func (u WorkUnit) Validate() error {
	if u.Schema != Schema {
		return fmt.Errorf("schema: want %q, got %q", Schema, u.Schema)
	}
	if strings.TrimSpace(u.ID) == "" {
		return errors.New("id is required")
	}
	for i, artifact := range u.Artifacts {
		if strings.TrimSpace(artifact.Path) == "" {
			return fmt.Errorf("artifacts[%d].path is required", i)
		}
	}
	if !oneOf(string(u.Axes.Authoring), string(AuthoringDraft), string(AuthoringRecorded)) {
		return fmt.Errorf("unknown authoring state %q", u.Axes.Authoring)
	}
	if !oneOf(string(u.Axes.Admission), string(AdmissionUndeclared), string(AdmissionExcluded), string(AdmissionAdmitted)) {
		return fmt.Errorf("unknown compile admission state %q", u.Axes.Admission)
	}
	if !oneOf(string(u.Axes.Verification), string(VerificationUnverified), string(VerificationPassed), string(VerificationFailed)) {
		return fmt.Errorf("unknown verification state %q", u.Axes.Verification)
	}
	if !oneOf(string(u.Axes.Integration), string(IntegrationUnintegrated), string(IntegrationIntegrated)) {
		return fmt.Errorf("unknown integration state %q", u.Axes.Integration)
	}
	if !oneOf(string(u.Axes.Release), string(ReleaseNotReady), string(ReleaseReady), string(ReleaseReleased)) {
		return fmt.Errorf("unknown release state %q", u.Axes.Release)
	}
	return nil
}

func (r Receipt) Validate(unit WorkUnit) error {
	if err := unit.Validate(); err != nil {
		return fmt.Errorf("unit: %w", err)
	}
	if r.Schema != Schema {
		return fmt.Errorf("schema: want %q, got %q", Schema, r.Schema)
	}
	if r.UnitID != unit.ID {
		return fmt.Errorf("unit_id %q does not match %q", r.UnitID, unit.ID)
	}
	if strings.TrimSpace(r.Gate) == "" {
		return errors.New("gate is required")
	}
	if r.ObservedAt.IsZero() {
		return errors.New("observed_at is required")
	}
	current, err := unit.Axes.state(r.Transition.Axis)
	if err != nil {
		return err
	}
	if r.Transition.From != current {
		return fmt.Errorf("stale transition: %s is %q, not %q", r.Transition.Axis, current, r.Transition.From)
	}
	if !validTransition(r.Transition.Axis, r.Transition.From, r.Transition.To) {
		return fmt.Errorf("illegal %s transition %q -> %q", r.Transition.Axis, r.Transition.From, r.Transition.To)
	}
	if r.Blocker != nil {
		if strings.TrimSpace(r.Blocker.Code) == "" || strings.TrimSpace(r.Blocker.Gate) == "" {
			return errors.New("blocker code and gate are required")
		}
		if r.Blocker.Gate != r.Gate {
			return errors.New("blocker gate does not match receipt gate")
		}
	}
	for i, evidence := range r.Evidence {
		if strings.TrimSpace(evidence.Kind) == "" || strings.TrimSpace(evidence.Reference) == "" {
			return fmt.Errorf("evidence[%d] kind and reference are required", i)
		}
	}
	return nil
}

// Apply updates only the axis named by a successful receipt. In particular,
// recording a commit cannot implicitly admit, verify, integrate, or release it.
func Apply(unit WorkUnit, receipt Receipt) (WorkUnit, error) {
	if err := receipt.Validate(unit); err != nil {
		return WorkUnit{}, err
	}
	if receipt.Blocker != nil {
		return unit, nil
	}
	if receipt.Transition.Axis == AxisActivation {
		if err := validateActivationReceipt(unit, receipt); err != nil {
			return WorkUnit{}, err
		}
	}
	if receipt.Transition.Axis == AxisAcceptance {
		if err := validateAcceptanceReceipt(unit, receipt); err != nil {
			return WorkUnit{}, err
		}
	}
	if err := unit.Axes.set(receipt.Transition.Axis, receipt.Transition.To); err != nil {
		return WorkUnit{}, err
	}
	if receipt.Transition.Axis == AxisActivation && receipt.Transition.To == string(ActivationActivated) {
		identity := *receipt.RuntimeIdentity
		unit.Activated = &identity
	}
	if receipt.Transition.Axis == AxisAcceptance && receipt.Transition.To == string(AcceptanceAccepted) {
		unit.Accepted = &OperatorAcceptance{RuntimeIdentity: *receipt.RuntimeIdentity, Journey: receipt.Journey}
	}
	return unit, nil
}

func (a Axes) state(axis Axis) (string, error) {
	switch axis {
	case AxisAuthoring:
		return string(a.Authoring), nil
	case AxisAdmission:
		return string(a.Admission), nil
	case AxisVerification:
		return string(a.Verification), nil
	case AxisIntegration:
		return string(a.Integration), nil
	case AxisRelease:
		return string(a.Release), nil
	case AxisActivation:
		return string(normalizeActivation(a.Activation)), nil
	case AxisAcceptance:
		return string(normalizeAcceptance(a.Acceptance)), nil
	default:
		return "", fmt.Errorf("unknown axis %q", axis)
	}
}

func (a *Axes) set(axis Axis, state string) error {
	switch axis {
	case AxisAuthoring:
		a.Authoring = AuthoringState(state)
	case AxisAdmission:
		a.Admission = AdmissionState(state)
	case AxisVerification:
		a.Verification = VerificationState(state)
	case AxisIntegration:
		a.Integration = IntegrationState(state)
	case AxisRelease:
		a.Release = ReleaseState(state)
	case AxisActivation:
		a.Activation = ActivationState(state)
	case AxisAcceptance:
		a.Acceptance = AcceptanceState(state)
	default:
		return fmt.Errorf("unknown axis %q", axis)
	}
	return nil
}

func validTransition(axis Axis, from, to string) bool {
	allowed := map[Axis]map[string][]string{
		AxisAuthoring:    {string(AuthoringDraft): {string(AuthoringRecorded)}, string(AuthoringRecorded): {string(AuthoringDraft)}},
		AxisAdmission:    {string(AdmissionUndeclared): {string(AdmissionExcluded), string(AdmissionAdmitted)}, string(AdmissionExcluded): {string(AdmissionAdmitted), string(AdmissionUndeclared)}, string(AdmissionAdmitted): {string(AdmissionExcluded), string(AdmissionUndeclared)}},
		AxisVerification: {string(VerificationUnverified): {string(VerificationPassed), string(VerificationFailed)}, string(VerificationPassed): {string(VerificationUnverified), string(VerificationFailed)}, string(VerificationFailed): {string(VerificationUnverified), string(VerificationPassed)}},
		AxisIntegration:  {string(IntegrationUnintegrated): {string(IntegrationIntegrated)}, string(IntegrationIntegrated): {string(IntegrationUnintegrated)}},
		AxisRelease:      {string(ReleaseNotReady): {string(ReleaseReady)}, string(ReleaseReady): {string(ReleaseNotReady), string(ReleaseReleased)}, string(ReleaseReleased): {string(ReleaseNotReady)}},
		AxisActivation:   {string(ActivationInactive): {string(ActivationActivated)}, string(ActivationActivated): {string(ActivationInactive)}},
		AxisAcceptance:   {string(AcceptanceUnaccepted): {string(AcceptanceAccepted)}, string(AcceptanceAccepted): {string(AcceptanceUnaccepted)}},
	}
	for _, candidate := range allowed[axis][from] {
		if candidate == to {
			return true
		}
	}
	return false
}

func oneOf(value string, values ...string) bool {
	for _, candidate := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

func normalizeActivation(v ActivationState) ActivationState {
	if v == "" {
		return ActivationInactive
	}
	return v
}
func normalizeAcceptance(v AcceptanceState) AcceptanceState {
	if v == "" {
		return AcceptanceUnaccepted
	}
	return v
}
func validateActivationReceipt(unit WorkUnit, receipt Receipt) error {
	if receipt.Transition.To != string(ActivationActivated) {
		return nil
	}
	if receipt.RuntimeIdentity == nil {
		return errors.New("activation receipt requires runtime_identity")
	}
	if unit.Revision == "" || receipt.RuntimeIdentity.Revision != unit.Revision {
		return fmt.Errorf("activation revision %q does not match unit revision %q", receipt.RuntimeIdentity.Revision, unit.Revision)
	}
	if strings.TrimSpace(receipt.RuntimeIdentity.BuildDigest) == "" || strings.TrimSpace(receipt.RuntimeIdentity.ConfigDigest) == "" {
		return errors.New("activation receipt requires build_digest and config_digest")
	}
	if !receiptHasAttestedArtifact(receipt.Evidence) {
		return errors.New("activation receipt requires witnessed evidence")
	}
	return nil
}
func validateAcceptanceReceipt(unit WorkUnit, receipt Receipt) error {
	if receipt.Transition.To != string(AcceptanceAccepted) {
		return nil
	}
	if unit.Activated == nil || unit.Axes.Activation != ActivationActivated {
		return errors.New("operator acceptance requires an activated runtime")
	}
	if receipt.RuntimeIdentity == nil || *receipt.RuntimeIdentity != *unit.Activated {
		return errors.New("operator acceptance runtime_identity does not match activated build/config")
	}
	if strings.TrimSpace(receipt.Journey) == "" {
		return errors.New("operator acceptance requires a named journey")
	}
	if !receiptHasAttestedArtifact(receipt.Evidence) {
		return errors.New("operator acceptance requires witnessed evidence")
	}
	return nil
}
func receiptHasAttestedArtifact(evidence []Evidence) bool {
	for _, item := range evidence {
		if item.Witnessed && strings.TrimSpace(item.Reference) != "" {
			return true
		}
	}
	return false
}
