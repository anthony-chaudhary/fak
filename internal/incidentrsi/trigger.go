// Package incidentrsi defines the durable, content-free contract emitted when
// repeated product incidents are considered for an RSI improvement launch.
package incidentrsi

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"time"
)

// Schema is the only trigger schema major accepted by this package.
const Schema = "fak-incident-rsi-trigger/1"

const maxSideEffectFailures = 8

var (
	safeName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)
	safeRef  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:#@+-]{0,191}$`)
	hex256   = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

type IncidentClass string

type PrivacyClass string

type RedactionClass string

type DebounceState string

type TriggerDecision string

type Reason string

type IssueAction string

type SideEffectStage string

const (
	IncidentHookFailure    IncidentClass = "HOOK_FAILURE"
	IncidentGatewayFailure IncidentClass = "GATEWAY_FAILURE"

	PrivacyContentFree PrivacyClass   = "CONTENT_FREE"
	RedactionSHA256    RedactionClass = "SHA256"

	StateCollecting         DebounceState = "COLLECTING"
	StateThresholdReady     DebounceState = "THRESHOLD_READY"
	StateMaxWaitReady       DebounceState = "MAX_WAIT_READY"
	StateCooldownSuppressed DebounceState = "COOLDOWN_SUPPRESSED"

	DecisionCollect  TriggerDecision = "COLLECT"
	DecisionAdmit    TriggerDecision = "ADMIT"
	DecisionSuppress TriggerDecision = "SUPPRESS"

	ReasonBelowThreshold Reason = "BELOW_THRESHOLD"
	ReasonThreshold      Reason = "THRESHOLD_REACHED"
	ReasonMaxWait        Reason = "MAX_WAIT_REACHED"
	ReasonCooldown       Reason = "COOLDOWN_ACTIVE"

	IssueNone   IssueAction = "NONE"
	IssueCreate IssueAction = "CREATE"
	IssueUpdate IssueAction = "UPDATE"

	StageLedger SideEffectStage = "LEDGER"
	StageIssue  SideEffectStage = "ISSUE"
	StageRoute  SideEffectStage = "ROUTE"
	StageLaunch SideEffectStage = "LAUNCH"
)

type TriggerSource struct {
	Component       string        `json:"component"`
	Operation       string        `json:"operation"`
	IncidentClass   IncidentClass `json:"incident_class"`
	ProducerSchema  string        `json:"producer_schema"`
	ProducerVersion string        `json:"producer_version"`
}

type Privacy struct {
	Classification PrivacyClass   `json:"classification"`
	Redaction      RedactionClass `json:"redaction"`
}

type Observation struct {
	ID              string    `json:"id"`
	ObservedAt      time.Time `json:"observed_at"`
	OccurrenceCount int       `json:"occurrence_count"`
	BurstCount      int       `json:"burst_count"`
	FirstSeen       time.Time `json:"first_seen"`
	LastSeen        time.Time `json:"last_seen"`
}

type DebounceInputs struct {
	Threshold               int `json:"threshold"`
	CollectionWindowSeconds int `json:"collection_window_s"`
	MaxWaitSeconds          int `json:"max_wait_s"`
	CooldownSeconds         int `json:"cooldown_s"`
}

type DebounceResult struct {
	State          DebounceState `json:"state"`
	WindowStarted  time.Time     `json:"window_started_at"`
	WindowEnds     time.Time     `json:"window_ends_at"`
	MaxWaitEnds    time.Time     `json:"max_wait_ends_at"`
	CooldownUntil  *time.Time    `json:"cooldown_until,omitempty"`
	NextEligibleAt time.Time     `json:"next_eligible_at"`
}

type DurableRefs struct {
	LedgerEvent   string      `json:"ledger_event,omitempty"`
	IssueAction   IssueAction `json:"issue_action"`
	Issue         string      `json:"issue,omitempty"`
	Route         string      `json:"route,omitempty"`
	LaunchReceipt string      `json:"launch_receipt,omitempty"`
}

type SideEffectFailure struct {
	Stage     SideEffectStage `json:"stage"`
	Code      string          `json:"code"`
	Retryable bool            `json:"retryable"`
}

// Trigger is immutable by convention after construction. Validate, Encode, and
// Identity only read it and are safe to call concurrently when callers do not
// mutate the value or its SideEffectFailures slice.
type Trigger struct {
	Schema                          string              `json:"schema"`
	Source                          TriggerSource       `json:"source"`
	Fingerprint                     string              `json:"fingerprint"`
	Privacy                         Privacy             `json:"privacy"`
	Observation                     Observation         `json:"observation"`
	DebounceInputs                  DebounceInputs      `json:"debounce_inputs"`
	DebounceResult                  DebounceResult      `json:"debounce_result"`
	Decision                        TriggerDecision     `json:"decision"`
	Reason                          Reason              `json:"reason"`
	Refs                            DurableRefs         `json:"refs"`
	SideEffectFailures              []SideEffectFailure `json:"side_effect_failures,omitempty"`
	OriginalProductFailurePreserved bool                `json:"original_product_failure_preserved"`
}

// DecodeStrict is the compatibility policy for v1: unknown fields are rejected.
// A future additive-field mode must be explicit rather than silently weakening
// this decoder.
func DecodeStrict(data []byte) (Trigger, error) {
	var trigger Trigger
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&trigger); err != nil {
		return Trigger{}, fmt.Errorf("decode incident RSI trigger: %w", err)
	}
	if err := ensureEOF(dec); err != nil {
		return Trigger{}, err
	}
	if err := trigger.Validate(); err != nil {
		return Trigger{}, err
	}
	return trigger, nil
}

func ensureEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("decode incident RSI trigger: trailing JSON value")
		}
		return fmt.Errorf("decode incident RSI trigger trailing data: %w", err)
	}
	return nil
}

// Encode validates and returns the canonical field-ordered JSON representation.
func (t Trigger) Encode() ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(t)
}

// Identity is the lowercase SHA-256 of Encode's canonical bytes.
func (t Trigger) Identity() (string, error) {
	encoded, err := t.Encode()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func (t Trigger) Validate() error {
	if t.Schema != Schema {
		return fmt.Errorf("schema must be %q", Schema)
	}
	if err := t.Source.validate(); err != nil {
		return err
	}
	if !hex256.MatchString(t.Fingerprint) {
		return errors.New("fingerprint must be a lowercase content-free SHA-256 digest")
	}
	if t.Privacy.Classification != PrivacyContentFree || t.Privacy.Redaction != RedactionSHA256 {
		return errors.New("privacy must declare CONTENT_FREE classification with SHA256 redaction")
	}
	if err := t.Observation.validate(); err != nil {
		return err
	}
	if err := t.DebounceInputs.validate(); err != nil {
		return err
	}
	if err := t.validateDebounceDecision(); err != nil {
		return err
	}
	if err := t.Refs.validate(t.Decision); err != nil {
		return err
	}
	if len(t.SideEffectFailures) > maxSideEffectFailures {
		return fmt.Errorf("side_effect_failures exceeds limit %d", maxSideEffectFailures)
	}
	for i, failure := range t.SideEffectFailures {
		if err := failure.validate(); err != nil {
			return fmt.Errorf("side_effect_failures[%d]: %w", i, err)
		}
	}
	if !t.OriginalProductFailurePreserved {
		return errors.New("original_product_failure_preserved must be true")
	}
	return nil
}

func (s TriggerSource) validate() error {
	if !safeName.MatchString(s.Component) || !safeName.MatchString(s.Operation) ||
		!safeName.MatchString(s.ProducerSchema) || !safeName.MatchString(s.ProducerVersion) {
		return errors.New("source fields must be bounded content-free identifiers")
	}
	switch s.IncidentClass {
	case IncidentHookFailure, IncidentGatewayFailure:
		return nil
	default:
		return fmt.Errorf("unknown incident_class %q", s.IncidentClass)
	}
}

func (o Observation) validate() error {
	if !safeRef.MatchString(o.ID) {
		return errors.New("observation id must be a bounded content-free reference")
	}
	if o.ObservedAt.IsZero() || o.FirstSeen.IsZero() || o.LastSeen.IsZero() {
		return errors.New("observation timestamps are required")
	}
	if o.FirstSeen.After(o.LastSeen) || !o.LastSeen.Equal(o.ObservedAt) {
		return errors.New("observation requires first_seen <= last_seen == observed_at")
	}
	if o.OccurrenceCount < 1 || o.BurstCount < 1 || o.BurstCount > o.OccurrenceCount {
		return errors.New("observation counts require occurrence_count >= burst_count >= 1")
	}
	return nil
}

func (d DebounceInputs) validate() error {
	if d.Threshold < 1 || d.CollectionWindowSeconds < 1 || d.MaxWaitSeconds < 1 || d.CooldownSeconds < 1 {
		return errors.New("debounce threshold, collection window, max wait, and cooldown must be positive")
	}
	if d.MaxWaitSeconds < d.CollectionWindowSeconds {
		return errors.New("debounce max wait must not be shorter than collection window")
	}
	return nil
}

func (t Trigger) validateDebounceDecision() error {
	d := t.DebounceResult
	o := t.Observation
	i := t.DebounceInputs
	if d.WindowStarted.IsZero() || d.WindowEnds.IsZero() || d.MaxWaitEnds.IsZero() || d.NextEligibleAt.IsZero() {
		return errors.New("debounce result timestamps are required")
	}
	if !d.WindowStarted.Equal(o.FirstSeen) || !d.WindowEnds.Equal(d.WindowStarted.Add(time.Duration(i.CollectionWindowSeconds)*time.Second)) ||
		!d.MaxWaitEnds.Equal(d.WindowStarted.Add(time.Duration(i.MaxWaitSeconds)*time.Second)) {
		return errors.New("debounce result bounds do not match first_seen and configured durations")
	}

	var expectedDecision TriggerDecision
	var expectedReason Reason
	var expectedNext time.Time

	// State is part of the closed decision contract. A future cooldown timestamp
	// on an admission is the newly established cooldown, not evidence that this
	// observation was suppressed by an existing cooldown.
	switch d.State {
	case StateThresholdReady:
		expectedDecision, expectedReason = DecisionAdmit, ReasonThreshold
		if o.BurstCount < i.Threshold {
			return errors.New("threshold admission requires burst_count at or above threshold")
		}
		expectedNext = o.ObservedAt.Add(time.Duration(i.CooldownSeconds) * time.Second)
	case StateMaxWaitReady:
		expectedDecision, expectedReason = DecisionAdmit, ReasonMaxWait
		if o.BurstCount >= i.Threshold {
			return errors.New("threshold admission takes precedence over max-wait admission")
		}
		if o.ObservedAt.Before(d.MaxWaitEnds) {
			return errors.New("max-wait admission requires observation at or after max_wait_ends_at")
		}
		expectedNext = o.ObservedAt.Add(time.Duration(i.CooldownSeconds) * time.Second)
	case StateCooldownSuppressed:
		expectedDecision, expectedReason = DecisionSuppress, ReasonCooldown
		if d.CooldownUntil == nil || !d.CooldownUntil.After(o.ObservedAt) {
			return errors.New("cooldown suppression requires an active cooldown after observed_at")
		}
		expectedNext = *d.CooldownUntil
	case StateCollecting:
		expectedDecision, expectedReason = DecisionCollect, ReasonBelowThreshold
		if o.BurstCount >= i.Threshold {
			return errors.New("collecting decision requires burst_count below threshold")
		}
		if !o.ObservedAt.Before(d.MaxWaitEnds) {
			return errors.New("collecting decision requires observation before max_wait_ends_at")
		}
		if !o.ObservedAt.Before(d.WindowEnds) {
			return errors.New("collecting observation must remain inside collection window; start a new burst window")
		}
		if d.CooldownUntil != nil {
			return errors.New("collecting decision must not carry cooldown_until")
		}
		expectedNext = d.WindowEnds
		if d.MaxWaitEnds.Before(expectedNext) {
			expectedNext = d.MaxWaitEnds
		}
	default:
		return fmt.Errorf("unknown debounce state %q", d.State)
	}

	if t.Decision != expectedDecision || t.Reason != expectedReason {
		return fmt.Errorf("debounce decision must be %s/%s for state %s", expectedDecision, expectedReason, d.State)
	}
	if !d.NextEligibleAt.Equal(expectedNext) {
		return errors.New("next_eligible_at does not match debounce/cooldown decision")
	}
	if expectedDecision == DecisionAdmit && (d.CooldownUntil == nil || !d.CooldownUntil.Equal(expectedNext)) {
		return errors.New("admitted decision requires cooldown_until equal to next_eligible_at")
	}
	return nil
}

func (r DurableRefs) validate(decision TriggerDecision) error {
	if r.IssueAction != IssueNone && r.IssueAction != IssueCreate && r.IssueAction != IssueUpdate {
		return fmt.Errorf("unknown issue_action %q", r.IssueAction)
	}
	for name, value := range map[string]string{
		"ledger_event": r.LedgerEvent, "issue": r.Issue, "route": r.Route, "launch_receipt": r.LaunchReceipt,
	} {
		if value != "" && !safeRef.MatchString(value) {
			return fmt.Errorf("%s must be a bounded content-free reference", name)
		}
	}
	if decision == DecisionAdmit {
		if r.LedgerEvent == "" || r.Issue == "" || r.Route == "" || r.LaunchReceipt == "" || (r.IssueAction != IssueCreate && r.IssueAction != IssueUpdate) {
			return errors.New("admitted decision requires ledger, issue create/update, route, and launch receipt references")
		}
		return nil
	}
	if r.LedgerEvent != "" || r.Issue != "" || r.Route != "" || r.LaunchReceipt != "" || r.IssueAction != IssueNone {
		return errors.New("non-admitted decision must not claim durable side-effect references")
	}
	return nil
}

func (f SideEffectFailure) validate() error {
	switch f.Stage {
	case StageLedger, StageIssue, StageRoute, StageLaunch:
	default:
		return fmt.Errorf("unknown side-effect stage %q", f.Stage)
	}
	if !safeName.MatchString(f.Code) {
		return errors.New("code must be a bounded content-free identifier")
	}
	return nil
}
