// Package cloudhandoff defines explicit app-owned transitions from local to remote execution.
// Invariant: local-required data classes are fail-closed and must never escape to remote destinations.
// Guard assumption: destinations and triggers must be explicitly permitted by policy.
package cloudhandoff

import (
	"errors"
	"sort"
	"time"
)

// Schema identifies the versioned cloud handoff package protocol.
const Schema = "fak.cloud-handoff/1"

// Consent specifies the authorization mode required before remote transition.
type Consent string

const (
	// ConsentPreapproved permits handoff without interactive approval.
	ConsentPreapproved Consent = "pre_consented"
	// ConsentAsk requires interactive user or caller approval.
	ConsentAsk Consent = "ask"
	// ConsentNever strictly prohibits remote handoff.
	ConsentNever Consent = "never"
)

// Trigger represents the reason motivating execution handoff to cloud.
type Trigger string

const (
	// TriggerUnsupported indicates the requested feature cannot run locally.
	TriggerUnsupported Trigger = "unsupported_feature"
	// TriggerDeadline indicates local execution risks missing completion deadlines.
	TriggerDeadline Trigger = "deadline_risk"
	// TriggerQuality indicates local generation failed quality thresholds.
	TriggerQuality Trigger = "quality_reject"
	// TriggerPressure indicates local compute or memory resource pressure.
	TriggerPressure Trigger = "pressure"
	// TriggerFault indicates an unrecoverable local runtime fault occurred.
	TriggerFault Trigger = "runtime_fault"
	// TriggerUser indicates an explicit user request for remote execution.
	TriggerUser Trigger = "user_choice"
)

// DataClass describes a data element and whether local residency is required.
type DataClass struct {
	Name          string `json:"name"`
	LocalRequired bool   `json:"local_required"`
}

// Policy governs cloud handoff eligibility, consent, allowed destinations and triggers.
type Policy struct {
	Eligible        bool      `json:"eligible"`
	Consent         Consent   `json:"consent"`
	Destinations    []string  `json:"destinations"`
	AllowedTriggers []Trigger `json:"allowed_triggers"`
}

// Request holds parameters for a proposed transition to remote execution.
type Request struct {
	OperationID      string      `json:"operation_id"`
	Trigger          Trigger     `json:"trigger"`
	Data             []DataClass `json:"data"`
	DestinationClass string      `json:"destination_class"`
	Consequence      string      `json:"consequence"`
	Alternatives     []string    `json:"alternatives"`
	Payload          []byte      `json:"payload"`
}

// Package represents the wire envelope delivered to remote transport.
type Package struct {
	Schema           string   `json:"schema"`
	OperationID      string   `json:"operation_id"`
	Trigger          Trigger  `json:"trigger"`
	DataClasses      []string `json:"data_classes"`
	DestinationClass string   `json:"destination_class"`
	Payload          []byte   `json:"payload"`
}

// Event records a lifecycle state change or transition decision in the broker.
type Event struct {
	Sequence         uint64    `json:"sequence"`
	At               time.Time `json:"at"`
	State            string    `json:"state"`
	OperationID      string    `json:"operation_id"`
	Reason           string    `json:"reason"`
	DataClasses      []string  `json:"data_classes"`
	Consequence      string    `json:"consequence"`
	DestinationClass string    `json:"destination_class"`
	Alternatives     []string  `json:"alternatives"`
}

// Attempt records an execution attempt by engine and location.
type Attempt struct {
	Engine   string `json:"engine"`
	Location string `json:"location"`
	Outcome  string `json:"outcome"`
}

// Receipt summarizes the terminal state and execution attempts for an operation.
type Receipt struct {
	Schema          string    `json:"schema"`
	OperationID     string    `json:"operation_id"`
	Terminal        string    `json:"terminal"`
	Attempts        []Attempt `json:"attempts"`
	LocalCompleted  bool      `json:"local_completed"`
	RemoteCompleted bool      `json:"remote_completed"`
}

// Approval evaluates a proposed handoff event and returns true if permitted.
type Approval func(Event) bool

// Transport dispatches the handoff package to the remote destination.
type Transport func(Package) error

// Broker manages stateful evaluation and event auditing for cloud handoffs.
type Broker struct {
	now    func() time.Time
	events []Event
}

// ErrDenied indicates handoff was refused by policy, data residency, or user denial.
var ErrDenied = errors.New("cloudhandoff: denied")

// ErrRemote indicates handoff failed during remote transport delivery.
var ErrRemote = errors.New("cloudhandoff: remote transport failed")

// New initializes a new cloud handoff Broker with UTC timestamp provider.
func New() *Broker { return &Broker{now: func() time.Time { return time.Now().UTC() }} }

// Events returns a copy of all lifecycle events recorded by the broker.
func (b *Broker) Events() []Event { return append([]Event(nil), b.events...) }

func (b *Broker) emit(r Request, state, reason string) Event {
	classes := make([]string, len(r.Data))
	for i, d := range r.Data {
		classes[i] = d.Name
	}
	sort.Strings(classes)
	e := Event{
		Sequence:         uint64(len(b.events) + 1),
		At:               b.now(),
		State:            state,
		OperationID:      r.OperationID,
		Reason:           reason,
		DataClasses:      classes,
		Consequence:      r.Consequence,
		DestinationClass: r.DestinationClass,
		Alternatives:     append([]string(nil), r.Alternatives...),
	}
	b.events = append(b.events, e)
	return e
}

func containsTrigger(v []Trigger, w Trigger) bool {
	for _, x := range v {
		if x == w {
			return true
		}
	}
	return false
}

func contains(v []string, w string) bool {
	for _, x := range v {
		if x == w {
			return true
		}
	}
	return false
}

// Handoff coordinates policy validation, approval, packaging, and dispatch.
func (b *Broker) Handoff(p Policy, r Request, approve Approval, send Transport, local Attempt) (Receipt, error) {
	receipt := Receipt{Schema: "fak.cloud-handoff-receipt/1", OperationID: r.OperationID, Attempts: []Attempt{local}}
	deny := func(reason string) (Receipt, error) {
		b.emit(r, "denied", reason)
		receipt.Terminal = "denied"
		return receipt, ErrDenied
	}
	if r.OperationID == "" || !p.Eligible {
		return deny("ineligible")
	}
	if !containsTrigger(p.AllowedTriggers, r.Trigger) {
		return deny("trigger_not_allowed")
	}
	if !contains(p.Destinations, r.DestinationClass) {
		return deny("destination_not_allowed")
	}
	for _, d := range r.Data {
		if d.LocalRequired {
			return deny("local_required_data")
		}
	}
	proposal := b.emit(r, "proposed", string(r.Trigger))
	switch p.Consent {
	case ConsentNever:
		return deny("authority_never")
	case ConsentAsk:
		if approve == nil || !approve(proposal) {
			return deny("user_denied")
		}
	case ConsentPreapproved:
	default:
		return deny("invalid_authority")
	}
	b.emit(r, "approved", string(p.Consent))
	classes := append([]string(nil), proposal.DataClasses...)
	pkg := Package{
		Schema:           Schema,
		OperationID:      r.OperationID,
		Trigger:          r.Trigger,
		DataClasses:      classes,
		DestinationClass: r.DestinationClass,
		Payload:          append([]byte(nil), r.Payload...),
	}
	if send == nil || send(pkg) != nil {
		b.emit(r, "failed", "remote_transport")
		receipt.Attempts = append(receipt.Attempts, Attempt{Engine: "app_provider", Location: "remote", Outcome: "failed"})
		receipt.Terminal = "failed"
		return receipt, ErrRemote
	}
	b.emit(r, "completed", "remote")
	receipt.Attempts = append(receipt.Attempts, Attempt{Engine: "app_provider", Location: "remote", Outcome: "completed"})
	receipt.Terminal = "completed"
	receipt.RemoteCompleted = true
	receipt.LocalCompleted = local.Location == "local" && local.Outcome == "completed"
	return receipt, nil
}
