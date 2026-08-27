// Package cloudhandoff defines explicit app-owned transitions from local to remote execution.
package cloudhandoff

import (
	"errors"
	"sort"
	"time"
)

const Schema = "fak.cloud-handoff/1"

type Consent string

const (
	ConsentPreapproved Consent = "pre_consented"
	ConsentAsk         Consent = "ask"
	ConsentNever       Consent = "never"
)

type Trigger string

const (
	TriggerUnsupported Trigger = "unsupported_feature"
	TriggerDeadline    Trigger = "deadline_risk"
	TriggerQuality     Trigger = "quality_reject"
	TriggerPressure    Trigger = "pressure"
	TriggerFault       Trigger = "runtime_fault"
	TriggerUser        Trigger = "user_choice"
)

type DataClass struct {
	Name          string `json:"name"`
	LocalRequired bool   `json:"local_required"`
}
type Policy struct {
	Eligible        bool      `json:"eligible"`
	Consent         Consent   `json:"consent"`
	Destinations    []string  `json:"destinations"`
	AllowedTriggers []Trigger `json:"allowed_triggers"`
}
type Request struct {
	OperationID      string      `json:"operation_id"`
	Trigger          Trigger     `json:"trigger"`
	Data             []DataClass `json:"data"`
	DestinationClass string      `json:"destination_class"`
	Consequence      string      `json:"consequence"`
	Alternatives     []string    `json:"alternatives"`
	Payload          []byte      `json:"payload"`
}
type Package struct {
	Schema           string   `json:"schema"`
	OperationID      string   `json:"operation_id"`
	Trigger          Trigger  `json:"trigger"`
	DataClasses      []string `json:"data_classes"`
	DestinationClass string   `json:"destination_class"`
	Payload          []byte   `json:"payload"`
}
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
type Attempt struct {
	Engine   string `json:"engine"`
	Location string `json:"location"`
	Outcome  string `json:"outcome"`
}
type Receipt struct {
	Schema          string    `json:"schema"`
	OperationID     string    `json:"operation_id"`
	Terminal        string    `json:"terminal"`
	Attempts        []Attempt `json:"attempts"`
	LocalCompleted  bool      `json:"local_completed"`
	RemoteCompleted bool      `json:"remote_completed"`
}
type Approval func(Event) bool
type Transport func(Package) error
type Broker struct {
	now    func() time.Time
	events []Event
}

var ErrDenied = errors.New("cloudhandoff: denied")
var ErrRemote = errors.New("cloudhandoff: remote transport failed")

func New() *Broker                { return &Broker{now: func() time.Time { return time.Now().UTC() }} }
func (b *Broker) Events() []Event { return append([]Event(nil), b.events...) }
func (b *Broker) emit(r Request, state, reason string) Event {
	classes := make([]string, len(r.Data))
	for i, d := range r.Data {
		classes[i] = d.Name
	}
	sort.Strings(classes)
	e := Event{Sequence: uint64(len(b.events) + 1), At: b.now(), State: state, OperationID: r.OperationID, Reason: reason, DataClasses: classes, Consequence: r.Consequence, DestinationClass: r.DestinationClass, Alternatives: append([]string(nil), r.Alternatives...)}
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
	pkg := Package{Schema: Schema, OperationID: r.OperationID, Trigger: r.Trigger, DataClasses: classes, DestinationClass: r.DestinationClass, Payload: append([]byte(nil), r.Payload...)}
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
