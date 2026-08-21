package orchestration

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const EffectSuccessorReceiptSchema = "fak-orchestration-effect-successor/1"

type AccessMode string

const (
	AccessObserve AccessMode = "observe"
	AccessEffect  AccessMode = "effect"
)

type EffectSuccessorRefusalReason string

const (
	EffectSuccessorInvalid          EffectSuccessorRefusalReason = "INVALID_EFFECT_SUCCESSOR"
	EffectSuccessorObserverWrite    EffectSuccessorRefusalReason = "OBSERVER_EFFECT_DENIED"
	EffectSuccessorStaleObservation EffectSuccessorRefusalReason = "STALE_OBSERVATION"
	EffectSuccessorEnvelopeWidening EffectSuccessorRefusalReason = "ENVELOPE_WIDENING"
	EffectSuccessorBudgetExhausted  EffectSuccessorRefusalReason = "BUDGET_EXHAUSTED"
	EffectSuccessorLeaseContention  EffectSuccessorRefusalReason = "COLLISION_RISK"
)

type EffectSuccessorRefusal struct {
	Reason EffectSuccessorRefusalReason `json:"reason"`
	Detail string                       `json:"detail"`
}

func (r *EffectSuccessorRefusal) Error() string {
	return fmt.Sprintf("%s: %s", r.Reason, r.Detail)
}

type NodeAccess struct {
	Mode     AccessMode `json:"mode"`
	ReadSet  []string   `json:"read_set,omitempty"`
	WriteSet []string   `json:"write_set,omitempty"`
}

type ObserverNode struct {
	ID     string     `json:"id"`
	Access NodeAccess `json:"access"`
}

type ObservationArtifact struct {
	ID         string   `json:"id"`
	ObserverID string   `json:"observer_id"`
	StateEpoch string   `json:"state_epoch"`
	ReadSet    []string `json:"read_set"`
}

type EffectEnvelope struct {
	Tools    []string `json:"tools"`
	WriteSet []string `json:"write_set"`
}

// EffectEnvelopeAttenuation records that the policy compiler proved an effect
// envelope is attenuated from its parent. EnvelopeDigest binds the decision to
// the proposal so an admission for one envelope cannot be replayed for another.
type EffectEnvelopeAttenuation struct {
	Admit          bool   `json:"admit"`
	EnvelopeDigest string `json:"envelope_digest"`
	Reason         string `json:"reason,omitempty"`
}

// SnapshotVerdict is the observation-validity decision produced against the
// current change feed.
type SnapshotVerdict struct {
	Current       bool     `json:"current"`
	ObservationID string   `json:"observation_id"`
	StateEpoch    string   `json:"state_epoch"`
	ReadSet       []string `json:"read_set"`
	Reason        string   `json:"reason,omitempty"`
}

// LeaseVerdict is the lease authority's decision for the proposed write set.
type LeaseVerdict struct {
	Admit     bool     `json:"admit"`
	Exclusive bool     `json:"exclusive"`
	LeaseID   string   `json:"lease_id,omitempty"`
	WriteSet  []string `json:"write_set"`
	Reason    string   `json:"reason,omitempty"`
}

type EffectSuccessorProposal struct {
	Observer           ObserverNode              `json:"observer"`
	Observation        ObservationArtifact       `json:"observation"`
	Effect             EffectEnvelope            `json:"effect"`
	Capability         EffectEnvelopeAttenuation `json:"capability"`
	Snapshot           SnapshotVerdict           `json:"snapshot"`
	Lease              LeaseVerdict              `json:"lease"`
	BudgetLimit        Budget                    `json:"budget_limit"`
	BudgetEvents       []BudgetEvent             `json:"budget_events"`
	ParentBudgetNodeID string                    `json:"parent_budget_node_id"`
	Reservation        Budget                    `json:"reservation"`
}

type EffectSuccessorNode struct {
	ID            string         `json:"id"`
	ObserverID    string         `json:"observer_id"`
	ObservationID string         `json:"observation_id"`
	Access        NodeAccess     `json:"access"`
	Envelope      EffectEnvelope `json:"envelope"`
	Budget        Budget         `json:"budget"`
}

type EffectSuccessorReceipt struct {
	Schema         string `json:"schema"`
	ID             string `json:"id"`
	NodeID         string `json:"node_id"`
	ObserverID     string `json:"observer_id"`
	ObservationID  string `json:"observation_id"`
	SnapshotEpoch  string `json:"snapshot_epoch"`
	EnvelopeDigest string `json:"envelope_digest"`
	LeaseID        string `json:"lease_id"`
	Budget         Budget `json:"budget"`
}

type EffectSuccessorAdmission struct {
	Node    EffectSuccessorNode    `json:"node"`
	Receipt EffectSuccessorReceipt `json:"receipt"`
	Edge    Edge                   `json:"edge"`
	Budget  BudgetLedger           `json:"budget"`
}

// ProposeEffectSuccessor creates a separately admitted effect node from a
// current observer artifact. All inputs are values or pure gate verdicts, so a
// refusal cannot widen or otherwise mutate the running observer.
func ProposeEffectSuccessor(proposal EffectSuccessorProposal) (EffectSuccessorAdmission, error) {
	observer := normalizeObserver(proposal.Observer)
	observation := normalizeObservation(proposal.Observation)
	effect := normalizeEffectEnvelope(proposal.Effect)
	parentBudgetID := strings.TrimSpace(proposal.ParentBudgetNodeID)

	if observer.ID == "" || observer.Access.Mode != AccessObserve {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, "source must identify an observe-mode node")
	}
	if len(observer.Access.WriteSet) != 0 {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorObserverWrite, "the running observer must keep an empty write set; request a separate effect node")
	}
	if observation.ID == "" || observation.ObserverID != observer.ID || observation.StateEpoch == "" || len(observation.ReadSet) == 0 {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, "observation must bind its source observer, state epoch, and read set")
	}
	if !sameStrings(observation.ReadSet, observer.Access.ReadSet) {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, "observation read set must match the observer declaration")
	}
	if !proposal.Snapshot.Current ||
		strings.TrimSpace(proposal.Snapshot.ObservationID) != observation.ID ||
		strings.TrimSpace(proposal.Snapshot.StateEpoch) != observation.StateEpoch ||
		!sameStrings(normalizeStrings(proposal.Snapshot.ReadSet), observation.ReadSet) {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorStaleObservation, firstDetail(proposal.Snapshot.Reason, "observation is not current at the admitted snapshot epoch"))
	}
	if len(effect.Tools) == 0 || len(effect.WriteSet) == 0 {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, "effect successor requires a tool envelope and a bounded write set")
	}
	envelopeDigest, err := digestValue(effect)
	if err != nil {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, err.Error())
	}
	if !proposal.Capability.Admit || strings.TrimSpace(proposal.Capability.EnvelopeDigest) != envelopeDigest {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorEnvelopeWidening, firstDetail(proposal.Capability.Reason, "effect envelope is not attenuated from the parent"))
	}
	if !proposal.Lease.Admit ||
		!proposal.Lease.Exclusive ||
		strings.TrimSpace(proposal.Lease.LeaseID) == "" ||
		!sameStrings(normalizeStrings(proposal.Lease.WriteSet), effect.WriteSet) {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorLeaseContention, firstDetail(proposal.Lease.Reason, "effect successor requires an admitted exclusive lease"))
	}
	if parentBudgetID == "" || proposal.Reservation.MaxWorkers <= 0 || proposal.Reservation.MaxTokens <= 0 {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, "effect successor requires a parent budget node and positive reservation")
	}

	identity := successorIdentity{
		ObserverID:     observer.ID,
		Observation:    observation,
		Effect:         effect,
		SnapshotEpoch:  observation.StateEpoch,
		LeaseID:        strings.TrimSpace(proposal.Lease.LeaseID),
		ParentBudgetID: parentBudgetID,
		Reservation:    proposal.Reservation,
	}
	nodeID, err := digestID("effect-", identity)
	if err != nil {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, err.Error())
	}

	events := append([]BudgetEvent(nil), proposal.BudgetEvents...)
	events = append(events, BudgetEvent{
		Kind:     BudgetReserve,
		NodeID:   nodeID,
		ParentID: parentBudgetID,
		Workers:  proposal.Reservation.MaxWorkers,
		Tokens:   proposal.Reservation.MaxTokens,
	})
	ledger, err := FoldBudgetEvents(proposal.BudgetLimit, events)
	if err != nil {
		var refusal *BudgetRefusal
		if errors.As(err, &refusal) && (refusal.Reason == BudgetRootExhausted || refusal.Reason == BudgetParentExhausted) {
			return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorBudgetExhausted, refusal.Error())
		}
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, err.Error())
	}

	receiptID, err := digestID("effect-receipt-", identity)
	if err != nil {
		return EffectSuccessorAdmission{}, effectSuccessorRefusal(EffectSuccessorInvalid, err.Error())
	}

	node := EffectSuccessorNode{
		ID:            nodeID,
		ObserverID:    observer.ID,
		ObservationID: observation.ID,
		Access:        NodeAccess{Mode: AccessEffect, WriteSet: append([]string(nil), effect.WriteSet...)},
		Envelope:      effect,
		Budget:        proposal.Reservation,
	}
	receipt := EffectSuccessorReceipt{
		Schema:         EffectSuccessorReceiptSchema,
		ID:             receiptID,
		NodeID:         nodeID,
		ObserverID:     observer.ID,
		ObservationID:  observation.ID,
		SnapshotEpoch:  observation.StateEpoch,
		EnvelopeDigest: envelopeDigest,
		LeaseID:        identity.LeaseID,
		Budget:         proposal.Reservation,
	}
	return EffectSuccessorAdmission{
		Node:    node,
		Receipt: receipt,
		Edge:    Edge{From: observer.ID, To: nodeID},
		Budget:  ledger,
	}, nil
}

type successorIdentity struct {
	ObserverID     string              `json:"observer_id"`
	Observation    ObservationArtifact `json:"observation"`
	Effect         EffectEnvelope      `json:"effect"`
	SnapshotEpoch  string              `json:"snapshot_epoch"`
	LeaseID        string              `json:"lease_id"`
	ParentBudgetID string              `json:"parent_budget_id"`
	Reservation    Budget              `json:"reservation"`
}

func normalizeObserver(observer ObserverNode) ObserverNode {
	return ObserverNode{
		ID: strings.TrimSpace(observer.ID),
		Access: NodeAccess{
			Mode:     AccessMode(strings.TrimSpace(string(observer.Access.Mode))),
			ReadSet:  normalizeStrings(observer.Access.ReadSet),
			WriteSet: normalizeStrings(observer.Access.WriteSet),
		},
	}
}

func normalizeObservation(observation ObservationArtifact) ObservationArtifact {
	return ObservationArtifact{
		ID:         strings.TrimSpace(observation.ID),
		ObserverID: strings.TrimSpace(observation.ObserverID),
		StateEpoch: strings.TrimSpace(observation.StateEpoch),
		ReadSet:    normalizeStrings(observation.ReadSet),
	}
}

func normalizeEffectEnvelope(effect EffectEnvelope) EffectEnvelope {
	return EffectEnvelope{
		Tools:    normalizeStrings(effect.Tools),
		WriteSet: normalizeStrings(effect.WriteSet),
	}
}

func normalizeStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			seen[value] = struct{}{}
		}
	}
	out := make([]string, 0, len(seen))
	for value := range seen {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func digestID(prefix string, value any) (string, error) {
	digest, err := digestBytes(value)
	if err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(digest[:16]), nil
}

func digestValue(value any) (string, error) {
	digest, err := digestBytes(value)
	if err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func digestBytes(value any) ([sha256.Size]byte, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return [sha256.Size]byte{}, fmt.Errorf("canonical successor identity: %w", err)
	}
	return sha256.Sum256(raw), nil
}

func firstDetail(detail, fallback string) string {
	if detail = strings.TrimSpace(detail); detail != "" {
		return detail
	}
	return fallback
}

func effectSuccessorRefusal(reason EffectSuccessorRefusalReason, detail string) *EffectSuccessorRefusal {
	return &EffectSuccessorRefusal{Reason: reason, Detail: detail}
}
