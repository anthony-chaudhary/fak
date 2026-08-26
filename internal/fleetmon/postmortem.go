package fleetmon

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// CausalityProvenance states how strongly an event is known to cause another.
// Inferred evidence is useful context, but never creates a causal edge.
type CausalityProvenance string

const (
	ProvenanceControlled CausalityProvenance = "controlled"
	ProvenanceWitnessed  CausalityProvenance = "witnessed"
	ProvenanceInferred   CausalityProvenance = "inferred"
)

// FaultRelation names a stable shared-fate relationship.
type FaultRelation string

const (
	RelationOwner            FaultRelation = "owner"
	RelationParentProcess    FaultRelation = "parent_process"
	RelationSupervisor       FaultRelation = "supervisor"
	RelationSharedService    FaultRelation = "shared_service"
	RelationResourceEnvelope FaultRelation = "resource_envelope"
	RelationLease            FaultRelation = "lease"
	RelationSharedFateGroup  FaultRelation = "shared_fate_group"
)

// FaultEventKind is an event retained in a postmortem causality ledger.
type FaultEventKind string

const (
	EventProcessStart            FaultEventKind = "process_start"
	EventProcessExit             FaultEventKind = "process_exit"
	EventSignal                  FaultEventKind = "signal"
	EventKill                    FaultEventKind = "kill"
	EventCancellationPropagation FaultEventKind = "cancellation_propagation"
	EventRestartOutcome          FaultEventKind = "restart_outcome"
	EventQuarantine              FaultEventKind = "quarantine"
	EventAdoption                FaultEventKind = "adoption"
	EventResourcePressure        FaultEventKind = "resource_pressure"
	EventSurvivorProgress        FaultEventKind = "survivor_progress"
)

// ProcessBirthID survives PID reuse and timestamp ambiguity. All fields are
// required, so joins cannot silently fall back to a bare PID.
type ProcessBirthID struct {
	RunID   string `json:"run_id"`
	OwnerID string `json:"owner_id"`
	BirthID string `json:"birth_id"`
}

func (id ProcessBirthID) valid() bool {
	return id.RunID != "" && id.OwnerID != "" && id.BirthID != ""
}

func (id ProcessBirthID) key() string { return id.RunID + "\x00" + id.OwnerID + "\x00" + id.BirthID }

// FaultDomainEdge records one typed relationship without embedding host names.
type FaultDomainEdge struct {
	Relation FaultRelation `json:"relation"`
	From     string        `json:"from"`
	To       string        `json:"to"`
}

// FaultEvent is ordered by Sequence. CauseSequence is the only way to assert a
// causal edge; close timestamps and matching domains remain correlation only.
type FaultEvent struct {
	Sequence      uint64              `json:"sequence"`
	Kind          FaultEventKind      `json:"kind"`
	Subject       ProcessBirthID      `json:"subject"`
	FaultDomain   string              `json:"fault_domain"`
	CauseSequence uint64              `json:"cause_sequence,omitempty"`
	Provenance    CausalityProvenance `json:"provenance"`
	Authority     string              `json:"authority,omitempty"`
	EvidenceID    string              `json:"evidence_id"`
}

// StopReceipt answers why one process stopped without promoting correlation to
// causation.
type StopReceipt struct {
	Victim          ProcessBirthID      `json:"victim"`
	StopSequence    uint64              `json:"stop_sequence"`
	RootSequence    uint64              `json:"root_sequence"`
	FaultDomain     string              `json:"fault_domain"`
	Provenance      CausalityProvenance `json:"provenance"`
	KillAuthority   string              `json:"kill_authority,omitempty"`
	CausalSequences []uint64            `json:"causal_sequences"`
}

// SurvivorReceipt proves work progressed after the selected root event.
type SurvivorReceipt struct {
	Subject          ProcessBirthID `json:"subject"`
	ProgressSequence uint64         `json:"progress_sequence"`
	EvidenceID       string         `json:"evidence_id"`
}

// PostmortemBundle is scrubbed by construction: it has no host, command line,
// environment, free-form message, PID, or wall-clock fields.
type PostmortemBundle struct {
	Schema    string            `json:"schema"`
	Domains   []FaultDomainEdge `json:"domains"`
	Events    []FaultEvent      `json:"events"`
	Stops     []StopReceipt     `json:"stops"`
	Survivors []SurvivorReceipt `json:"survivors"`
}

// BuildPostmortem validates and joins a fault ledger into a deterministic,
// replayable bundle. Only explicit cause_sequence links form cascades.
func BuildPostmortem(domains []FaultDomainEdge, events []FaultEvent) (PostmortemBundle, error) {
	bundle := PostmortemBundle{Schema: "fak.fleetmon.postmortem.v1"}
	bundle.Domains = append([]FaultDomainEdge(nil), domains...)
	bundle.Events = append([]FaultEvent(nil), events...)
	sort.Slice(bundle.Domains, func(i, j int) bool {
		a, b := bundle.Domains[i], bundle.Domains[j]
		if a.Relation != b.Relation {
			return a.Relation < b.Relation
		}
		if a.From != b.From {
			return a.From < b.From
		}
		return a.To < b.To
	})
	sort.Slice(bundle.Events, func(i, j int) bool { return bundle.Events[i].Sequence < bundle.Events[j].Sequence })

	bySequence := make(map[uint64]FaultEvent, len(bundle.Events))
	for _, event := range bundle.Events {
		if event.Sequence == 0 || !event.Subject.valid() || event.EvidenceID == "" {
			return PostmortemBundle{}, errors.New("fault events require sequence, stable run/owner/birth identity, and evidence_id")
		}
		if event.Provenance != ProvenanceControlled && event.Provenance != ProvenanceWitnessed && event.Provenance != ProvenanceInferred {
			return PostmortemBundle{}, fmt.Errorf("event %d has invalid provenance %q", event.Sequence, event.Provenance)
		}
		if _, exists := bySequence[event.Sequence]; exists {
			return PostmortemBundle{}, fmt.Errorf("duplicate event sequence %d", event.Sequence)
		}
		bySequence[event.Sequence] = event
	}
	for _, event := range bundle.Events {
		if event.CauseSequence == 0 {
			continue
		}
		cause, ok := bySequence[event.CauseSequence]
		if !ok || cause.Sequence >= event.Sequence {
			return PostmortemBundle{}, fmt.Errorf("event %d has invalid cause sequence %d", event.Sequence, event.CauseSequence)
		}
		if event.Provenance == ProvenanceInferred {
			return PostmortemBundle{}, fmt.Errorf("event %d: inferred correlation cannot form a causal edge", event.Sequence)
		}
	}

	affectedByRoot := make(map[uint64]map[string]bool)
	for _, event := range bundle.Events {
		if event.Kind != EventProcessExit {
			continue
		}
		path := []uint64{event.Sequence}
		root := event
		provenance := event.Provenance
		for root.CauseSequence != 0 {
			root = bySequence[root.CauseSequence]
			path = append(path, root.Sequence)
			if root.Provenance == ProvenanceWitnessed {
				provenance = ProvenanceWitnessed
			}
		}
		sort.Slice(path, func(i, j int) bool { return path[i] < path[j] })
		bundle.Stops = append(bundle.Stops, StopReceipt{Victim: event.Subject, StopSequence: event.Sequence, RootSequence: root.Sequence, FaultDomain: event.FaultDomain, Provenance: provenance, KillAuthority: event.Authority, CausalSequences: path})
		if affectedByRoot[root.Sequence] == nil {
			affectedByRoot[root.Sequence] = map[string]bool{}
		}
		affectedByRoot[root.Sequence][event.Subject.key()] = true
	}
	survivorSeen := make(map[string]bool)
	for root, affected := range affectedByRoot {
		for _, event := range bundle.Events {
			if event.Sequence > root && event.Kind == EventSurvivorProgress && !affected[event.Subject.key()] {
				key := fmt.Sprintf("%d\x00%s", event.Sequence, event.Subject.key())
				if !survivorSeen[key] {
					survivorSeen[key] = true
					bundle.Survivors = append(bundle.Survivors, SurvivorReceipt{Subject: event.Subject, ProgressSequence: event.Sequence, EvidenceID: event.EvidenceID})
				}
			}
		}
	}
	sort.Slice(bundle.Stops, func(i, j int) bool { return bundle.Stops[i].StopSequence < bundle.Stops[j].StopSequence })
	sort.Slice(bundle.Survivors, func(i, j int) bool {
		if bundle.Survivors[i].ProgressSequence != bundle.Survivors[j].ProgressSequence {
			return bundle.Survivors[i].ProgressSequence < bundle.Survivors[j].ProgressSequence
		}
		return bundle.Survivors[i].Subject.key() < bundle.Survivors[j].Subject.key()
	})
	return bundle, nil
}

// MarshalPostmortem renders stable operator/GitHub-issue output.
func MarshalPostmortem(bundle PostmortemBundle) ([]byte, error) {
	return json.MarshalIndent(bundle, "", "  ")
}
