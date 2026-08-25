package orchestration

import (
	"errors"
	"reflect"
	"testing"
)

func TestProposeEffectSuccessor(t *testing.T) {
	proposal := validEffectSuccessorProposal()
	originalObserver := proposal.Observer

	got, err := ProposeEffectSuccessor(proposal)
	if err != nil {
		t.Fatalf("ProposeEffectSuccessor: %v", err)
	}
	if got.Node.ID == "" || got.Node.ID == proposal.Observer.ID {
		t.Fatalf("successor id = %q, observer id = %q", got.Node.ID, proposal.Observer.ID)
	}
	if got.Node.Access.Mode != AccessEffect {
		t.Fatalf("successor access = %q, want %q", got.Node.Access.Mode, AccessEffect)
	}
	if got.Edge != (Edge{From: proposal.Observer.ID, To: got.Node.ID}) {
		t.Fatalf("dependency edge = %+v", got.Edge)
	}
	if got.Receipt.ObserverID != proposal.Observer.ID || got.Receipt.NodeID != got.Node.ID || got.Receipt.ObservationID != proposal.Observation.ID {
		t.Fatalf("receipt does not bind observer, node, and observation: %+v", got.Receipt)
	}
	if got.Receipt.SnapshotEpoch != proposal.Observation.StateEpoch || got.Receipt.LeaseID != proposal.Lease.LeaseID {
		t.Fatalf("receipt does not bind snapshot and lease: %+v", got.Receipt)
	}
	reserved, ok := got.Budget.Nodes[got.Node.ID]
	if !ok || reserved.ParentID != proposal.ParentBudgetNodeID || reserved.Allocation != proposal.Reservation {
		t.Fatalf("successor budget reservation = %+v, found=%v", reserved, ok)
	}
	if !reflect.DeepEqual(proposal.Observer, originalObserver) {
		t.Fatalf("observer mutated in place: got %+v want %+v", proposal.Observer, originalObserver)
	}

	again, err := ProposeEffectSuccessor(proposal)
	if err != nil {
		t.Fatalf("repeat proposal: %v", err)
	}
	if !reflect.DeepEqual(got, again) {
		t.Fatalf("transition is not deterministic:\nfirst:  %+v\nsecond: %+v", got, again)
	}

	tests := []struct {
		name   string
		mutate func(*EffectSuccessorProposal)
		want   EffectSuccessorRefusalReason
	}{
		{
			name: "observer cannot be upgraded in place",
			mutate: func(p *EffectSuccessorProposal) {
				p.Observer.Access.WriteSet = []string{"internal/orchestration/successor.go"}
			},
			want: EffectSuccessorObserverWrite,
		},
		{
			name: "stale observation",
			mutate: func(p *EffectSuccessorProposal) {
				p.Snapshot.Current = false
				p.Snapshot.Reason = "matching read-set change after epoch"
			},
			want: EffectSuccessorStaleObservation,
		},
		{
			name: "snapshot verdict for another observation",
			mutate: func(p *EffectSuccessorProposal) {
				p.Snapshot.ObservationID = "observation-other"
			},
			want: EffectSuccessorStaleObservation,
		},
		{
			name: "widened capability",
			mutate: func(p *EffectSuccessorProposal) {
				p.Capability.Admit = false
				p.Capability.Reason = "requested tool is outside the parent envelope"
			},
			want: EffectSuccessorEnvelopeWidening,
		},
		{
			name: "capability verdict for another envelope",
			mutate: func(p *EffectSuccessorProposal) {
				p.Capability.EnvelopeDigest = "sha256:other"
			},
			want: EffectSuccessorEnvelopeWidening,
		},
		{
			name: "exhausted parent budget",
			mutate: func(p *EffectSuccessorProposal) {
				p.Reservation = Budget{MaxWorkers: 2, MaxTokens: 70}
			},
			want: EffectSuccessorBudgetExhausted,
		},
		{
			name: "lease contention",
			mutate: func(p *EffectSuccessorProposal) {
				p.Lease.Admit = false
				p.Lease.Reason = "tree overlaps live lease effect-lane-peer"
			},
			want: EffectSuccessorLeaseContention,
		},
		{
			name: "lease verdict for another write set",
			mutate: func(p *EffectSuccessorProposal) {
				p.Lease.WriteSet = []string{"internal/orchestration/other.go"}
			},
			want: EffectSuccessorLeaseContention,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := validEffectSuccessorProposal()
			tt.mutate(&p)
			observerBefore := p.Observer
			_, err := ProposeEffectSuccessor(p)
			var refusal *EffectSuccessorRefusal
			if !errors.As(err, &refusal) {
				t.Fatalf("error = %v, want EffectSuccessorRefusal(%s)", err, tt.want)
			}
			if refusal.Reason != tt.want {
				t.Fatalf("reason = %s, want %s (detail %q)", refusal.Reason, tt.want, refusal.Detail)
			}
			if !reflect.DeepEqual(p.Observer, observerBefore) {
				t.Fatalf("refusal mutated observer: got %+v want %+v", p.Observer, observerBefore)
			}
		})
	}
}

func validEffectSuccessorProposal() EffectSuccessorProposal {
	effect := EffectEnvelope{
		Tools:    []string{"edit", "read"},
		WriteSet: []string{"internal/orchestration/successor.go", "internal/orchestration/successor_test.go"},
	}
	envelopeDigest, err := digestValue(normalizeEffectEnvelope(effect))
	if err != nil {
		panic(err)
	}
	return EffectSuccessorProposal{
		RunID: "run-8843",
		Observer: ObserverNode{
			ID: "observer-scout",
			Access: NodeAccess{
				Mode:    AccessObserve,
				ReadSet: []string{"internal/orchestration/orchestration.go"},
			},
		},
		Observation: ObservationArtifact{
			ID:         "observation-7",
			ObserverID: "observer-scout",
			StateEpoch: "git:abc123",
			ReadSet:    []string{"internal/orchestration/orchestration.go"},
		},
		Effect:     effect,
		Capability: EffectEnvelopeAttenuation{Admit: true, EnvelopeDigest: envelopeDigest},
		Snapshot: SnapshotVerdict{
			Current:       true,
			ObservationID: "observation-7",
			StateEpoch:    "git:abc123",
			ReadSet:       []string{"internal/orchestration/orchestration.go"},
		},
		Lease: LeaseVerdict{
			Admit:     true,
			Exclusive: true,
			LeaseID:   "effect-lane-orchestration",
			WriteSet:  append([]string(nil), effect.WriteSet...),
		},
		BudgetLimit: Budget{MaxWorkers: 2, MaxTokens: 100},
		BudgetEvents: []BudgetEvent{
			{Kind: BudgetReserve, NodeID: "observer-scout", ParentID: RootBudgetNodeID, Workers: 1, Tokens: 60},
		},
		ParentBudgetNodeID: "observer-scout",
		Reservation:        Budget{MaxWorkers: 1, MaxTokens: 20},
	}
}
