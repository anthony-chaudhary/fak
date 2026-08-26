package fleetmon

import (
	"bytes"
	"testing"
)

func pid(owner, birth string) ProcessBirthID {
	return ProcessBirthID{RunID: "run-9055", OwnerID: owner, BirthID: birth}
}

func TestPostmortemCrashClassesAndSurvivors(t *testing.T) {
	classes := []struct {
		name, domain, authority string
		kind                    FaultEventKind
		provenance              CausalityProvenance
	}{
		{"direct-child", "supervisor/a", "child", EventProcessExit, ProvenanceWitnessed},
		{"supervisor", "supervisor/a", "supervisor", EventProcessExit, ProvenanceWitnessed},
		{"shared-gateway", "service/gateway", "gateway", EventProcessExit, ProvenanceWitnessed},
		{"resource-limit", "resource/gpu-0", "kernel-cgroup", EventResourcePressure, ProvenanceControlled},
		{"fleet-cancel", "fleet/run-9055", "operator", EventCancellationPropagation, ProvenanceControlled},
	}
	for _, tc := range classes {
		t.Run(tc.name, func(t *testing.T) {
			victim, survivor := pid("owner-a", "victim"), pid("owner-b", "survivor")
			events := []FaultEvent{
				{Sequence: 10, Kind: tc.kind, Subject: victim, FaultDomain: tc.domain, Provenance: tc.provenance, Authority: tc.authority, EvidenceID: "root-receipt"},
				{Sequence: 11, Kind: EventProcessExit, Subject: victim, FaultDomain: tc.domain, CauseSequence: 10, Provenance: tc.provenance, Authority: tc.authority, EvidenceID: "exit-receipt"},
				{Sequence: 12, Kind: EventSurvivorProgress, Subject: survivor, FaultDomain: "supervisor/b", Provenance: ProvenanceWitnessed, EvidenceID: "progress-receipt"},
			}
			bundle, err := BuildPostmortem([]FaultDomainEdge{{Relation: RelationSupervisor, From: "victim", To: tc.domain}}, events)
			if err != nil {
				t.Fatal(err)
			}
			var stop *StopReceipt
			for i := range bundle.Stops {
				if bundle.Stops[i].StopSequence == 11 {
					stop = &bundle.Stops[i]
				}
			}
			if stop == nil || stop.RootSequence != 10 || stop.FaultDomain != tc.domain || stop.Provenance != tc.provenance || stop.KillAuthority != tc.authority {
				t.Fatalf("wrong stop receipt: %+v", bundle.Stops)
			}
			if len(bundle.Survivors) != 1 || bundle.Survivors[0].Subject != survivor || bundle.Survivors[0].EvidenceID != "progress-receipt" {
				t.Fatalf("wrong survivors: %+v", bundle.Survivors)
			}
		})
	}
}

func TestPostmortemIndependentExitsAreNotCascade(t *testing.T) {
	events := []FaultEvent{
		{Sequence: 20, Kind: EventProcessExit, Subject: pid("a", "one"), FaultDomain: "shared/service", Provenance: ProvenanceInferred, EvidenceID: "exit-a"},
		{Sequence: 21, Kind: EventProcessExit, Subject: pid("b", "two"), FaultDomain: "shared/service", Provenance: ProvenanceInferred, EvidenceID: "exit-b"},
	}
	bundle, err := BuildPostmortem(nil, events)
	if err != nil {
		t.Fatal(err)
	}
	if len(bundle.Stops) != 2 || bundle.Stops[0].RootSequence != 20 || bundle.Stops[1].RootSequence != 21 {
		t.Fatalf("independent exits rendered as cascade: %+v", bundle.Stops)
	}
	if _, err := BuildPostmortem(nil, append(events, FaultEvent{Sequence: 22, Kind: EventProcessExit, Subject: pid("c", "three"), CauseSequence: 20, Provenance: ProvenanceInferred, EvidenceID: "guess"})); err == nil {
		t.Fatal("inferred correlation formed a causal edge")
	}
}

func TestPostmortemRequiresStableIdentityAndScrubsOutput(t *testing.T) {
	_, err := BuildPostmortem(nil, []FaultEvent{{Sequence: 1, Kind: EventProcessExit, Subject: ProcessBirthID{RunID: "run", OwnerID: "owner"}, Provenance: ProvenanceWitnessed, EvidenceID: "exit"}})
	if err == nil {
		t.Fatal("accepted identity without process birth")
	}
	bundle, err := BuildPostmortem(nil, []FaultEvent{{Sequence: 1, Kind: EventProcessExit, Subject: pid("owner", "birth"), Provenance: ProvenanceWitnessed, EvidenceID: "exit"}})
	if err != nil {
		t.Fatal(err)
	}
	first, err := MarshalPostmortem(bundle)
	if err != nil {
		t.Fatal(err)
	}
	second, err := MarshalPostmortem(bundle)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("postmortem output is not deterministic")
	}
	for _, forbidden := range [][]byte{[]byte(`"pid"`), []byte(`"host"`), []byte(`"timestamp"`), []byte(`"command"`)} {
		if bytes.Contains(first, forbidden) {
			t.Fatalf("bundle leaked private/unstable field %s: %s", forbidden, first)
		}
	}
}
