package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func TestUltracodeStatusRendersNodeGraph(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	wait := int64(73)
	hold := int64(410)
	plan := orchestration.WorkflowPlan{
		Schema: "fak-orchestration-plan/1",
		Roles: []orchestration.Role{
			{ID: "lead", Purpose: "coordinate", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve}},
			{ID: "observer", Purpose: "inspect", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve}},
			{ID: "effect", Purpose: "apply", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessEffect, WriteSet: []string{"cmd/fak"}}},
			{ID: "witnessed", Purpose: "verify", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessEffect, WriteSet: []string{"cmd/fak"}}},
			{ID: "reconciler", Purpose: "reconcile", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve}},
		},
		DAG: []orchestration.Edge{{From: "observer", To: "effect"}, {From: "effect", To: "witnessed"}, {From: "witnessed", To: "reconciler"}},
	}
	budget, err := orchestration.NewUltracodeEnvelopeReceipt(900, time.Minute, time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC), []string{"observer", "effect", "witnessed"})
	if err != nil {
		t.Fatal(err)
	}
	for i := range budget.Children {
		switch budget.Children[i].ChildID {
		case "observer":
			budget.Children[i].ProviderTokens, budget.Children[i].Covered = 12, true
		case "witnessed":
			budget.Children[i].ProviderTokens, budget.Children[i].Covered = 240, true
		}
	}
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "graph-session", RunID: "graph-run",
		RequestedProfile: "ultracode", ResolvedProfile: "ultracode", Status: "launched",
		Workers: []codexOrchestrationWorkerLaunch{
			{RoleID: "observer", Status: "exited"},
			{RoleID: "effect", Status: "starting"},
			{RoleID: "witnessed", Status: "exited"},
			{RoleID: "reconciler", Status: "blocked"},
		},
		Budget: budget,
		Graph: &ultracodeNodeGraphReceipt{
			Plan: &plan,
			Contexts: []ultracodeChildContextReceipt{
				{NodeID: "observer", ParentID: "lead", ContextDigest: "sha256:observer", StateEpoch: "epoch-1"},
				{NodeID: "effect", ParentID: "lead", ContextDigest: "sha256:effect", StateEpoch: "epoch-1"},
				{NodeID: "witnessed", ParentID: "lead", ContextDigest: "sha256:witnessed", StateEpoch: "epoch-2"},
			},
			Leases: []ultracodeNodeLeaseReceipt{
				{NodeID: "effect", Verdict: "contended", WaitMS: &wait},
				{NodeID: "witnessed", Verdict: "held", HoldMS: &hold},
			},
			Terminals: []ultracodeNodeTerminalReceipt{{NodeID: "observer", State: "exited"}, {NodeID: "witnessed", State: "exited"}, {NodeID: "reconciler", State: "blocked"}},
			Artifacts: []ultracodeNodeArtifactReceipt{
				{NodeID: "observer", Refs: []string{"artifact://finding.json"}},
				{NodeID: "witnessed", Refs: []string{"artifact://patch.diff", "artifact://readback.json"}},
			},
			Successors: []orchestration.EffectSuccessorReceipt{{NodeID: "effect", LeaseID: "lease-effect"}, {NodeID: "witnessed", LeaseID: "lease-witnessed"}},
			EffectWitness: []orchestration.EffectReceipt{{
				ChildID: "witnessed", State: orchestration.EffectVerified,
				Witness:        orchestration.EffectWitnessAuthority{AuthorityID: "independent-readback"},
				Reconciliation: orchestration.EffectReconciled,
			}},
		},
	}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}

	var jsonOut, humanOut, stderr bytes.Buffer
	if rc := runUltracodeStatus(&jsonOut, &stderr, []string{"--home", home, "--session", receipt.SessionID, "--json"}); rc != 0 {
		t.Fatalf("json status rc=%d stderr=%s", rc, stderr.String())
	}
	var got ultracodeStatus
	if err := json.Unmarshal(jsonOut.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Nodes) != 4 {
		t.Fatalf("nodes=%d want 4: %s", len(got.Nodes), jsonOut.String())
	}
	byID := make(map[string]ultracodeNodeStatus, len(got.Nodes))
	for _, node := range got.Nodes {
		byID[node.NodeID] = node
	}
	if observer := byID["observer"]; observer.ContextDigest != "sha256:observer" || observer.TerminalState != "exited" || observer.WitnessState != ultracodeNodeUnobserved || observer.UsedTokens == nil || *observer.UsedTokens != 12 {
		t.Fatalf("observer node=%+v", observer)
	}
	if effect := byID["effect"]; effect.LeaseVerdict != "contended" || effect.Attention != "lease_contended" || effect.UsedTokens != nil || effect.EffectState != "admitted" {
		t.Fatalf("effect node=%+v", effect)
	}
	if witnessed := byID["witnessed"]; witnessed.WorkerState != "exited" || witnessed.EffectState != string(orchestration.EffectVerified) || witnessed.WitnessState != "independent-readback" || witnessed.ReconcileState != string(orchestration.EffectReconciled) {
		t.Fatalf("witnessed node=%+v", witnessed)
	}
	if reconciler := byID["reconciler"]; reconciler.ContextDigest != ultracodeNodeUnobserved || reconciler.LeaseVerdict != ultracodeNodeUnobserved || reconciler.TerminalState != "blocked" || reconciler.Attention != "terminal_blocked" {
		t.Fatalf("reconciler node=%+v", reconciler)
	}
	if rc := runUltracodeStatus(&humanOut, &stderr, []string{"--home", home, "--session", receipt.SessionID}); rc != 0 {
		t.Fatalf("human status rc=%d stderr=%s", rc, stderr.String())
	}
	for _, want := range []string{
		"node effect", "deps=observer", "lease=contended", "budget=unobserved/300", "effect=admitted", "attention=lease_contended",
		"node witnessed", "effect=VERIFIED", "witness=independent-readback", "reconcile=RECONCILED", "terminal=exited",
		"node reconciler", "parent=unobserved", "terminal=blocked", "attention=terminal_blocked",
	} {
		if !strings.Contains(humanOut.String(), want) {
			t.Fatalf("human status missing %q:\n%s", want, humanOut.String())
		}
	}
	if strings.Contains(jsonOut.String(), "success\"") {
		t.Fatalf("status inferred success: %s", jsonOut.String())
	}
}
