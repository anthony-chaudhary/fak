package sessionjournal

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkSessionReplayRepairsTailFencesWriterAndPreservesUncertainEffect(t *testing.T) {
	path := filepath.Join(t.TempDir(), "work.journal")
	appendEvent := func(event WorkEvent) {
		t.Helper()
		if err := AppendWorkEvent(path, event); err != nil {
			t.Fatal(err)
		}
	}
	identity := ResidencyIdentity{WorkspaceHead: "abc", WorkspaceDirty: "clean", PolicyHash: "p1", ToolSchema: "t1", CredentialEpoch: "c1", AdapterIdentity: "a1"}
	appendEvent(WorkEvent{SessionID: "work-a", Kind: WorkSessionOpened, WriterEpoch: "epoch-old", Residency: &identity})
	appendEvent(WorkEvent{SessionID: "work-a", Kind: WorkTerminalOutput, WriterEpoch: "epoch-old", Terminal: []byte("same\r\n")})
	appendEvent(WorkEvent{SessionID: "work-a", Kind: WorkEffectIntent, WriterEpoch: "epoch-old", EffectID: "before", Command: "make marker", Check: "test marker"})
	appendEvent(WorkEvent{SessionID: "work-a", Kind: WorkEffectResolved, WriterEpoch: "epoch-old", EffectID: "before", Verdict: EffectConfirmed})
	appendEvent(WorkEvent{SessionID: "work-a", Kind: WorkEffectIntent, WriterEpoch: "epoch-old", EffectID: "after", Command: "external write", Check: "query receipt"})
	appendEvent(WorkEvent{SessionID: "work-a", Kind: WorkSessionOpened, WriterEpoch: "epoch-new", Residency: &identity})
	appendEvent(WorkEvent{SessionID: "work-a", Kind: WorkTerminalOutput, WriterEpoch: "epoch-new", Terminal: []byte("again\r\n")})
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte("torn-tail")); err != nil {
		t.Fatal(err)
	}
	f.Close()

	replay, err := ReplayWork(path)
	if err != nil {
		t.Fatal(err)
	}
	got := replay.Sessions["work-a"]
	if !replay.RecoveredTail {
		t.Fatal("torn append was not reported")
	}
	if got.WriterEpoch != "epoch-new" {
		t.Fatalf("writer epoch=%q", got.WriterEpoch)
	}
	changed := identity
	changed.WorkspaceDirty = "sha256:changed"
	changed.PolicyHash = "p2"
	mismatch := got.Residency.Mismatch(changed)
	if len(mismatch) != 2 || mismatch[0] != "workspace_dirty" || mismatch[1] != "policy_hash" {
		t.Fatalf("residency mismatch=%v", mismatch)
	}
	if string(got.Transcript) != "same\r\nagain\r\n" {
		t.Fatalf("transcript=%q", got.Transcript)
	}
	if got.Effects["before"].Verdict != EffectConfirmed || got.Effects["after"].Verdict != EffectUncertain || got.Effects["after"].Check == "" {
		t.Fatalf("effects=%+v", got.Effects)
	}
	if err := AppendWorkEvent(path, WorkEvent{SessionID: "work-a", Kind: WorkTerminalOutput, WriterEpoch: "epoch-old", Terminal: []byte("stale")}); err == nil {
		t.Fatal("old writer epoch was not fenced at append")
	}
}

func TestResidencyIdentityEveryUnsafeDimensionInvalidates(t *testing.T) {
	base := ResidencyIdentity{WorkspaceHead: "head", WorkspaceDirty: "dirty", PolicyHash: "policy", ToolSchema: "tools", CredentialEpoch: "cred", AdapterIdentity: "adapter"}
	if got := base.RecoveryDependency(); got != "" {
		t.Fatalf("complete residency dependency=%q", got)
	}
	missing := base
	missing.AdapterIdentity = ""
	if got := missing.RecoveryDependency(); got != "adapter identity unavailable" {
		t.Fatalf("dependency=%q", got)
	}
	cases := []struct {
		name   string
		mutate func(*ResidencyIdentity)
	}{
		{"workspace_head", func(v *ResidencyIdentity) { v.WorkspaceHead = "other" }},
		{"workspace_dirty", func(v *ResidencyIdentity) { v.WorkspaceDirty = "other" }},
		{"policy_hash", func(v *ResidencyIdentity) { v.PolicyHash = "other" }},
		{"tool_schema", func(v *ResidencyIdentity) { v.ToolSchema = "other" }},
		{"credential_epoch", func(v *ResidencyIdentity) { v.CredentialEpoch = "other" }},
		{"adapter_identity", func(v *ResidencyIdentity) { v.AdapterIdentity = "other" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			current := base
			tc.mutate(&current)
			got := base.Mismatch(current)
			if len(got) != 1 || got[0] != tc.name {
				t.Fatalf("mismatch=%v", got)
			}
		})
	}
}

func TestWorkSessionReplayPreservesMovementLineageWithoutCredentialMaterial(t *testing.T) {
	path := filepath.Join(t.TempDir(), "move.journal")
	identity := ResidencyIdentity{WorkspaceHead: "head", WorkspaceDirty: "clean", PolicyHash: "policy", ToolSchema: "tools", CredentialEpoch: "credential-epoch-reference", AdapterIdentity: "adapter"}
	if err := AppendWorkEvent(path, WorkEvent{SessionID: "portable", Kind: WorkSessionOpened, WriterEpoch: "writer-1", Residency: &identity}); err != nil {
		t.Fatal(err)
	}
	destination := PlacementIdentity{Provider: "provider-b", AccountRef: "account-ref-b", Model: "model-b", Compute: "compute-b", Capabilities: []string{"tools"}, CacheLineage: "cache-b", SemanticDegradations: []string{"vision unavailable"}}
	for _, phase := range []string{"SAFE_POINT_REQUESTED", "CHECKPOINTED", "DESTINATION_ADMITTED", "RESTORED", "CUTOVER_COMMITTED"} {
		if err := AppendWorkEvent(path, WorkEvent{SessionID: "portable", Kind: WorkMoveTransitionEvent, WriterEpoch: "writer-1", MovePhase: phase, SourceEpoch: "epoch-a", Destination: &destination, Checkpoint: "sha256:checkpoint"}); err != nil {
			t.Fatal(err)
		}
	}
	replay, err := ReplayWork(path)
	if err != nil {
		t.Fatal(err)
	}
	view := replay.Sessions["portable"]
	if len(view.MoveTransitions) != 5 || view.MoveTransitions[4].Phase != "CUTOVER_COMMITTED" || view.MoveTransitions[4].Destination.AccountRef != "account-ref-b" {
		t.Fatalf("lineage=%+v", view.MoveTransitions)
	}
	bytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(bytes), "secret") || strings.Contains(string(bytes), "token") {
		t.Fatalf("credential material leaked: %s", bytes)
	}
}
