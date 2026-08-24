package harnesscompose

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestGenerateArtifactPreservesReceiptOrigin(t *testing.T) {
	receipts := []ProfileReceipt{
		{Kind: ReceiptTool, ID: "shell", Verified: true, EvidenceRef: "evidence:tool", TaskID: "task-2", ParentTaskID: "task-1", VerifierOutcome: "accepted", Asset: Asset{Kind: "tool", ID: "shell", Ref: "workspace-only"}},
		{Kind: ReceiptArchitecture, ID: "runtime", Verified: true, EvidenceRef: "evidence:arch", TaskID: "task-1", VerifierOutcome: "accepted", Asset: Asset{Kind: "instruction", ID: "runtime", Value: "fak-native"}},
	}
	artifact, err := GenerateArtifact(receipts)
	if err != nil {
		t.Fatalf("GenerateArtifact: %v", err)
	}
	if len(artifact.Provenance) != 2 || artifact.Provenance[0].TaskID != "task-1" || artifact.Provenance[1].ParentTaskID != "task-1" {
		t.Fatalf("provenance = %#v", artifact.Provenance)
	}
	raw, err := json.Marshal(artifact)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	for _, want := range []string{"task-2", "task-1", "evidence:tool", "accepted"} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("artifact metadata omits %q: %s", want, raw)
		}
	}
}

func TestGenerateArtifactRefusesMissingLineage(t *testing.T) {
	receipt := ProfileReceipt{Kind: ReceiptTool, ID: "shell", Verified: true, EvidenceRef: "evidence:tool", VerifierOutcome: "accepted", Asset: Asset{Kind: "tool", ID: "shell", Ref: "workspace-only"}}
	if _, err := GenerateArtifact([]ProfileReceipt{receipt}); !errors.Is(err, ErrInvalidReceipt) {
		t.Fatalf("error = %v, want ErrInvalidReceipt", err)
	}
}
