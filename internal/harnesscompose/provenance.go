package harnesscompose

import (
	"fmt"
	"sort"
	"strings"
)

// ReceiptOrigin is audit metadata retained beside a generated harness.
// It records lineage and verifier outcome without retaining child transcripts.
type ReceiptOrigin struct {
	Kind            ReceiptKind `json:"kind"`
	DecisionID      string      `json:"decision_id"`
	TaskID          string      `json:"task_id"`
	ParentTaskID    string      `json:"parent_task_id,omitempty"`
	EvidenceRef     string      `json:"evidence_ref"`
	VerifierOutcome string      `json:"verifier_outcome"`
}

// GeneratedArtifact couples the composed harness with deterministic metadata
// sufficient to audit or regenerate each accepted decision.
type GeneratedArtifact struct {
	Harness    Result          `json:"harness"`
	Provenance []ReceiptOrigin `json:"provenance"`
}

// GenerateArtifact composes receipts and preserves their decision provenance.
func GenerateArtifact(receipts []ProfileReceipt) (GeneratedArtifact, error) {
	harness, err := ComposeReceipts(receipts)
	if err != nil {
		return GeneratedArtifact{}, err
	}
	provenance := make([]ReceiptOrigin, 0, len(receipts))
	for _, receipt := range receipts {
		if strings.TrimSpace(receipt.TaskID) == "" || strings.TrimSpace(receipt.VerifierOutcome) == "" {
			return GeneratedArtifact{}, fmt.Errorf("%w: receipt %q lacks task lineage or verifier outcome", ErrInvalidReceipt, receipt.ID)
		}
		provenance = append(provenance, ReceiptOrigin{
			Kind: receipt.Kind, DecisionID: receipt.ID, TaskID: receipt.TaskID,
			ParentTaskID: receipt.ParentTaskID, EvidenceRef: receipt.EvidenceRef,
			VerifierOutcome: receipt.VerifierOutcome,
		})
	}
	sort.Slice(provenance, func(i, j int) bool {
		if provenance[i].TaskID != provenance[j].TaskID {
			return provenance[i].TaskID < provenance[j].TaskID
		}
		if provenance[i].Kind != provenance[j].Kind {
			return provenance[i].Kind < provenance[j].Kind
		}
		return provenance[i].DecisionID < provenance[j].DecisionID
	})
	return GeneratedArtifact{Harness: harness, Provenance: provenance}, nil
}
