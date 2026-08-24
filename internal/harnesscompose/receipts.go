package harnesscompose

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// ReceiptKind is a closed harness-construction decision class.
type ReceiptKind string

const (
	ReceiptArchitecture ReceiptKind = "architecture"
	ReceiptTool         ReceiptKind = "tool"
	ReceiptModel        ReceiptKind = "model"
	ReceiptConstraints  ReceiptKind = "policy"
	ReceiptProof        ReceiptKind = "proof"
)

// ProfileReceipt is the consumer-owned boundary between delegated computation
// and harness composition. It intentionally carries a typed asset and evidence
// reference, never a child transcript.
type ProfileReceipt struct {
	Kind            ReceiptKind
	ID              string
	Verified        bool
	EvidenceRef     string
	TaskID          string
	ParentTaskID    string
	VerifierOutcome string
	Asset           Asset
}

var ErrInvalidReceipt = errors.New("harnesscompose: invalid profile receipt")

// ComposeReceipts folds verified child decisions into one deterministic harness
// plan without importing the producer package or any child transcript.
func ComposeReceipts(receipts []ProfileReceipt) (Result, error) {
	ordered := append([]ProfileReceipt(nil), receipts...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Kind != ordered[j].Kind {
			return ordered[i].Kind < ordered[j].Kind
		}
		return ordered[i].ID < ordered[j].ID
	})
	seen := make(map[string]struct{}, len(ordered))
	assets := make([]Asset, 0, len(ordered))
	for i, receipt := range ordered {
		if !validReceiptKind(receipt.Kind) || strings.TrimSpace(receipt.ID) == "" {
			return Result{}, fmt.Errorf("%w: receipt[%d] has invalid kind or id", ErrInvalidReceipt, i)
		}
		if !receipt.Verified || strings.TrimSpace(receipt.EvidenceRef) == "" {
			return Result{}, fmt.Errorf("%w: receipt %q lacks accepted verification evidence", ErrInvalidReceipt, receipt.ID)
		}
		key := string(receipt.Kind) + "\x00" + receipt.ID
		if _, ok := seen[key]; ok {
			return Result{}, fmt.Errorf("%w: duplicate %s receipt %q", ErrInvalidReceipt, receipt.Kind, receipt.ID)
		}
		seen[key] = struct{}{}
		assets = append(assets, receipt.Asset)
	}
	manifest := Manifest{Schema: Schema, Layers: []Layer{{ID: "verified-microagent-receipts", Scope: "task", Assets: assets}}}
	return Compose(manifest, []string{"verified-microagent-receipts"})
}

func validReceiptKind(kind ReceiptKind) bool {
	switch kind {
	case ReceiptArchitecture, ReceiptTool, ReceiptModel, ReceiptConstraints, ReceiptProof:
		return true
	default:
		return false
	}
}
