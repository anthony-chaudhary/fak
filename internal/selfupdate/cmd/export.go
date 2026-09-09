package selfupdatecmd

import "github.com/anthony-chaudhary/fak/internal/selfupdate"

// Receipt is the stable fak.self-update.receipt/v1 payload.
type Receipt = selfupdate.Receipt

// ReceiptTarget is one target entry in a self-update receipt.
type ReceiptTarget = selfupdate.ReceiptTarget

const (
	ReceiptSchema        = selfupdate.ReceiptSchema
	ReceiptSchemaVersion = selfupdate.ReceiptSchemaVersion
)

// BuildProvenance carries reproducible build stamps and artifacts.
type BuildProvenance = selfupdate.BuildProvenance

// TransferReceipt records differential or full artifact transfer metrics.
type TransferReceipt = selfupdate.TransferReceipt

// HandoffReceipt records successor process handoff results.
type HandoffReceipt = selfupdate.HandoffReceipt

// PhaseMS records timing across update phases.
type PhaseMS = selfupdate.PhaseMS

// Phase is a named step in the self-update pipeline.
type Phase = selfupdate.Phase

// Outcome classifies the terminal decision or execution result.
type Outcome = selfupdate.Outcome

// Builder constructs versioned receipts.
type Builder = selfupdate.Builder

// NewBuilder creates a new Builder with defaults.
func NewBuilder() *Builder {
	return selfupdate.NewBuilder()
}

// RepoRevOf preserves the native command's repository revision helper.
func RepoRevOf(root, ref string) string { return repoRevOf(root, ref) }
