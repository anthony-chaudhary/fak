package deploykit

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

// ReceiptSchema is the versioned schema identity for deploy receipts.
const ReceiptSchema = "fak.deploy.receipt/v1"

// ReceiptSchemaVersion is the integer version of ReceiptSchema.
const ReceiptSchemaVersion = 1

// Receipt is the stable payload one deployable run emits. Its shape is lifted
// from the self-update receipt; fak self-update keeps its own schema. Two
// fields are new: Deployable names the descriptor the run applied, and Stage is
// the last stage the run reached (omitted when unset).
type Receipt struct {
	Schema          string          `json:"schema"`
	SchemaVersion   int             `json:"schema_version"`
	CorrelationID   string          `json:"correlation_id"`
	Deployable      string          `json:"deployable"`
	Status          string          `json:"status"`
	Stage           Stage           `json:"stage,omitempty"`
	OldRevision     *string         `json:"old_revision"`
	NewRevision     *string         `json:"new_revision"`
	Targets         []ReceiptTarget `json:"targets"`
	Attempted       int             `json:"attempted"`
	Changed         int             `json:"changed"`
	RollbackStatus  string          `json:"rollback_status"`
	RollbackErrors  []string        `json:"rollback_errors"`
	RestartRequired bool            `json:"restart_required"`
	NextCommand     string          `json:"next_command"`
	Detail          string          `json:"detail,omitempty"`
	TotalMS         int64           `json:"total_ms"`
	StageMS         StageMS         `json:"stage_ms"`
}

// ReceiptTarget is one installed file's row in a deploy receipt.
type ReceiptTarget struct {
	Role                    string `json:"role"`
	Path                    string `json:"path"`
	CompatibilityGroup      string `json:"compatibility_group,omitempty"`
	DesiredArtifactDigest   string `json:"desired_artifact_digest,omitempty"`
	InstalledArtifactDigest string `json:"installed_artifact_digest,omitempty"`
	Acquisition             string `json:"acquisition,omitempty"`
	Activation              string `json:"activation,omitempty"`
	Rollback                string `json:"rollback,omitempty"`
}

// StageMS is a fixed-shape per-stage timing object, so consumers always see
// the full stage vocabulary, including zero durations for skipped stages.
type StageMS struct {
	Resolve   int64 `json:"resolve"`
	Acquire   int64 `json:"acquire"`
	Verify    int64 `json:"verify"`
	Preflight int64 `json:"preflight"`
	Drain     int64 `json:"drain"`
	Stage     int64 `json:"stage"`
	Swap      int64 `json:"swap"`
	Activate  int64 `json:"activate"`
	Probe     int64 `json:"probe"`
	Commit    int64 `json:"commit"`
	Rollback  int64 `json:"rollback"`
	Receipt   int64 `json:"receipt"`
}

// Set records the duration of one stage. An unknown stage is refused.
func (m *StageMS) Set(s Stage, ms int64) error {
	switch s {
	case StageResolve:
		m.Resolve = ms
	case StageAcquire:
		m.Acquire = ms
	case StageVerify:
		m.Verify = ms
	case StagePreflight:
		m.Preflight = ms
	case StageDrain:
		m.Drain = ms
	case StageStage:
		m.Stage = ms
	case StageSwap:
		m.Swap = ms
	case StageActivate:
		m.Activate = ms
	case StageProbe:
		m.Probe = ms
	case StageCommit:
		m.Commit = ms
	case StageRollback:
		m.Rollback = ms
	case StageReceipt:
		m.Receipt = ms
	default:
		return fmt.Errorf("%w %q", ErrUnknownStage, string(s))
	}
	return nil
}

// NewReceipt returns a receipt stamped with ReceiptSchema for one deployable.
// An empty correlationID is replaced by a random one. Slices start empty, not
// nil, so an unfilled receipt still encodes them as [].
func NewReceipt(deployable, correlationID string) Receipt {
	if correlationID == "" {
		correlationID = selfupdate.RandomCorrelationID()
	}
	return Receipt{
		Schema:         ReceiptSchema,
		SchemaVersion:  ReceiptSchemaVersion,
		CorrelationID:  correlationID,
		Deployable:     deployable,
		Targets:        []ReceiptTarget{},
		RollbackStatus: "not_attempted",
		RollbackErrors: []string{},
	}
}

// Outcome is the cause a deployable run ended with. The values are the
// matching self-update outcomes, so both classify under one rule set.
type Outcome string

const (
	OutcomeInstalled        Outcome = "installed"
	OutcomeTargetCurrent    Outcome = "target-current"
	OutcomeCheckOnly        Outcome = "check-only"
	OutcomeBusy             Outcome = "busy"
	OutcomeGateFailed       Outcome = "gate-failed"
	OutcomePrepareFailed    Outcome = "prepare-failed"
	OutcomePinSkew          Outcome = "pin-skew"
	OutcomeRolledBack       Outcome = "rolled-back"
	OutcomeRollbackFailed   Outcome = "rollback-failed"
	OutcomeHandoffRefused   Outcome = "handoff-refused"
	OutcomeHotCopyDivergent Outcome = "hot-copy-divergent"
	OutcomeRestartRequired  Outcome = "restart_required"
)

var knownOutcomes = map[Outcome]bool{
	OutcomeInstalled:        true,
	OutcomeTargetCurrent:    true,
	OutcomeCheckOnly:        true,
	OutcomeBusy:             true,
	OutcomeGateFailed:       true,
	OutcomePrepareFailed:    true,
	OutcomePinSkew:          true,
	OutcomeRolledBack:       true,
	OutcomeRollbackFailed:   true,
	OutcomeHandoffRefused:   true,
	OutcomeHotCopyDivergent: true,
	OutcomeRestartRequired:  true,
}

// StatusUnclassified is the status ClassifyOutcome reports for a cause it does
// not know. An unknown cause is never reported as current or updated.
const StatusUnclassified = "unclassified"

// NextCommands are the operator commands a receipt can point to. The caller
// supplies them because the command that re-runs a deployable depends on the
// driver, not on this contract.
type NextCommands struct {
	Inspect string // the run settled; look at what is installed
	Retry   string // the run did not settle and is safe to re-run
	Check   string // a human should check state before re-running
}

// ClassifyOutcome maps a cause to receipt posture. Status, rollback status,
// restart flag, and rollback errors come from selfupdate.ClassifyOutcome, so
// the two receipts cannot drift apart. The next command is picked from next by
// the resulting status. An unknown cause fails closed: StatusUnclassified with
// next.Check.
func ClassifyOutcome(cause Outcome, oldRev, newRev, detail string, next NextCommands) (status, rollbackStatus string, restartRequired bool, nextCommand string, rollbackErrors []string) {
	if !knownOutcomes[cause] {
		return StatusUnclassified, "not_attempted", false, next.Check, []string{}
	}
	status, rollbackStatus, restartRequired, _, rollbackErrors = selfupdate.ClassifyOutcome(selfupdate.Outcome(cause), oldRev, newRev, detail)
	switch status {
	case "updated", "current":
		nextCommand = next.Inspect
	case "gate_failed", "prepare_failed", "busy", "rolled_back", "stale", "divergent":
		nextCommand = next.Retry
	default:
		nextCommand = next.Check
	}
	return status, rollbackStatus, restartRequired, nextCommand, rollbackErrors
}
