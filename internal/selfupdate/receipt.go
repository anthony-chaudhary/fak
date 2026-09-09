package selfupdate

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// ReceiptSchema is the versioned schema identity for self-update receipts.
const ReceiptSchema = "fak.self-update.receipt/v1"

// ReceiptSchemaVersion is the integer version of ReceiptSchema.
const ReceiptSchemaVersion = 1

// Receipt is the stable versioned payload describing self-update execution, decisions, and posture.
type Receipt struct {
	Schema          string                     `json:"schema"`
	SchemaVersion   int                        `json:"schema_version"`
	CorrelationID   string                     `json:"correlation_id"`
	Status          string                     `json:"status"`
	OldRevision     *string                    `json:"old_revision"`
	NewRevision     *string                    `json:"new_revision"`
	Targets         []ReceiptTarget            `json:"targets"`
	Attempted       int                        `json:"attempted"`
	Changed         int                        `json:"changed"`
	RollbackStatus  string                     `json:"rollback_status"`
	RollbackErrors  []string                   `json:"rollback_errors"`
	RestartRequired bool                       `json:"restart_required"`
	NextCommand     string                     `json:"next_command"`
	Detail          string                     `json:"detail,omitempty"`
	BuildProvenance *BuildProvenance           `json:"build_provenance,omitempty"`
	Transfer        *TransferReceipt           `json:"transfer,omitempty"`
	Handoff         *HandoffReceipt            `json:"handoff,omitempty"`
	CandidateCache  *CandidateCacheDisposition `json:"candidate_cache,omitempty"`
	TotalMS         int64                      `json:"total_ms"`
	PhaseMS         PhaseMS                    `json:"phase_ms"`
}

// ReceiptTarget represents one binary target managed by the self-update transaction.
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

// BuildProvenance carries provenance and validation metadata for reproducible builds.
type BuildProvenance struct {
	SourceCommit         string            `json:"source_commit"`
	ArtifactSourceCommit string            `json:"artifact_source_commit"`
	BuildInputDigest     string            `json:"build_input_digest"`
	BuildEnvelope        map[string]string `json:"build_envelope"`
	ArtifactDigest       string            `json:"artifact_digest"`
	ArtifactSize         int64             `json:"artifact_size"`
	AppVersion           string            `json:"app_version"`
	Reused               bool              `json:"reused"`
}

// Phase is a named step in the self-update execution pipeline.
type Phase string

const (
	PhaseCheck     Phase = "check"
	PhaseLock      Phase = "lock"
	PhaseCleanup   Phase = "cleanup"
	PhasePrepare   Phase = "prepare"
	PhaseCompanion Phase = "companion"
	PhaseBuild     Phase = "build"
	PhaseVet       Phase = "vet"
	PhaseSmoke     Phase = "smoke"
	PhaseInstall   Phase = "install"
	PhaseVerify    Phase = "verify"
	PhaseHandoff   Phase = "handoff"
)

// PhaseOrder is the canonical sequence of self-update phases.
var PhaseOrder = [...]Phase{
	PhaseCheck,
	PhaseLock,
	PhaseCleanup,
	PhasePrepare,
	PhaseCompanion,
	PhaseBuild,
	PhaseVet,
	PhaseSmoke,
	PhaseInstall,
	PhaseVerify,
	PhaseHandoff,
}

// PhaseMS is a fixed-shape timing object so receipt consumers always observe
// identical phase vocabulary, including zero durations for skipped phases.
type PhaseMS struct {
	Check     int64 `json:"check"`
	Lock      int64 `json:"lock"`
	Cleanup   int64 `json:"cleanup"`
	Prepare   int64 `json:"prepare"`
	Companion int64 `json:"companion"`
	Build     int64 `json:"build"`
	Vet       int64 `json:"vet"`
	Smoke     int64 `json:"smoke"`
	Install   int64 `json:"install"`
	Verify    int64 `json:"verify"`
	Handoff   int64 `json:"handoff"`
}

// Set updates the duration in milliseconds for the specified phase.
func (p *PhaseMS) Set(phase Phase, value int64) {
	switch phase {
	case PhaseCheck:
		p.Check = value
	case PhaseLock:
		p.Lock = value
	case PhaseCleanup:
		p.Cleanup = value
	case PhasePrepare:
		p.Prepare = value
	case PhaseCompanion:
		p.Companion = value
	case PhaseBuild:
		p.Build = value
	case PhaseVet:
		p.Vet = value
	case PhaseSmoke:
		p.Smoke = value
	case PhaseInstall:
		p.Install = value
	case PhaseVerify:
		p.Verify = value
	case PhaseHandoff:
		p.Handoff = value
	}
}

// TransferReceipt records metrics from differential or fallback full artifact transport.
type TransferReceipt struct {
	ChosenPath     string `json:"chosen_path"`
	DeltaBytes     int64  `json:"delta_bytes"`
	FullBytes      int64  `json:"full_bytes"`
	TotalMS        int64  `json:"elapsed_total_ms"`
	Verification   string `json:"verification"`
	FallbackReason string `json:"fallback_reason,omitempty"`
	FallbackBytes  int64  `json:"fallback_bytes"`
	FallbackMS     int64  `json:"fallback_ms"`
}

// HandoffReceipt records the result of graceful process replacement.
type HandoffReceipt struct {
	State             selfinstall.HandoffState `json:"state"`
	SessionID         string                   `json:"session_id"`
	SuccessorRevision string                   `json:"successor_revision"`
	Detail            string                   `json:"detail,omitempty"`
}

// CandidateCacheState classifies the candidate build-cache disposition.
type CandidateCacheState string

const (
	CandidateCacheDisabled CandidateCacheState = "disabled"
	CandidateCacheMiss     CandidateCacheState = "miss"
	CandidateCacheHit      CandidateCacheState = "hit"
	CandidateCacheRejected CandidateCacheState = "rejected"
)

// CandidateCacheDisposition records candidate build-cache reuse state and reasoning.
type CandidateCacheDisposition struct {
	State  CandidateCacheState `json:"state"`
	Reason string              `json:"reason,omitempty"`
}

// Outcome classifies the cause and terminal result of a self-update execution.
type Outcome string

const (
	OutcomeInstalled        Outcome = "installed"
	OutcomeMetadataOnly     Outcome = "metadata_only"
	OutcomeTargetCurrent    Outcome = "target-current"
	OutcomeSelfFresh        Outcome = "self-fresh"
	OutcomeSelfAhead        Outcome = "self-ahead"
	OutcomeSelfLocal        Outcome = "self-local"
	OutcomeSelfUnknown      Outcome = "self-unknown"
	OutcomeBusy             Outcome = "busy"
	OutcomeCheckOnly        Outcome = "check-only"
	OutcomeGateFailed       Outcome = "gate-failed"
	OutcomePrepareFailed    Outcome = "prepare-failed"
	OutcomeRolledBack       Outcome = "rolled-back"
	OutcomeRollbackFailed   Outcome = "rollback-failed"
	OutcomeHandoffRefused   Outcome = "handoff-refused"
	OutcomeHotCopyDivergent Outcome = "hot-copy-divergent"
	OutcomePinSkew          Outcome = "pin-skew"
	OutcomeRestartRequired  Outcome = "restart_required"
)

// ClassifyOutcome maps an execution outcome, revision states, and error detail
// to receipt posture fields (status, rollbackStatus, restartRequired, nextCommand, rollbackErrors).
func ClassifyOutcome(cause Outcome, oldRev, newRev, detail string) (status string, rollbackStatus string, restartRequired bool, nextCommand string, rollbackErrors []string) {
	status = "current"
	rollbackStatus = "not_attempted"
	restartRequired = false
	nextCommand = "fak version"
	switch cause {
	case OutcomeInstalled:
		status = "updated"
	case OutcomeMetadataOnly:
		status = "current"
	case OutcomeGateFailed:
		status, nextCommand = "gate_failed", "fak self-update"
	case OutcomePrepareFailed:
		status, nextCommand = "prepare_failed", "fak self-update"
	case OutcomePinSkew:
		status, nextCommand = "pin_skew", "fak self-update --check"
	case OutcomeRolledBack:
		status, rollbackStatus, nextCommand = "rolled_back", "succeeded", "fak self-update"
	case OutcomeRollbackFailed:
		status, rollbackStatus, nextCommand = "rollback_failed", "failed", "fak self-update --check"
	case OutcomeBusy:
		status, nextCommand = "busy", "fak self-update"
	case OutcomeCheckOnly:
		if oldRevision, newRevision := strings.TrimSpace(oldRev), strings.TrimSpace(newRev); oldRevision != "" && newRevision != "" && oldRevision != newRevision {
			status, nextCommand = string(StatusStale), "fak self-update"
		}
	case OutcomeHotCopyDivergent:
		status, nextCommand = string(StatusDivergent), "fak self-update"
	case OutcomeHandoffRefused:
		status, nextCommand = "handoff_refused", "fak self-update --check"
	case OutcomeRestartRequired:
		status, restartRequired, nextCommand = "restart_required", true, "fak self-update --check"
	}
	rollbackErrors = []string{}
	if status == "rollback_failed" && strings.TrimSpace(detail) != "" {
		rollbackErrors = append(rollbackErrors, detail)
	}
	return status, rollbackStatus, restartRequired, nextCommand, rollbackErrors
}

// OptionalRevision returns nil when revision is empty or whitespace, or a pointer to rev.
func OptionalRevision(rev string) *string {
	if strings.TrimSpace(rev) == "" {
		return nil
	}
	return &rev
}

// NormalizeTargets returns a clean slice of targets. When empty and fallbackTarget
// is provided (and not "<self>"), a single primary target entry is generated.
func NormalizeTargets(targets []ReceiptTarget, fallbackTarget string) []ReceiptTarget {
	res := append([]ReceiptTarget(nil), targets...)
	if len(res) == 0 && strings.TrimSpace(fallbackTarget) != "" && fallbackTarget != "<self>" {
		res = append(res, ReceiptTarget{
			Role: "primary",
			Path: filepath.ToSlash(filepath.Clean(fallbackTarget)),
		})
	}
	if res == nil {
		res = []ReceiptTarget{}
	}
	return res
}

// RandomCorrelationID generates a 16-byte random hex correlation identifier.
func RandomCorrelationID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err == nil {
		return hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("pid-%d", os.Getpid())
}

// Builder constructs versioned Receipt instances with explicit configuration.
type Builder struct {
	correlationID   string
	oldRevision     string
	newRevision     string
	targets         []ReceiptTarget
	attempted       int
	changed         int
	status          string
	rollbackStatus  string
	rollbackErrors  []string
	restartRequired bool
	nextCommand     string
	detail          string
	buildProvenance *BuildProvenance
	transfer        *TransferReceipt
	handoff         *HandoffReceipt
	candidateCache  *CandidateCacheDisposition
	totalMS         int64
	phaseMS         PhaseMS
}

// NewBuilder creates a new Builder with initialized default slices.
func NewBuilder() *Builder {
	return &Builder{
		targets:        []ReceiptTarget{},
		rollbackErrors: []string{},
	}
}

// Reset clears all configured builder state back to defaults.
func (b *Builder) Reset() *Builder {
	*b = Builder{
		targets:        []ReceiptTarget{},
		rollbackErrors: []string{},
	}
	return b
}

// SetCorrelationID specifies the tracing token linking the execution across logs and parent invocations.
func (b *Builder) SetCorrelationID(id string) *Builder {
	b.correlationID = id
	return b
}

// SetOldRevision records the commit hash or version stamp of the installed binary prior to update.
func (b *Builder) SetOldRevision(rev string) *Builder {
	b.oldRevision = rev
	return b
}

// SetNewRevision records the target commit hash or version stamp being installed or verified.
func (b *Builder) SetNewRevision(rev string) *Builder {
	b.newRevision = rev
	return b
}

// SetRevisions updates both the pre-existing and target revisions in one step.
func (b *Builder) SetRevisions(oldRev, newRev string) *Builder {
	b.oldRevision = oldRev
	b.newRevision = newRev
	return b
}

// AddTarget registers an additional destination binary into the planned or executed target set.
func (b *Builder) AddTarget(target ReceiptTarget) *Builder {
	b.targets = append(b.targets, target)
	return b
}

// SetTargets installs a replacement slice of destination binaries, defensively copying non-nil inputs.
func (b *Builder) SetTargets(targets []ReceiptTarget) *Builder {
	if targets == nil {
		b.targets = []ReceiptTarget{}
	} else {
		b.targets = append([]ReceiptTarget(nil), targets...)
	}
	return b
}

// SetAttempted records the number of destination binaries the updater scheduled for replacement.
func (b *Builder) SetAttempted(attempted int) *Builder {
	b.attempted = attempted
	return b
}

// SetChanged records the count of destination binaries successfully written and activated.
func (b *Builder) SetChanged(changed int) *Builder {
	b.changed = changed
	return b
}

// SetCounts records both scheduled attempts and successful activations.
func (b *Builder) SetCounts(attempted, changed int) *Builder {
	b.attempted = attempted
	b.changed = changed
	return b
}

// SetStatus explicitly overrides the automatic outcome-to-status classification.
func (b *Builder) SetStatus(status string) *Builder {
	b.status = status
	return b
}

// SetRollbackStatus overrides the default rollback posture.
func (b *Builder) SetRollbackStatus(status string) *Builder {
	b.rollbackStatus = status
	return b
}

// SetRollbackErrors supplies captured failure messages encountered during transaction reversion.
func (b *Builder) SetRollbackErrors(errors []string) *Builder {
	if errors == nil {
		b.rollbackErrors = []string{}
	} else {
		b.rollbackErrors = append([]string(nil), errors...)
	}
	return b
}

// SetRestartRequired marks whether running daemon or background processes must be restarted to adopt changes.
func (b *Builder) SetRestartRequired(required bool) *Builder {
	b.restartRequired = required
	return b
}

// SetNextCommand overrides the suggested CLI action displayed to operators.
func (b *Builder) SetNextCommand(cmd string) *Builder {
	b.nextCommand = cmd
	return b
}

// SetDetail attaches diagnostic or error text to the receipt.
func (b *Builder) SetDetail(detail string) *Builder {
	b.detail = detail
	return b
}

// SetBuildProvenance attaches verified compiler inputs, source commit, and artifact digests for reproducible builds.
func (b *Builder) SetBuildProvenance(p *BuildProvenance) *Builder {
	b.buildProvenance = p
	return b
}

// SetTransfer attaches network transfer accounting including patch compression ratios and fallback metrics.
func (b *Builder) SetTransfer(t *TransferReceipt) *Builder {
	b.transfer = t
	return b
}

// SetHandoff records state transitions and successor PID information during process graceful replacement.
func (b *Builder) SetHandoff(h *HandoffReceipt) *Builder {
	b.handoff = h
	return b
}

// SetCandidateCache records whether candidate compilation reused shared warm worktree objects or populated cold entries.
func (b *Builder) SetCandidateCache(c *CandidateCacheDisposition) *Builder {
	b.candidateCache = c
	return b
}

// SetTiming populates elapsed execution wall-clock duration and per-phase breakdown metrics.
func (b *Builder) SetTiming(totalMS int64, phaseMS PhaseMS) *Builder {
	b.totalMS = totalMS
	b.phaseMS = phaseMS
	return b
}

// Build constructs a Receipt using configured builder state and outcome classification.
func (b *Builder) Build(cause Outcome, target, detail string) Receipt {
	return b.BuildWithTiming(cause, target, detail, b.totalMS, b.phaseMS)
}

// BuildWithTiming constructs a Receipt incorporating timing metrics.
func (b *Builder) BuildWithTiming(cause Outcome, target, detail string, totalMS int64, phaseMS PhaseMS) Receipt {
	detailVal := detail
	if detailVal == "" {
		detailVal = b.detail
	}
	detailVal = strings.TrimSpace(detailVal)

	status, rollbackStatus, restartRequired, nextCommand, rollbackErrors := ClassifyOutcome(cause, b.oldRevision, b.newRevision, detailVal)
	if b.status != "" {
		status = b.status
	}
	if b.rollbackStatus != "" {
		rollbackStatus = b.rollbackStatus
	}
	if len(b.rollbackErrors) > 0 {
		rollbackErrors = append([]string(nil), b.rollbackErrors...)
	}
	if b.restartRequired {
		restartRequired = true
	}
	if b.nextCommand != "" {
		nextCommand = b.nextCommand
	}

	targets := NormalizeTargets(b.targets, target)

	corrID := b.correlationID
	if corrID == "" {
		corrID = RandomCorrelationID()
	}

	timingTotal := totalMS
	if timingTotal == 0 && b.totalMS != 0 {
		timingTotal = b.totalMS
	}
	timingPhase := phaseMS
	if timingPhase == (PhaseMS{}) && b.phaseMS != (PhaseMS{}) {
		timingPhase = b.phaseMS
	}

	return Receipt{
		Schema:          ReceiptSchema,
		SchemaVersion:   ReceiptSchemaVersion,
		CorrelationID:   corrID,
		Status:          status,
		OldRevision:     OptionalRevision(b.oldRevision),
		NewRevision:     OptionalRevision(b.newRevision),
		Targets:         targets,
		Attempted:       b.attempted,
		Changed:         b.changed,
		RollbackStatus:  rollbackStatus,
		RollbackErrors:  rollbackErrors,
		RestartRequired: restartRequired,
		NextCommand:     nextCommand,
		Detail:          detailVal,
		BuildProvenance: b.buildProvenance,
		Transfer:        b.transfer,
		Handoff:         b.handoff,
		CandidateCache:  b.candidateCache,
		TotalMS:         timingTotal,
		PhaseMS:         timingPhase,
	}
}
