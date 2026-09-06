package safecommit

import (
	"sync"
	"time"
)

// CommitPhase identifies a discrete lifecycle phase during commit execution (#11844).
type CommitPhase string

const (
	PhaseValidation  CommitPhase = "validation"   // Pre-commit validation and gate checks
	PhaseReview      CommitPhase = "review"       // Optional pre-commit review
	PhaseLockWait    CommitPhase = "lock_wait"    // Advisory commit lock acquisition/wait
	PhaseLockAcquire CommitPhase = "lock_wait"    // Alias for PhaseLockWait
	PhaseWriterLease CommitPhase = "writer_lease" // Cooperative worktree writer lease
	PhaseAuthority   CommitPhase = "authority"    // Authority fence revalidation
	PhaseIndexNotes  CommitPhase = "index_notes"  // Auto-indexing notes and pre-staging
	PhaseStage       CommitPhase = "stage"        // Path staging (git add)
	PhaseCommit      CommitPhase = "commit"       // Commit creation (git commit)
	PhaseVerify      CommitPhase = "verify"       // Post-commit effect verification
	PhasePush        CommitPhase = "push"         // Optional push of verified commit
)

func (p CommitPhase) String() string {
	return string(p)
}

// Phase status constants.
const (
	PhaseStatusPending = "pending"
	PhaseStatusStarted = "started"
	PhaseStatusPassed  = "passed"
	PhaseStatusFailed  = "failed"
	PhaseStatusStalled = "stalled"
	PhaseStatusTimeout = "timeout"
	PhaseStatusSkipped = "skipped"
)

// Reason constants for post-validation commit stalls (#11844).
const (
	ReasonCommitStalled         = "COMMIT_STALLED"
	ReasonPostValidationTimeout = "POST_VALIDATION_TIMEOUT"
)

// PhaseEvidenceSchema is the versioned schema string for phase evidence.
const PhaseEvidenceSchema = "fak-safecommit-phase-evidence/1"

// DefaultPostValidationTimeout bounds total post-validation execution when no explicit
// timeout is configured on Options and the context has no shorter deadline.
const DefaultPostValidationTimeout = 30 * time.Second

// PhaseReceipt records timing and outcome for a single commit execution phase (#11844).
type PhaseReceipt struct {
	Phase      CommitPhase `json:"phase"`
	Status     string      `json:"status"`
	ElapsedNS  int64       `json:"elapsed_ns,omitempty"`
	DeadlineNS int64       `json:"deadline_ns,omitempty"`
	Detail     string      `json:"detail,omitempty"`
	Error      string      `json:"error,omitempty"`
}

// PhaseEvidence provides structured, versioned proof of phase progression and any stall (#11844).
type PhaseEvidence struct {
	Schema       string         `json:"schema"`
	CurrentPhase CommitPhase    `json:"current_phase"`
	Phases       []PhaseReceipt `json:"phases"`
	Stalled      bool           `json:"stalled,omitempty"`
	StallPhase   CommitPhase    `json:"stall_phase,omitempty"`
	StallReason  string         `json:"stall_reason,omitempty"`
	ElapsedNS    int64          `json:"elapsed_ns"`
	DeadlineNS   int64          `json:"deadline_ns,omitempty"`
}

// PhaseProgressionFunc receives real-time phase progression updates.
type PhaseProgressionFunc func(receipt PhaseReceipt)

// PhaseReporter is an optional interface for reporting phase progression.
type PhaseReporter interface {
	ReportPhase(receipt PhaseReceipt)
}

// PhaseObserver is an alternative interface for phase observation.
type PhaseObserver interface {
	ObservePhase(receipt PhaseReceipt)
}

type phaseTracker struct {
	mu           sync.Mutex
	opts         Options
	start        time.Time
	currentPhase CommitPhase
	phaseStart   time.Time
	receipts     []PhaseReceipt
	stalled      bool
	stallPhase   CommitPhase
	stallReason  string
	deadline     time.Duration
}

func newPhaseTracker(opts Options, deadline time.Duration) *phaseTracker {
	start := time.Now()
	return &phaseTracker{
		opts:       opts,
		start:      start,
		phaseStart: start,
		deadline:   deadline,
	}
}

func (pt *phaseTracker) emit(receipt PhaseReceipt) {
	if pt.opts.OnPhase != nil {
		pt.opts.OnPhase(receipt)
	}
	if pt.opts.PhaseReporter != nil {
		pt.opts.PhaseReporter.ReportPhase(receipt)
	}
	if observer, ok := any(pt.opts.PhaseReporter).(PhaseObserver); ok && observer != nil {
		observer.ObservePhase(receipt)
	}
}

func (pt *phaseTracker) Start(phase CommitPhase) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	pt.currentPhase = phase
	pt.phaseStart = time.Now()

	r := PhaseReceipt{
		Phase:      phase,
		Status:     PhaseStatusStarted,
		DeadlineNS: pt.deadline.Nanoseconds(),
	}
	pt.emit(r)
}

func (pt *phaseTracker) End(phase CommitPhase, status string, reason, detail string) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	elapsed := time.Since(pt.phaseStart)
	r := PhaseReceipt{
		Phase:      phase,
		Status:     status,
		ElapsedNS:  elapsed.Nanoseconds(),
		DeadlineNS: pt.deadline.Nanoseconds(),
		Detail:     detail,
	}
	if status == PhaseStatusFailed || status == PhaseStatusStalled || status == PhaseStatusTimeout {
		if reason != "" {
			r.Error = reason
		} else {
			r.Error = detail
		}
	}
	pt.receipts = append(pt.receipts, r)
	pt.emit(r)
}

func (pt *phaseTracker) MarkStalled(phase CommitPhase, reason string) {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	pt.stalled = true
	pt.stallPhase = phase
	pt.stallReason = reason
}

func (pt *phaseTracker) Finalize(res *Result) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	totalElapsed := time.Since(pt.start)

	if pt.currentPhase != "" && (len(pt.receipts) == 0 || pt.receipts[len(pt.receipts)-1].Phase != pt.currentPhase) {
		status := PhaseStatusPassed
		if res.Reason != "" || pt.stalled {
			if pt.stalled {
				status = PhaseStatusStalled
			} else {
				status = PhaseStatusFailed
			}
		}
		elapsed := time.Since(pt.phaseStart)
		r := PhaseReceipt{
			Phase:      pt.currentPhase,
			Status:     status,
			ElapsedNS:  elapsed.Nanoseconds(),
			DeadlineNS: pt.deadline.Nanoseconds(),
			Detail:     res.Detail,
			Error:      res.Reason,
		}
		pt.receipts = append(pt.receipts, r)
	}

	res.Phase = pt.currentPhase
	res.Phases = append([]PhaseReceipt(nil), pt.receipts...)

	evidence := PhaseEvidence{
		Schema:       PhaseEvidenceSchema,
		CurrentPhase: pt.currentPhase,
		Phases:       res.Phases,
		Stalled:      pt.stalled,
		StallPhase:   pt.stallPhase,
		StallReason:  pt.stallReason,
		ElapsedNS:    totalElapsed.Nanoseconds(),
		DeadlineNS:   pt.deadline.Nanoseconds(),
	}
	res.PhaseEvidence = &evidence
}

// Stalled reports whether commit execution stalled in any post-validation phase (#11844).
func (r Result) Stalled() bool {
	return r.PhaseEvidence != nil && r.PhaseEvidence.Stalled
}

// CurrentPhase returns the terminal or active commit phase (#11844).
func (r Result) CurrentPhase() CommitPhase {
	if r.PhaseEvidence != nil && r.PhaseEvidence.CurrentPhase != "" {
		return r.PhaseEvidence.CurrentPhase
	}
	return r.Phase
}
