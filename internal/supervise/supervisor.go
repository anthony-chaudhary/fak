package supervise

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TurnRecord captures tool interaction data for stuck loop detection.
type TurnRecord struct {
	Timestamp          time.Time `json:"timestamp"`
	ToolInputSignature string    `json:"tool_input_signature"`
	FilesChangedCount  int       `json:"files_changed_count"`
}

// SupervisorConfig configures contract supervision.
type SupervisorConfig struct {
	RepoRoot            string
	Store               ContractStore
	WorktreeDeallocator WorktreeDeallocator
	TreeLeaseReleaser   TreeLeaseReleaser
	Holder              string
	StuckLoopThreshold  int // default 3 (>2 consecutive identical turns with 0 file changes)
}

// SuperviseReport provides summary metrics from a supervisor tick.
type SuperviseReport struct {
	InspectedCount int                `json:"inspected_count"`
	ActiveCount    int                `json:"active_count"`
	ReapedCount    int                `json:"reaped_count"`
	Reaped         []FailureDiagnosis `json:"reaped,omitempty"`
}

// Supervisor monitors live contracts for liveness expiration, token budget overruns,
// and pathological stuck loops.
type Supervisor struct {
	mu     sync.Mutex
	cfg    SupervisorConfig
	turns  map[string][]TurnRecord
	holder string
}

// NewSupervisor creates a new contract supervisor.
func NewSupervisor(cfg SupervisorConfig) *Supervisor {
	threshold := cfg.StuckLoopThreshold
	if threshold <= 0 {
		threshold = 3 // default: >2 turns
	}
	cfg.StuckLoopThreshold = threshold

	holder := cfg.Holder
	if holder == "" {
		holder = "contract-supervisor"
	}

	return &Supervisor{
		cfg:    cfg,
		turns:  make(map[string][]TurnRecord),
		holder: holder,
	}
}

// RecordTurn records a worker turn's tool signature and file change count for stuck-loop analysis.
func (s *Supervisor) RecordTurn(ticketID, toolInputSignature string, filesChangedCount int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	cleanID := CleanTicketID(ticketID)
	rec := TurnRecord{
		Timestamp:          time.Now(),
		ToolInputSignature: toolInputSignature,
		FilesChangedCount:  filesChangedCount,
	}

	history := s.turns[cleanID]
	history = append(history, rec)
	// Retain only the last 10 turns
	if len(history) > 10 {
		history = history[len(history)-10:]
	}
	s.turns[cleanID] = history
}

// isStuckLoopLocked checks if a ticket has emitted identical tool input signatures
// across >= StuckLoopThreshold consecutive turns with zero file changes.
func (s *Supervisor) isStuckLoopLocked(cleanID string) (bool, string) {
	history := s.turns[cleanID]
	threshold := s.cfg.StuckLoopThreshold
	if len(history) < threshold {
		return false, ""
	}

	lastSig := history[len(history)-1].ToolInputSignature
	if lastSig == "" {
		return false, ""
	}

	for i := len(history) - threshold; i < len(history); i++ {
		t := history[i]
		if t.ToolInputSignature != lastSig || t.FilesChangedCount > 0 {
			return false, ""
		}
	}

	return true, fmt.Sprintf("repeated tool signature %q across %d consecutive turns with 0 file changes", lastSig, threshold)
}

// ClearTicket clears turn history when a ticket completes or is deallocated.
func (s *Supervisor) ClearTicket(ticketID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.turns, CleanTicketID(ticketID))
}

// Tick evaluates all live contracts and reaps any expired, over-budget, or stuck workers.
func (s *Supervisor) Tick(ctx context.Context, now ...time.Time) (*SuperviseReport, error) {
	if s.cfg.Store == nil {
		return nil, fmt.Errorf("supervisor: nil ContractStore configured")
	}

	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}

	contracts, err := s.cfg.Store.LiveContracts(ctx, tNow)
	if err != nil {
		return nil, fmt.Errorf("supervisor: failed to fetch live contracts: %w", err)
	}

	report := &SuperviseReport{
		InspectedCount: len(contracts),
	}

	reaperOpts := ReaperOptions{
		RepoRoot:            s.cfg.RepoRoot,
		Holder:              s.holder,
		Store:               s.cfg.Store,
		WorktreeDeallocator: s.cfg.WorktreeDeallocator,
		TreeLeaseReleaser:   s.cfg.TreeLeaseReleaser,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for _, rec := range contracts {
		cleanID := CleanTicketID(rec.TicketID)

		// Ignore already terminal contracts
		if rec.State == leaseref.ContractStateSucceeded || rec.State == leaseref.ContractStateFailed {
			continue
		}

		var reapReason, reapDetail string

		// Rule 1: Liveness expiration
		// Notice: rec.Expired checks if tNow >= effectiveActiveAt + TTL.
		// If the worker is sending heartbeats (RenewContract), RenewedAt is fresh, so Expired is false!
		if rec.Expired(tNow) {
			activeAt := rec.AcquiredAt
			if rec.RenewedAt > activeAt {
				activeAt = rec.RenewedAt
			}
			reapReason = ReasonTimeout
			reapDetail = fmt.Sprintf("contract lease expired at %s (TTL: %ds, RenewedAt: %d, AcquiredAt: %d)",
				time.Unix(activeAt+int64(rec.TTLSeconds), 0).Format(time.RFC3339),
				rec.TTLSeconds, rec.RenewedAt, rec.AcquiredAt)
		} else if rec.TokenBudget > 0 && rec.TokensUsed > rec.TokenBudget {
			// Rule 2: Token budget overrun
			reapReason = ReasonBudgetExhausted
			reapDetail = fmt.Sprintf("tokens used (%d) exceeded allocated token budget (%d)",
				rec.TokensUsed, rec.TokenBudget)
		} else if isStuck, detail := s.isStuckLoopLocked(cleanID); isStuck {
			// Rule 3: Stuck loop detection
			reapReason = ReasonStuckLoop
			reapDetail = detail
		}

		if reapReason != "" {
			diagnosis, err := ReapContract(ctx, rec, reapReason, reapDetail, reaperOpts, tNow)
			if err != nil {
				// Record diagnosis even if state update had partial failure
				diagnosis = &FailureDiagnosis{
					TicketID: rec.TicketID,
					Reason:   reapReason,
					Detail:   fmt.Sprintf("%s (reap err: %v)", reapDetail, err),
					FailedAt: tNow,
					Contract: rec,
				}
			}
			report.Reaped = append(report.Reaped, *diagnosis)
			report.ReapedCount++
			delete(s.turns, cleanID)
		} else {
			report.ActiveCount++
		}
	}

	return report, nil
}
