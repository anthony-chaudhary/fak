package supervise

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/worktree"
)

// Failure reason tokens for contract reaping.
const (
	ReasonTimeout         = "TIMEOUT"
	ReasonBudgetExhausted = "BUDGET_EXHAUSTED"
	ReasonStuckLoop       = "STUCK_LOOP"
)

// ContractStore is the contract-persistence seam the supervisor drives.
// *leaseref.Store satisfies it directly, which keeps the supervisor invocable
// against the real ref-backed table rather than mock-only. The two methods are
// pinned to the store's actual shapes on purpose:
//   - ListContracts returns the FULL table (live AND expired). The store's
//     LiveContracts view hides expired records, which would make the liveness
//     rule in Supervisor.Tick untriggerable against real data; the supervisor
//     evaluates each record's TTL itself at tick time.
//   - UpdateContractState is the CAS state transition; the store preserves
//     holder, token accounting, and generation across it, so the reaper never
//     rewrites those fields.
type ContractStore interface {
	ListContracts(ctx context.Context) ([]leaseref.ContractRecord, error)
	UpdateContractState(ctx context.Context, ticketID string, newState leaseref.ContractState, now ...time.Time) (leaseref.ContractRecord, error)
}

// WorktreeDeallocator describes worktree deallocation.
type WorktreeDeallocator interface {
	Deallocate(ctx context.Context, ticketID string, keepBranch bool) error
}

// TreeLeaseReleaser releases tree locks held by a session.
type TreeLeaseReleaser interface {
	ReleaseSessionLeases(ctx context.Context, sessionID string) error
}

// FailureDiagnosis captures post-mortem information when a contract is reaped.
type FailureDiagnosis struct {
	TicketID string                  `json:"ticket_id"`
	Reason   string                  `json:"reason"`
	Detail   string                  `json:"detail"`
	FailedAt time.Time               `json:"failed_at"`
	Contract leaseref.ContractRecord `json:"contract"`
}

// ReaperOptions configures a contract reap execution.
type ReaperOptions struct {
	RepoRoot            string
	Holder              string
	Store               ContractStore
	WorktreeDeallocator WorktreeDeallocator
	TreeLeaseReleaser   TreeLeaseReleaser
}

// ReapContract transitions an expired or pathological contract to FAILED,
// removes its ephemeral worktree, releases associated tree leases, and records
// a structured failure diagnosis file under .dispatch-runs/failures/.
func ReapContract(ctx context.Context, rec leaseref.ContractRecord, reason, detail string, opts ReaperOptions, now ...time.Time) (*FailureDiagnosis, error) {
	tNow := time.Now()
	if len(now) > 0 && !now[0].IsZero() {
		tNow = now[0]
	}

	holder := opts.Holder
	if holder == "" {
		holder = "contract-reaper"
	}

	// 1. Transition contract to FAILED in leaseref store
	if opts.Store != nil {
		_, err := opts.Store.UpdateContractState(ctx, rec.TicketID, leaseref.ContractStateFailed, tNow)
		if err != nil && !errors.Is(err, leaseref.ErrContractExpired) {
			return nil, fmt.Errorf("reaper: failed to update contract state: %w", err)
		}
		// ErrContractExpired: the CAS store refuses to write a state onto an already
		// lapsed lease, so FAILED-marking is best-effort for TIMEOUT reaps. The ref
		// stays readable (and deletable via `fak contract reap`); cleanup and the
		// diagnosis below still run.
	}

	// 2. Deallocate worktree
	if opts.WorktreeDeallocator != nil {
		_ = opts.WorktreeDeallocator.Deallocate(ctx, rec.TicketID, false)
	}

	// 3. Release tree leases if session is bound
	if opts.TreeLeaseReleaser != nil && rec.SessionID != "" {
		_ = opts.TreeLeaseReleaser.ReleaseSessionLeases(ctx, rec.SessionID)
	}

	diagnosis := &FailureDiagnosis{
		TicketID: rec.TicketID,
		Reason:   reason,
		Detail:   detail,
		FailedAt: tNow,
		Contract: rec,
	}

	// 4. Persist failure diagnosis JSON file
	if opts.RepoRoot != "" {
		cleanID := worktree.CleanTicketID(rec.TicketID)
		failDir := filepath.Join(opts.RepoRoot, ".dispatch-runs", "failures")
		if err := os.MkdirAll(failDir, 0755); err == nil {
			filePath := filepath.Join(failDir, fmt.Sprintf("ticket-%s.json", cleanID))
			if data, err := json.MarshalIndent(diagnosis, "", "  "); err == nil {
				_ = os.WriteFile(filePath, data, 0644)
			}
		}
	}

	return diagnosis, nil
}

// CleanTicketID is an alias for worktree.CleanTicketID.
func CleanTicketID(id string) string {
	return worktree.CleanTicketID(strings.TrimSpace(id))
}
