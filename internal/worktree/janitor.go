package worktree

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// JanitorReport summarizes discovered and cleaned worktrees.
type JanitorReport struct {
	ScannedCount  int      `json:"scanned_count"`
	CleanedCount  int      `json:"cleaned_count"`
	RetainedCount int      `json:"retained_count"`
	Cleaned       []string `json:"cleaned,omitempty"`
	Retained      []string `json:"retained,omitempty"`
	Errors        []string `json:"errors,omitempty"`
}

// Janitor inspects the worktrees directory, detects orphaned/abandoned worktrees
// whose contracts are expired, missing, or terminal, and deallocates them.
func (m *Manager) Janitor(ctx context.Context, liveContracts []leaseref.ContractRecord) (*JanitorReport, error) {
	worktreesRoot := m.worktreesDir
	if !filepath.IsAbs(worktreesRoot) {
		worktreesRoot = filepath.Join(m.repoRoot, m.worktreesDir)
	}

	report := &JanitorReport{}

	entries, err := os.ReadDir(worktreesRoot)
	if os.IsNotExist(err) {
		return report, nil
	}
	if err != nil {
		return nil, fmt.Errorf("worktree janitor: failed to read worktrees root: %w", err)
	}

	// Build lookup of active contract ticket IDs
	activeContracts := make(map[string]bool, len(liveContracts))
	now := time.Now()
	for _, rec := range liveContracts {
		cleanID := CleanTicketID(rec.TicketID)
		if !rec.Expired(now) && rec.State != leaseref.ContractStateSucceeded && rec.State != leaseref.ContractStateFailed {
			activeContracts[cleanID] = true
		}
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, "ticket-") {
			continue
		}

		report.ScannedCount++
		cleanID := strings.TrimPrefix(name, "ticket-")

		if activeContracts[cleanID] {
			report.RetainedCount++
			report.Retained = append(report.Retained, cleanID)
			continue
		}

		// Orphaned worktree: deallocate
		if err := m.Deallocate(ctx, cleanID, false); err != nil {
			report.Errors = append(report.Errors, fmt.Sprintf("clean %s: %v", cleanID, err))
		} else {
			report.CleanedCount++
			report.Cleaned = append(report.Cleaned, cleanID)
		}
	}

	return report, nil
}

// Sweep runs Janitor to detect and deallocate orphaned or terminal worktrees.
func (m *Manager) Sweep(ctx context.Context, liveContracts []leaseref.ContractRecord) (*JanitorReport, error) {
	return m.Janitor(ctx, liveContracts)
}
