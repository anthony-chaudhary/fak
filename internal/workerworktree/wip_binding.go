package workerworktree

import (
	"errors"
	"fmt"
	"strings"
)

// WIPBindingSchema is the stable wire-format version for managed-worktree WIP bindings.
const WIPBindingSchema = "fak-worker-worktree-wip-binding/1"

// WIPBinding is the explicit, read-only provenance needed to associate a
// managed worktree with a logical WIP unit. Repository paths, commits, file
// contents, and timestamps are deliberately absent: none is ownership proof.
type WIPBinding struct {
	Schema     string `json:"schema"`
	WorktreeID string `json:"worktree_id"`
	WIPUnitID  string `json:"wip_unit_id"`
	WorkerID   string `json:"worker_id"`
	Lane       string `json:"lane"`
	LeaseID    string `json:"lease_id"`
	Registered bool   `json:"registered"`
	Dirty      bool   `json:"dirty"`
}

// Validate checks only declared provenance. It performs no filesystem or Git
// reads and never changes, lands, moves, or reaps a worktree.
func (b WIPBinding) Validate() error {
	var errs []error
	if b.Schema != WIPBindingSchema {
		errs = append(errs, fmt.Errorf("schema %q: want %q", b.Schema, WIPBindingSchema))
	}
	for name, value := range map[string]string{
		"worktree_id": b.WorktreeID,
		"wip_unit_id": b.WIPUnitID,
		"worker_id":   b.WorkerID,
		"lane":        b.Lane,
		"lease_id":    b.LeaseID,
	} {
		if strings.TrimSpace(value) == "" {
			errs = append(errs, fmt.Errorf("%s is required", name))
		}
	}
	if !b.Registered {
		errs = append(errs, errors.New("managed worktree registration is required"))
	}
	return errors.Join(errs...)
}

// WIPWorktreeID returns the receipt's authoritative managed-worktree ID.
func (b WIPBinding) WIPWorktreeID() string { return b.WorktreeID }

// WIPUnit returns the explicitly registered logical unit ID.
func (b WIPBinding) WIPUnit() string { return b.WIPUnitID }

// WIPWorkerID returns the worker identity recorded at allocation.
func (b WIPBinding) WIPWorkerID() string { return b.WorkerID }

// WIPLane returns the declared lane, not a path-derived guess.
func (b WIPBinding) WIPLane() string { return b.Lane }

// WIPLeaseID returns the authoritative lease receipt ID.
func (b WIPBinding) WIPLeaseID() string { return b.LeaseID }

// WIPRegistered reports whether the binding was registered explicitly.
func (b WIPBinding) WIPRegistered() bool { return b.Registered }
