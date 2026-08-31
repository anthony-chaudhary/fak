package coordination

import (
	"errors"
	"fmt"
	"slices"
)

// BootstrapRepairRequest describes the single guarded worker that may repair the
// managed-worker substrate when both canonical launch paths failed in one run.
type BootstrapRepairRequest struct {
	Issue                 int
	BaseSHA               string
	CandidatePath         string
	Lease                 string
	Files                 []string
	PrepareFailureDigest  string
	DispatchFailureDigest string
	CandidateDirty        bool
	CandidateLiveOwned    bool
	LeaseActive           bool
	PathsOverlap          bool
	CanonicalPrepareOK    bool
	CanonicalDispatchOK   bool
}

// BootstrapRepairAdmission is the immutable contract handed to the normal
// guarded worker launcher after bootstrap admission succeeds.
type BootstrapRepairAdmission struct {
	Issue                 int
	BaseSHA               string
	CandidatePath         string
	Lease                 string
	Files                 []string
	PrepareFailureDigest  string
	DispatchFailureDigest string
}

// BootstrapRepairGate admits at most one substrate-repair worker. Construct a
// new gate only for a new coordinator run; successful canonical recovery
// permanently retires the gate for that run.
type BootstrapRepairGate struct {
	launched bool
	retired  bool
}

var ErrBootstrapRepairRefused = errors.New("bootstrap repair refused")

func (g *BootstrapRepairGate) Admit(req BootstrapRepairRequest) (BootstrapRepairAdmission, error) {
	if g.retired || req.CanonicalPrepareOK || req.CanonicalDispatchOK {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: canonical managed-worker substrate is healthy", ErrBootstrapRepairRefused)
	}
	if g.launched {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: one repair worker already launched", ErrBootstrapRepairRefused)
	}
	if req.PrepareFailureDigest == "" || req.DispatchFailureDigest == "" {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: both canonical failure receipts are required", ErrBootstrapRepairRefused)
	}
	if req.Issue <= 0 || req.BaseSHA == "" || req.CandidatePath == "" || req.Lease == "" || len(req.Files) == 0 {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: issue, base SHA, candidate path, lease, and explicit files are required", ErrBootstrapRepairRefused)
	}
	if req.CandidateDirty {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: candidate worktree is dirty", ErrBootstrapRepairRefused)
	}
	if req.CandidateLiveOwned {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: candidate worktree has a live owner", ErrBootstrapRepairRefused)
	}
	if !req.LeaseActive {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: repair lease is not active", ErrBootstrapRepairRefused)
	}
	if req.PathsOverlap {
		return BootstrapRepairAdmission{}, fmt.Errorf("%w: explicit files overlap an active lease", ErrBootstrapRepairRefused)
	}

	g.launched = true
	return BootstrapRepairAdmission{
		Issue:                 req.Issue,
		BaseSHA:               req.BaseSHA,
		CandidatePath:         req.CandidatePath,
		Lease:                 req.Lease,
		Files:                 slices.Clone(req.Files),
		PrepareFailureDigest:  req.PrepareFailureDigest,
		DispatchFailureDigest: req.DispatchFailureDigest,
	}, nil
}

// Retire disables bootstrap admission after both canonical launch paths pass.
func (g *BootstrapRepairGate) Retire(canonicalPrepareOK, canonicalDispatchOK bool) error {
	if !canonicalPrepareOK || !canonicalDispatchOK {
		return fmt.Errorf("%w: both canonical launch paths must be healthy before retirement", ErrBootstrapRepairRefused)
	}
	g.retired = true
	return nil
}
