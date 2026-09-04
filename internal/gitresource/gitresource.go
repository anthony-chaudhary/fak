// Package gitresource defines the ownership vocabulary used to reason about
// shared Git repositories, linked worktrees, and cleanup authority.
//
// Invariant: git resource operations are fail-closed and bounded.
// Every resource acquisition, validation, and cleanup decision requires explicit
// canonical identities, active lease verification, and valid epoch fencing before
// mutating or deleting any underlying git working tree or shared reference state.
// Guard: AdmitCleanup refuses any candidate with unpushed, untracked, dirty, or
// live working-tree state to guarantee zero data loss.
package gitresource

import (
	"errors"
	"fmt"
	"strings"
)

// IDs are opaque, canonical identities supplied by the component that
// resolves the corresponding resource. A display path or branch name is not
// an identity.
type (
	RepositoryID string
	WorktreeID   string
	ResourceID   string
	OwnerID      string
	LeaseID      string
	WorkspaceID  string
	HostID       string
	SandboxID    string
	GitDirID     string
	CommonDirID  string
	ResultID     string
	SnapshotID   string
)

// Epoch is a monotonically increasing fencing token for a lease.
type Epoch uint64

// ResourceKind is a closed vocabulary of resources whose ownership matters.
type ResourceKind string

const (
	// Repository-common Git resources.
	CommonRefs        ResourceKind = "common_refs"
	CommonConfig      ResourceKind = "common_config"
	CommonHooks       ResourceKind = "common_hooks"
	CommonObjectStore ResourceKind = "common_object_store"

	// Per-worktree Git resources.
	WorktreeHEAD   ResourceKind = "worktree_head"
	WorktreeIndex  ResourceKind = "worktree_index"
	WorktreeFiles  ResourceKind = "worktree_files"
	WorktreeConfig ResourceKind = "worktree_config"

	// Explicit filesystem and runtime resources.
	PathSetResource ResourceKind = "path_set"
	RuntimeProcess  ResourceKind = "runtime_process"
	RuntimeCWD      ResourceKind = "runtime_cwd"
	RuntimeLock     ResourceKind = "runtime_lock"
)

func (k ResourceKind) valid() bool {
	switch k {
	case CommonRefs, CommonConfig, CommonHooks, CommonObjectStore,
		WorktreeHEAD, WorktreeIndex, WorktreeFiles, WorktreeConfig,
		PathSetResource, RuntimeProcess, RuntimeCWD, RuntimeLock:
		return true
	default:
		return false
	}
}

func (k ResourceKind) common() bool {
	switch k {
	case CommonRefs, CommonConfig, CommonHooks, CommonObjectStore:
		return true
	default:
		return false
	}
}

func (k ResourceKind) perWorktree() bool {
	switch k {
	case WorktreeHEAD, WorktreeIndex, WorktreeFiles, WorktreeConfig:
		return true
	default:
		return false
	}
}

// ResourceScope states the identity boundary at which a resource is owned.
type ResourceScope string

const (
	RepositoryScope ResourceScope = "repository"
	WorktreeScope   ResourceScope = "worktree"
	PathSetScope    ResourceScope = "path_set"
	RuntimeScope    ResourceScope = "runtime"
)

// LeaseMode states whether other owners may concurrently hold the resource.
type LeaseMode string

const (
	SharedLease    LeaseMode = "shared"
	ExclusiveLease LeaseMode = "exclusive"
)

// LeaseState is the closed lifecycle of a resource lease.
type LeaseState string

const (
	LeaseOffered  LeaseState = "offered"
	LeaseActive   LeaseState = "active"
	LeaseTerminal LeaseState = "terminal"
	LeaseReleased LeaseState = "released"
)

// WorkspaceHandle binds a worker to canonical host, sandbox, repository, and
// Git-resolved worktree identities. Root is diagnostic only.
type WorkspaceHandle struct {
	ID         WorkspaceID
	Host       HostID
	Sandbox    SandboxID
	Repository RepositoryID
	Worktree   WorktreeID
	GitDir     GitDirID
	CommonDir  CommonDirID
	Root       string
}

// Validate fails closed unless every canonical identity is present.
func (h WorkspaceHandle) Validate() error {
	var errs []error
	for name, value := range map[string]string{
		"workspace":  string(h.ID),
		"host":       string(h.Host),
		"sandbox":    string(h.Sandbox),
		"repository": string(h.Repository),
		"worktree":   string(h.Worktree),
		"git_dir":    string(h.GitDir),
		"common_dir": string(h.CommonDir),
	} {
		if strings.TrimSpace(value) == "" {
			errs = append(errs, fmt.Errorf("canonical %s identity is required", name))
		}
	}
	return errors.Join(errs...)
}

// Resource is a typed ownership target. Repository is always required;
// Worktree is additionally required for per-worktree resources.
type Resource struct {
	ID         ResourceID
	Kind       ResourceKind
	Scope      ResourceScope
	Repository RepositoryID
	Worktree   WorktreeID
}

// Validate checks that kind and scope agree with Git's ownership boundary.
func (r Resource) Validate() error {
	if strings.TrimSpace(string(r.ID)) == "" {
		return errors.New("resource identity is required")
	}
	if !r.Kind.valid() {
		return fmt.Errorf("unknown resource kind %q", r.Kind)
	}
	if strings.TrimSpace(string(r.Repository)) == "" {
		return errors.New("repository identity is required")
	}
	switch {
	case r.Kind.common() && r.Scope != RepositoryScope:
		return fmt.Errorf("common Git resource %q requires repository scope", r.Kind)
	case r.Kind.perWorktree() && r.Scope != WorktreeScope:
		return fmt.Errorf("per-worktree Git resource %q requires worktree scope", r.Kind)
	case r.Kind.perWorktree() && strings.TrimSpace(string(r.Worktree)) == "":
		return fmt.Errorf("per-worktree Git resource %q requires worktree identity", r.Kind)
	case r.Kind == PathSetResource && r.Scope != PathSetScope:
		return errors.New("path-set resource requires path-set scope")
	case (r.Kind == RuntimeProcess || r.Kind == RuntimeCWD || r.Kind == RuntimeLock) && r.Scope != RuntimeScope:
		return fmt.Errorf("runtime resource %q requires runtime scope", r.Kind)
	}
	return nil
}

// Lease carries ownership and its fencing epoch.
type Lease struct {
	ID       LeaseID
	Resource Resource
	Owner    OwnerID
	Mode     LeaseMode
	State    LeaseState
	Epoch    Epoch
}

// Validate checks the closed lease vocabulary and its required identities.
func (l Lease) Validate() error {
	if err := l.Resource.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(string(l.ID)) == "" || strings.TrimSpace(string(l.Owner)) == "" {
		return errors.New("lease and owner identities are required")
	}
	if l.Epoch == 0 {
		return errors.New("positive fencing epoch is required")
	}
	if l.Mode != SharedLease && l.Mode != ExclusiveLease {
		return fmt.Errorf("unknown lease mode %q", l.Mode)
	}
	switch l.State {
	case LeaseOffered, LeaseActive, LeaseTerminal, LeaseReleased:
		return nil
	default:
		return fmt.Errorf("unknown lease state %q", l.State)
	}
}

// ValidateMutation rejects mutations that do not carry a valid active lease.
// Repository-common Git mutation is always exclusive at repository scope.
func (l Lease) ValidateMutation() error {
	if err := l.Validate(); err != nil {
		return err
	}
	if l.State != LeaseActive {
		return fmt.Errorf("mutation requires active lease, got %q", l.State)
	}
	if l.Resource.Kind.common() && l.Mode != ExclusiveLease {
		return errors.New("common Git mutation requires repository-scoped exclusive lease")
	}
	return nil
}

// CleanupReason is a closed cleanup-admission result vocabulary.
type CleanupReason string

const (
	CleanupAdmitted       CleanupReason = "admitted"
	PreserveInvalid       CleanupReason = "invalid"
	PreserveOwnerMismatch CleanupReason = "owner_mismatch"
	PreserveStaleEpoch    CleanupReason = "stale_epoch"
	PreserveNonTerminal   CleanupReason = "non_terminal"
	PreserveNoResult      CleanupReason = "no_persisted_result"
	PreserveResumable     CleanupReason = "resumable"
	PreserveDirty         CleanupReason = "dirty"
	PreserveUntracked     CleanupReason = "untracked"
	PreserveUnpushed      CleanupReason = "unpushed"
	PreserveLiveCWD       CleanupReason = "live_cwd"
	PreserveForeignLock   CleanupReason = "foreign_lock"
)

// CleanupCandidate contains observed state used for compare-and-delete
// admission. Result or Snapshot proves that terminal output survived reap.
type CleanupCandidate struct {
	Lease            Lease
	ObservedOwner    OwnerID
	ObservedEpoch    Epoch
	TerminalEvidence bool
	Result           ResultID
	Snapshot         SnapshotID
	Resumable        bool
	Dirty            bool
	Untracked        bool
	Unpushed         bool
	LiveCWD          bool
	ForeignLock      bool
}

// CleanupDecision is fail-closed: only Admitted authorizes deletion.
type CleanupDecision struct {
	Admitted bool
	Reason   CleanupReason
}

// AdmitCleanup requires exact ownership, a current fencing epoch, terminal
// evidence, a persisted result or snapshot, and no state that must survive.
func AdmitCleanup(c CleanupCandidate) CleanupDecision {
	if err := c.Lease.Validate(); err != nil || c.Lease.Mode != ExclusiveLease {
		return CleanupDecision{Reason: PreserveInvalid}
	}
	if c.ObservedOwner != c.Lease.Owner {
		return CleanupDecision{Reason: PreserveOwnerMismatch}
	}
	if c.ObservedEpoch != c.Lease.Epoch {
		return CleanupDecision{Reason: PreserveStaleEpoch}
	}
	if c.Lease.State != LeaseTerminal || !c.TerminalEvidence {
		return CleanupDecision{Reason: PreserveNonTerminal}
	}
	if strings.TrimSpace(string(c.Result)) == "" && strings.TrimSpace(string(c.Snapshot)) == "" {
		return CleanupDecision{Reason: PreserveNoResult}
	}
	preservationChecks := []struct {
		present bool
		reason  CleanupReason
	}{
		{c.Resumable, PreserveResumable},
		{c.Dirty, PreserveDirty},
		{c.Untracked, PreserveUntracked},
		{c.Unpushed, PreserveUnpushed},
		{c.LiveCWD, PreserveLiveCWD},
		{c.ForeignLock, PreserveForeignLock},
	}
	for _, check := range preservationChecks {
		if check.present {
			return CleanupDecision{Reason: check.reason}
		}
	}
	return CleanupDecision{Admitted: true, Reason: CleanupAdmitted}
}
