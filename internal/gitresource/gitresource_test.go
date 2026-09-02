package gitresource

import (
	"strings"
	"testing"
)

func validWorkspace() WorkspaceHandle {
	return WorkspaceHandle{
		ID:         "workspace:run-7",
		Host:       "host:control-1",
		Sandbox:    "sandbox:worker-7",
		Repository: "repo:git-common-dir-abc",
		Worktree:   "worktree:git-dir-def",
		GitDir:     "gitdir:def",
		CommonDir:  "commondir:abc",
		Root:       `C:\scratch\worker-7`,
	}
}

func worktreeResource(kind ResourceKind, worktree WorktreeID) Resource {
	return Resource{
		ID:         ResourceID(string(kind) + ":" + string(worktree)),
		Kind:       kind,
		Scope:      WorktreeScope,
		Repository: "repo:1",
		Worktree:   worktree,
	}
}

func terminalCandidate() CleanupCandidate {
	return CleanupCandidate{
		Lease: Lease{
			ID: "lease:1",
			Resource: Resource{
				ID:         "resource:worktree-files:1",
				Kind:       WorktreeFiles,
				Scope:      WorktreeScope,
				Repository: "repo:1",
				Worktree:   "worktree:1",
			},
			Owner: "owner:worker-1",
			Mode:  ExclusiveLease,
			State: LeaseTerminal,
			Epoch: 7,
		},
		ObservedOwner:    "owner:worker-1",
		ObservedEpoch:    7,
		TerminalEvidence: true,
		Result:           "result:sha256:abc",
	}
}

func TestCommonGitMutationRequiresRepositoryScopedExclusiveLease(t *testing.T) {
	base := Lease{
		ID:    "lease:refs",
		Owner: "owner:1",
		Mode:  ExclusiveLease,
		State: LeaseActive,
		Epoch: 4,
		Resource: Resource{
			ID:         "resource:refs",
			Kind:       CommonRefs,
			Scope:      RepositoryScope,
			Repository: "repo:1",
		},
	}
	if err := base.ValidateMutation(); err != nil {
		t.Fatalf("valid repository-exclusive mutation rejected: %v", err)
	}

	shared := base
	shared.Mode = SharedLease
	if err := shared.ValidateMutation(); err == nil || !strings.Contains(err.Error(), "exclusive") {
		t.Fatalf("shared common-resource mutation error = %v, want exclusive refusal", err)
	}

	worktreeScoped := base
	worktreeScoped.Resource.Scope = WorktreeScope
	worktreeScoped.Resource.Worktree = "worktree:1"
	if err := worktreeScoped.ValidateMutation(); err == nil || !strings.Contains(err.Error(), "repository scope") {
		t.Fatalf("worktree-scoped common-resource mutation error = %v, want repository-scope refusal", err)
	}
}

func TestPerWorktreeHEADAndIndexRemainDistinct(t *testing.T) {
	for _, kind := range []ResourceKind{WorktreeHEAD, WorktreeIndex} {
		a := worktreeResource(kind, "worktree:a")
		b := worktreeResource(kind, "worktree:b")
		if err := a.Validate(); err != nil {
			t.Fatalf("%s resource A invalid: %v", kind, err)
		}
		if err := b.Validate(); err != nil {
			t.Fatalf("%s resource B invalid: %v", kind, err)
		}
		if a == b {
			t.Fatalf("%s resources collapsed across worktree identities", kind)
		}
	}
}

func TestWorkspaceHandleRejectsProseOrPathOnlyBinding(t *testing.T) {
	h := WorkspaceHandle{Root: `C:\repo\friendly-worker-name`}
	if err := h.Validate(); err == nil {
		t.Fatal("path-only workspace binding admitted")
	}

	h = validWorkspace()
	h.CommonDir = ""
	if err := h.Validate(); err == nil || !strings.Contains(err.Error(), "common_dir") {
		t.Fatalf("missing canonical common-dir error = %v", err)
	}
}

func TestWorkspaceHandleAcceptsCanonicalBinding(t *testing.T) {
	if err := validWorkspace().Validate(); err != nil {
		t.Fatalf("canonical workspace binding rejected: %v", err)
	}
}

func TestCleanupRejectsStaleOwnerAndEpoch(t *testing.T) {
	owner := terminalCandidate()
	owner.ObservedOwner = "owner:replacement"
	if got := AdmitCleanup(owner); got.Admitted || got.Reason != PreserveOwnerMismatch {
		t.Fatalf("owner mismatch decision = %+v", got)
	}

	epoch := terminalCandidate()
	epoch.ObservedEpoch--
	if got := AdmitCleanup(epoch); got.Admitted || got.Reason != PreserveStaleEpoch {
		t.Fatalf("stale epoch decision = %+v", got)
	}
}

func TestCleanupPreservesUnsafeOrResumableWorktrees(t *testing.T) {
	tests := []struct {
		name   string
		reason CleanupReason
		alter  func(*CleanupCandidate)
	}{
		{"nonterminal", PreserveNonTerminal, func(c *CleanupCandidate) { c.Lease.State = LeaseActive }},
		{"no result", PreserveNoResult, func(c *CleanupCandidate) { c.Result = "" }},
		{"resumable", PreserveResumable, func(c *CleanupCandidate) { c.Resumable = true }},
		{"dirty", PreserveDirty, func(c *CleanupCandidate) { c.Dirty = true }},
		{"untracked", PreserveUntracked, func(c *CleanupCandidate) { c.Untracked = true }},
		{"unpushed", PreserveUnpushed, func(c *CleanupCandidate) { c.Unpushed = true }},
		{"live cwd", PreserveLiveCWD, func(c *CleanupCandidate) { c.LiveCWD = true }},
		{"foreign lock", PreserveForeignLock, func(c *CleanupCandidate) { c.ForeignLock = true }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := terminalCandidate()
			tt.alter(&candidate)
			if got := AdmitCleanup(candidate); got.Admitted || got.Reason != tt.reason {
				t.Fatalf("decision = %+v, want preserved as %q", got, tt.reason)
			}
		})
	}
}

func TestCleanupAdmitsExactTerminalOwnerWithPersistedResult(t *testing.T) {
	if got := AdmitCleanup(terminalCandidate()); !got.Admitted || got.Reason != CleanupAdmitted {
		t.Fatalf("decision = %+v, want admitted", got)
	}

	snapshot := terminalCandidate()
	snapshot.Result = ""
	snapshot.Snapshot = "snapshot:sha256:def"
	if got := AdmitCleanup(snapshot); !got.Admitted {
		t.Fatalf("persisted snapshot decision = %+v, want admitted", got)
	}
}
