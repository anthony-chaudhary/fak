package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// fak-test:runtime fast est=1s
func TestWorktreeWorkerLifecyclePreservesPorcelainColumns(t *testing.T) {
	for _, tt := range []struct {
		name       string
		newTracked []string
		dirtyPaths []string
		staged     bool
	}{
		{name: "unstaged tracked", dirtyPaths: []string{"owned.txt"}},
		{name: "sole one-character unstaged", newTracked: []string{"x"}, dirtyPaths: []string{"x"}},
		{name: "clean"},
		{name: "staged tracked", dirtyPaths: []string{"owned.txt"}, staged: true},
		{name: "untracked", dirtyPaths: []string{"untracked.txt"}},
		{name: "two unstaged tracked", newTracked: []string{"second.txt"}, dirtyPaths: []string{"owned.txt", "second.txt"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			repo, worktree, _ := newSingleReapFixture(t)
			write := func(path, body string) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(worktree, path), []byte(body), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if len(tt.newTracked) != 0 {
				for _, path := range tt.newTracked {
					write(path, "base\n")
				}
				worktreeWorkerTestGit(t, worktree, append([]string{"add", "--"}, tt.newTracked...)...)
				worktreeWorkerTestGit(t, worktree, "commit", "-qm", "tracked status fixture")
			}
			head := worktreeWorkerTestGit(t, worktree, "rev-parse", "HEAD")
			base := worktreeWorkerTestGit(t, repo, "merge-base", "HEAD", head)
			for _, path := range tt.dirtyPaths {
				write(path, "changed\n")
			}
			if tt.staged {
				worktreeWorkerTestGit(t, worktree, append([]string{"add", "--"}, tt.dirtyPaths...)...)
			}

			got, err := inspectWorktreeWorkerContext(context.Background(), repo, worktree)
			if err != nil {
				t.Fatal(err)
			}
			if got.HeadSHA != head || got.BaseSHA != base {
				t.Errorf("revision = %q/%q, want unchanged HEAD/base %q/%q", got.HeadSHA, got.BaseSHA, head, base)
			}
			if !slices.Equal(got.DirtyPaths, tt.dirtyPaths) {
				t.Errorf("dirty paths = %q, want %q", got.DirtyPaths, tt.dirtyPaths)
			}
			wantCleanliness := worktreeEvidenceClean
			if len(tt.dirtyPaths) != 0 {
				wantCleanliness = worktreeEvidenceDirty
			}
			if got.Cleanliness != wantCleanliness {
				t.Errorf("cleanliness = %s, want %s", got.Cleanliness, wantCleanliness)
			}
			if tt.name == "sole one-character unstaged" {
				// Project the real status evidence without invoking a reaper.
				state, readiness := lifecycleVerdict(worktreeWorkerLifecycleRow{
					HeadSHA: got.HeadSHA,
					BaseSHA: got.BaseSHA,
					Association: worktreeWorkerAssociation{
						State: worktreeEvidenceAssociated, Lane: "cmd", OwnerPID: 1, LeaseID: "fixture",
					},
					Liveness: worktreeWorkerLiveness{
						Owner: worktreeEvidenceDead, Lease: worktreeEvidenceReleased,
					},
					Cleanliness: worktreeWorkerCleanliness{State: got.Cleanliness, DirtyPaths: got.DirtyPaths},
				})
				if state != worktreeLifecycleDirty || readiness.Reapable || readiness.Verdict != worktreeKeep {
					t.Errorf("dirty one-character lifecycle = %s, readiness = %+v; want DIRTY/KEEP", state, readiness)
				}
			}
		})
	}
}
