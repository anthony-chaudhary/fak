package safesync

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func setupTestOriginAndClone(t *testing.T) (string, string) {
	t.Helper()
	tmp := t.TempDir()
	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "work")
	git(t, origin, "config", "core.autocrlf", "false")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	git(t, origin, "config", "receive.denyCurrentBranch", "updateInstead")
	writeFile(t, filepath.Join(origin, "a.txt"), "v1\n")
	writeFile(t, filepath.Join(origin, "shared.txt"), "shared\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "initial")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "-c", "core.autocrlf=false", "clone", origin, clone)
	git(t, clone, "config", "core.autocrlf", "false")
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")
	return origin, clone
}

func TestRouteReconciliationTable(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		setup          func(t *testing.T, origin, clone string) ReconcileOptions
		wantRoute      string
		wantOK         bool
		wantState      string
		wantReason     string
		checkColliding []string
	}{
		{
			name: "clean (in-sync)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute: RouteNoop,
			wantOK:    true,
			wantState: StateInSync,
		},
		{
			name: "ahead",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				writeFile(t, filepath.Join(clone, "local_only.txt"), "local\n")
				git(t, clone, "add", ".")
				git(t, clone, "commit", "-m", "local commit")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute: RoutePush,
			wantOK:    true,
			wantState: StateAhead,
		},
		{
			name: "behind (safe)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				writeFile(t, filepath.Join(origin, "remote_only.txt"), "remote\n")
				git(t, origin, "add", ".")
				git(t, origin, "commit", "-m", "remote commit")
				git(t, clone, "fetch", "origin")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute: RouteApply,
			wantOK:    true,
			wantState: StateBehind,
		},
		{
			name: "behind (dirty collision)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				writeFile(t, filepath.Join(origin, "a.txt"), "v2 from remote\n")
				git(t, origin, "add", ".")
				git(t, origin, "commit", "-m", "remote edit a.txt")
				git(t, clone, "fetch", "origin")
				writeFile(t, filepath.Join(clone, "a.txt"), "uncommitted local change\n")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute:      RouteHoldDirtyCollision,
			wantOK:         false,
			wantState:      StateBehind,
			wantReason:     ReasonDirtyWriteOverlap,
			checkColliding: []string{"a.txt"},
		},
		{
			name: "diverged (disjoint)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				writeFile(t, filepath.Join(origin, "remote_file.txt"), "remote\n")
				git(t, origin, "add", ".")
				git(t, origin, "commit", "-m", "remote file")

				writeFile(t, filepath.Join(clone, "local_file.txt"), "local\n")
				git(t, clone, "add", ".")
				git(t, clone, "commit", "-m", "local file")

				git(t, clone, "fetch", "origin")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute:  RouteDisjointIntegrate,
			wantOK:     true,
			wantState:  StateDiverged,
			wantReason: ReasonDivergedDisjoint,
		},
		{
			name: "diverged (disjoint untracked collision)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				writeFile(t, filepath.Join(origin, "remote_file.txt"), "remote\n")
				git(t, origin, "add", ".")
				git(t, origin, "commit", "-m", "remote file")

				writeFile(t, filepath.Join(clone, "local_file.txt"), "local\n")
				git(t, clone, "add", ".")
				git(t, clone, "commit", "-m", "local file")

				git(t, clone, "fetch", "origin")

				// Untracked local file collides with incoming remote_file.txt
				writeFile(t, filepath.Join(clone, "remote_file.txt"), "untracked collision\n")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute:      RouteHoldDirtyCollision,
			wantOK:         false,
			wantState:      StateDiverged,
			wantReason:     ReasonCollisionRisk,
			checkColliding: []string{"remote_file.txt"},
		},
		{
			name: "diverged (overlap)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				writeFile(t, filepath.Join(origin, "shared.txt"), "conflicting remote content\n")
				git(t, origin, "add", ".")
				git(t, origin, "commit", "-m", "remote edit shared")

				writeFile(t, filepath.Join(clone, "shared.txt"), "conflicting local content\n")
				git(t, clone, "add", ".")
				git(t, clone, "commit", "-m", "local edit shared")

				git(t, clone, "fetch", "origin")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute:      RouteReconcilePacket,
			wantOK:         false,
			wantState:      StateDiverged,
			wantReason:     ReasonDivergedOverlap,
			checkColliding: []string{"shared.txt"},
		},
		{
			name: "merge-active (MERGE_HEAD)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				head := revString(t, clone, "HEAD")
				writeFile(t, filepath.Join(clone, ".git", "MERGE_HEAD"), head+"\n")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute:  RouteHoldMergeActive,
			wantOK:     false,
			wantReason: ReasonMergeActivePeerOwned,
		},
		{
			name: "contention (active lease / drain)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				return ReconcileOptions{
					Repo:       clone,
					Remote:     "origin",
					Branch:     "work",
					Goal:       "publish",
					Contention: true,
				}
			},
			wantRoute:  RouteDrain,
			wantOK:     false,
			wantReason: ReasonQueuedAwaitingQuiescence,
		},
		{
			name: "diverged (trivial superset)",
			setup: func(t *testing.T, origin, clone string) ReconcileOptions {
				// Both origin and clone commit the exact same change independently
				writeFile(t, filepath.Join(origin, "same.txt"), "same\n")
				git(t, origin, "add", ".")
				git(t, origin, "commit", "-m", "same change on origin")

				writeFile(t, filepath.Join(clone, "same.txt"), "same\n")
				git(t, clone, "add", ".")
				git(t, clone, "commit", "-m", "same change on clone")

				git(t, clone, "fetch", "origin")
				return ReconcileOptions{
					Repo:   clone,
					Remote: "origin",
					Branch: "work",
					Goal:   "publish",
				}
			},
			wantRoute: RouteSupersetMerge,
			wantOK:    true,
			wantState: StateDiverged,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			origin, clone := setupTestOriginAndClone(t)
			opts := tc.setup(t, origin, clone)

			res, err := RouteReconciliation(context.Background(), opts)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if res.Schema != ReconcileSchema {
				t.Errorf("schema = %q, want %q", res.Schema, ReconcileSchema)
			}
			if res.Route != tc.wantRoute {
				t.Errorf("route = %q, want %q", res.Route, tc.wantRoute)
			}
			if res.OK != tc.wantOK {
				t.Errorf("ok = %v, want %v", res.OK, tc.wantOK)
			}
			if tc.wantState != "" && res.State != tc.wantState {
				t.Errorf("state = %q, want %q", res.State, tc.wantState)
			}
			if tc.wantReason != "" && res.Reason != tc.wantReason {
				t.Errorf("reason = %q, want %q", res.Reason, tc.wantReason)
			}
			for _, p := range tc.checkColliding {
				found := false
				for _, c := range res.CollidingPaths {
					if c == p {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected colliding path %q not found in %v", p, res.CollidingPaths)
				}
			}
		})
	}
}

func TestRouteReconciliationApplyExecution(t *testing.T) {
	t.Run("ahead with apply executes push", func(t *testing.T) {
		origin, clone := setupTestOriginAndClone(t)
		writeFile(t, filepath.Join(clone, "push_me.txt"), "pushed\n")
		git(t, clone, "add", ".")
		git(t, clone, "commit", "-m", "local commit for push")

		opts := ReconcileOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
			Goal:   "publish",
			Apply:  true,
		}
		res, err := RouteReconciliation(context.Background(), opts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Route != RoutePush {
			t.Fatalf("route = %q, want %q", res.Route, RoutePush)
		}
		if !res.Applied {
			t.Fatalf("res.Applied = false, want true")
		}
		if res.Execution == nil || !res.Execution.Success {
			t.Fatalf("execution failed: %+v", res.Execution)
		}
		// Confirm origin received the commit
		cloneHead := revString(t, clone, "HEAD")
		originHead := revString(t, origin, "work")
		if cloneHead != originHead {
			t.Fatalf("origin HEAD %s != clone HEAD %s after push", originHead, cloneHead)
		}
	})

	t.Run("behind safe with apply executes apply", func(t *testing.T) {
		origin, clone := setupTestOriginAndClone(t)
		writeFile(t, filepath.Join(origin, "apply_me.txt"), "applied\n")
		git(t, origin, "add", ".")
		git(t, origin, "commit", "-m", "remote commit for apply")
		git(t, clone, "fetch", "origin")

		opts := ReconcileOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
			Goal:   "publish",
			Apply:  true,
		}
		res, err := RouteReconciliation(context.Background(), opts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Route != RouteApply {
			t.Fatalf("route = %q, want %q", res.Route, RouteApply)
		}
		if !res.Applied {
			t.Fatalf("res.Applied = false, want true")
		}
		if res.Execution == nil || !res.Execution.Success {
			t.Fatalf("execution failed: %+v", res.Execution)
		}
		// Confirm clone fast-forwarded to origin
		cloneHead := revString(t, clone, "HEAD")
		originHead := revString(t, origin, "work")
		if cloneHead != originHead {
			t.Fatalf("clone HEAD %s != origin HEAD %s after apply", cloneHead, originHead)
		}
	})

	t.Run("diverged disjoint with apply executes integration", func(t *testing.T) {
		origin, clone := setupTestOriginAndClone(t)
		writeFile(t, filepath.Join(origin, "disjoint_remote.txt"), "remote\n")
		git(t, origin, "add", ".")
		git(t, origin, "commit", "-m", "remote disjoint")

		writeFile(t, filepath.Join(clone, "disjoint_local.txt"), "local\n")
		git(t, clone, "add", ".")
		git(t, clone, "commit", "-m", "local disjoint")
		git(t, clone, "fetch", "origin")

		opts := ReconcileOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
			Goal:   "publish",
			Apply:  true,
		}
		res, err := RouteReconciliation(context.Background(), opts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Route != RouteDisjointIntegrate {
			t.Fatalf("route = %q, want %q", res.Route, RouteDisjointIntegrate)
		}
		if !res.Applied {
			t.Fatalf("res.Applied = false, want true")
		}
		if res.Execution == nil || !res.Execution.Success {
			t.Fatalf("execution failed: %+v", res.Execution)
		}
		// Confirm remote file exists in synthetic merge tree and both files exist in working tree
		if got := strings.TrimSpace(gitOutput(t, clone, "show", "HEAD:disjoint_remote.txt")); got != "remote" {
			t.Fatalf("disjoint_remote.txt in HEAD = %q, want %q", got, "remote")
		}
		if got := readFile(t, filepath.Join(clone, "disjoint_remote.txt")); got != "remote\n" {
			t.Fatalf("disjoint_remote.txt in clone working tree = %q, want %q", got, "remote\n")
		}
		if got := readFile(t, filepath.Join(clone, "disjoint_local.txt")); got != "local\n" {
			t.Fatalf("disjoint_local.txt = %q", got)
		}

		// Confirm git status in clone has no staged deletions or uncommitted index drift
		status := strings.TrimSpace(gitOutput(t, clone, "status", "--porcelain"))
		if status != "" {
			t.Fatalf("git status after disjoint integrate = %q, want clean status (no false deletions)", status)
		}
	})
}

// fak-test:runtime integration est=3s lane=default
func TestRouteReconciliationQueuedIntegrate(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, goal, blocker, wantRoute string
		apply                          bool
	}{
		{name: "preview", goal: "integrate", wantRoute: RouteDisjointIntegrate},
		{name: "apply", goal: "integrate origin/work", apply: true, wantRoute: RouteDisjointIntegrate},
		{name: "default still drains", apply: true, wantRoute: RouteDrain},
		{name: "publish still drains", goal: "publish", apply: true, wantRoute: RouteDrain},
		{name: "writer lease", goal: "integrate", blocker: "writer", apply: true, wantRoute: RouteDrain},
		{name: "index lock", goal: "integrate", blocker: "index", apply: true, wantRoute: RouteDrain},
		{name: "active merge", goal: "integrate", blocker: "merge", apply: true, wantRoute: RouteHoldMergeActive},
		{name: "dirty collision", goal: "integrate", blocker: "dirty", apply: true, wantRoute: RouteHoldDirtyCollision},
		{name: "overlapping divergence", goal: "integrate", blocker: "overlap", apply: true, wantRoute: RouteReconcilePacket},
	} {
		t.Run(tc.name, func(t *testing.T) {
			origin, clone := setupTestOriginAndClone(t)
			localPath, remotePath := "disjoint_local.txt", "disjoint_remote.txt"
			if tc.blocker == "overlap" {
				localPath, remotePath = "a.txt", "a.txt"
			}
			writeFile(t, filepath.Join(clone, localPath), "local\n")
			git(t, clone, "add", localPath)
			git(t, clone, "commit", "-m", "queued local commit")
			localHead := revString(t, clone, "HEAD")
			writeFile(t, filepath.Join(origin, remotePath), "remote\n")
			git(t, origin, "add", remotePath)
			git(t, origin, "commit", "-m", "remote commit")
			git(t, clone, "fetch", "origin")
			targetHead := revString(t, clone, "origin/work")

			// Match the drain queue's persisted shape, including backpressure metadata.
			// The queued SHA must remain reachable and the queue must remain untouched.
			writeFile(t, filepath.Join(clone, ".git", "info", "exclude"), ".fak/\n")
			queuePath := filepath.Join(clone, ".fak", "sync-drain-queue.json")
			mkdir(t, filepath.Dir(queuePath))
			queue := `{"schema":"fak.sync_drain.v1","entries":[{"sha":"` + localHead + `","subject":"queued local commit","refusal_reason":"diverged","queued_at_unix":1}],"attempts":3,"next_retry_unix":99}` + "\n"
			writeFile(t, queuePath, queue)
			t.Cleanup(func() {
				if got := readFile(t, queuePath); got != queue {
					t.Errorf("drain queue changed: got %q, want %q", got, queue)
				}
				if got := revString(t, origin, "work"); got != targetHead {
					t.Errorf("integration published unexpectedly: origin = %s, want %s", got, targetHead)
				}
			})

			switch tc.blocker {
			case "writer":
				lease, err := AcquireWriterLease(clone, "peer-writer", time.Now, time.Minute)
				if err != nil {
					t.Fatal(err)
				}
				defer lease.Release()
			case "index":
				writeFile(t, filepath.Join(clone, ".git", "index.lock"), "peer lock\n")
			case "merge":
				writeFile(t, filepath.Join(clone, ".git", "MERGE_HEAD"), targetHead+"\n")
			case "dirty":
				writeFile(t, filepath.Join(clone, remotePath), "untracked peer work\n")
			}

			var refUpdates [][]string
			res, err := RouteReconciliation(context.Background(), ReconcileOptions{
				Repo: clone, Remote: "origin", Branch: "work", Goal: tc.goal, Apply: tc.apply,
				Runner: func(ctx context.Context, repo string, args ...string) RunResult {
					if len(args) > 0 && args[0] == "update-ref" {
						refUpdates = append(refUpdates, append([]string(nil), args...))
					}
					return RealRunner(ctx, repo, args...)
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			if res.Route != tc.wantRoute {
				t.Fatalf("queued %q route = %q, want %q", tc.goal, res.Route, tc.wantRoute)
			}
			wantApplied := tc.apply && tc.wantRoute == RouteDisjointIntegrate
			if res.Applied != wantApplied || res.OK != (tc.wantRoute == RouteDisjointIntegrate) {
				t.Fatalf("unexpected queued reconciliation result: %+v", res)
			}
			if !wantApplied {
				if got := revString(t, clone, "HEAD"); got != localHead || len(refUpdates) != 0 {
					t.Fatalf("preview or hold moved local HEAD: got %s, want %s; updates=%v", got, localHead, refUpdates)
				}
				if tc.blocker == "dirty" && readFile(t, filepath.Join(clone, remotePath)) != "untracked peer work\n" {
					t.Fatal("dirty collision changed peer work")
				}
				return
			}

			newHead := revString(t, clone, "HEAD")
			if res.Execution == nil || !res.Execution.Success || res.Execution.NewHead != newHead {
				t.Fatalf("integration execution = %+v, want successful HEAD %s", res.Execution, newHead)
			}
			wantCAS := "update-ref refs/heads/work " + newHead + " " + localHead
			if len(refUpdates) != 1 || strings.Join(refUpdates[0], " ") != wantCAS {
				t.Fatalf("ref updates = %v, want expected-parent CAS %q", refUpdates, wantCAS)
			}
			if got := strings.TrimSpace(gitOutput(t, clone, "show", "-s", "--format=%P", "HEAD")); got != localHead+" "+targetHead {
				t.Fatalf("merge parents = %q, want queued local commit and pinned target", got)
			}
			for path, want := range map[string]string{localPath: "local\n", remotePath: "remote\n"} {
				if head, index, worktree := gitOutput(t, clone, "show", "HEAD:"+path), gitOutput(t, clone, "show", ":"+path), readFile(t, filepath.Join(clone, path)); head != want || index != want || worktree != want {
					t.Errorf("%s: HEAD=%q index=%q worktree=%q, want %q", path, head, index, worktree, want)
				}
			}
			if got := strings.TrimSpace(gitOutput(t, clone, "status", "--porcelain")); got != "" {
				t.Fatalf("integration left index/worktree drift: %q", got)
			}
		})
	}
}

func TestParseGoal(t *testing.T) {
	t.Parallel()

	cases := []struct {
		raw           string
		defaultRemote string
		defaultBranch string
		wantKind      string
		wantSource    string
		wantTarget    string
		wantErr       bool
	}{
		{
			raw:           "",
			defaultRemote: "origin",
			defaultBranch: "main",
			wantKind:      "publish",
			wantSource:    "HEAD",
			wantTarget:    "origin/main",
		},
		{
			raw:           "",
			defaultRemote: "origin",
			defaultBranch: "",
			wantKind:      "publish",
			wantSource:    "HEAD",
			wantTarget:    "origin/main",
		},
		{
			raw:           "integrate",
			defaultRemote: "origin",
			defaultBranch: "",
			wantKind:      "integrate",
			wantSource:    "HEAD",
			wantTarget:    "origin/main",
		},
		{
			raw:           "publish",
			defaultRemote: "origin",
			defaultBranch: "work",
			wantKind:      "publish",
			wantSource:    "HEAD",
			wantTarget:    "origin/work",
		},
		{
			raw:           "publish 1234abcd",
			defaultRemote: "origin",
			defaultBranch: "work",
			wantKind:      "publish",
			wantSource:    "1234abcd",
			wantTarget:    "origin/work",
		},
		{
			raw:           "integrate",
			defaultRemote: "origin",
			defaultBranch: "work",
			wantKind:      "integrate",
			wantSource:    "HEAD",
			wantTarget:    "origin/work",
		},
		{
			raw:           "integrate origin/main",
			defaultRemote: "origin",
			defaultBranch: "work",
			wantKind:      "integrate",
			wantSource:    "HEAD",
			wantTarget:    "origin/main",
		},
		{
			raw:     "invalid_goal",
			wantErr: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.raw, func(t *testing.T) {
			got, err := ParseGoal(tc.raw, tc.defaultRemote, tc.defaultBranch)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ParseGoal(%q) succeeded, want error", tc.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseGoal(%q) error = %v", tc.raw, err)
			}
			if got.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", got.Kind, tc.wantKind)
			}
			if got.Source != tc.wantSource {
				t.Errorf("Source = %q, want %q", got.Source, tc.wantSource)
			}
			if got.Target != tc.wantTarget {
				t.Errorf("Target = %q, want %q", got.Target, tc.wantTarget)
			}
		})
	}
}

func TestParseGoal_BranchRole(t *testing.T) {
	tmp := t.TempDir()
	dosFile := filepath.Join(tmp, "dos.toml")
	if err := os.WriteFile(dosFile, []byte("[branch_roles]\ndevelopment_branch = \"dev\"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(tmp)

	cases := []struct {
		name          string
		raw           string
		defaultRemote string
		defaultBranch string
		wantKind      string
		wantSource    string
		wantTarget    string
	}{
		{
			name:          "empty goal derives dev",
			raw:           "",
			defaultRemote: "origin",
			defaultBranch: "",
			wantKind:      "publish",
			wantSource:    "HEAD",
			wantTarget:    "origin/dev",
		},
		{
			name:          "publish goal derives dev",
			raw:           "publish",
			defaultRemote: "origin",
			defaultBranch: "",
			wantKind:      "publish",
			wantSource:    "HEAD",
			wantTarget:    "origin/dev",
		},
		{
			name:          "integrate goal derives dev",
			raw:           "integrate",
			defaultRemote: "origin",
			defaultBranch: "",
			wantKind:      "integrate",
			wantSource:    "HEAD",
			wantTarget:    "origin/dev",
		},
		{
			name:          "explicit branch takes precedence",
			raw:           "",
			defaultRemote: "origin",
			defaultBranch: "feature",
			wantKind:      "publish",
			wantSource:    "HEAD",
			wantTarget:    "origin/feature",
		},
		{
			name:          "explicit integrate ref takes precedence",
			raw:           "integrate upstream/custom",
			defaultRemote: "origin",
			defaultBranch: "",
			wantKind:      "integrate",
			wantSource:    "HEAD",
			wantTarget:    "upstream/custom",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseGoal(tc.raw, tc.defaultRemote, tc.defaultBranch)
			if err != nil {
				t.Fatalf("ParseGoal(%q) error = %v", tc.raw, err)
			}
			if got.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", got.Kind, tc.wantKind)
			}
			if got.Source != tc.wantSource {
				t.Errorf("Source = %q, want %q", got.Source, tc.wantSource)
			}
			if got.Target != tc.wantTarget {
				t.Errorf("Target = %q, want %q", got.Target, tc.wantTarget)
			}
		})
	}
}

func TestRouteReconciliationDetached_BranchRole(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()

	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "dev")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "init.txt"), "init\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "init")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "-c", "core.autocrlf=false", "clone", "-b", "dev", origin, clone)
	git(t, clone, "config", "core.autocrlf", "false")
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")

	writeFile(t, filepath.Join(clone, "dos.toml"), "[branch_roles]\ndevelopment_branch = \"dev\"\n")
	git(t, clone, "checkout", "--detach", "HEAD")

	opts := ReconcileOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "",
		Goal:   "publish",
	}

	assessment, err := RouteReconciliation(ctx, opts)
	if err != nil {
		t.Fatalf("RouteReconciliation error: %v", err)
	}
	if assessment.Branch != "dev" {
		t.Fatalf("assessment.Branch = %q, want dev", assessment.Branch)
	}
	if assessment.TargetRef != "origin/dev" {
		t.Fatalf("assessment.TargetRef = %q, want origin/dev", assessment.TargetRef)
	}
	if assessment.Route != RouteNoop {
		t.Fatalf("assessment.Route = %q, want %s", assessment.Route, RouteNoop)
	}
}

func TestActiveWriterLease(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)
	_ = origin

	now := time.Now()
	nowFunc := func() time.Time { return now }

	// No lease initially
	if _, held := ActiveWriterLease(clone, nowFunc, time.Minute); held {
		t.Fatalf("expected no active lease")
	}

	// Acquire lease
	l, err := AcquireWriterLease(clone, "test-holder", nowFunc, time.Minute)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}

	// Should be active
	info, held := ActiveWriterLease(clone, nowFunc, time.Minute)
	if !held || info == nil || info.Owner != "test-holder" {
		t.Fatalf("expected active lease with owner test-holder, got held=%v info=%+v", held, info)
	}

	// Release lease
	_ = l.Release()

	// Should be no lease
	if _, held := ActiveWriterLease(clone, nowFunc, time.Minute); held {
		t.Fatalf("expected no active lease after release")
	}
}

func TestRouteReconciliationSuspendPaths(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	// Set base in origin and clone to have top/middle/bottom
	writeFile(t, filepath.Join(origin, "a.txt"), "top\nmiddle\nbottom\n")
	git(t, origin, "add", "a.txt")
	git(t, origin, "commit", "-m", "base a")
	git(t, clone, "pull", "origin", "work")

	writeFile(t, filepath.Join(origin, "a.txt"), "top\nincoming\nmiddle\nbottom\n")
	git(t, origin, "add", "a.txt")
	git(t, origin, "commit", "-m", "remote update a")
	git(t, clone, "fetch", "origin")

	writeFile(t, filepath.Join(clone, "a.txt"), "top\nincoming\nmiddle\nbottom\nunique\n")

	// 1. Without suspend paths: should hold collision
	optsNoSuspend := ReconcileOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
		Goal:   "integrate",
		Apply:  false,
	}
	resNoSuspend, err := RouteReconciliation(context.Background(), optsNoSuspend)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resNoSuspend.Route != RouteHoldDirtyCollision {
		t.Fatalf("route = %q, want %q", resNoSuspend.Route, RouteHoldDirtyCollision)
	}

	// 2. With suspend paths preview (Apply: false)
	optsPreview := ReconcileOptions{
		Repo:         clone,
		Remote:       "origin",
		Branch:       "work",
		Goal:         "integrate",
		Apply:        false,
		SuspendPaths: []string{"a.txt"},
		Session:      "recon-test-sess",
	}
	resPreview, err := RouteReconciliation(context.Background(), optsPreview)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resPreview.Route != RouteDisjointIntegrate || !resPreview.OK {
		t.Fatalf("preview = %+v, want RouteDisjointIntegrate and OK", resPreview)
	}
	if resPreview.Park == nil || resPreview.Park.Status != ParkStatusDryRun {
		t.Fatalf("park receipt = %+v, want DRY_RUN", resPreview.Park)
	}

	// 3. With suspend paths apply (Apply: true)
	optsApply := ReconcileOptions{
		Repo:         clone,
		Remote:       "origin",
		Branch:       "work",
		Goal:         "integrate",
		Apply:        true,
		SuspendPaths: []string{"a.txt"},
		Session:      "recon-test-sess",
	}
	resApply, err := RouteReconciliation(context.Background(), optsApply)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resApply.Route != RouteDisjointIntegrate || !resApply.OK || !resApply.Applied {
		t.Fatalf("apply = %+v, want RouteDisjointIntegrate, OK, Applied", resApply)
	}
	if resApply.Park == nil || resApply.Park.Status != ParkStatusRestored {
		t.Fatalf("park receipt = %+v, want RESTORED", resApply.Park)
	}

	wantA := "top\nincoming\nmiddle\nbottom\nunique\n"
	if gotA := readFile(t, filepath.Join(clone, "a.txt")); gotA != wantA {
		t.Fatalf("a.txt = %q, want %q", gotA, wantA)
	}
}

func TestRouteReconciliationDisjointUntrackedCollision(t *testing.T) {
	t.Run("disjoint with colliding untracked file fails closed with COLLISION_RISK", func(t *testing.T) {
		origin, clone := setupTestOriginAndClone(t)
		writeFile(t, filepath.Join(origin, "disjoint_remote.txt"), "remote content\n")
		git(t, origin, "add", ".")
		git(t, origin, "commit", "-m", "remote disjoint")

		writeFile(t, filepath.Join(clone, "disjoint_local.txt"), "local content\n")
		git(t, clone, "add", ".")
		git(t, clone, "commit", "-m", "local disjoint")
		git(t, clone, "fetch", "origin")

		// Create untracked local file colliding with incoming remote file
		untrackedContent := "untracked local work that must not be overwritten\n"
		untrackedPath := filepath.Join(clone, "disjoint_remote.txt")
		writeFile(t, untrackedPath, untrackedContent)

		// 1. Preview assessment (Apply: false) must fail closed with COLLISION_RISK
		opts := ReconcileOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
			Goal:   "integrate",
			Apply:  false,
		}
		res, err := RouteReconciliation(context.Background(), opts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.OK {
			t.Fatalf("expected res.OK to be false due to untracked collision, got true")
		}
		if res.Reason != ReasonCollisionRisk {
			t.Fatalf("expected res.Reason = %q, got %q", ReasonCollisionRisk, res.Reason)
		}
		if res.Route != RouteHoldDirtyCollision {
			t.Fatalf("expected res.Route = %q, got %q", RouteHoldDirtyCollision, res.Route)
		}
		foundColliding := false
		for _, c := range res.CollidingPaths {
			if c == "disjoint_remote.txt" {
				foundColliding = true
				break
			}
		}
		if !foundColliding {
			t.Fatalf("expected colliding path disjoint_remote.txt, got %v", res.CollidingPaths)
		}

		// 2. Apply execution (Apply: true) without quarantine must fail closed and NOT clobber file
		optsApply := ReconcileOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
			Goal:   "integrate",
			Apply:  true,
		}
		resApply, err := RouteReconciliation(context.Background(), optsApply)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resApply.OK || resApply.Applied {
			t.Fatalf("expected Apply to fail closed, got OK=%v Applied=%v", resApply.OK, resApply.Applied)
		}
		if resApply.Reason != ReasonCollisionRisk {
			t.Fatalf("expected resApply.Reason = %q, got %q", ReasonCollisionRisk, resApply.Reason)
		}

		// Untracked local file MUST remain intact
		if got := readFile(t, untrackedPath); got != untrackedContent {
			t.Fatalf("local untracked file was modified/overwritten! got %q, want %q", got, untrackedContent)
		}
	})

	t.Run("disjoint with colliding untracked file and auto-quarantine succeeds", func(t *testing.T) {
		origin, clone := setupTestOriginAndClone(t)
		remoteContent := "remote content\n"
		writeFile(t, filepath.Join(origin, "disjoint_remote.txt"), remoteContent)
		git(t, origin, "add", ".")
		git(t, origin, "commit", "-m", "remote disjoint")

		writeFile(t, filepath.Join(clone, "disjoint_local.txt"), "local content\n")
		git(t, clone, "add", ".")
		git(t, clone, "commit", "-m", "local disjoint")
		git(t, clone, "fetch", "origin")

		// Create untracked local file colliding with incoming remote file
		untrackedContent := "untracked local work to quarantine\n"
		untrackedPath := filepath.Join(clone, "disjoint_remote.txt")
		writeFile(t, untrackedPath, untrackedContent)
		expectedHash, _, err := FileSHA256(untrackedPath)
		if err != nil {
			t.Fatalf("FileSHA256: %v", err)
		}

		opts := ReconcileOptions{
			Repo:           clone,
			Remote:         "origin",
			Branch:         "work",
			Goal:           "integrate",
			Apply:          true,
			AutoQuarantine: true,
			Session:        "disjoint-quarantine-test",
		}
		res, err := RouteReconciliation(context.Background(), opts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK || !res.Applied {
			t.Fatalf("expected apply with AutoQuarantine to succeed, got OK=%v Applied=%v Reason=%s", res.OK, res.Applied, res.Reason)
		}
		if res.Route != RouteDisjointIntegrate {
			t.Fatalf("expected res.Route = %q, got %q", RouteDisjointIntegrate, res.Route)
		}
		if res.Quarantine == nil {
			t.Fatalf("expected non-nil Quarantine receipt")
		}
		if res.Quarantine.Preserved["disjoint_remote.txt"] != expectedHash && res.Quarantine.SHA256 != expectedHash {
			t.Fatalf("quarantine receipt did not preserve hash %s: %+v", expectedHash, res.Quarantine)
		}

		// Working tree file must now have remote content
		if got := readFile(t, untrackedPath); got != remoteContent {
			t.Fatalf("working tree disjoint_remote.txt = %q, want %q", got, remoteContent)
		}
	})
}

// fak-test:runtime fast est=2s
func TestRouteReconciliationApplyExactTarget(t *testing.T) {
	t.Parallel()
	origin, clone := setupTestOriginAndClone(t)
	base := revString(t, clone, "HEAD")

	writeFile(t, filepath.Join(origin, "intermediate.txt"), "intermediate\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "intermediate target")
	git(t, clone, "fetch", "origin")
	tracking := revString(t, clone, "origin/work")

	writeFile(t, filepath.Join(origin, "exact_target.txt"), "exact target\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "exact integration target")
	target := revString(t, origin, "HEAD")
	// Obtain the requested object without advancing the deliberately stale tracking ref.
	git(t, clone, "fetch", "--no-tags", "--refmap=", "origin", target)
	if revString(t, clone, "HEAD") != base || revString(t, clone, "origin/work") != tracking || revString(t, clone, "FETCH_HEAD") != target {
		t.Fatal("fixture must retain HEAD at A and tracking at B while fetching C")
	}

	res, err := RouteReconciliation(context.Background(), ReconcileOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
		Goal:   "integrate " + target,
		Apply:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Route != RouteApply || res.Target != target || !res.Applied || res.Execution == nil || !res.Execution.Success {
		t.Fatalf("exact-target apply failed: %+v", res)
	}
	if res.Execution.NewHead != target || revString(t, clone, "HEAD") != target {
		t.Fatalf("applied HEAD = %q, receipt = %q, want exact target %q", revString(t, clone, "HEAD"), res.Execution.NewHead, target)
	}
	got, err := os.ReadFile(filepath.Join(clone, "exact_target.txt"))
	if err != nil || string(got) != "exact target\n" {
		t.Fatalf("exact target file = %q, error = %v", got, err)
	}
	if got := revString(t, clone, "origin/work"); got != tracking {
		t.Fatalf("tracking ref moved to %q, want unchanged %q", got, tracking)
	}
}
