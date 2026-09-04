package safesync

import (
	"context"
	"path/filepath"
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
	git(t, tmp, "clone", origin, clone)
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
		// Confirm both files exist in working tree
		if got := readFile(t, filepath.Join(clone, "disjoint_remote.txt")); got != "remote\n" {
			t.Fatalf("disjoint_remote.txt = %q", got)
		}
		if got := readFile(t, filepath.Join(clone, "disjoint_local.txt")); got != "local\n" {
			t.Fatalf("disjoint_local.txt = %q", got)
		}
	})
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
