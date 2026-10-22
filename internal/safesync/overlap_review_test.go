package safesync

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// Git invokes this test executable through an owned fixture hook symlink. The
// strict opt-in and path checks keep ordinary test startup inert; no shell or
// newly compiled helper is needed.
func init() {
	if os.Getenv("FAK_REVIEWED_OVERLAP_HOOK_HELPER") != "1" || filepath.Base(os.Args[0]) != "pre-merge-commit" {
		return
	}
	if err := runReviewedOverlapHook(); err != nil {
		_, _ = os.Stderr.WriteString("reviewed overlap fixture hook: " + err.Error() + "\n")
		os.Exit(1)
	}
	os.Exit(0)
}

func runReviewedOverlapHook() error {
	repo := os.Getenv("FAK_REVIEWED_OVERLAP_HOOK_REPO")
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	cwd, err = filepath.EvalSymlinks(cwd)
	if err != nil {
		return err
	}
	hook, err := filepath.Abs(os.Args[0])
	if err != nil {
		return err
	}
	if !filepath.IsAbs(repo) || cwd != repo || hook != filepath.Join(repo, ".git", "hooks", "pre-merge-commit") {
		return errors.New("hook is outside its explicitly owned fixture")
	}
	info, err := os.Lstat(filepath.Join(repo, ".git"))
	if err != nil || !info.IsDir() {
		return errors.New("hook fixture requires a normal .git directory")
	}
	mode := os.Getenv("FAK_REVIEWED_OVERLAP_HOOK_MODE")
	if mode != "marker" && mode != "mutate-index" {
		return errors.New("unknown hook fixture mode")
	}
	if err := os.WriteFile(filepath.Join(repo, ".git", "review-hook-ran"), []byte("reviewed"), 0600); err != nil {
		return err
	}
	if mode == "mutate-index" {
		if err := os.WriteFile(filepath.Join(repo, "hook.txt"), []byte("changed"), 0600); err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, "git", "add", "--", "hook.txt")
		cmd.Dir = repo
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		return cmd.Run()
	}
	return nil
}

func installReviewedOverlapHook(t *testing.T, repo, mode string) {
	t.Helper()
	repo, err := filepath.EvalSymlinks(repo)
	if err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(executable, filepath.Join(repo, ".git", "hooks", "pre-merge-commit")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_REVIEWED_OVERLAP_HOOK_HELPER", "1")
	t.Setenv("FAK_REVIEWED_OVERLAP_HOOK_REPO", repo)
	t.Setenv("FAK_REVIEWED_OVERLAP_HOOK_MODE", mode)
}

func reviewedOverlapFixture(t *testing.T) (string, string, *ReconciliationPacket, ReviewedOverlapRequest) {
	t.Helper()
	origin, repo := setupTestOriginAndClone(t)
	content := "first\n" + strings.Repeat("unchanged\n", 20) + "last\n"
	writeFile(t, filepath.Join(origin, "shared.txt"), content)
	git(t, origin, "add", "shared.txt")
	git(t, origin, "commit", "-m", "shared baseline")
	git(t, repo, "pull", "--ff-only")
	writeFile(t, filepath.Join(origin, "shared.txt"), strings.TrimSuffix(content, "last\n")+"remote\n")
	git(t, origin, "add", "shared.txt")
	git(t, origin, "commit", "-m", "remote shared hunk")
	writeFile(t, filepath.Join(repo, "shared.txt"), strings.Replace(content, "first", "local", 1))
	writeFile(t, filepath.Join(repo, "go.mod"), "module example.com/reviewedcheckpoint\n\ngo 1.22\n")
	writeFile(t, filepath.Join(repo, "checkpoint.go"), "package checkpoint\nfunc Value() int { return 7 }\n")
	writeFile(t, filepath.Join(repo, "checkpoint_test.go"), "package checkpoint\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value()!=7 { t.Fatal(Value()) } }\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "local shared hunk")
	git(t, repo, "fetch", "origin")
	packet, err := BuildReconciliationPacket(context.Background(), PacketOptions{Repo: repo, Branch: "work"})
	if err != nil || packet == nil || packet.Disposition != DispositionOwnerHandoffRequired || packet.Dispatchable {
		t.Fatalf("packet=%+v err=%v", packet, err)
	}
	review, err := PrepareCleanOverlapReview(context.Background(), CleanOverlapReviewOptions{Repo: repo, TargetSHA: packet.TargetSHA, Reviewer: "fixture reviewer", Evidence: "fixture delegated review"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(review.LocalDiff, "+remote") || !strings.Contains(review.TargetDiff, "+local") {
		t.Fatal("missing parent-to-result review evidence")
	}
	return origin, repo, packet, ReviewedOverlapRequest{Review: *review, Approval: ReviewedOverlapApproval{Digest: review.Digest}}
}

func TestReviewedOverlapCheckpoint(t *testing.T) {
	t.Setenv("GOWORK", "off")
	t.Setenv("GOTOOLCHAIN", "local")
	origin, repo, packet, request := reviewedOverlapFixture(t)
	// An unrelated explicit claim with an arbitrary ID and no dos.toml must
	// not become universal ownership through an empty optional taxonomy entry.
	if _, err := leaseref.NewInDir(repo).Acquire(context.Background(), leaseref.Record{ID: "unrelated-arbitrary", Holder: "peer", TreeGlobs: []string{"unrelated.txt"}, AcquiredAt: time.Now().Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatal(err)
	}
	installReviewedOverlapHook(t, repo, "marker")
	builds, tests := 0, 0
	runner := func(ctx context.Context, dir string, args ...string) RunResult {
		if len(args) > 0 && (args[0] == "fetch" || args[0] == "push") {
			t.Fatalf("executor transport: %v", args)
		}
		return RealRunner(ctx, dir, args...)
	}
	receipt, err := ExecutePacket(context.Background(), packet, ExecuteOptions{Repo: repo, Branch: "work", LocalOnly: true, ReviewedOverlap: &request, Runner: runner,
		BuildVerifier: func(ctx context.Context, dir string) error { builds++; return defaultBuildVerifier(ctx, dir) },
		TestVerifier:  func(ctx context.Context, dir string) error { tests++; return defaultTestVerifier(ctx, dir) },
	})
	if err != nil || receipt == nil || receipt.Status != ExecuteStatusExecuted || receipt.Pushed || receipt.ReviewDigest != request.Review.Digest || receipt.NewTree != request.Review.MergedTree || receipt.ExecutionMode != "reviewed-clean-overlap" {
		t.Fatalf("receipt=%+v err=%v", receipt, err)
	}
	if builds != 1 || tests != 1 {
		t.Fatalf("builds=%d tests=%d", builds, tests)
	}
	if _, err := os.Stat(filepath.Join(repo, ".git", "review-hook-ran")); err != nil {
		t.Fatal("normal merge hook did not run:", err)
	}
	if packet.Dispatchable || packet.Disposition != DispositionOwnerHandoffRequired {
		t.Fatal("packet was rewritten")
	}
	remote, _ := rev(context.Background(), RealRunner, origin, "HEAD")
	if remote != packet.TargetSHA {
		t.Fatal("remote changed")
	}
	parents, _ := reviewOutput(context.Background(), RealRunner, repo, "rev-list", "--parents", "-n", "1", receipt.NewHEAD)
	if strings.TrimSpace(parents) != receipt.NewHEAD+" "+packet.LocalHead+" "+packet.TargetSHA {
		t.Fatal("wrong ordered parents")
	}
	tree, _ := rev(context.Background(), RealRunner, repo, "HEAD^{tree}")
	if tree != request.Review.MergedTree {
		t.Fatal("wrong final tree")
	}
}

func TestReviewedOverlapRefusesUnboundOrUnsafeInputs(t *testing.T) {
	for _, kind := range []string{"missing", "unaccepted", "digest", "schema", "repository", "worktree", "head", "target", "tree", "paths", "diff", "nonlocal", "suspend", "dirty", "staged", "merge-active", "path-lease", "lease-inspection", "conflict", "false-disjoint"} {
		t.Run(kind, func(t *testing.T) {
			_, repo, packet, request := reviewedOverlapFixture(t)
			opts := ExecuteOptions{Repo: repo, Branch: "work", LocalOnly: true, ReviewedOverlap: &request,
				BuildVerifier: func(context.Context, string) error { t.Fatal("refusal reached build"); return nil },
				TestVerifier:  func(context.Context, string) error { t.Fatal("refusal reached test"); return nil },
			}
			switch kind {
			case "missing":
				opts.ReviewedOverlap = nil
			case "unaccepted":
				request.Approval.Digest = ""
			case "digest":
				request.Review.Digest = strings.Repeat("0", 64)
			case "schema":
				request.Review.Schema = "unknown"
			case "repository":
				request.Review.Repository = repo
			case "worktree":
				request.Review.Worktree = filepath.Dir(repo)
			case "head":
				request.Review.LocalHead = packet.TargetSHA
			case "target":
				request.Review.TargetSHA = packet.LocalHead
			case "tree":
				request.Review.MergedTree = packet.LocalHead
			case "paths":
				request.Review.OverlapPaths = []string{"other.txt"}
			case "diff":
				request.Review.LocalDiff = "invented review display"
			case "nonlocal":
				opts.LocalOnly = false
			case "suspend":
				opts.SuspendPaths = []string{"shared.txt"}
			case "dirty", "staged":
				writeFile(t, filepath.Join(repo, "dirty.txt"), "retain these bytes")
				if kind == "staged" {
					git(t, repo, "add", "dirty.txt")
				}
				packet.DirtyPaths = []string{"dirty.txt"}
			case "merge-active":
				writeFile(t, filepath.Join(repo, ".git", "MERGE_HEAD"), packet.TargetSHA+"\n")
			case "path-lease":
				_, err := leaseref.NewInDir(repo).Acquire(context.Background(), leaseref.Record{ID: "peer", Holder: "peer", SessionID: "same-text", TreeGlobs: []string{"shared.txt"}, AcquiredAt: time.Now().Unix(), TTLSeconds: 3600})
				if err != nil {
					t.Fatal(err)
				}
				opts.Session = "same-text"
			case "lease-inspection":
				opts.Runner = func(ctx context.Context, dir string, args ...string) RunResult {
					if len(args) > 0 && args[0] == "for-each-ref" {
						return RunResult{Code: 1, Err: errors.New("lease lookup unavailable")}
					}
					return RealRunner(ctx, dir, args...)
				}
			case "conflict":
				opts.Runner = func(ctx context.Context, dir string, args ...string) RunResult {
					if len(args) > 0 && args[0] == "merge-tree" {
						return RunResult{Code: 1, Err: errors.New("conflicting preview")}
					}
					return RealRunner(ctx, dir, args...)
				}
			case "false-disjoint":
				packet.Disposition, packet.Dispatchable = DispositionSafeDisjoint, true
				request.Review.OverlapPaths = []string{"invented"}
			}
			// Re-sign modified manifests to prove native recomputation, not merely a
			// digest check, rejects stale metadata and forged review display fields.
			switch kind {
			case "repository", "worktree", "head", "target", "tree", "paths", "diff", "false-disjoint":
				request.Review.Digest = reviewDigest(request.Review)
				request.Approval.Digest = request.Review.Digest
			}
			receipt, err := ExecutePacket(context.Background(), packet, opts)
			if err == nil || receipt == nil || receipt.Status != ExecuteStatusRefused || receipt.Pushed {
				t.Fatalf("receipt=%+v err=%v", receipt, err)
			}
			head, _ := rev(context.Background(), RealRunner, repo, "HEAD")
			if head != packet.LocalHead {
				t.Fatal("refusal mutated HEAD")
			}
		})
	}
}

func TestReviewedOverlapMutationBoundaries(t *testing.T) {
	for _, kind := range []string{"writer-lost-before-merge", "head-race", "target-race", "hook-tree-change", "gate-tree-change", "gate-failure"} {
		t.Run(kind, func(t *testing.T) {
			_, repo, packet, request := reviewedOverlapFixture(t)
			opts := ExecuteOptions{Repo: repo, Branch: "work", LocalOnly: true, ReviewedOverlap: &request,
				BuildVerifier: func(context.Context, string) error { return nil }, TestVerifier: func(context.Context, string) error { return nil }}
			merged := false
			opts.Runner = func(ctx context.Context, dir string, args ...string) RunResult {
				// Trigger after preview/diff and path inspection, before the final fence.
				if len(args) > 0 && args[0] == "for-each-ref" {
					r := RealRunner(ctx, dir, args...)
					switch kind {
					case "writer-lost-before-merge":
						if err := os.Remove(filepath.Join(repo, ".git", writerLeaseFile)); err != nil {
							t.Fatal(err)
						}
					case "head-race":
						git(t, repo, "commit", "--allow-empty", "-m", "concurrent HEAD drift")
					case "target-race":
						git(t, repo, "update-ref", packet.TargetRef, packet.LocalHead)
					}
					return r
				}
				if len(args) > 0 && args[0] == "merge" {
					merged = true
				}
				return RealRunner(ctx, dir, args...)
			}
			switch kind {
			case "hook-tree-change":
				installReviewedOverlapHook(t, repo, "mutate-index")
			case "gate-tree-change":
				opts.TestVerifier = func(context.Context, string) error {
					writeFile(t, filepath.Join(repo, "gate.txt"), "late change")
					git(t, repo, "add", "gate.txt")
					git(t, repo, "commit", "-m", "gate changes HEAD")
					return nil
				}
			case "gate-failure":
				opts.BuildVerifier = func(context.Context, string) error { return errors.New("real gate failure") }
			}
			receipt, err := ExecutePacket(context.Background(), packet, opts)
			if err == nil || receipt == nil || receipt.Status == ExecuteStatusExecuted || receipt.Pushed {
				t.Fatalf("receipt=%+v err=%v", receipt, err)
			}
			if strings.HasPrefix(kind, "hook") || strings.HasPrefix(kind, "gate") {
				if !merged || receipt.Status != ExecuteStatusFailed {
					t.Fatal("mutation failure mislabeled")
				}
				head, _ := rev(context.Background(), RealRunner, repo, "HEAD")
				if head == packet.LocalHead {
					t.Fatal("failed mutation was reset")
				}
			} else if merged {
				t.Fatal("lost exclusion or source drift reached merge")
			}
		})
	}
}

func TestReviewedOverlapManifestIsNotApproval(t *testing.T) {
	_, repo, packet, request := reviewedOverlapFixture(t)
	// Preparation has no ref/index/worktree effects, and real textual conflicts
	// remain outside this capability even when the target is an immutable commit.
	head, _ := rev(context.Background(), RealRunner, repo, "HEAD")
	if head != packet.LocalHead {
		t.Fatal("manifest generation moved HEAD")
	}
	dirty, err := workingTreeDirtyPaths(context.Background(), RealRunner, repo)
	if err != nil || len(dirty) != 0 {
		t.Fatalf("manifest generation dirtied source: %v %v", dirty, err)
	}
	request.Approval.Digest = ""
	receipt, err := ExecutePacket(context.Background(), packet, ExecuteOptions{Repo: repo, LocalOnly: true, ReviewedOverlap: &request})
	if err == nil || receipt == nil || receipt.Status != ExecuteStatusRefused {
		t.Fatal("manifest alone admitted execution")
	}
	data, err := os.ReadFile(filepath.Join(repo, "shared.txt"))
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(repo, "shared.txt"), strings.TrimSuffix(string(data), "last\n")+"local conflict\n")
	git(t, repo, "add", "shared.txt")
	git(t, repo, "commit", "-m", "conflicting final line")
	if _, err := PrepareCleanOverlapReview(context.Background(), CleanOverlapReviewOptions{Repo: repo, TargetSHA: packet.TargetSHA, Reviewer: "fixture reviewer", Evidence: "fixture delegated review"}); err == nil {
		t.Fatal("textual conflict received clean review manifest")
	}
}

func TestReviewedOverlapReconcileAPIRejectsNonlocal(t *testing.T) {
	for _, opts := range []ReconcileOptions{
		{ReviewedOverlap: &ReviewedOverlapRequest{}},
		{ReviewedOverlap: &ReviewedOverlapRequest{}, LocalOnly: true, Execute: true, Apply: true},
		{ReviewedOverlap: &ReviewedOverlapRequest{}, LocalOnly: true, Execute: true, SuspendPaths: []string{"shared.txt"}},
	} {
		opts.Runner = func(context.Context, string, ...string) RunResult {
			t.Fatal("invalid API options reached Git")
			return RunResult{}
		}
		if _, err := RouteReconciliation(context.Background(), opts); err == nil {
			t.Fatal("invalid review API options admitted")
		}
	}
}

func TestReviewedOverlapLiteralPathBoundary(t *testing.T) {
	for _, name := range []string{"quoted\"name.txt", " leading.txt", "trailing.txt ", "line\nbreak.txt", "tab\tname.txt", "wild*.txt"} {
		t.Run(name, func(t *testing.T) {
			runner := func(_ context.Context, _ string, args ...string) RunResult {
				if strings.Join(args, " ") != "diff --no-ext-diff --no-textconv --no-renames --name-only -z base tip --" {
					t.Fatalf("not exact NUL path inspection: %v", args)
				}
				return RunResult{Stdout: []byte(name + "\x00")}
			}
			paths, err := reviewTreePaths(context.Background(), runner, "repo", "base", "tip")
			if name == "quoted\"name.txt" {
				if err != nil || len(paths) != 1 || paths[0] != name {
					t.Fatalf("literal name lost: %q %v", paths, err)
				}
			} else if err == nil {
				t.Fatalf("unsupported literal name silently normalized: %q", paths)
			}
		})
	}
}

func TestReviewedOverlapMergeIntroducedPathLease(t *testing.T) {
	_, repo, _, _ := reviewedOverlapFixture(t)
	git(t, repo, "checkout", "-b", "side")
	writeFile(t, filepath.Join(repo, "side.txt"), "side")
	git(t, repo, "add", "side.txt")
	git(t, repo, "commit", "-m", "side branch")
	git(t, repo, "checkout", "work")
	git(t, repo, "commit", "--allow-empty", "-m", "local branch")
	git(t, repo, "merge", "--no-ff", "--no-commit", "side")
	writeFile(t, filepath.Join(repo, "merge-only.txt"), "introduced only in merge result")
	git(t, repo, "add", "merge-only.txt")
	git(t, repo, "commit", "-m", "merge with additional result path")
	packet, err := BuildReconciliationPacket(context.Background(), PacketOptions{Repo: repo, Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	review, err := PrepareCleanOverlapReview(context.Background(), CleanOverlapReviewOptions{Repo: repo, TargetSHA: packet.TargetSHA, Reviewer: "fixture reviewer", Evidence: "fixture delegated review"})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, p := range review.WritePaths {
		if p == "merge-only.txt" {
			found = true
		}
	}
	if !found {
		t.Fatal("native tree inspection missed merge-introduced path")
	}
	if _, err := leaseref.NewInDir(repo).Acquire(context.Background(), leaseref.Record{ID: "merge-path-owner", Holder: "peer", TreeGlobs: []string{"merge-only.txt"}, AcquiredAt: time.Now().Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatal(err)
	}
	request := ReviewedOverlapRequest{Review: *review, Approval: ReviewedOverlapApproval{Digest: review.Digest}}
	receipt, err := ExecutePacket(context.Background(), packet, ExecuteOptions{Repo: repo, LocalOnly: true, ReviewedOverlap: &request})
	if err == nil || receipt == nil || receipt.Status != ExecuteStatusRefused {
		t.Fatalf("merge path owner lost: %+v %v", receipt, err)
	}
	head, _ := rev(context.Background(), RealRunner, repo, "HEAD")
	if head != packet.LocalHead {
		t.Fatal("path refusal changed HEAD")
	}
}

func TestReviewedOverlapStrictLaneAdmission(t *testing.T) {
	cases := []struct {
		name, taxonomy, id string
		geometry           []string
		refuse             bool
	}{
		{"unknown-disjoint", "", "random-id", []string{"unrelated.txt"}, false},
		{"known-disjoint", "[lanes.trees]\npeer = [\"unrelated.txt\"]\n", "peer", []string{"unrelated.txt"}, false},
		{"lane-only-disjoint", "[lanes.trees]\npeer = [\"unrelated.txt\"]\n", "peer", nil, false},
		{"exclusive-live", "[lanes]\nexclusive = [\"peer\"]\n[lanes.trees]\npeer = [\"unrelated.txt\"]\n", "peer", []string{"unrelated.txt"}, true},
		{"exclusive-requested", "[lanes]\nexclusive = [\"requested\"]\n[lanes.trees]\nrequested = [\"shared.txt\"]\n", "random-id", []string{"unrelated.txt"}, true},
		{"exclusive-inferred", "[lanes]\nexclusive = [\"exclusive\"]\n[lanes.trees]\nexclusive = [\"private/**\"]\n", "random-id", []string{"private/one.txt"}, true},
		{"unsupported-recursive-claim", "", "random-id", []string{"docs/**/owned.txt"}, true},
		{"unsupported-recursive-taxonomy", "[lanes.trees]\npeer = [\"docs/**/owned.txt\"]\n", "peer", []string{"unrelated.txt"}, true},
		{"malformed-glob", "[lanes.trees]\npeer = [\"unrelated[\"]\n", "peer", []string{"unrelated.txt"}, true},
		{"incomplete-array", "[lanes]\nexclusive = [\"peer\",\n[lanes.trees]\npeer = [\"unrelated.txt\"]\n", "peer", []string{"unrelated.txt"}, true},
		{"invalid-scalar", "[lanes]\nexclusive = \"peer\"\n", "peer", []string{"unrelated.txt"}, true},
		{"duplicate-key", "[lanes]\nexclusive = [\"peer\"]\nexclusive = []\n", "peer", []string{"unrelated.txt"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, repo := setupTestOriginAndClone(t)
			if tc.taxonomy != "" {
				writeFile(t, filepath.Join(repo, "dos.toml"), tc.taxonomy)
			}
			if _, err := leaseref.NewInDir(repo).Acquire(context.Background(), leaseref.Record{ID: tc.id, Holder: "peer", TreeGlobs: tc.geometry, AcquiredAt: time.Now().Unix(), TTLSeconds: 3600}); err != nil {
				t.Fatal(err)
			}
			err := checkReviewPathOwners(context.Background(), CleanOverlapReviewOptions{Repo: repo}, CleanOverlapReview{WritePaths: []string{"shared.txt"}})
			if (err != nil) != tc.refuse {
				t.Fatalf("refuse=%v err=%v", tc.refuse, err)
			}
		})
	}
}
