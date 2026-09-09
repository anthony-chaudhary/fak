package safesync

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func setupConcurrentTestOrigin(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	origin := filepath.Join(tmp, "origin.git")
	git(t, tmp, "init", "--bare", "-b", "main", origin)

	// Seed origin with initial commit via a seed clone
	seedDir := filepath.Join(tmp, "seed")
	git(t, tmp, "-c", "core.autocrlf=false", "clone", origin, seedDir)
	git(t, seedDir, "config", "core.autocrlf", "false")
	git(t, seedDir, "config", "user.name", "test")
	git(t, seedDir, "config", "user.email", "test@example.com")

	// Create valid Go module with base package and tests
	writeFile(t, filepath.Join(seedDir, "go.mod"), "module example.com/safesynctest\n\ngo 1.22\n")

	basePkg := filepath.Join(seedDir, "pkg", "base")
	mkdir(t, basePkg)
	writeFile(t, filepath.Join(basePkg, "base.go"), "package base\n\nfunc Base() string {\n\treturn \"base\"\n}\n")
	writeFile(t, filepath.Join(basePkg, "base_test.go"), "package base\n\nimport \"testing\"\n\nfunc TestBase(t *testing.T) {\n\tif Base() != \"base\" {\n\t\tt.Fatalf(\"unexpected value: %s\", Base())\n\t}\n}\n")

	git(t, seedDir, "add", ".")
	git(t, seedDir, "commit", "-m", "init: base package and go.mod")
	git(t, seedDir, "push", "origin", "main")

	return origin
}

func cloneNode(t *testing.T, origin, name string) string {
	t.Helper()
	tmp := filepath.Dir(origin)
	nodeDir := filepath.Join(tmp, name)
	git(t, tmp, "-c", "core.autocrlf=false", "clone", origin, nodeDir)
	git(t, nodeDir, "config", "core.autocrlf", "false")
	git(t, nodeDir, "config", "user.name", name)
	git(t, nodeDir, "config", "user.email", name+"@example.com")
	return nodeDir
}

func addNodePackage(t *testing.T, nodeDir, pkgName string) {
	t.Helper()
	pkgDir := filepath.Join(nodeDir, "pkg", pkgName)
	mkdir(t, pkgDir)
	writeFile(t, filepath.Join(pkgDir, pkgName+".go"), fmt.Sprintf("package %s\n\nfunc Val() string {\n\treturn %q\n}\n", pkgName, pkgName))
	writeFile(t, filepath.Join(pkgDir, pkgName+"_test.go"), fmt.Sprintf("package %s\n\nimport \"testing\"\n\nfunc TestVal(t *testing.T) {\n\tif Val() != %q {\n\t\tt.Fatalf(\"unexpected: %%s\", Val())\n\t}\n}\n", pkgName, pkgName))
	git(t, nodeDir, "add", ".")
	git(t, nodeDir, "commit", "-m", fmt.Sprintf("feat(%s): add package", pkgName))
}

func runGoTests(t *testing.T, dir string) {
	t.Helper()
	cmd := exec.Command("go", "test", "-v", "./...")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go test ./... in %s failed: %v\n%s", dir, err, out)
	}
}

func assertCleanConvergence(t *testing.T, dir string, expectedFiles []string) {
	t.Helper()
	// 1. Verify all expected files exist on disk (0 missing files)
	for _, rel := range expectedFiles {
		fullPath := filepath.Join(dir, rel)
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			t.Errorf("node %s is missing expected file: %s", filepath.Base(dir), rel)
		}
	}

	// 2. Verify clean index state (no phantom deletions, no staged or unstaged changes)
	status := strings.TrimSpace(gitOutput(t, dir, "status", "--porcelain"))
	if status != "" {
		t.Errorf("node %s index not clean, git status:\n%s", filepath.Base(dir), status)
	}

	// 3. Verify tests pass cleanly across all packages
	runGoTests(t, dir)
}

func iterativelyReconcileAndPush(ctx context.Context, t *testing.T, nodeDir, branch string, maxRetries int) error {
	initialHead := revString(t, nodeDir, "HEAD")
	for i := 0; i < maxRetries; i++ {
		// Fetch latest remote
		git(t, nodeDir, "fetch", "origin", branch)
		remoteRef := "origin/" + branch
		remoteHead := revString(t, nodeDir, remoteRef)

		// Check if our initial commit is already merged into remote
		if isAnc, _ := isAncestor(ctx, RealRunner, nodeDir, initialHead, remoteHead); isAnc {
			// Work is already integrated into remote; fast-forward local branch
			ffRes := RealRunner(ctx, nodeDir, "merge", "--ff-only", "--no-autostash", "--no-overwrite-ignore", remoteRef)
			if ffRes.Err == nil && ffRes.Code == 0 {
				return nil
			}
		}

		pushOpts := PushOptions{
			Repo:   nodeDir,
			Remote: "origin",
			Branch: branch,
		}
		pushRes, err := SafePush(ctx, pushOpts)
		if err == nil && pushRes.Pushed {
			return nil
		}

		reconOpts := ReconcileOptions{
			Repo:   nodeDir,
			Remote: "origin",
			Branch: branch,
			Goal:   "publish",
			Apply:  true,
			Fetch:  true,
		}
		reconRes, err := RouteReconciliation(ctx, reconOpts)
		if err != nil {
			return fmt.Errorf("reconciliation failed: %w", err)
		}
		if reconRes.Execution != nil && reconRes.Execution.Pushed {
			return nil
		}
		if !reconRes.OK {
			time.Sleep(50 * time.Millisecond)
			continue
		}

		time.Sleep(50 * time.Millisecond)
	}
	return fmt.Errorf("exceeded max retries (%d) during iterative reconciliation", maxRetries)
}

func TestConcurrentIntegrationRace(t *testing.T) {
	t.Run("sequential_disjoint_publication_and_reconciliation", func(t *testing.T) {
		origin := setupConcurrentTestOrigin(t)
		node1 := cloneNode(t, origin, "node1")
		node2 := cloneNode(t, origin, "node2")

		// Pre-flight check: both nodes pass baseline tests
		runGoTests(t, node1)
		runGoTests(t, node2)

		// Node 1 commits package alpha
		addNodePackage(t, node1, "alpha")

		// Node 2 commits package beta (disjoint from alpha)
		addNodePackage(t, node2, "beta")

		ctx := context.Background()

		// Node 1 publishes to origin
		pushOpts1 := PushOptions{Repo: node1, Remote: "origin", Branch: "main"}
		p1Res, err := SafePush(ctx, pushOpts1)
		if err != nil || !p1Res.Pushed {
			t.Fatalf("node 1 push failed: %v, res: %+v", err, p1Res)
		}

		// Node 2 attempts push: should stop because origin has advanced (non-fast-forward)
		pushOpts2 := PushOptions{Repo: node2, Remote: "origin", Branch: "main"}
		p2Res, err := SafePush(ctx, pushOpts2)
		if err != nil {
			t.Fatalf("unexpected error on node 2 push: %v", err)
		}
		if p2Res.Pushed {
			t.Fatalf("expected node 2 push to be rejected as diverged, got pushed")
		}

		// Node 2 reconciles via RouteDisjointIntegrate
		reconOpts2 := ReconcileOptions{
			Repo:   node2,
			Remote: "origin",
			Branch: "main",
			Goal:   "publish",
			Apply:  true,
			Fetch:  true,
		}
		rec2, err := RouteReconciliation(ctx, reconOpts2)
		if err != nil {
			t.Fatalf("node 2 RouteReconciliation failed: %v", err)
		}
		if rec2.Route != RouteDisjointIntegrate {
			t.Fatalf("expected RouteDisjointIntegrate, got %s", rec2.Route)
		}
		if !rec2.OK || !rec2.Applied {
			t.Fatalf("expected RouteDisjointIntegrate OK and Applied, got %+v", rec2)
		}

		// Node 2 now publishes the integrated merge commit
		p2ResRetried, err := SafePush(ctx, pushOpts2)
		if err != nil || !p2ResRetried.Pushed {
			t.Fatalf("node 2 push after integration failed: %v, res: %+v", err, p2ResRetried)
		}

		// Node 1 is now behind origin/main (which holds the merge commit)
		// Node 1 reconciles via RouteApply (fast-forward)
		reconOpts1 := ReconcileOptions{
			Repo:   node1,
			Remote: "origin",
			Branch: "main",
			Goal:   "integrate",
			Apply:  true,
			Fetch:  true,
		}
		rec1, err := RouteReconciliation(ctx, reconOpts1)
		if err != nil {
			t.Fatalf("node 1 RouteReconciliation failed: %v", err)
		}
		if rec1.Route != RouteApply {
			t.Fatalf("expected node 1 RouteApply, got %s", rec1.Route)
		}
		if !rec1.OK || !rec1.Applied {
			t.Fatalf("expected node 1 Apply OK and Applied, got %+v", rec1)
		}

		// Verify convergence across both nodes
		expectedFiles := []string{
			"go.mod",
			filepath.Join("pkg", "base", "base.go"),
			filepath.Join("pkg", "base", "base_test.go"),
			filepath.Join("pkg", "alpha", "alpha.go"),
			filepath.Join("pkg", "alpha", "alpha_test.go"),
			filepath.Join("pkg", "beta", "beta.go"),
			filepath.Join("pkg", "beta", "beta_test.go"),
		}

		assertCleanConvergence(t, node1, expectedFiles)
		assertCleanConvergence(t, node2, expectedFiles)

		// Verify identical commit SHAs matching origin
		originHead := revString(t, origin, "refs/heads/main")
		node1Head := revString(t, node1, "HEAD")
		node2Head := revString(t, node2, "HEAD")

		if node1Head != originHead {
			t.Errorf("node 1 HEAD (%s) != origin main (%s)", node1Head, originHead)
		}
		if node2Head != originHead {
			t.Errorf("node 2 HEAD (%s) != origin main (%s)", node2Head, originHead)
		}
	})

	t.Run("concurrent_goroutine_disjoint_race", func(t *testing.T) {
		origin := setupConcurrentTestOrigin(t)
		node1 := cloneNode(t, origin, "node1")
		node2 := cloneNode(t, origin, "node2")

		addNodePackage(t, node1, "alpha")
		addNodePackage(t, node2, "beta")

		ctx := context.Background()
		var wg sync.WaitGroup
		errCh := make(chan error, 2)

		startBarrier := make(chan struct{})

		wg.Add(2)
		go func() {
			defer wg.Done()
			<-startBarrier
			if err := iterativelyReconcileAndPush(ctx, t, node1, "main", 15); err != nil {
				errCh <- fmt.Errorf("node 1 error: %w", err)
			}
		}()

		go func() {
			defer wg.Done()
			<-startBarrier
			if err := iterativelyReconcileAndPush(ctx, t, node2, "main", 15); err != nil {
				errCh <- fmt.Errorf("node 2 error: %w", err)
			}
		}()

		close(startBarrier)
		wg.Wait()
		close(errCh)

		for err := range errCh {
			t.Fatalf("concurrent race failed: %v", err)
		}

		// Ensure both nodes bring themselves to origin/main
		for _, node := range []string{node1, node2} {
			syncOpts := ReconcileOptions{
				Repo:   node,
				Remote: "origin",
				Branch: "main",
				Goal:   "integrate",
				Apply:  true,
				Fetch:  true,
			}
			_, err := RouteReconciliation(ctx, syncOpts)
			if err != nil {
				t.Fatalf("final sync for %s failed: %v", filepath.Base(node), err)
			}
		}

		expectedFiles := []string{
			"go.mod",
			filepath.Join("pkg", "base", "base.go"),
			filepath.Join("pkg", "base", "base_test.go"),
			filepath.Join("pkg", "alpha", "alpha.go"),
			filepath.Join("pkg", "alpha", "alpha_test.go"),
			filepath.Join("pkg", "beta", "beta.go"),
			filepath.Join("pkg", "beta", "beta_test.go"),
		}

		assertCleanConvergence(t, node1, expectedFiles)
		assertCleanConvergence(t, node2, expectedFiles)
	})

	t.Run("disjoint_integration_preserves_uncommitted_wip", func(t *testing.T) {
		origin := setupConcurrentTestOrigin(t)
		node1 := cloneNode(t, origin, "node1")
		node2 := cloneNode(t, origin, "node2")

		addNodePackage(t, node1, "alpha")
		addNodePackage(t, node2, "beta")

		// Add uncommitted WIP in node 2 in a disjoint package gamma
		gammaPkg := filepath.Join(node2, "pkg", "gamma")
		mkdir(t, gammaPkg)
		gammaContent := "package gamma\n\nvar Uncommitted = true\n"
		writeFile(t, filepath.Join(gammaPkg, "gamma.go"), gammaContent)

		ctx := context.Background()

		// Node 1 publishes alpha
		pushOpts1 := PushOptions{Repo: node1, Remote: "origin", Branch: "main"}
		p1Res, err := SafePush(ctx, pushOpts1)
		if err != nil || !p1Res.Pushed {
			t.Fatalf("node 1 push failed: %v", err)
		}

		// Node 2 reconciles via RouteDisjointIntegrate with dirty uncommitted WIP present
		reconOpts2 := ReconcileOptions{
			Repo:   node2,
			Remote: "origin",
			Branch: "main",
			Goal:   "integrate",
			Apply:  true,
			Fetch:  true,
		}
		rec2, err := RouteReconciliation(ctx, reconOpts2)
		if err != nil {
			t.Fatalf("node 2 reconciliation failed: %v", err)
		}
		if rec2.Route != RouteDisjointIntegrate || !rec2.Applied {
			t.Fatalf("expected RouteDisjointIntegrate Applied, got %+v", rec2)
		}

		// Node 2 must preserve uncommitted gamma file unmodified
		gotGamma := readFile(t, filepath.Join(gammaPkg, "gamma.go"))
		if gotGamma != gammaContent {
			t.Fatalf("uncommitted gamma.go was modified! got %q, want %q", gotGamma, gammaContent)
		}

		// Node 2 must have received alpha.go on disk (0 missing incoming files)
		alphaPath := filepath.Join(node2, "pkg", "alpha", "alpha.go")
		if _, err := os.Stat(alphaPath); os.IsNotExist(err) {
			t.Fatalf("node 2 is missing incoming disjoint file alpha.go")
		}

		// Git status in node 2 must only report the untracked gamma file, NO deleted alpha files!
		status2 := gitOutput(t, node2, "status", "--porcelain")
		if strings.Contains(status2, "D ") || strings.Contains(status2, " D") {
			t.Fatalf("node 2 has phantom deletions in git status:\n%s", status2)
		}
		if !strings.Contains(status2, "?? pkg/gamma") {
			t.Fatalf("node 2 git status missing untracked gamma:\n%s", status2)
		}
	})
}
