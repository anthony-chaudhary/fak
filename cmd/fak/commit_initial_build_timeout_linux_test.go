package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// This test binary doubles as a native go stand-in, without compiling another
// executable or introducing a shell helper. Only the explicitly marked subprocess
// enters this path, before testing parses the go-build arguments.
func init() {
	root := os.Getenv("FAK_COMMIT_BUILD_TIMEOUT_HELPER")
	if root == "" || len(os.Args) < 2 {
		return
	}
	if os.Args[1] == "commit-build-timeout-child" {
		time.Sleep(8 * time.Second)
		os.Exit(0)
	}
	if os.Args[1] != "build" {
		return
	}
	phase := os.Getenv("FAK_COMMIT_BUILD_TIMEOUT_PHASE")
	if strings.HasPrefix(phase, "head") {
		f, err := os.OpenFile(filepath.Join(root, "first-build"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_ = f.Close()
			fmt.Fprintln(os.Stderr, "p/p.go:3: undefined: Missing")
			os.Exit(1)
		}
		if !os.IsExist(err) {
			os.Exit(2)
		}
	}
	child := exec.Command(os.Args[0], "commit-build-timeout-child")
	child.Stdout, child.Stderr = os.Stdout, os.Stderr
	if err := child.Start(); err != nil {
		os.Exit(2)
	}
	if err := os.WriteFile(filepath.Join(root, "child-pid"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
		_ = child.Process.Kill()
		os.Exit(2)
	}
	if phase == "head-root-exits" {
		fmt.Fprintln(os.Stderr, "p/p.go:3: undefined: Missing")
		os.Exit(1) // descendant still holds stdout/stderr after this failed root
	}
	_ = child.Wait()
	os.Exit(0)
}

// fak-test:runtime integration est=9s lane=default
func TestCommitInitialBuildTimeoutRefusesAndReaps(t *testing.T) {
	for _, phase := range []string{"prospective", "head", "head-root-exits"} {
		t.Run(phase, func(t *testing.T) {
			repo, git := seedBuildCheckRepo(t)
			writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
			writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 1 }\n")
			commitBuildCheckPlumbing(t, repo, git, "seed build deadline witness", "go.mod", "p/p.go")
			writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 2 }\n")
			head, index := buildCheckGitState(t, git)
			helper := t.TempDir()
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(exe, filepath.Join(helper, "go")); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", helper+string(os.PathListSeparator)+os.Getenv("PATH"))
			t.Setenv("FAK_COMMIT_BUILD_TIMEOUT_HELPER", helper)
			t.Setenv("FAK_COMMIT_BUILD_TIMEOUT_PHASE", phase)
			previous := validatePhaseHook
			t.Cleanup(func() { validatePhaseHook = previous })
			validated := false
			validatePhaseHook = func(context.Context, string) { validated = true }
			var stderr bytes.Buffer
			timeout := time.Second
			if phase == "head-root-exits" {
				// Outlast exec.Cmd's former 5s WaitDelay to expose an ExitError
				// masking an incomplete pipe drain before the context expires.
				timeout = 6 * time.Second
			}
			start := time.Now()
			outcome, detail := runCommitBuildCheck(&stderr, repo, []string{"p/p.go"}, timeout)
			elapsed := time.Since(start)
			pidBytes, err := os.ReadFile(filepath.Join(helper, "child-pid"))
			if err != nil {
				t.Fatalf("stalled native helper did not start its child: %v; %s %s", err, outcome, detail)
			}
			pid, err := strconv.Atoi(string(pidBytes))
			if err != nil || pid <= 0 {
				t.Fatalf("invalid helper PID %q: %v", pidBytes, err)
			}
			t.Cleanup(func() {
				// The helper exits after 8s even if cancellation is broken. Do not
				// signal a numeric PID: it may have been reused after helper exit.
				until := time.Now().Add(9 * time.Second)
				for commitBuildTimeoutChildRunning(pid) && time.Now().Before(until) {
					time.Sleep(10 * time.Millisecond)
				}
				if commitBuildTimeoutChildRunning(pid) {
					t.Logf("PID %d is still present after the helper lifetime; identity is uncertain, so cleanup did not signal it", pid)
				}
			})
			res, admit, reason := safecommit.DecideBuildCheck(outcome, detail, false)
			if elapsed > timeout+3*time.Second || outcome != safecommit.BuildCheckSkippedTimeout || res.Compiled || admit || reason != safecommit.ReasonBuildCheckTimeout || validated {
				t.Fatalf("elapsed=%s outcome=%s compiled=%v admit=%v reason=%s validated=%v detail=%s", elapsed, outcome, res.Compiled, admit, reason, validated, detail)
			}
			deadline := time.Now().Add(time.Second)
			for commitBuildTimeoutChildRunning(pid) && time.Now().Before(deadline) {
				time.Sleep(10 * time.Millisecond)
			}
			if commitBuildTimeoutChildRunning(pid) {
				t.Fatalf("compiler descendant %d survived cancellation", pid)
			}
			requireBuildCheckGitState(t, git, head, index)
		})
	}
}

func commitBuildTimeoutChildRunning(pid int) bool {
	stat, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if os.IsNotExist(err) {
		return false
	}
	if err != nil {
		return true
	}
	// An orphan killed with its parent may await init's reaping; a zombie cannot
	// execute or retain the compiler's descriptors and is not a surviving child.
	end := strings.LastIndexByte(string(stat), ')')
	return end < 0 || !strings.HasPrefix(string(stat[end+1:]), " Z ")
}
