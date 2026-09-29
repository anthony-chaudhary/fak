package workerworktree

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const candidateLeakHelperEnv = "FAK_CANDIDATE_LEAK_HELPER_ROOT"

func TestCandidateLeakHelperProcess(t *testing.T) {
	root := os.Getenv(candidateLeakHelperEnv)
	if root == "" {
		t.Skip("helper process for TestKilledLandLeaksCandidateUntilSweep")
	}
	verifyTopologyCandidate(root, "HEAD", "", func(dir string) (bool, string) {
		os.Stdout.WriteString("CANDIDATE " + dir + "\n")
		time.Sleep(time.Hour)
		return false, "unreachable"
	}, nil)
}

func TestKilledLandLeaksCandidateUntilSweep(t *testing.T) {
	root, parent := newSiblingTopologyRepo(t)

	cmd := exec.Command(os.Args[0], "-test.run=^TestCandidateLeakHelperProcess$", "-test.v")
	cmd.Env = append(os.Environ(), candidateLeakHelperEnv+"="+root)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	found := make(chan string, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			if dir, ok := strings.CutPrefix(scanner.Text(), "CANDIDATE "); ok {
				found <- strings.TrimSpace(dir)
				return
			}
		}
		close(found)
	}()
	var cand string
	select {
	case cand = <-found:
	case <-time.After(60 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatal("helper never reported its candidate")
	}
	if cand == "" {
		_ = cmd.Process.Kill()
		t.Fatal("helper exited without reporting a candidate")
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Wait()

	if filepath.Dir(cand) != parent {
		t.Fatalf("candidate %q not created beside the repository in %q", cand, parent)
	}
	owner, ok := parseTopologyCandidateOwner(filepath.Base(cand))
	if !ok || owner.PID != cmd.Process.Pid {
		t.Fatalf("candidate name %q does not carry the helper's pid %d (owner=%+v ok=%v)", filepath.Base(cand), cmd.Process.Pid, owner, ok)
	}
	if _, err := os.Stat(filepath.Join(cand, ".git")); err != nil {
		t.Fatalf("killed land should leave its checkout behind (the leak): %v", err)
	}
	if got := registeredCandidates(t, root); len(got) != 1 {
		t.Fatalf("killed land should leave one registration behind, got %v", got)
	}

	report := SweepTopologyCandidates(root, parent, nil, CandidateSweepOptions{Apply: true})
	if report.Reaped != 1 || report.WouldReap != 1 || len(report.Failures) != 0 {
		t.Fatalf("sweep did not collect the dead owner's candidate: %+v", report)
	}
	if report.Candidates[0].Reason != "owner_exited" {
		t.Fatalf("reason = %q, want owner_exited", report.Candidates[0].Reason)
	}
	if _, err := os.Stat(cand); !os.IsNotExist(err) {
		t.Fatalf("candidate directory survived the sweep: %v", err)
	}
	if got := registeredCandidates(t, root); len(got) != 0 {
		t.Fatalf("sweep left registrations behind: %v", got)
	}
}

func TestVerifyTopologyCandidateSweepsBeforeCreatingAndCleansUp(t *testing.T) {
	sweepTopologyCandidatesBeforeCreate = realSweepTopologyCandidatesBeforeCreate
	t.Cleanup(func() { sweepTopologyCandidatesBeforeCreate = func(string, string, GitRunner) {} })
	root, parent := newSiblingTopologyRepo(t)
	candidateSweepMu.Lock()
	delete(candidateSweepLast, parent)
	candidateSweepMu.Unlock()

	// PID 2^31-1 is never a live process on the supported platforms.
	stale := filepath.Join(parent, topologyCandidatePrefix+"2147483647-0-1")
	if err := os.MkdirAll(filepath.Join(stale, "pkg"), 0o755); err != nil {
		t.Fatal(err)
	}

	var verified string
	ok, detail := verifyTopologyCandidate(root, "HEAD", "", func(dir string) (bool, string) {
		verified = dir
		if _, err := os.Stat(stale); !os.IsNotExist(err) {
			return false, "stale candidate still present during verify"
		}
		return true, ""
	}, nil)
	if !ok {
		t.Fatalf("verify failed: %s", detail)
	}
	if owner, parsed := parseTopologyCandidateOwner(filepath.Base(verified)); !parsed || owner.PID != os.Getpid() {
		t.Fatalf("candidate %q is not owner-named for this process", verified)
	}
	if _, err := os.Stat(verified); !os.IsNotExist(err) {
		t.Fatalf("candidate survived cleanup: %v", err)
	}
	if got := registeredCandidates(t, root); len(got) != 0 {
		t.Fatalf("cleanup left registrations: %v", got)
	}
}
