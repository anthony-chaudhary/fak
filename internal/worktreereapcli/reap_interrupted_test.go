package worktreereapcli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// residueLeaseID is the fixed lease identity the fixture's residue owner carries, so
// the live-lease refusal can seed that exact id into the leaseref store.
const residueLeaseID = "land-dependency-test"

// deadPID returns a PID that is reliably dead on this host: it starts a short-lived
// git child, captures its PID, and waits for it to exit. processalive.Check then
// reports it dead on both Windows (OpenProcess -> ERROR_INVALID_PARAMETER) and Unix
// (Kill(pid,0) -> ESRCH).
func deadPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command("git", "rev-parse", "--version")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start dead-pid probe: %v", err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		t.Fatalf("wait dead-pid probe: %v", err)
	}
	return pid
}

// interruptedResidueFixture builds a root repo plus a managed worker whose .git marker
// and Git registration are gone but whose owner and pool sidecars remain, with a dead
// owner and a released lease — the exact post-interruption state. It returns the root,
// the residue path, and a neighboring foreign worker that must survive byte-identical.
func interruptedResidueFixture(t *testing.T) (root, residue, foreign string) {
	t.Helper()
	root = t.TempDir()
	gitTest(t, root, "init", "-q", "-b", "main")
	gitTest(t, root, "config", "user.name", "test")
	gitTest(t, root, "config", "user.email", "test@example.invalid")
	gitTest(t, root, "config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(root, "base"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitTest(t, root, "add", "base")
	gitTest(t, root, "commit", "-qm", "base")
	base := gitTest(t, root, "rev-parse", "HEAD")
	wtRoot := t.TempDir()

	// The residue: a managed worker owned by a dead PID with a released lease.
	pid := deadPID(t)
	res := workerworktree.PrepareOwned(root, "cli", "residue", base, wtRoot, nil,
		workerworktree.OwnerStamp{PID: pid, LeaseID: residueLeaseID, CreatedAt: time.Now().UTC()})
	if !res.OK {
		t.Fatalf("prepare residue: %+v", res)
	}
	residue = res.Path

	// The foreign neighbor: a second managed worker that must remain byte-identical.
	fres := workerworktree.PrepareOwned(root, "cli", "foreign", base, wtRoot, nil,
		workerworktree.OwnerStamp{PID: os.Getpid(), LeaseID: "land-dependency-foreign", CreatedAt: time.Now().UTC()})
	if !fres.OK {
		t.Fatalf("prepare foreign: %+v", fres)
	}
	foreign = fres.Path

	// Simulate the interrupted reap: the .git marker is gone and the registration is
	// pruned, but the checkout and its owner/pool sidecars remain.
	if err := os.Remove(filepath.Join(residue, ".git")); err != nil {
		t.Fatalf("remove residue .git: %v", err)
	}
	gitTest(t, root, "worktree", "prune")
	return root, residue, foreign
}

// poolSidecarPath mirrors workerworktree.poolMemberPath for the v2 pool record.
func poolSidecarPath(wtPath string) string {
	return filepath.Join(filepath.Dir(wtPath), ".fak-wt-pool", filepath.Base(wtPath)+".json")
}

// rewriteOwnerPID rewrites the owner sidecar's PID, preserving the schema and lease.
func rewriteOwnerPID(t *testing.T, wtPath string, pid int) {
	t.Helper()
	b, err := os.ReadFile(workerworktree.OwnerStampPath(wtPath))
	if err != nil {
		t.Fatal(err)
	}
	var stamp workerworktree.OwnerStamp
	if err := json.Unmarshal(b, &stamp); err != nil {
		t.Fatal(err)
	}
	stamp.PID = pid
	out, err := json.Marshal(stamp)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(workerworktree.OwnerStampPath(wtPath), out, 0o600); err != nil {
		t.Fatal(err)
	}
}

// seedLease registers a live lease with the given id in the root's leaseref store.
func seedLease(t *testing.T, root, id string) {
	t.Helper()
	ref, err := leaseref.NewInDir(root).Acquire(context.Background(), leaseref.Record{ID: id, TTLSeconds: 3600})
	if err != nil {
		t.Fatalf("seed lease: %v", err)
	}
	if ref == "" {
		t.Fatal("seed lease returned no ref")
	}
}

// rewritePoolOwnerPID rewrites the pool sidecar's owner PID, leaving the schema,
// lane, state, and timestamps intact so readPoolMember still accepts it. The
// resulting owner/pool PID disagreement is the mismatched-sidecars refusal.
func rewritePoolOwnerPID(t *testing.T, wtPath string, pid int) {
	t.Helper()
	path := poolSidecarPath(wtPath)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode pool sidecar: %v", err)
	}
	owner, ok := m["owner"].(map[string]any)
	if !ok {
		t.Fatalf("pool sidecar has no owner object: %s", b)
	}
	owner["pid"] = pid
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime unit est=1s lane=default
func TestSingleReapInterruptedResidueRecovery(t *testing.T) {
	root, residue, foreign := interruptedResidueFixture(t)
	foreignBefore, err := os.ReadFile(filepath.Join(foreign, "base"))
	if err != nil {
		t.Fatal(err)
	}
	foreignGitBefore, err := os.ReadFile(filepath.Join(foreign, ".git"))
	if err != nil {
		t.Fatal(err)
	}

	var out, stderr bytes.Buffer
	rc := RunSingle(&out, &stderr, root, residue, "", 5*time.Second)
	var result workerworktree.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v out=%s err=%s", err, &out, &stderr)
	}
	if rc != 0 || !result.OK || !result.Removed || result.Code != "INTERRUPTED_RESIDUE_REAPED" {
		t.Fatalf("recovery rc=%d result=%+v err=%s", rc, result, &stderr)
	}
	// The exact residue and its matching sidecars are gone.
	if _, err := os.Stat(residue); !os.IsNotExist(err) {
		t.Fatalf("residue remains: %v", err)
	}
	if _, err := os.Stat(workerworktree.OwnerStampPath(residue)); !os.IsNotExist(err) {
		t.Fatalf("owner sidecar remains: %v", err)
	}
	if _, err := os.Stat(poolSidecarPath(residue)); !os.IsNotExist(err) {
		t.Fatalf("pool sidecar remains: %v", err)
	}
	// The neighboring foreign dependency is byte-identical.
	foreignAfter, err := os.ReadFile(filepath.Join(foreign, "base"))
	if err != nil {
		t.Fatalf("foreign lost: %v", err)
	}
	if string(foreignBefore) != string(foreignAfter) {
		t.Fatalf("foreign content changed: %q -> %q", foreignBefore, foreignAfter)
	}
	foreignGitAfter, err := os.ReadFile(filepath.Join(foreign, ".git"))
	if err != nil {
		t.Fatalf("foreign .git lost: %v", err)
	}
	if string(foreignGitBefore) != string(foreignGitAfter) {
		t.Fatalf("foreign .git changed: %q -> %q", foreignGitBefore, foreignGitAfter)
	}
}

// fak-test:runtime unit est=1s lane=default
func TestSingleReapInterruptedResidueRefusals(t *testing.T) {
	for _, kind := range []string{
		"live-owner", "live-lease", "registered", "git-present",
		"missing-owner-sidecar", "missing-pool-sidecar", "mismatched-sidecars", "foreign-root",
	} {
		t.Run(kind, func(t *testing.T) {
			root, residue, _ := interruptedResidueFixture(t)
			switch kind {
			case "live-owner":
				// Both sidecars must agree on a LIVE pid so the mismatch gate passes
				// and the live-owner gate is the one that fires.
				rewriteOwnerPID(t, residue, os.Getpid())
				rewritePoolOwnerPID(t, residue, os.Getpid())
			case "live-lease":
				seedLease(t, root, residueLeaseID)
			case "registered":
				// A still-registered worker must NOT reach the recovery branch: the
				// normal path handles it. Make it dirty so the normal path refuses
				// (preserving every byte) instead of reaping it, proving the
				// recovery branch is not invoked for a registered target.
				reg := workerworktree.PrepareOwned(root, "cli", "registered", gitTest(t, root, "rev-parse", "HEAD"), filepath.Dir(residue), nil,
					workerworktree.OwnerStamp{PID: os.Getpid(), LeaseID: "land-dependency-registered", CreatedAt: time.Now().UTC()})
				if !reg.OK {
					t.Fatalf("re-register: %+v", reg)
				}
				if err := os.WriteFile(filepath.Join(reg.Path, "dirty.txt"), []byte("unlanded\n"), 0o644); err != nil {
					t.Fatal(err)
				}
				residue = reg.Path
			case "git-present":
				if err := os.WriteFile(filepath.Join(residue, ".git"), []byte("gitdir: /nowhere\n"), 0o644); err != nil {
					t.Fatal(err)
				}
			case "missing-owner-sidecar":
				if err := os.Remove(workerworktree.OwnerStampPath(residue)); err != nil {
					t.Fatal(err)
				}
			case "missing-pool-sidecar":
				if err := os.Remove(poolSidecarPath(residue)); err != nil {
					t.Fatal(err)
				}
			case "mismatched-sidecars":
				rewritePoolOwnerPID(t, residue, os.Getpid())
			case "foreign-root":
				residue = filepath.Join(t.TempDir(), "fak-worker-wt-foreign")
				if err := os.Mkdir(residue, 0o755); err != nil {
					t.Fatal(err)
				}
			}
			var out, stderr bytes.Buffer
			rc := RunSingle(&out, &stderr, root, residue, "", 5*time.Second)
			var result workerworktree.Result
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatalf("decode: %v out=%s err=%s", err, &out, &stderr)
			}
			if rc != 1 || result.OK || !result.Preserved {
				t.Fatalf("refusal rc=%d result=%+v err=%s", rc, result, &stderr)
			}
			// Every byte is preserved: the residue directory still exists.
			if _, err := os.Stat(residue); err != nil {
				t.Fatalf("refused residue lost: %v", err)
			}
		})
	}
}
