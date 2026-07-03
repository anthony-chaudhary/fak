package loopdrive

import (
	"context"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func TestLeaseRefSyncTwoCloneLaneadmitWitness(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skipf("git not available: %v", err)
	}
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	nodeA := filepath.Join(root, "node-a")
	nodeB := filepath.Join(root, "node-b")

	runGit(t, root, "init", "--bare", remote)
	runGit(t, root, "clone", remote, nodeA)
	runGit(t, root, "clone", remote, nodeB)

	ctx := context.Background()
	storeA := leaseref.NewInDir(nodeA)
	if _, err := storeA.Acquire(ctx, leaseref.Record{
		ID:         "resolve-loopdrive",
		TreeGlobs:  []string{"internal/loopdrive/**"},
		Holder:     "node-a",
		AcquiredAt: time.Now().Unix(),
		TTLSeconds: 3600,
	}); err != nil {
		t.Fatalf("node A acquire: %v", err)
	}

	afterWrite := LeaseRefSyncPlanForSurface(LeaseRefSyncSurfaceDispatchPreflight, LeaseRefSyncPlanInput{LeaseRefsWritten: true})[1]
	push, fetch, err := LeaseRefSyncDirections(afterWrite)
	if err != nil {
		t.Fatal(err)
	}
	if res, err := storeA.Sync(ctx, "origin", push, fetch); err != nil {
		t.Fatalf("node A after-write sync: %+v %v", res, err)
	}

	storeB := leaseref.NewInDir(nodeB)
	beforeDecide := LeaseRefSyncPlanForSurface(LeaseRefSyncSurfaceDispatchPreflight, LeaseRefSyncPlanInput{})[0]
	push, fetch, err = LeaseRefSyncDirections(beforeDecide)
	if err != nil {
		t.Fatal(err)
	}
	if res, err := storeB.Sync(ctx, "origin", push, fetch); err != nil {
		t.Fatalf("node B before-decide sync: %+v %v", res, err)
	}

	live, expired, err := storeB.Live(ctx, time.Now())
	if err != nil {
		t.Fatalf("node B live leases: %v", err)
	}
	if len(expired) != 0 {
		t.Fatalf("expired leases = %v, want none", expired)
	}
	leases := make([]laneadmit.Lease, 0, len(live))
	for _, rec := range live {
		leases = append(leases, laneadmit.Lease{
			ID:     rec.ID,
			Tree:   append([]string(nil), rec.TreeGlobs...),
			Holder: rec.Holder,
		})
	}

	verdict := laneadmit.Decide(
		laneadmit.Request{
			Surface: laneadmit.SurfaceDispatch,
			Lane:    "loopdrive",
			Tree:    []string{"internal/loopdrive/**"},
			Holder:  "node-b",
			LeaseID: "resolve-loopdrive-node-b",
		},
		leases,
		laneadmit.Taxonomy{
			Loaded:    true,
			Exclusive: map[string]bool{},
			Trees:     map[string][]string{"loopdrive": {"internal/loopdrive/**"}},
		},
	)
	if verdict.Admit || verdict.Reason != laneadmit.ReasonCollisionRisk {
		t.Fatalf("laneadmit verdict = %+v, want COLLISION_RISK refusal", verdict)
	}
	if len(verdict.Conflicts) != 1 || verdict.Conflicts[0].LeaseID != "resolve-loopdrive" {
		t.Fatalf("conflicts = %+v, want node A lease conflict", verdict.Conflicts)
	}
}

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v failed: %v\n%s", args, err, out)
	}
}
