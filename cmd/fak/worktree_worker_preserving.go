package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// worktreePreservingGate performs only reads against an already admitted lease.
// It never acquires, renews, releases, reclaims or publishes a lease.
func worktreePreservingGate(root, target string, owner workerworktree.OwnerStamp, holder string, generation int64, paths []string, base string, reserve int64) workerworktree.PreservationGate {
	return func(ctx context.Context) error {
		if strings.TrimSpace(holder) == "" || generation <= 0 || len(paths) == 0 || reserve <= 0 {
			return fmt.Errorf("fenced lease holder/generation, admitted source paths and caller disk-reserve policy required")
		}
		store := leaseref.NewInDir(root)
		verdict, err := store.Fence(ctx, leaseref.Record{ID: owner.LeaseID, Holder: holder, Generation: generation}, time.Now().UTC())
		if err != nil {
			return err
		}
		if !verdict.OK {
			return fmt.Errorf("lease admission refused: %s", verdict.Reason)
		}
		record, exists, err := store.Get(ctx, owner.LeaseID)
		if err != nil {
			return err
		}
		if !exists || record.Expired(time.Now().UTC()) || record.Holder != holder || record.Generation != generation || len(record.TreeGlobs) == 0 {
			return fmt.Errorf("admitted lease changed or has no path ownership")
		}
		if err := workerworktree.ValidatePreservingPaths(paths); err != nil {
			return err
		}
		if err := workerworktree.ValidateWorkerTreeDisjointness(paths, record.TreeGlobs); err != nil {
			return err
		}
		volume := filepath.Dir(target)
		for {
			if _, err := os.Stat(volume); err == nil {
				break
			} else if !os.IsNotExist(err) {
				return err
			}
			parent := filepath.Dir(volume)
			if parent == volume {
				return fmt.Errorf("target volume unavailable")
			}
			volume = parent
		}
		free, err := treedoctor.GoCacheFreeBytes(volume)
		if err != nil {
			return err
		}
		// Estimate tracked checkout bytes from the pinned object, without extracting
		// source. Leave the caller policy reserve after twice its bytes for checkout overhead.
		cmd := exec.CommandContext(ctx, "git", "ls-tree", "-rl", base)
		cmd.Dir = root
		data, err := cmd.Output()
		if err != nil {
			return fmt.Errorf("checkout size unavailable: %w", err)
		}
		var size int64
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			fields := strings.Fields(strings.SplitN(line, "\t", 2)[0])
			if len(fields) != 4 {
				return fmt.Errorf("checkout size entry unavailable")
			}
			if fields[1] != "blob" {
				continue
			}
			n, err := strconv.ParseInt(fields[3], 10, 64)
			if err != nil || n < 0 || n > (1<<60)-size {
				return fmt.Errorf("invalid checkout size")
			}
			size += n
		}
		if free < 2*size || free-2*size < reserve {
			return fmt.Errorf("insufficient disk headroom: free=%d checkout=%d reserve=%d", free, size, reserve)
		}
		publicationFence, err := store.Fence(ctx, leaseref.Record{ID: owner.LeaseID, Holder: holder, Generation: generation}, time.Now().UTC())
		if err != nil {
			return err
		}
		if !publicationFence.OK {
			return fmt.Errorf("publication fence refused: %s", publicationFence.Reason)
		}
		return ctx.Err()
	}
}

func worktreePreservingCLIIdentity(explicit map[string]bool, ownerPID int, leaseID string, sandbox bool) error {
	if !explicit["owner-pid"] || !explicit["lease-id"] || ownerPID <= 0 || ownerPID == os.Getpid() || strings.TrimSpace(leaseID) == "" || sandbox {
		return fmt.Errorf("preservation mode requires explicit nonempty lease ID and actual long-lived owner PID; sandbox translation is unsupported")
	}
	return nil
}

// Preservation output has no mutable CLI postprocessing. Intent publication was
// completed inside the primitive's target lock; failure never exports an env.
func worktreePreservingResultOut(res workerworktree.Result, capacity workerworktree.CapacityAdvisory, gate workerworktree.PreservationGate, owner workerworktree.OwnerStamp) worktreePrepareOut {
	out := worktreePrepareOut{Result: res, Capacity: capacity, PreserveExisting: true}
	if res.OK {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		var err error
		if gate == nil || !processalive.Check(owner.PID) {
			err = fmt.Errorf("publication authority or owner unavailable")
		} else {
			err = gate(ctx)
		}
		if err != nil {
			out.OK = false
			out.Code = "PRESERVATION_ADMISSION_REFUSED"
			_, statErr := os.Lstat(res.Path)
			out.Preserved = !os.IsNotExist(statErr)
			out.Reason = "publication refused: " + err.Error()
			return out
		}
		out.Env = workerworktree.WorktreeEnv(nil, res.Path)
	}
	return out
}

// Preservation telemetry never inspects cleanliness/lifecycle or recommends
// cleanup, even above the advisory setpoint or after a refusal.
func worktreePreservingCapacityAdvisory(census workerworktree.CapacityCensus, prospective int, reason string) workerworktree.CapacityAdvisory {
	return workerworktree.AssessCapacity(len(census.Paths), prospective, census.Known, reason, []workerworktree.RetainedTree{})
}
