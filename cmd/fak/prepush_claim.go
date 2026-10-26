package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

const prepushClaimWaitBudget = 30 * time.Second

func prepushClaimPath(root, tip string) string {
	commonDir := prepushSuccessCommonDir(root)
	if commonDir == "" || tip == "" {
		return ""
	}
	return filepath.Join(commonDir, "fak-prepush-"+tip+".lock")
}

// claimPrepushTip coalesces cross-process checks for the same immutable tip.
// Contention is bounded and never reclaims a claim by age: a slow owner may still
// be validating. Keep its file open until release so identity cannot be recycled.
func claimPrepushTip(root, tip string, now func() time.Time) (bool, func(), error) {
	return claimPrepushTipWithReuse(root, tip, now, func() bool { return prepushSuccessReusable(root, tip, now()) })
}

func claimPrepushTipWithReuse(root, tip string, now func() time.Time, reusable func() bool, onWait ...func(string)) (bool, func(), error) {
	path := prepushClaimPath(root, tip)
	noop := func() {}
	if path == "" {
		return true, noop, nil
	}
	deadline := now().Add(prepushClaimWaitBudget)
	waitReported := false
	for {
		if reusable() {
			return false, noop, nil
		}
		f, err := toolproc.OpenExclusiveShareDelete(path, 0o644)
		if err == nil {
			info, statErr := f.Stat()
			var released sync.Once
			release := func() {
				released.Do(func() {
					// Preserve a claim already replaced before this check. This
					// is not atomic against external legacy age-based unlink;
					// quiesce legacy verifiers before coordinated rollout.
					if current, err := os.Stat(path); statErr == nil && err == nil && os.SameFile(info, current) {
						_ = os.Remove(path)
					}
					_ = f.Close()
				})
			}
			if statErr != nil {
				release()
				return false, noop, fmt.Errorf("inspect pre-push claim: %w", statErr)
			}
			if reusable() {
				release()
				return false, noop, nil
			}
			if _, err := fmt.Fprintf(f, "%d\n", os.Getpid()); err != nil {
				release()
				return false, noop, fmt.Errorf("write pre-push claim: %w", err)
			}
			return true, release, nil
		}
		if !os.IsExist(err) {
			return false, noop, fmt.Errorf("acquire pre-push claim: %w", err)
		}
		if reusable() {
			return false, noop, nil
		}
		if !waitReported {
			for _, report := range onWait {
				if report != nil {
					report(fmt.Sprintf("PREPUSH_WAITING: tip %s is being validated by another claimant; waiting up to %s on %s", tip, prepushClaimWaitBudget, path))
				}
			}
			waitReported = true
		}
		if !now().Before(deadline) {
			return false, noop, fmt.Errorf("PREPUSH_WAIT_TIMEOUT: waited %s for tip %s; claim preserved at %s; inspect its owner and retry after validation completes", prepushClaimWaitBudget, tip, path)
		}
		prepushSuccessSleep(100 * time.Millisecond)
	}
}
