package selfinstall

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// LaunchPriorPath is the one retained last-known-good executable for target.
// Keep the .exe suffix last on Windows so CreateProcess can execute the copy.
func LaunchPriorPath(target string) string {
	target = filepath.Clean(target)
	if strings.EqualFold(filepath.Ext(target), ".exe") {
		return strings.TrimSuffix(target, filepath.Ext(target)) + ".self-update-prior.exe"
	}
	return target + ".self-update-prior"
}

// BeginLaunchTransaction publishes a complete prior executable before the
// replacement window becomes visible to stable launchers.
func BeginLaunchTransaction(target string) (func(), error) {
	target, err := filepath.Abs(filepath.Clean(target))
	if err != nil {
		return nil, err
	}
	prior := LaunchPriorPath(target)
	// Reclaim old Windows swap-asides for the retained prior copy before
	// replacing it. Live prior launches remain protected by their owning PID.
	_ = ReapStaleAsides(prior, os.Getpid(), safecommit.ProcessAlive)

	staged, err := stageCopy(target, target, "launch-prior")
	if err != nil {
		return nil, fmt.Errorf("stage launch prior: %w", err)
	}
	if err := OSSwap(staged, prior); err != nil {
		_ = os.Remove(staged)
		return nil, fmt.Errorf("activate launch prior: %w", err)
	}
	if err := launchshim.WriteUpdateState(target, prior); err != nil {
		return nil, fmt.Errorf("publish launch state: %w", err)
	}

	var once sync.Once
	// Keep one prior copy after completion. A launcher may have read the state
	// and not reached exec yet; deleting it here would recreate the read-to-exec
	// race. The next transaction atomically refreshes this bounded copy.
	return func() {
		once.Do(func() { _ = os.Remove(launchshim.UpdateStatePath(target)) })
	}, nil
}
