package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/storagepressure"
)

const guardStoragePressureReason = "HOST_FREE_SPACE_BUDGET"

type guardStoragePressureDeps struct {
	getwd    func() (string, error)
	diskInfo func(string) (int64, int64, bool)
}

func defaultGuardStoragePressureDeps() guardStoragePressureDeps {
	return guardStoragePressureDeps{getwd: os.Getwd, diskInfo: compute.DiskInfo}
}

// runGuardStoragePressureGate performs only the fast filesystem-capacity probe.
// The fuller storage-pressure census stays a read-only operator follow-up because
// walking worktrees and caches on every launch would make the guard itself add I/O
// pressure during the condition it is trying to diagnose.
func runGuardStoragePressureGate(stderr io.Writer, deps guardStoragePressureDeps) int {
	root, err := deps.getwd()
	if err != nil || root == "" {
		fmt.Fprintf(stderr, "fak guard: storage headroom UNKNOWN (current directory unavailable: %v); continuing because unknown capacity is not evidence of disk full\n", err)
		return 0
	}
	if abs, absErr := filepath.Abs(root); absErr == nil {
		root = abs
	}
	total, free, known := deps.diskInfo(root)
	filesystem := storagepressure.AssessFilesystem(storagepressure.Filesystem{
		Path: root, TotalBytes: total, FreeBytes: free, Known: known,
		WarningFreeBytes: storagepressure.DefaultWarningFreeBytes,
		RefuseFreeBytes:  storagepressure.DefaultRefuseFreeBytes,
	})
	if !filesystem.Known {
		fmt.Fprintf(stderr, "fak guard: storage headroom UNKNOWN path=%q; continuing because unknown capacity is not evidence of disk full\n", root)
		return 0
	}
	if filesystem.Refuse {
		fmt.Fprintf(stderr, "fak guard: storage headroom REFUSE reason=%s path=%q free_bytes=%d reserve_bytes=%d; run `fak storage-pressure --root %q --json` for a read-only owner census before using an owner-specific cleanup command\n",
			guardStoragePressureReason, root, filesystem.FreeBytes, filesystem.RefuseFreeBytes, root)
		return 1
	}
	if filesystem.Warning {
		fmt.Fprintf(stderr, "fak guard: storage headroom WARNING path=%q free_bytes=%d warning_free_bytes=%d reserve_bytes=%d; run `fak storage-pressure --root %q --json` for a read-only owner census\n",
			root, filesystem.FreeBytes, filesystem.WarningFreeBytes, filesystem.RefuseFreeBytes, root)
	}
	return 0
}
