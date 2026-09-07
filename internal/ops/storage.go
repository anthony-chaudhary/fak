package ops

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

// StorageReclaimResult captures the details of an executed storage GC pass.
type StorageReclaimResult struct {
	Tier0Bytes int64    `json:"tier0_bytes"`
	Tier1Bytes int64    `json:"tier1_bytes"`
	Tier2Bytes int64    `json:"tier2_bytes"`
	Tier3Bytes int64    `json:"tier3_bytes"`
	TotalBytes int64    `json:"total_bytes"`
	FilesCount int      `json:"files_count"`
	Actions    []string `json:"actions"`
}

// StorageManager orchestrates watermarks and the 4-tier reclamation cascade.
type StorageManager struct {
	RepoRoot     string
	Config       Config
	ProcessAlive func(int) bool
	DiskFree     func(string) (int64, error)
}

// NewStorageManager creates a StorageManager.
func NewStorageManager(repoRoot string, cfg Config) *StorageManager {
	return &StorageManager{
		RepoRoot:     repoRoot,
		Config:       cfg,
		ProcessAlive: defaultProcessAlive,
		DiskFree:     treedoctor.GoCacheFreeBytes,
	}
}

// EvaluateWatermark checks if free disk space falls below Warning or Refuse thresholds.
func (sm *StorageManager) EvaluateWatermark(freeBytes uint64) (warning bool, refuse bool) {
	if freeBytes <= sm.Config.RefuseFreeBytes {
		return true, true
	}
	if freeBytes <= sm.Config.WarningFreeBytes {
		return true, false
	}
	return false, false
}

// CheckDiskSpace evaluates free disk space on the filesystem containing RepoRoot,
// and evaluates the Warning and Refuse watermarks against Config.
func (sm *StorageManager) CheckDiskSpace() (freeBytes uint64, warning bool, refuse bool, err error) {
	if sm.RepoRoot == "" {
		return 0, false, false, nil
	}
	df := sm.DiskFree
	if df == nil {
		df = treedoctor.GoCacheFreeBytes
	}
	free, err := df(sm.RepoRoot)
	if err != nil {
		return 0, false, false, err
	}
	if free < 0 {
		free = 0
	}
	freeBytes = uint64(free)
	warning, refuse = sm.EvaluateWatermark(freeBytes)
	return freeBytes, warning, refuse, nil
}

// CheckDiskSpace evaluates free disk space and watermark status via the storage manager.
func (e *Engine) CheckDiskSpace() (freeBytes uint64, warning bool, refuse bool, err error) {
	if e == nil || e.Storage == nil {
		return 0, false, false, nil
	}
	return e.Storage.CheckDiskSpace()
}

// ReclaimCascade runs Tier 0 through Tier 3 reclamation.
// If dryRun is true, it only calculates reclaimable bytes without deleting.
func (sm *StorageManager) ReclaimCascade(ctx context.Context, dryRun bool) (StorageReclaimResult, error) {
	var res StorageReclaimResult

	// Tier 0: Root-level stray build binaries and compiler temp (_scratch/go-tmp).
	t0Bytes, t0Count, t0Actions, err := sm.reclaimTier0(ctx, dryRun)
	if err == nil {
		res.Tier0Bytes = t0Bytes
		res.FilesCount += t0Count
		res.Actions = append(res.Actions, t0Actions...)
	}

	// Tier 1: Expired producer scratch directories in _scratch/*
	t1Bytes, t1Count, t1Actions, err := sm.reclaimTier1(ctx, dryRun)
	if err == nil {
		res.Tier1Bytes = t1Bytes
		res.FilesCount += t1Count
		res.Actions = append(res.Actions, t1Actions...)
	}

	// Tier 2: Go build cache LRU prune down to low_bytes when high_bytes is exceeded.
	t2Bytes, t2Actions, err := sm.reclaimTier2(ctx, dryRun)
	if err == nil {
		res.Tier2Bytes = t2Bytes
		res.Actions = append(res.Actions, t2Actions...)
	}

	// Tier 3: External temp dumps (%TEMP%\opencode, /tmp/fak-*)
	t3Bytes, t3Count, t3Actions, err := sm.reclaimTier3(ctx, dryRun)
	if err == nil {
		res.Tier3Bytes = t3Bytes
		res.FilesCount += t3Count
		res.Actions = append(res.Actions, t3Actions...)
	}

	res.TotalBytes = res.Tier0Bytes + res.Tier1Bytes + res.Tier2Bytes + res.Tier3Bytes
	return res, nil
}

// reclaimTier0 cleans stray build binaries and old compiler temp directories.
func (sm *StorageManager) reclaimTier0(ctx context.Context, dryRun bool) (int64, int, []string, error) {
	var bytesReclaimed int64
	var count int
	var actions []string

	if sm.RepoRoot == "" {
		return 0, 0, nil, nil
	}

	// 1. Root stray binaries
	cmdDirSet := make(map[string]bool)
	entries, err := os.ReadDir(filepath.Join(sm.RepoRoot, "cmd"))
	if err == nil {
		for _, e := range entries {
			if e.IsDir() {
				cmdDirSet[e.Name()] = true
			}
		}
	}

	rootEntries, err := os.ReadDir(sm.RepoRoot)
	if err == nil {
		for _, e := range rootEntries {
			if !e.Type().IsRegular() {
				continue
			}
			name := e.Name()
			isBin := false
			if cmdDirSet[name] {
				isBin = true
			} else if strings.HasSuffix(name, ".exe") || strings.HasSuffix(name, ".exe~") {
				base := strings.TrimSuffix(strings.TrimSuffix(name, ".exe~"), ".exe")
				if cmdDirSet[base] {
					isBin = true
				}
			}
			if !isBin || name == "fak" || name == "fak.exe" {
				continue
			}

			// Verify it is git-ignored before touching
			ignored, _ := gitCheckIgnored(sm.RepoRoot, name)
			if !ignored {
				continue
			}

			info, err := e.Info()
			if err != nil {
				continue
			}
			size := info.Size()

			if !dryRun {
				if err := os.Remove(filepath.Join(sm.RepoRoot, name)); err == nil {
					bytesReclaimed += size
					count++
					actions = append(actions, fmt.Sprintf("pruned root binary %s (%d bytes)", name, size))
				}
			} else {
				bytesReclaimed += size
				count++
				actions = append(actions, fmt.Sprintf("dry-run: would prune root binary %s (%d bytes)", name, size))
			}
		}
	}

	// 2. Expired compiler temp _scratch/go-tmp
	goTmpDir := filepath.Join(sm.RepoRoot, "_scratch", "go-tmp")
	if tmpEntries, err := os.ReadDir(goTmpDir); err == nil {
		now := time.Now()
		for _, te := range tmpEntries {
			info, err := te.Info()
			if err != nil {
				continue
			}
			if now.Sub(info.ModTime()) > 1*time.Hour {
				p := filepath.Join(goTmpDir, te.Name())
				sz := dirSize(p)
				if !dryRun {
					if err := os.RemoveAll(p); err == nil {
						bytesReclaimed += sz
						count++
						actions = append(actions, fmt.Sprintf("pruned old go-tmp %s (%d bytes)", te.Name(), sz))
					}
				} else {
					bytesReclaimed += sz
					count++
					actions = append(actions, fmt.Sprintf("dry-run: would prune old go-tmp %s (%d bytes)", te.Name(), sz))
				}
			}
		}
	}

	return bytesReclaimed, count, actions, nil
}

// isScratchDirActive checks whether a scratch directory is actively locked or in use
// by inspecting lock files, PID files, lease files, and owner stamps.
func (sm *StorageManager) isScratchDirActive(dirPath string, now time.Time) bool {
	processAlive := sm.ProcessAlive
	if processAlive == nil {
		processAlive = defaultProcessAlive
	}

	active := false
	maxEntries := 2000
	seen := 0

	_ = filepath.WalkDir(dirPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			active = true
			return filepath.SkipAll
		}
		if path == dirPath {
			return nil
		}
		seen++
		if seen > maxEntries {
			return nil
		}
		if d.IsDir() {
			return nil
		}

		name := strings.ToLower(d.Name())
		info, err := d.Info()
		if err != nil {
			active = true
			return filepath.SkipAll
		}

		// 1. Check for PID files (*.pid or pid)
		if strings.HasSuffix(name, ".pid") || name == "pid" {
			data, readErr := os.ReadFile(path)
			if readErr == nil {
				pidStr := strings.TrimSpace(string(data))
				if pid, convErr := strconv.Atoi(pidStr); convErr == nil && pid > 0 {
					if processAlive(pid) {
						active = true
						return filepath.SkipAll
					}
				}
			}
		}

		// 2. Check for lock files (*.lock or lock)
		if strings.HasSuffix(name, ".lock") || name == "lock" {
			// Try non-blocking advisory flock to detect active holders
			if f, openErr := os.OpenFile(path, os.O_RDWR, 0); openErr == nil {
				lockErr := flock.TryLock(f)
				if errors.Is(lockErr, flock.ErrLockBusy) {
					_ = f.Close()
					active = true
					return filepath.SkipAll
				}
				if lockErr == nil {
					_ = flock.Unlock(f)
				}
				_ = f.Close()
			} else if os.IsPermission(openErr) {
				active = true
				return filepath.SkipAll
			}

			// Check if the lock file records a live PID
			data, readErr := os.ReadFile(path)
			if readErr == nil {
				pidStr := strings.TrimSpace(string(data))
				if pid, convErr := strconv.Atoi(pidStr); convErr == nil && pid > 0 {
					if processAlive(pid) {
						active = true
						return filepath.SkipAll
					}
				}
			}

			// If lock file has no PID or unparseable, check if modified recently (within 15 minutes)
			if now.Sub(info.ModTime()) < 15*time.Minute {
				active = true
				return filepath.SkipAll
			}
		}

		// 3. Check for lease.json or owner.json / *.stamp
		if name == "lease.json" || name == "owner.json" || strings.HasSuffix(name, ".stamp") {
			data, readErr := os.ReadFile(path)
			if readErr == nil {
				var record struct {
					PID         int       `json:"pid"`
					HeartbeatTS time.Time `json:"heartbeat_ts"`
				}
				if json.Unmarshal(data, &record) == nil {
					if record.PID > 0 && processAlive(record.PID) {
						active = true
						return filepath.SkipAll
					}
					if !record.HeartbeatTS.IsZero() && now.Sub(record.HeartbeatTS) < 15*time.Minute {
						active = true
						return filepath.SkipAll
					}
				}
			}
		}

		return nil
	})

	return active
}

// reclaimTier1 sweeps expired producer scratch directories in _scratch/*
func (sm *StorageManager) reclaimTier1(ctx context.Context, dryRun bool) (int64, int, []string, error) {
	var bytesReclaimed int64
	var count int
	var actions []string

	if sm.RepoRoot == "" {
		return 0, 0, nil, nil
	}

	scratchDir := filepath.Join(sm.RepoRoot, "_scratch")
	entries, err := os.ReadDir(scratchDir)
	if err != nil {
		return 0, 0, nil, nil
	}

	ttl := sm.Config.ScratchTTL
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	now := time.Now()

	for _, e := range entries {
		if !e.IsDir() || e.Name() == "go-tmp" || e.Name() == "gotmp" || strings.HasPrefix(e.Name(), "fak-worker-wt") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}

		// Reclaim if older than TTL and not actively locked or in use
		if now.Sub(info.ModTime()) > ttl {
			p := filepath.Join(scratchDir, e.Name())
			if sm.isScratchDirActive(p, now) {
				continue
			}
			sz := dirSize(p)
			if !dryRun {
				if err := os.RemoveAll(p); err == nil {
					bytesReclaimed += sz
					count++
					actions = append(actions, fmt.Sprintf("pruned expired scratch %s (%d bytes)", e.Name(), sz))
				}
			} else {
				bytesReclaimed += sz
				count++
				actions = append(actions, fmt.Sprintf("dry-run: would prune expired scratch %s (%d bytes)", e.Name(), sz))
			}
		}
	}

	return bytesReclaimed, count, actions, nil
}

// reclaimTier2 runs Go build cache LRU prune down to low_bytes.
func (sm *StorageManager) reclaimTier2(ctx context.Context, dryRun bool) (int64, []string, error) {
	cacheRoot := treedoctor.GoCacheRootFromEnv(os.Getenv, os.UserCacheDir)
	if cacheRoot == "" {
		return 0, nil, nil
	}

	opts := treedoctor.GoCacheOptions{
		Root:      cacheRoot,
		Now:       time.Now(),
		HighBytes: int64(sm.Config.GoCacheHighBytes),
		LowBytes:  int64(sm.Config.GoCacheLowBytes),
		MinAge:    sm.Config.GoCacheMinAge,
		Context:   ctx,
	}
	if dryRun {
		opts.Remove = func(string) error { return nil }
	}

	rep := treedoctor.SweepGoCache(opts, !dryRun)
	if rep.ReclaimedBytes > 0 {
		return rep.ReclaimedBytes, []string{fmt.Sprintf("pruned Go build cache (%d bytes reclaimed)", rep.ReclaimedBytes)}, nil
	}
	return 0, nil, nil
}

// reclaimTier3 cleans external stale temp dumps (%TEMP%\opencode, /tmp/fak-*).
func (sm *StorageManager) reclaimTier3(ctx context.Context, dryRun bool) (int64, int, []string, error) {
	var bytesReclaimed int64
	var count int
	var actions []string

	tempRoots := []string{}
	if runtime.GOOS == "windows" {
		if t := os.Getenv("TEMP"); t != "" {
			tempRoots = append(tempRoots, filepath.Join(t, "opencode"))
		}
	} else {
		tempRoots = append(tempRoots, "/tmp")
	}

	now := time.Now()
	for _, tr := range tempRoots {
		if tr == "/tmp" {
			// clean /tmp/fak-* older than 24h
			entries, err := os.ReadDir(tr)
			if err != nil {
				continue
			}
			for _, e := range entries {
				if strings.HasPrefix(e.Name(), "fak-") {
					info, err := e.Info()
					if err == nil && now.Sub(info.ModTime()) > 24*time.Hour {
						p := filepath.Join(tr, e.Name())
						sz := dirSize(p)
						if !dryRun {
							_ = os.RemoveAll(p)
						}
						bytesReclaimed += sz
						count++
						actions = append(actions, fmt.Sprintf("pruned temp %s (%d bytes)", p, sz))
					}
				}
			}
		} else {
			// clean whole directory if older than 24h
			entries, err := os.ReadDir(tr)
			if err != nil {
				continue
			}
			for _, e := range entries {
				info, err := e.Info()
				if err == nil && now.Sub(info.ModTime()) > 24*time.Hour {
					p := filepath.Join(tr, e.Name())
					sz := dirSize(p)
					if !dryRun {
						_ = os.RemoveAll(p)
					}
					bytesReclaimed += sz
					count++
					actions = append(actions, fmt.Sprintf("pruned temp %s (%d bytes)", p, sz))
				}
			}
		}
	}

	return bytesReclaimed, count, actions, nil
}

func gitCheckIgnored(repoRoot, name string) (bool, error) {
	cmd := exec.Command("git", "check-ignore", "-q", "--", name)
	cmd.Dir = repoRoot
	configureDispatchHelperCommand(cmd)
	err := cmd.Run()
	if err == nil {
		return true, nil
	}
	if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
		return false, nil
	}
	return false, err
}

func dirSize(path string) int64 {
	var size int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}
