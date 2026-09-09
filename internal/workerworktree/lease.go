package workerworktree

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

const (
	// WorkerLeaseFileName is the standard lease file written inside each worker worktree.
	WorkerLeaseFileName = "lease.json"

	// DefaultHeartbeatStaleThreshold is the cutoff after which a worktree with an un-updated
	// heartbeat is considered stale and eligible for reaping (#11239, #11508).
	DefaultHeartbeatStaleThreshold = 15 * time.Minute
)

// WorkerLease represents the durable heartbeat lease stored inside each worker worktree (#11239).
type WorkerLease struct {
	PID         int       `json:"pid"`
	SessionID   string    `json:"session_id"`
	CreatedAt   time.Time `json:"created_at"`
	HeartbeatTS time.Time `json:"heartbeat_ts"`
}

// WriteWorkerLease writes lease.json inside wtPath atomically.
func WriteWorkerLease(wtPath string, lease WorkerLease) error {
	if wtPath == "" {
		return fmt.Errorf("worker worktree path cannot be empty")
	}
	if lease.PID <= 0 {
		lease.PID = os.Getpid()
	}
	now := time.Now().UTC()
	if lease.CreatedAt.IsZero() {
		lease.CreatedAt = now
	}
	if lease.HeartbeatTS.IsZero() {
		lease.HeartbeatTS = now
	}
	data, err := json.MarshalIndent(lease, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal worker lease: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(wtPath, 0o755); err != nil {
		return fmt.Errorf("create worker worktree dir: %w", err)
	}
	target := filepath.Join(wtPath, WorkerLeaseFileName)
	tmp := target + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write temp worker lease: %w", err)
	}
	var renameErr error
	for attempt := 0; attempt < 10; attempt++ {
		renameErr = os.Rename(tmp, target)
		if renameErr == nil {
			return nil
		}
		_ = os.Remove(target)
		time.Sleep(25 * time.Millisecond)
	}
	_ = os.Remove(tmp)
	return fmt.Errorf("replace worker lease: %w", renameErr)
}

// ReadWorkerLease reads and parses lease.json from wtPath.
func ReadWorkerLease(wtPath string) (WorkerLease, error) {
	var lease WorkerLease
	if wtPath == "" {
		return lease, fmt.Errorf("worker worktree path cannot be empty")
	}
	data, err := os.ReadFile(filepath.Join(wtPath, WorkerLeaseFileName))
	if err != nil {
		return lease, err
	}
	if err := json.Unmarshal(data, &lease); err != nil {
		return lease, fmt.Errorf("unmarshal worker lease: %w", err)
	}
	return lease, nil
}

// UpdateHeartbeat updates the heartbeat_ts in lease.json of wtPath to the current UTC time.
func UpdateHeartbeat(wtPath string) error {
	lease, err := ReadWorkerLease(wtPath)
	if err != nil {
		return err
	}
	lease.HeartbeatTS = time.Now().UTC()
	return WriteWorkerLease(wtPath, lease)
}

// ensureGitExclude appends pattern to the git info/exclude file so metadata like
// lease.json is excluded from git status and untracked file listings.
func ensureGitExclude(root, pattern string) {
	if root == "" || pattern == "" {
		return
	}
	excludePath := filepath.Join(root, ".git", "info", "exclude")
	if fi, err := os.Stat(filepath.Join(root, ".git")); err == nil && !fi.IsDir() {
		if data, err := os.ReadFile(filepath.Join(root, ".git")); err == nil {
			s := strings.TrimSpace(string(data))
			if strings.HasPrefix(s, "gitdir: ") {
				gd := strings.TrimPrefix(s, "gitdir: ")
				if !filepath.IsAbs(gd) {
					gd = filepath.Join(root, gd)
				}
				excludePath = filepath.Join(gd, "info", "exclude")
			}
		}
	}
	if data, err := os.ReadFile(excludePath); err == nil {
		if strings.Contains(string(data), pattern) {
			return
		}
	}
	_ = os.MkdirAll(filepath.Dir(excludePath), 0o755)
	f, err := os.OpenFile(excludePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err == nil {
		_, _ = f.WriteString("\n" + pattern + "\n")
		_ = f.Close()
	}
}

// DeadWorktreeSweepReport records the outcome of a dead worktree sweep.
type DeadWorktreeSweepReport struct {
	Inspected int      `json:"inspected"`
	Pruned    int      `json:"pruned"`
	Unlocked  int      `json:"unlocked"`
	Paths     []string `json:"paths,omitempty"`
	PruneErr  string   `json:"prune_err,omitempty"`
}

// isWorktreeDirty reports whether wtPath has uncommitted git changes.
// Metadata files like lease.json are ignored.
func isWorktreeDirty(wtPath string, git GitRunner) bool {
	if wtPath == "" {
		return false
	}
	rc, out := run(git, wtPath, []string{"status", "--porcelain"})
	if rc != 0 {
		return false
	}
	return strings.TrimSpace(cleanStatusWithoutLease(out)) != ""
}

// SweepDeadWorktrees runs a non-blocking sweep of managed worker worktrees.
// It checks .git/worktrees/fak-worker-wt-*, _scratch/fak-worker-wt-*, and wtRoot.
// Only a dead recorded PID or a missing checkout authorizes removal. An old
// heartbeat or creation time does not prove death; unknown owners are kept.
// Active processes and dirty worktrees are always preserved.
func SweepDeadWorktrees(root, wtRoot string, git GitRunner) DeadWorktreeSweepReport {
	var report DeadWorktreeSweepReport
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cleanupGit := git
	if cleanupGit == nil {
		cleanupGit = BoundedGitRunner(ctx)
	}

	gitCommon := filepath.Join(root, ".git")
	if fi, err := os.Stat(gitCommon); err == nil && !fi.IsDir() {
		if data, err := os.ReadFile(gitCommon); err == nil {
			s := strings.TrimSpace(string(data))
			if strings.HasPrefix(s, "gitdir: ") {
				gd := strings.TrimPrefix(s, "gitdir: ")
				if !filepath.IsAbs(gd) {
					gd = filepath.Join(root, gd)
				}
				if filepath.Base(filepath.Dir(gd)) == "worktrees" {
					gitCommon = filepath.Dir(filepath.Dir(gd))
				} else {
					gitCommon = gd
				}
			}
		}
	}

	worktreesDir := filepath.Join(gitCommon, "worktrees")
	if entries, err := os.ReadDir(worktreesDir); err == nil {
		for _, entry := range entries {
			if !entry.IsDir() || !IsWorkerWorktree(entry.Name()) {
				continue
			}
			report.Inspected++
			wtAdminDir := filepath.Join(worktreesDir, entry.Name())
			gitdirFile := filepath.Join(wtAdminDir, "gitdir")
			lockedFile := filepath.Join(wtAdminDir, "locked")

			var wtPath string
			var rawGitdir string
			if content, err := os.ReadFile(gitdirFile); err == nil {
				rawGitdir = strings.TrimSpace(string(content))
				rawGitdir = strings.TrimPrefix(rawGitdir, "gitdir: ")
				rawGitdir = strings.TrimSpace(rawGitdir)
				wtPath = filepath.Dir(rawGitdir)
			}

			dead := false
			stale := false

			if isForeignPlatformRegistration(rawGitdir, wtPath) {
				// Foreign-platform worktree that cannot be stat-ed locally.
				// Preserve the registration so cross-platform worker checkouts are not wiped (#11814).
				continue
			}

			if wtPath == "" {
				dead = true
			} else if _, err := os.Stat(wtPath); os.IsNotExist(err) {
				dead = true
			} else {
				// Do not reap idle pool members
				if record, err := readPoolMember(wtPath); err == nil && record.State == poolStateIdle {
					continue
				}
				if lease, lerr := ReadWorkerLease(wtPath); lerr == nil {
					if lease.PID > 0 && processalive.Check(lease.PID) {
						continue
					}
					if lease.PID > 0 && !processalive.Check(lease.PID) {
						dead = true
					}
					if !lease.HeartbeatTS.IsZero() && time.Since(lease.HeartbeatTS) > DefaultHeartbeatStaleThreshold {
						if lease.PID <= 0 || !processalive.Check(lease.PID) {
							stale = true
						}
					}
				}
				if stamp, serr := readOwnerStamp(wtPath); serr == nil {
					if stamp.PID > 0 && processalive.Check(stamp.PID) {
						continue
					}
					if stamp.PID > 0 && !processalive.Check(stamp.PID) {
						dead = true
					}
					if !stamp.CreatedAt.IsZero() && time.Since(stamp.CreatedAt) > DefaultHeartbeatStaleThreshold {
						if stamp.PID <= 0 || !processalive.Check(stamp.PID) {
							stale = true
						}
					}
				}
			}

			if dead || stale {
				if alive, ok := OwnerProcessLive(wtPath, processalive.Check); ok && alive {
					continue
				}
				if lease, lerr := ReadWorkerLease(wtPath); lerr == nil && lease.PID > 0 && processalive.Check(lease.PID) {
					continue
				}
				if stamp, serr := readOwnerStamp(wtPath); serr == nil && stamp.PID > 0 && processalive.Check(stamp.PID) {
					continue
				}
				if isWorktreeDirty(wtPath, cleanupGit) {
					continue
				}
				_ = os.Remove(lockedFile)
				run(cleanupGit, root, []string{"worktree", "unlock", entry.Name()})
				if wtPath != "" {
					run(cleanupGit, root, []string{"worktree", "unlock", wtPath})
					_ = os.RemoveAll(wtPath)
					_ = os.Remove(OwnerStampPath(wtPath))
				}
				_ = os.RemoveAll(wtAdminDir)
				report.Pruned++
				report.Unlocked++
				if wtPath != "" {
					report.Paths = append(report.Paths, wtPath)
				}
			}
		}
	}

	var scratchDirs []string
	if root != "" {
		scratchDirs = append(scratchDirs, filepath.Join(root, "_scratch"))
	}
	if wtRoot != "" && wtRoot != filepath.Join(root, "_scratch") {
		scratchDirs = append(scratchDirs, wtRoot)
	}

	for _, sdir := range scratchDirs {
		entries, err := os.ReadDir(sdir)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if !entry.IsDir() || !IsWorkerWorktree(entry.Name()) {
				continue
			}
			wtPath := filepath.Join(sdir, entry.Name())
			if record, err := readPoolMember(wtPath); err == nil && record.State == poolStateIdle {
				continue
			}
			dead := false
			stale := false
			if lease, lerr := ReadWorkerLease(wtPath); lerr == nil {
				if lease.PID > 0 && processalive.Check(lease.PID) {
					continue
				}
				if lease.PID > 0 && !processalive.Check(lease.PID) {
					dead = true
				}
				if !lease.HeartbeatTS.IsZero() && time.Since(lease.HeartbeatTS) > DefaultHeartbeatStaleThreshold {
					if lease.PID <= 0 || !processalive.Check(lease.PID) {
						stale = true
					}
				}
			}
			if stamp, serr := readOwnerStamp(wtPath); serr == nil {
				if stamp.PID > 0 && processalive.Check(stamp.PID) {
					continue
				}
				if stamp.PID > 0 && !processalive.Check(stamp.PID) {
					dead = true
				}
				if !stamp.CreatedAt.IsZero() && time.Since(stamp.CreatedAt) > DefaultHeartbeatStaleThreshold {
					if stamp.PID <= 0 || !processalive.Check(stamp.PID) {
						stale = true
					}
				}
			}
			if dead || stale {
				if alive, ok := OwnerProcessLive(wtPath, processalive.Check); ok && alive {
					continue
				}
				if lease, lerr := ReadWorkerLease(wtPath); lerr == nil && lease.PID > 0 && processalive.Check(lease.PID) {
					continue
				}
				if stamp, serr := readOwnerStamp(wtPath); serr == nil && stamp.PID > 0 && processalive.Check(stamp.PID) {
					continue
				}
				if isWorktreeDirty(wtPath, cleanupGit) {
					continue
				}
				run(cleanupGit, root, []string{"worktree", "unlock", wtPath})
				run(cleanupGit, root, []string{"worktree", "unlock", entry.Name()})
				_ = os.RemoveAll(wtPath)
				_ = os.Remove(OwnerStampPath(wtPath))
				report.Pruned++
				report.Paths = append(report.Paths, wtPath)
			}
		}
	}

	if report.Pruned > 0 {
		if rc, out := run(cleanupGit, root, []string{"worktree", "prune", "--expire", "now"}); rc != 0 {
			report.PruneErr = strings.TrimSpace(out)
		}
	}

	return report
}

func sweepDeadWorktrees(root, wtRoot string, git GitRunner) {
	_ = DebouncedSweepDeadWorktrees(root, wtRoot, git)
}

// isForeignPlatformRegistration reports whether rawGitdir or wtPath represents a
// foreign-platform path that cannot be stat-ed locally or is from a foreign OS
// namespace (e.g. POSIX /mnt/... or paths without a Windows volume on Windows,
// or Windows drive letter / backslash paths on Linux/macOS) (#11813, #11814).
func isForeignPlatformRegistration(rawGitdir, wtPath string) bool {
	if rawGitdir == "" && wtPath == "" {
		return false
	}
	if !isForeignOSPath(rawGitdir) && !isForeignOSPath(wtPath) {
		return false
	}
	// The path belongs to a foreign OS namespace. If it cannot be translated
	// or is from a foreign OS namespace, preserve it rather than treating it as dead (#11813).
	if translated, ok := translateForeignPath(rawGitdir); ok {
		if canStatLocally(translated) || canStatLocally(filepath.Dir(translated)) {
			// Even if the translated path exists on disk, it belongs to a foreign
			// OS worker whose PID and git metadata cannot be safely verified locally.
			return true
		}
	}
	if translated, ok := translateForeignPath(wtPath); ok {
		if canStatLocally(translated) {
			return true
		}
	}
	return true
}

// isForeignOSPath reports whether p looks like a path from a foreign operating system
// (e.g. on Linux/Unix, starts with a Windows drive letter ^[a-zA-Z]:[/\\] or contains '\\';
// on Windows, starts with '/' or doesn't have a Windows volume) (#11813).
func isForeignOSPath(p string) bool {
	if p == "" {
		return false
	}
	if runtime.GOOS == "windows" {
		// On Windows, a foreign-platform path starts with '/' or doesn't have a Windows volume.
		if strings.HasPrefix(p, "/") || !hasWindowsVolume(p) {
			return true
		}
		return false
	}
	// On Linux/Unix, a foreign path starts with a Windows drive letter, contains '\\', or starts with '\\\\'.
	if isWindowsDrivePath(p) || strings.Contains(p, `\`) || strings.HasPrefix(p, `\\`) {
		return true
	}
	return false
}

// translateForeignPath attempts to translate a foreign-platform path to the local platform path.
// On Windows, it translates /mnt/<drive>/... to <drive>:\...
// On Linux/macOS, it translates <drive>:\... or <drive>:/... to /mnt/<drive>/...
// Returns ("", false) if translation is not possible (#11813).
func translateForeignPath(p string) (string, bool) {
	if p == "" {
		return "", false
	}
	if runtime.GOOS == "windows" {
		normalized := filepath.ToSlash(p)
		if strings.HasPrefix(normalized, "/mnt/") && len(normalized) >= 6 && isDriveLetter(normalized[5]) {
			if len(normalized) == 6 || normalized[6] == '/' {
				drive := strings.ToUpper(string(normalized[5])) + ":"
				rest := ""
				if len(normalized) > 7 {
					rest = normalized[7:]
				}
				return filepath.Join(drive+`\`, filepath.FromSlash(rest)), true
			}
		}
		return "", false
	}
	if isWindowsDrivePath(p) {
		drive := strings.ToLower(string(p[0]))
		rest := p[2:]
		rest = strings.TrimPrefix(rest, `\`)
		rest = strings.TrimPrefix(rest, `/`)
		rest = strings.ReplaceAll(rest, `\`, `/`)
		return "/mnt/" + drive + "/" + rest, true
	}
	return "", false
}

func isWindowsAbsolutePath(p string) bool {
	return isWindowsDrivePath(p) || strings.HasPrefix(p, `\\`)
}

func isWindowsDrivePath(p string) bool {
	return len(p) >= 3 && isDriveLetter(p[0]) && p[1] == ':' && (p[2] == '/' || p[2] == '\\')
}

func hasWindowsVolume(p string) bool {
	if len(p) >= 2 && isDriveLetter(p[0]) && p[1] == ':' {
		return true
	}
	if strings.HasPrefix(p, `\\`) || strings.HasPrefix(p, "//") {
		return true
	}
	return false
}

func isDriveLetter(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func canStatLocally(p string) bool {
	if p == "" {
		return false
	}
	_, err := os.Stat(p)
	return err == nil
}
