package selfinstall

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ErrBusy is returned by TrySingleFlight when another self-update already holds the lock.
var ErrBusy = errors.New("selfinstall: another self-update is in progress")

// TrySingleFlight takes a NON-BLOCKING advisory lock so at most one self-update builds at a
// time on a host. A second concurrent invocation returns ErrBusy immediately instead of
// stacking another expensive origin checkout + build — critical on a saturated box where the
// scheduled tick could otherwise pile builds on top of a slow one. The returned release frees
// the lock; the OS also drops it if the process exits. dir is where the lockfile lives (""
// => OS temp); the lock file is named fak-selfupdate.lock there.
func TrySingleFlight(dir string) (release func(), err error) {
	if dir == "" {
		dir = os.TempDir()
	}
	path := filepath.Join(dir, "fak-selfupdate.lock")
	f, oerr := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if oerr != nil {
		return nil, oerr
	}
	if lerr := flock.TryLock(f); lerr != nil {
		f.Close()
		if errors.Is(lerr, flock.ErrLockBusy) {
			return nil, ErrBusy
		}
		return nil, lerr
	}
	return func() { _ = flock.Unlock(f); _ = f.Close() }, nil
}

// RealRunner runs the command for real, merging stdout+stderr, and reports ok=false on any
// non-zero exit or exec failure (so a failed gate is a clean ok=false, not a panic).
//
// Every child it spawns (git fetch/rev-parse/worktree add, go build, the target's `version`
// smoke) is a console-prone tool, and self-update's caller is a windowless scheduled task
// (`conhost --headless cmd.exe /c fak self-update …`, every N minutes). Without the no-window
// hook those children allocate their own conhost and FLASH a foreground console window each
// tick — the `--headless` wrapper only covers the top process, not descendants it spawns.
// windowgate.ConfigureBackgroundCommand sets HideWindow + CREATE_NO_WINDOW (a no-op off
// Windows), so the whole self-update subprocess tree stays off the desktop.
func RealRunner(ctx context.Context, dir, name string, args ...string) (string, bool) {
	return runCommandWithEnv(ctx, dir, name, args, goRunnerEnv(name, os.Environ(), os.TempDir(), ""))
}

type commandEnvRunner func(ctx context.Context, dir, name string, args []string, env []string) (string, bool)

func runCommandWithEnv(ctx context.Context, dir, name string, args []string, env []string) (string, bool) {
	cmd := exec.CommandContext(ctx, name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	if env != nil {
		cmd.Env = env
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// goRunnerEnv keeps self-update's compiler work outside repo scratch that another session may
// sweep. cacheDir, when non-empty, also isolates the compiler build cache from ambient `go
// clean -cache` and developer cleanup. Non-Go children inherit the caller environment unchanged.
func goRunnerEnv(name string, env []string, tempDir, cacheDir string) []string {
	base := filepath.Base(name)
	if !strings.EqualFold(base, "go") && !strings.EqualFold(base, "go.exe") {
		return nil
	}
	out := make([]string, 0, len(env)+2)
	for _, entry := range env {
		key, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(key, "GOTMPDIR") || (cacheDir != "" && strings.EqualFold(key, "GOCACHE")) {
			continue
		}
		out = append(out, entry)
	}
	out = append(out, "GOTMPDIR="+tempDir)
	if cacheDir != "" {
		out = append(out, "GOCACHE="+cacheDir)
	}
	return out
}

// NewSelfUpdateRunner returns one runner for the complete self-update build/vet transaction.
// Its Go children share a stable fak-owned compiler cache, preserving warm builds without
// depending on the ambient GOCACHE that developer cleanup tools commonly remove. If that
// owned cache is itself removed while a Go command is running, the runner retries that command
// exactly once in a fresh fixed recovery cache and uses that cache for the rest of the
// transaction. cleanup removes recovery state; the stable cache remains warm and Go's own
// age-based cache trimming reclaims old entries. Fixed paths ensure a killed attempt cannot
// leak one cache directory per invocation.
func NewSelfUpdateRunner() (Runner, func(), error) {
	cacheBase, err := os.UserCacheDir()
	if err != nil || strings.TrimSpace(cacheBase) == "" {
		cacheBase = os.TempDir()
	}
	primary, recovery := selfUpdateGoCachePaths(cacheBase, os.TempDir())
	report := func(counts GoCacheOutcomeCounts) {
		fmt.Fprintln(os.Stderr, FormatGoCacheOutcomeCounts(counts))
	}
	return newGoCacheRunner(primary, recovery, os.TempDir(), runCommandWithEnv, report)
}

func selfUpdateGoCachePaths(cacheBase, tempDir string) (primary, recovery string) {
	return filepath.Join(cacheBase, "fak", "self-update", "go-build-v1"),
		filepath.Join(tempDir, "fak-self-update-go-build-recovery-v1")
}

// GoCacheOutcomeCounts summarizes commands run through the self-update-owned Go cache.
// Success means the command completed, Refusal means cache lifecycle recovery was
// exhausted or unavailable, and Error means the command failed for another reason.
type GoCacheOutcomeCounts struct {
	Success int
	Refusal int
	Error   int
}

// FormatGoCacheOutcomeCounts renders the existing self-update progress readout.
func FormatGoCacheOutcomeCounts(counts GoCacheOutcomeCounts) string {
	return fmt.Sprintf("self-update: go-cache outcomes success=%d refusal=%d error=%d",
		counts.Success, counts.Refusal, counts.Error)
}

type goCacheRunner struct {
	mu           sync.Mutex
	primary      string
	recovery     string
	tempDir      string
	active       string
	recoveryUsed bool
	counts       GoCacheOutcomeCounts
	report       func(GoCacheOutcomeCounts)
	run          commandEnvRunner
}

func newGoCacheRunner(primary, recovery, tempDir string, run commandEnvRunner, reports ...func(GoCacheOutcomeCounts)) (Runner, func(), error) {
	primary = filepath.Clean(primary)
	recovery = filepath.Clean(recovery)
	if primary == "." || recovery == "." || primary == recovery {
		return nil, func() {}, fmt.Errorf("selfinstall: invalid Go build-cache paths")
	}
	if err := os.MkdirAll(primary, 0o755); err != nil {
		return nil, func() {}, fmt.Errorf("selfinstall: create update-owned Go build cache %s: %w", primary, err)
	}
	// A hard-killed prior attempt can leave the fixed recovery cache behind. Single-flight
	// excludes a live peer, so startup is the safe point to reclaim that bounded residue.
	if err := os.RemoveAll(recovery); err != nil {
		return nil, func() {}, fmt.Errorf("selfinstall: clean stale Go build-cache recovery %s: %w", recovery, err)
	}
	r := &goCacheRunner{primary: primary, recovery: recovery, tempDir: tempDir, active: primary, run: run}
	if len(reports) != 0 {
		r.report = reports[0]
	}
	cleanup := func() { _ = os.RemoveAll(recovery) }
	return r.runCommand, cleanup, nil
}

func (r *goCacheRunner) runCommand(ctx context.Context, dir, name string, args ...string) (string, bool) {
	base := filepath.Base(name)
	if !strings.EqualFold(base, "go") && !strings.EqualFold(base, "go.exe") {
		return r.run(ctx, dir, name, args, nil)
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if err := os.MkdirAll(r.active, 0o755); err != nil {
		r.recordOutcome(&r.counts.Refusal)
		return fmt.Sprintf("self-update Go build cache %s is unavailable: %v; stop concurrent cache cleanup and rerun fak self-update", r.active, err), false
	}
	env := goRunnerEnv(name, os.Environ(), r.tempDir, r.active)
	out, ok := r.run(ctx, dir, name, args, env)
	if ok {
		r.recordOutcome(&r.counts.Success)
		return out, true
	}
	if !goCacheLifecycleFailure(out, r.active) {
		r.recordOutcome(&r.counts.Error)
		return out, false
	}
	if r.recoveryUsed {
		r.recordOutcome(&r.counts.Refusal)
		return appendGoCacheDiagnostic(out, fmt.Sprintf("Go build cache %s became unavailable after the one bounded recovery; stop concurrent cache cleanup and rerun fak self-update", r.active)), false
	}

	failedCache := r.active
	r.recoveryUsed = true
	if err := os.RemoveAll(r.recovery); err != nil {
		r.recordOutcome(&r.counts.Refusal)
		return appendGoCacheDiagnostic(out, fmt.Sprintf("Go build cache %s became unavailable and the one bounded recovery could not clean %s: %v; stop concurrent cache cleanup and rerun fak self-update", failedCache, r.recovery, err)), false
	}
	if err := os.MkdirAll(r.recovery, 0o755); err != nil {
		r.recordOutcome(&r.counts.Refusal)
		return appendGoCacheDiagnostic(out, fmt.Sprintf("Go build cache %s became unavailable and the one bounded recovery could not create %s: %v; stop concurrent cache cleanup and rerun fak self-update", failedCache, r.recovery, err)), false
	}
	r.active = r.recovery
	retryEnv := goRunnerEnv(name, os.Environ(), r.tempDir, r.active)
	retryOut, retryOK := r.run(ctx, dir, name, args, retryEnv)
	if retryOK {
		r.recordOutcome(&r.counts.Success)
		return retryOut, true
	}
	r.recordOutcome(&r.counts.Refusal)
	detail := fmt.Sprintf("Go build cache %s became unavailable; one recovery attempt used fresh cache %s, but the retried command failed and will not be retried", failedCache, r.recovery)
	return appendGoCacheDiagnostic(retryOut, detail), false
}

func (r *goCacheRunner) recordOutcome(count *int) {
	*count++
	if r.report != nil {
		r.report(r.counts)
	}
}

func goCacheLifecycleFailure(out, cacheDir string) bool {
	normalizedOut := normalizeCacheDiagnostic(out)
	normalizedCache := normalizeCacheDiagnostic(filepath.Clean(cacheDir))
	if normalizedCache == "" || normalizedCache == "." || !strings.Contains(normalizedOut, normalizedCache) {
		return false
	}
	for _, signal := range []string{
		"no such file or directory",
		"cannot find the file",
		"cannot find the path",
		"file does not exist",
		"not a directory",
		"failed to initialize build cache",
		"build cache is required",
		"permission denied",
		"access is denied",
	} {
		if strings.Contains(normalizedOut, signal) {
			return true
		}
	}
	return false
}

func normalizeCacheDiagnostic(value string) string {
	// Go can report Windows paths even when a cross-platform unit test runs on Unix. Normalize
	// both separator forms explicitly instead of relying only on the host-specific ToSlash.
	return strings.ToLower(strings.ReplaceAll(filepath.ToSlash(value), `\`, "/"))
}

func appendGoCacheDiagnostic(out, diagnostic string) string {
	out = strings.TrimSpace(out)
	if out == "" {
		return "self-update: " + diagnostic
	}
	return out + "\nself-update: " + diagnostic
}

// OSSwap atomically replaces dst with src. On unix os.Rename over an existing (even
// running) binary is atomic. On Windows a mapped .exe cannot be overwritten in place, so we
// rename the existing target ASIDE first, then move the new one in; a concurrent reader sees
// either the intact old or the intact new binary, never a partial file. The renamed-aside copy
// is best-effort removed. If a prior aside file is still held by a stale process, we choose a
// unique aside name rather than letting one locked dst.old wedge every future self-update.
func OSSwap(src, dst string) error {
	if runtime.GOOS != "windows" {
		return os.Rename(src, dst)
	}
	_ = os.Remove(dst + ".old") // clear the conventional aside when no stale handle holds it
	old := windowsSwapAsidePath(dst, os.Getpid(), pathExists)
	if _, err := os.Stat(dst); err == nil {
		if err := os.Rename(dst, old); err != nil {
			return err
		}
	}
	if err := os.Rename(src, dst); err != nil {
		// Roll back: put the original binary back so the fleet is never left without one.
		_ = os.Rename(old, dst)
		return err
	}
	_ = os.Remove(old) // best-effort; a held handle just leaves dst.old until it closes
	return nil
}

func windowsSwapAsidePath(dst string, pid int, exists func(string) bool) string {
	base := dst + ".old"
	if !exists(base) {
		return base
	}
	for i := 0; i < 1000; i++ {
		candidate := fmt.Sprintf("%s.%d.%d", base, pid, i)
		if !exists(candidate) {
			return candidate
		}
	}
	return fmt.Sprintf("%s.%d.overflow", base, pid)
}

// pidFromAside extracts the PID encoded in a swap-aside binary's basename, or (0,false) when
// base is not a "<binary>.old.<pid>.<i>" name produced by windowsSwapAsidePath. It anchors on
// the ".old." segment and requires a positive numeric PID and a numeric index suffix, so it
// never matches the conventional plain "<binary>.old", a manual ".old-<sha>"/".old-<date>"
// backup, or any other neighbour in the install dir. Single-sourced with the producer above so
// the reaper and the namer can never drift into reaping the wrong file.
func pidFromAside(dstBase, base string) (int, bool) {
	rest, ok := strings.CutPrefix(base, dstBase+".old.")
	if !ok {
		return 0, false
	}
	pidStr, idxStr, ok := strings.Cut(rest, ".")
	if !ok {
		return 0, false
	}
	pid, err := strconv.Atoi(pidStr)
	if err != nil || pid <= 0 {
		return 0, false
	}
	if _, err := strconv.Atoi(idxStr); err != nil { // require the "<pid>.<i>" shape, not "<pid>.overflow"
		return 0, false
	}
	return pid, true
}

// ReapStaleAsides deletes the "<binary>.old.<pid>.<i>" swap-aside files that OSSwap leaves
// behind on Windows when the running binary it renamed aside is still handle-locked at swap
// time (a long-lived guarded session keeps the old .exe mapped). OSSwap's own best-effort
// os.Remove of the aside fails while that handle is open, and it then picks a fresh unique
// aside name on the NEXT swap — so nothing ever reclaims them and one aside leaks per
// self-update tick (a real host leaked 211 of them, ~9 GB, before this existed). Calling this
// at the START of every self-update makes the leak self-healing, exactly as ReapStaleBuilds
// does for leaked build worktrees.
//
// It is safe to run blind:
//   - it only ever deletes files whose basename is "<Target-base>.old.<pid>.<i>" — the shape
//     windowsSwapAsidePath produces — so the live binary, its plain ".old", and manual
//     ".old-<sha>"/".old-<date>" backups are never targets;
//   - it deletes one ONLY when alive(pid) is false (the process that held the old binary is
//     gone, so the file is no longer mapped) and pid != selfPID (never an aside we just made);
//   - a delete that still fails (a lingering handle) is skipped silently and retried next tick.
//
// Liveness comes through the injected alive predicate and every file effect is local, so the
// decision tree is testable with no real processes. It returns the paths it removed.
func ReapStaleAsides(target string, selfPID int, alive func(int) bool) []string {
	dir := filepath.Dir(target)
	dstBase := filepath.Base(target)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var reaped []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		pid, ok := pidFromAside(dstBase, e.Name())
		if !ok || pid == selfPID || alive(pid) {
			continue
		}
		full := filepath.Join(dir, e.Name())
		if err := os.Remove(full); err == nil {
			reaped = append(reaped, full)
		}
	}
	return reaped
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// AsideFootprint reports the swap-aside binaries currently sitting next to target. It is the
// OBSERVABILITY half of the leak fix: the reason 211 asides (~9 GB) could pile up unnoticed is
// that nothing ever looked at the install dir, so accumulation was invisible until someone ran
// `ls`. Surfacing Count/Bytes/DeadCount in `self-update --check` turns a silent 9 GB leak into
// a one-line signal a human or the fleet notices at 5 files, not 500.
//
// It counts only "<target-base>.old.<pid>.<i>" names (the exact shape ReapStaleAsides reaps),
// so manual .bak-*/.old-<sha> backups and the live binary never inflate the number. DeadCount
// is how many of them the NEXT self-update would reclaim (owning PID dead and not selfPID) —
// the actionable subset. It reads only metadata (os.ReadDir + entry Size) and never deletes.
type AsideFootprint struct {
	Count     int   // total swap-aside files next to target
	Bytes     int64 // their combined size
	DeadCount int   // subset whose owning PID is dead (reclaimable on the next self-update)
	DeadBytes int64 // combined size of the reclaimable subset
}

// MeasureAsides walks target's directory and tallies the swap-aside footprint. alive reports
// whether a PID is still running (dead-owner asides are the reclaimable ones); selfPID is
// excluded from the dead subset so an aside this very process made is never counted as
// reclaimable. A missing/unreadable dir is a valid empty footprint, not an error.
func MeasureAsides(target string, selfPID int, alive func(int) bool) AsideFootprint {
	var fp AsideFootprint
	dir := filepath.Dir(target)
	dstBase := filepath.Base(target)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fp
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		pid, ok := pidFromAside(dstBase, e.Name())
		if !ok {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		fp.Count++
		fp.Bytes += info.Size()
		if pid != selfPID && !alive(pid) {
			fp.DeadCount++
			fp.DeadBytes += info.Size()
		}
	}
	return fp
}
