package main

// guard_tempreap.go — the bounded reaper for the `fak-guard-<hook>-*` session
// temp-dir family (#3299, part of the unbounded-accumulation leak epic #3287).
//
// Every `fak guard -- <agent>` writes the child's settings/config into one or
// more per-session temp dirs (installGuardMCPRegistration, the PreCompact /
// Stop / SessionStart / ToolProc hook installers, the Pi extension, and the
// task-handoff file). The child reads them for its WHOLE lifetime, so none can
// be removed with a setup-time defer — and until now none were removed at all.
// The only prior sweep (guardRestartSeedDirs) globs `fak-guard-reset-*` seeds
// and is read-only, so over a fleet's life (200-370 sessions/night) the hook
// dirs pile up in %TEMP%/tmp with no fak-owned bound.
//
// The fix mirrors the two reapers already proven in-tree —
// selfinstall.ReapStaleAsides and ReapStaleBuilds — using the epic's prescribed
// pattern for kill-residue: DEAD-OWNER-PID-KEYED reaping. guardSessionTempDir
// stamps the creating guard's PID into the dir name (fak-guard-<hook>-<pid>-*),
// exactly as BuildDirName encodes it for build worktrees, so the reaper can
// prove a dir is abandoned (its guard is gone) before removing it — no age
// guess that could race a long-running session's still-open settings file.
//
// Legacy dirs written by a pre-#3299 binary carry no PID and are deliberately
// NOT reaped here (there is no owner to prove dead); they are a bounded
// one-time residue the OS temp cleaner collects. Every dir created from this
// commit forward is reaped the first time a later guard allocates a hook dir
// (guardSessionTempDir's once-guarded sweep) once its owner has exited.

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// guardTempDirPrefix is the shared marker on every guard session temp dir. It
// both tags a dir as guard-owned throwaway state and anchors the reaper's glob.
const guardTempDirPrefix = "fak-guard-"

// guardTempDirHooks is the CLOSED set of hook tokens the reaper owns — one per
// per-session temp-dir call site. "reset" is deliberately absent: the restart
// carryover seeds (fak-guard-reset-*) are durable forensic evidence scanned by
// `fak guard restart-audit`, reaped on their own schedule, and must never be
// swept here. A new hook dir MUST add its token here or the reaper will not
// recognize (and so never reap) it — the same single-source-of-truth coupling
// BuildDirName/pidFromBuildDir keep between creator and reaper.
//
// "replay" is the one token that is not a Claude-Code hook: it is a `fak guard
// replay` run reserving the dir that holds its default audit journal (#5524).
// It allocates through this same seam and so belongs to the same reaped family —
// omitting it would trade the old unbounded pile of loose fak-guard-replay-*.jsonl
// files for an unbounded pile of dirs.
var guardTempDirHooks = map[string]bool{
	"handoff":      true,
	"mcp":          true,
	"pi":           true,
	"precompact":   true,
	"replay":       true,
	"seedprompt":   true,
	"sessionstart": true,
	"stophook":     true,
	"toolproc":     true,
}

// guardSessionTempDir creates one per-session guard temp dir for the named hook,
// stamping the current process's PID into the name so guardReapStaleTempDirs can
// later prove the owning guard is dead before reaping it. The layout is
// "fak-guard-<hook>-<pid>-<random>" (os.MkdirTemp fills the trailing *). It is
// the single creation seam every hook installer routes through, keeping the name
// shape the reaper parses and the shape the installers write from ever drifting.
//
// It also self-wires the boot sweep: the FIRST hook dir a guard process allocates
// triggers a once-guarded reap of prior dead-owner dirs (guardReapStaleHookTempDirsOnce),
// so the family stays bounded without a dedicated cmdGuard boot call. The sweep runs
// BEFORE this dir is created, and is dead-owner-keyed + selfPID-excluded, so it never
// touches a live peer's dir or one this process is about to make.
func guardSessionTempDir(hook string) (string, error) {
	guardReapStaleHookTempDirsOnce()
	return os.MkdirTemp("", fmt.Sprintf("%s%s-%d-*", guardTempDirPrefix, hook, os.Getpid()))
}

// guardTempDirOwner parses a guard temp-dir basename back into its hook token and
// owning PID. ok is true only for a "fak-guard-<hook>-<pid>-<random>" name whose
// hook is in the closed guardTempDirHooks set and whose pid segment is a positive
// integer — so a legacy no-PID dir, a fak-guard-reset-* seed dir, and any
// unrelated temp entry all return ok=false and are left untouched.
func guardTempDirOwner(base string) (hook string, pid int, ok bool) {
	rest, had := strings.CutPrefix(base, guardTempDirPrefix)
	if !had {
		return "", 0, false
	}
	// hook, pid, random — the random suffix os.MkdirTemp appends is digits only,
	// but split into 3 to tolerate it defensively; extra fields are the suffix.
	fields := strings.SplitN(rest, "-", 3)
	if len(fields) < 3 {
		return "", 0, false
	}
	hook = fields[0]
	if !guardTempDirHooks[hook] {
		return "", 0, false
	}
	pid, err := strconv.Atoi(fields[1])
	if err != nil || pid <= 0 {
		return "", 0, false
	}
	return hook, pid, true
}

// guardReapStaleTempDirs removes guard session temp dirs left behind by PRIOR
// guards whose owning process is gone. It is the boot-time self-healing sweep:
// each new `fak guard` buries the temp dirs of the sessions that exited (cleanly
// or killed) before it, so steady-state count is bounded by live concurrency,
// not by fleet lifetime.
//
// It is safe to run blind, exactly like selfinstall.ReapStaleAsides:
//   - it only ever removes a dir whose basename is "fak-guard-<hook>-<pid>-*"
//     with hook in the closed guardTempDirHooks set — a shape only
//     guardSessionTempDir produces — so the reset-* seed dirs, legacy no-PID
//     dirs, and any unrelated temp entry are never targets;
//   - it removes one ONLY when alive(pid) is false (the owning guard has exited,
//     so no child is still reading its settings) and pid != selfPID (never a dir
//     this very guard just made);
//   - a RemoveAll that still fails (a lingering handle) is skipped silently and
//     retried on the next boot.
//
// Liveness comes through the injected alive predicate and every file effect is
// local, so the whole decision tree is testable with no real processes. It
// returns the paths it removed.
func guardReapStaleTempDirs(tempRoot string, selfPID int, alive func(int) bool) []string {
	entries, err := os.ReadDir(tempRoot)
	if err != nil {
		return nil
	}
	var reaped []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		_, pid, ok := guardTempDirOwner(e.Name())
		if !ok || pid == selfPID || alive(pid) {
			continue
		}
		full := filepath.Join(tempRoot, e.Name())
		if err := os.RemoveAll(full); err == nil {
			reaped = append(reaped, full)
		}
	}
	return reaped
}

// guardTempDirFootprint is the observability half of the leak fix: a count/byte
// tally of the guard session temp dirs currently sitting in the temp root, with
// the reclaimable (dead-owner) subset broken out. Surfacing it turns a silent
// unbounded pile into a one-line signal a human or the fleet notices early.
type guardTempDirFootprint struct {
	Count     int   // total fak-guard-<hook>-<pid>-* dirs in the temp root
	Bytes     int64 // their combined on-disk size (settings/config JSON)
	DeadCount int   // subset whose owning guard PID is dead (reaped next boot)
	DeadBytes int64 // combined size of the reclaimable subset
}

// guardMeasureTempDirs tallies the guard session temp-dir footprint under
// tempRoot without removing anything. alive reports whether a PID is still
// running (dead-owner dirs are the reclaimable ones); selfPID is excluded from
// the dead subset so a dir this process just made is never counted reclaimable.
// A missing/unreadable temp root is a valid empty footprint, not an error.
func guardMeasureTempDirs(tempRoot string, selfPID int, alive func(int) bool) guardTempDirFootprint {
	var fp guardTempDirFootprint
	entries, err := os.ReadDir(tempRoot)
	if err != nil {
		return fp
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		_, pid, ok := guardTempDirOwner(e.Name())
		if !ok {
			continue
		}
		size := guardDirSize(filepath.Join(tempRoot, e.Name()))
		fp.Count++
		fp.Bytes += size
		if pid != selfPID && !alive(pid) {
			fp.DeadCount++
			fp.DeadBytes += size
		}
	}
	return fp
}

// guardDirSize sums the sizes of the regular files under dir (one guard temp dir
// holds a couple of small JSON files). Unreadable entries contribute 0 rather
// than failing the whole measurement.
func guardDirSize(dir string) int64 {
	var total int64
	_ = filepath.WalkDir(dir, func(_ string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if info, err := d.Info(); err == nil {
			total += info.Size()
		}
		return nil
	})
	return total
}

// guardReapStaleHookTempDirs sweeps the OS temp dir for dead-owner guard hook dirs,
// keyed off this guard's own PID and the real process-liveness predicate. Returns
// the number reaped so a caller can surface it (an explicit cmdGuard boot call can
// print the count; the once-wire below discards it).
func guardReapStaleHookTempDirs() int {
	return len(guardReapStaleTempDirs(os.TempDir(), os.Getpid(), dispatchPIDAlive))
}

// guardReapHookTempDirsOnce serializes the boot sweep to exactly one run per process.
var guardReapHookTempDirsOnce sync.Once

// guardReapStaleHookTempDirsOnce runs the dead-owner sweep at most once per process,
// on the first guard hook temp-dir allocation (guardSessionTempDir). This self-wires
// the reaper at guard boot without a dedicated call site in cmdGuard: the first hook a
// guard installs buries the temp dirs of the sessions that exited before it, exactly as
// selfinstall.ReapStaleAsides fires at the start of each self-update. Idempotent and
// dead-owner-safe, so it composes harmlessly with any explicit boot call.
func guardReapStaleHookTempDirsOnce() {
	guardReapHookTempDirsOnce.Do(func() { guardReapStaleHookTempDirs() })
}
