// watchctx.go — the durable watch-context descriptor (#2384): the record a WATCHER
// (the agent babysitting a long supervised compute job — not the job itself, and not
// the watcher's own Claude session) writes the moment it begins supervising, so a
// restarted watcher re-attaches to the job it was babysitting instead of launching a
// duplicate or abandoning the original.
//
// This is the second no-babysit failure mode (epic #2269): the watcher crashes —
// context compaction, a closed terminal, an OOM — while the job keeps running. It is
// DISTINCT from job-crash (#2086/#2108) and agent-turn-crash (#1352/#1363/#2217):
// those checkpoint the agent's own served turn/session; none record the identity of a
// separate supervised job. resume.WatchdogPlanRow resumes a dead Claude SESSION and
// carries no supervised-job fields, and the nightrun ledger row lands only AFTER a
// task ends — so before this descriptor, a mid-task watcher crash left no durable
// pointer to the in-flight job or its artifact.
//
// The descriptor closes exactly that gap: job_pid + artifact_path (computable from
// absArtifact BEFORE launch — see NewWatchContext) + last_progress, keyed to the
// watcher's identity. On restart the watcher reads its descriptor and asks the #2277
// pid-liveness seam (looprecover.Probe, reuse-safe via start-time identity) for the
// verdict: a live pid ⇒ re-attach (never a duplicate launch); a dead pid ⇒ the job is
// gone. Scope is deliberately the descriptor + its write/read path + that verdict;
// the re-attach POLICY (kill-and-restart vs adopt) is a follow-on, not here.
package nightrun

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/looprecover"
)

// WatchContext binds one watcher to the one job it supervises. Plain data only, so
// the record round-trips losslessly through its JSON file.
type WatchContext struct {
	// WatcherID is the supervising watcher's identity — the descriptor's key: one
	// watcher, one file, so a restarted watcher finds ITS job, never a sibling's.
	WatcherID string `json:"watcher_id"`
	// JobPID is the supervised process id a restarted watcher probes for liveness.
	JobPID int `json:"job_pid"`
	// JobStart is the job's start-time identity — the reuse-safe fingerprint
	// looprecover.Probe uses to defeat pid reuse ("" when the watcher has none, in
	// which case a held pid is trusted as alive).
	JobStart string `json:"job_start,omitempty"`
	// ArtifactPath is the absolute artifact/checkpoint path the job writes,
	// populated BEFORE launch (NewWatchContext derives it from absArtifact) rather
	// than after the ledger row lands — a mid-task crash still leaves the pointer.
	ArtifactPath string `json:"artifact_path"`
	// LastProgress is the last progress line the watcher saw, rewritten on each
	// advance (see Advance); LastProgressUnix is when it was seen.
	LastProgress     string `json:"last_progress,omitempty"`
	LastProgressUnix int64  `json:"last_progress_unix,omitempty"`
	// ExpectedDoneByUnix is the optional deadline after which a silent job is a
	// stall rather than a long run (0 = no deadline). Recorded for the follow-on
	// policy; nothing here acts on it.
	ExpectedDoneByUnix int64 `json:"expected_done_by_unix,omitempty"`
}

// NewWatchContext builds the descriptor for a task the watcher is ABOUT to launch:
// the artifact path comes from absArtifact(opts, t) — derivable from RunOptions +
// Task alone, before the process exists and long before its ledger row lands.
func NewWatchContext(watcherID string, jobPID int, jobStart string, opts RunOptions, t Task) WatchContext {
	return WatchContext{
		WatcherID:    watcherID,
		JobPID:       jobPID,
		JobStart:     jobStart,
		ArtifactPath: absArtifact(opts, t),
	}
}

// Advance rewrites the last-seen progress line + timestamp, returning the updated
// descriptor by value (the caller re-writes it durably; nothing mutates in place).
func (wc WatchContext) Advance(line string, atUnix int64) WatchContext {
	wc.LastProgress, wc.LastProgressUnix = line, atUnix
	return wc
}

// WatchContextPath is the one file a watcher's descriptor lives at: keyed by the
// sanitized watcher identity, so the write and the restart read can never disagree.
func WatchContextPath(dir, watcherID string) string {
	return filepath.Join(dir, "watchctx-"+sanitize(watcherID)+".json")
}

// WriteWatchContext persists the descriptor durably (temp file + rename, so a crash
// mid-write never leaves a torn record) and returns the path written.
func WriteWatchContext(dir string, wc WatchContext) (string, error) {
	if strings.TrimSpace(wc.WatcherID) == "" {
		return "", fmt.Errorf("watch context: empty watcher id")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("watch context dir: %w", err)
	}
	b, err := json.MarshalIndent(wc, "", "  ")
	if err != nil {
		return "", fmt.Errorf("watch context encode: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".watchctx-*")
	if err != nil {
		return "", fmt.Errorf("watch context temp: %w", err)
	}
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return "", fmt.Errorf("watch context write: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return "", fmt.Errorf("watch context close: %w", err)
	}
	path := WatchContextPath(dir, wc.WatcherID)
	if err := os.Rename(tmp.Name(), path); err != nil {
		os.Remove(tmp.Name())
		return "", fmt.Errorf("watch context rename: %w", err)
	}
	return path, nil
}

// ReadWatchContext is the restart read: the descriptor for this watcher's identity,
// or the underlying error (os.IsNotExist ⇒ the watcher never began supervising).
func ReadWatchContext(dir, watcherID string) (WatchContext, error) {
	b, err := os.ReadFile(WatchContextPath(dir, watcherID))
	if err != nil {
		return WatchContext{}, err
	}
	var wc WatchContext
	if err := json.Unmarshal(b, &wc); err != nil {
		return WatchContext{}, fmt.Errorf("watch context decode: %w", err)
	}
	return wc, nil
}

// ReattachVerdict is the closed vocabulary a restarted watcher acts on.
type ReattachVerdict string

const (
	// ReattachLive: the supervised job is confirmed alive — re-attach to it; a
	// duplicate launch is exactly the failure this descriptor exists to prevent.
	ReattachLive ReattachVerdict = "re-attach"
	// ReattachGone: the supervised job is confirmed dead (or its pid was reused by
	// a different process) — report the job gone; whether to relaunch is the
	// follow-on policy's call, not made here.
	ReattachGone ReattachVerdict = "job_gone"
	// ReattachUnknown: no probeable pid or no prober — no liveness verdict.
	ReattachUnknown ReattachVerdict = "unknown"
)

// Reattach is the restart decision over the descriptor: probe the recorded job pid
// through the #2277 seam (looprecover.Probe — reuse-safe via JobStart) and map the
// verdict to what the watcher reports. Pure given the injected Liveness lookup.
func (wc WatchContext) Reattach(live looprecover.Liveness) ReattachVerdict {
	switch looprecover.Probe(wc.JobPID, wc.JobStart, live) {
	case looprecover.ProbeAlive:
		return ReattachLive
	case looprecover.ProbeDead:
		return ReattachGone
	default:
		return ReattachUnknown
	}
}
