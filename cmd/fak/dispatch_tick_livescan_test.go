package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestCooldownCoolsIssueFromWitnessSidecar pins the ported parity behavior: the dispatch
// cooldown must cool an issue off its durable .witness audit sidecar, not only its .log
// transcript. prune_dead_sidecars retains .witness as the durable cooldown evidence and a
// worker's .log may be swept externally, so an issue witnessed recently -- even one whose
// .log is gone -- must still be held OUT of the wave. This mirrors
// tools/issue_resolve_dispatch.py's recently_attempted_issues, which globs BOTH
// resolve-*.log and resolve-*.witness. Before the port the Go scan read .log only, so a
// .witness-only slot re-entered the pool immediately.
func TestCooldownCoolsIssueFromWitnessSidecar(t *testing.T) {
	runsDir := t.TempDir()
	now := time.Unix(1700000000, 0).UTC()
	writeWitness := func(issue int, stamp string, mod time.Time) {
		t.Helper()
		path := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-%s%s", issue, stamp, dispatchtick.WitnessSidecarSuffix))
		if err := os.WriteFile(path, []byte(`{"claim":"CLAIM_NO_COMMIT"}`), 0o644); err != nil {
			t.Fatalf("write witness sidecar: %v", err)
		}
		if err := os.Chtimes(path, mod, mod); err != nil {
			t.Fatalf("chtime witness sidecar: %v", err)
		}
	}
	// #4100: only a .witness survives (its .log was swept), touched 30 min ago -> cooling.
	writeWitness(4100, "20260701-010000", now.Add(-30*time.Minute))
	// #4200: only a .witness, aged past the 120-min window -> ready (not cooling).
	writeWitness(4200, "20260701-000000", now.Add(-3*time.Hour))

	cooled := recentlyAttemptedIssuesAt(runsDir, 120, now)
	if !cooled[4100] {
		t.Fatalf("recently attempted = %#v, want #4100 cooling from its .witness sidecar", cooled)
	}
	if cooled[4200] {
		t.Fatalf("recently attempted = %#v, want #4200 aged out (not cooling)", cooled)
	}
}

// TestCooldownExtendsFromLaterWitnessMtime pins that when BOTH a .log and its .witness
// exist for an issue, the cooldown is measured from the LATER of the two mtimes. The
// .witness is written post-mortem by the audit sweep, so it carries the most-recent
// attempt touch. This case would FAIL under the pre-port log-only scan: the .log alone is
// aged past the window, so only folding the fresher .witness keeps the issue cooling --
// matching Python's "recent if ANY attempt artifact is recent".
func TestCooldownExtendsFromLaterWitnessMtime(t *testing.T) {
	runsDir := t.TempDir()
	now := time.Unix(1700000000, 0).UTC()
	stem := "resolve-4300-20260701-000000"

	logPath := filepath.Join(runsDir, stem+".log")
	if err := os.WriteFile(logPath, []byte("# fak-spawn\n"), 0o644); err != nil {
		t.Fatalf("write log: %v", err)
	}
	// .log aged 130 min ago -- past the 120-min window on its own.
	logMod := now.Add(-130 * time.Minute)
	if err := os.Chtimes(logPath, logMod, logMod); err != nil {
		t.Fatalf("chtime log: %v", err)
	}

	witPath := filepath.Join(runsDir, stem+dispatchtick.WitnessSidecarSuffix)
	if err := os.WriteFile(witPath, []byte(`{"claim":"CLAIM_WITNESSED"}`), 0o644); err != nil {
		t.Fatalf("write witness: %v", err)
	}
	// .witness written 10 min ago -- the true most-recent attempt touch.
	witMod := now.Add(-10 * time.Minute)
	if err := os.Chtimes(witPath, witMod, witMod); err != nil {
		t.Fatalf("chtime witness: %v", err)
	}

	rows := cooldownIssueRowsAt(runsDir, 120, now)
	if len(rows) != 1 {
		t.Fatalf("cooldown rows = %+v, want exactly one row for #4300", rows)
	}
	if !rows[0].Cooling {
		t.Fatalf("row = %+v, want #4300 still cooling from its fresher .witness mtime", rows[0])
	}
	// The window must run from the witness mtime (10 min old), not the log mtime (130 min).
	if rows[0].LastAttemptUnix != witMod.Unix() {
		t.Fatalf("last_attempt_unix = %d, want witness mtime %d", rows[0].LastAttemptUnix, witMod.Unix())
	}
	cooled := recentlyAttemptedIssuesAt(runsDir, 120, now)
	if !cooled[4300] {
		t.Fatalf("recently attempted = %#v, want #4300 cooling from the witness mtime", cooled)
	}
}

// countingReadCloser wraps a log open so a test can measure exactly how many
// bytes laneFromSpawnHeader pulls off a live worker's streaming transcript.
type countingReadCloser struct {
	rc io.ReadCloser
	n  *int64
}

func (c *countingReadCloser) Read(p []byte) (int, error) {
	n, err := c.rc.Read(p)
	*c.n += int64(n)
	return n, err
}

func (c *countingReadCloser) Close() error { return c.rc.Close() }

// TestLaneFromSpawnHeaderReadsBoundedPrefix pins the #3466 fix: the lane probe
// must parse `lane=` off a worker log's FIRST line while reading only a bounded
// prefix of the file — never the whole multi-MB, still-growing transcript the
// legacy os.ReadFile pulled in on every livescan pass. The body is made far
// larger than any header buffer, and the fsOpen seam counts the bytes actually
// read to prove the read stayed under the cap.
func TestLaneFromSpawnHeaderReadsBoundedPrefix(t *testing.T) {
	runsDir := t.TempDir()
	path := filepath.Join(runsDir, "resolve-3466-20260709-000000.log")
	header := "# fak-spawn 20260709-000000 issue=3466 lane=docs backend=opencode argv0=node\n"
	body := strings.Repeat("streaming transcript filler line well past any first-line buffer\n", 40000) // ~2.6 MB
	if err := os.WriteFile(path, []byte(header+body), 0o644); err != nil {
		t.Fatalf("write log: %v", err)
	}

	var bytesRead int64
	orig := fsOpen
	fsOpen = func(p string) (io.ReadCloser, error) {
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		return &countingReadCloser{rc: f, n: &bytesRead}, nil
	}
	t.Cleanup(func() { fsOpen = orig })

	if got := laneFromSpawnHeader(path); got != "docs" {
		t.Fatalf("lane = %q, want %q (parsed from the spawn-header first line)", got, "docs")
	}
	if bytesRead > laneHeaderReadCapBytes {
		t.Fatalf("lane probe read %d bytes of a %d-byte transcript, want <= %d (first line only, #3466)",
			bytesRead, len(header)+len(body), laneHeaderReadCapBytes)
	}
}

// TestLaneFromSpawnHeaderHeaderOnlyNoNewline pins the EOF edge of the bounded
// read: a header-only log with no trailing newline (a worker that died at
// spawn) must still yield its lane, matching the legacy whole-file parse.
func TestLaneFromSpawnHeaderHeaderOnlyNoNewline(t *testing.T) {
	runsDir := t.TempDir()
	path := filepath.Join(runsDir, "resolve-3467-20260709-000001.log")
	if err := os.WriteFile(path,
		[]byte("# fak-spawn 20260709-000001 issue=3467 lane=gateway backend=claude argv0=claude"), 0o644); err != nil {
		t.Fatalf("write log: %v", err)
	}
	if got := laneFromSpawnHeader(path); got != "gateway" {
		t.Fatalf("lane = %q, want %q (header-only log, no trailing newline)", got, "gateway")
	}
	if got := laneFromSpawnHeader(filepath.Join(runsDir, "missing.log")); got != "" {
		t.Fatalf("lane = %q for a missing log, want \"\" (fail closed)", got)
	}
}
