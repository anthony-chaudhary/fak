package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

// Concern: the host spawn-burst ("churn") backpressure term of the dispatch preflight.
//
// This file owns that term end to end -- its FAK_CHURN_* env knobs (freshness, burst
// threshold, cold-start worker floor), the stallscan self-monitor ledger it reads, the
// bounded tail read that gets the last record, and the fold of that reading into a
// dispatchtick.ChurnCheck plus its stallscan.Arming verdict.
//
// It is a seam, not a line-count cut: churn is the only preflight term sourced from the
// stallscan ledger on disk rather than from a live probe, it shares no state with the
// gate / rate-limit / account / seat paths in dispatch_tick_preflight.go, and it reaches
// them only through the two values dispatchPreflightChurnState returns. Its fail-open
// contract (no ledger, garbled tail, stale reading, or a disabled threshold all yield the
// zero-value check) is therefore reviewable in one place.
//
// Pure code motion out of dispatch_tick_preflight.go -- no behavior change.

// dispatchChurnDefaultFreshness bounds how old the stallscan self-monitor reading may be
// and still gate admission. The burst signal is a point-in-time host census; a reading
// older than a couple sample intervals no longer describes the CURRENT scheduler load, and
// gating on a stale storm would wrongly freeze a fleet whose burst has long since drained.
// A reading older than this yields a zero-value ChurnCheck (the fold abstains). The impure
// shell overlays FAK_CHURN_FRESHNESS.
const dispatchChurnDefaultFreshness = 90 * time.Second

// dispatchChurnFreshness resolves the max age of a usable stallscan reading from
// FAK_CHURN_FRESHNESS (a Go duration; "0"/"off" disables the freshness gate and trusts the
// last reading regardless of age), falling back to the default on empty/unparseable input.
func dispatchChurnFreshness() time.Duration {
	return parseDurationEnv("FAK_CHURN_FRESHNESS", dispatchChurnDefaultFreshness)
}

// dispatchChurnThreshold resolves the spawn-burst arming threshold from
// FAK_CHURN_BURST_THRESHOLD, falling back to dispatchtick.DefaultChurnBurstThreshold. A zero
// or negative override is ignored so the term cannot be armed on ordinary process turnover.
// A value of "off" disables the term entirely (returns 0 -> the shell yields a zero-value
// check that never gates).
func dispatchChurnThreshold() int {
	raw := strings.TrimSpace(os.Getenv("FAK_CHURN_BURST_THRESHOLD"))
	if raw == "" {
		return dispatchtick.DefaultChurnBurstThreshold
	}
	if strings.EqualFold(raw, "off") {
		return 0
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchtick.DefaultChurnBurstThreshold
}

// dispatchChurnMinWorkers resolves the cold-start floor from FAK_CHURN_MIN_WORKERS, falling
// back to dispatchtick.DefaultChurnMinWorkers. A negative override is ignored; the pure
// fold's floor() re-clamps a zero back to the default, so the one-probe liveness carve-out
// cannot be removed through the env.
func dispatchChurnMinWorkers() int {
	raw := strings.TrimSpace(os.Getenv("FAK_CHURN_MIN_WORKERS"))
	if raw == "" {
		return dispatchtick.DefaultChurnMinWorkers
	}
	if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
		return n
	}
	return dispatchtick.DefaultChurnMinWorkers
}

// dispatchStallLedgerPath resolves the stallscan self-monitor ledger the churn fold reads.
// It mirrors cmd/fak/stallscan.go's defaultStallLogPath EXACTLY (FAK_STALL_DIR, else the
// Windows LOCALAPPDATA\Fleet location, else ~/.fak) so the reader and the writer agree on
// one path without importing the writer.
func dispatchStallLedgerPath() string {
	if d := os.Getenv("FAK_STALL_DIR"); d != "" {
		return filepath.Join(d, "stallscan.jsonl")
	}
	if la := os.Getenv("LOCALAPPDATA"); la != "" {
		return filepath.Join(la, "Fleet", "stallscan.jsonl")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".fak", "stallscan.jsonl")
}

// dispatchPreflightChurn folds the MEASURED whole-host spawn burst into the host_churn
// admission term (dispatchtick.ApplyChurnBackpressure). It reads the LAST line of the
// stallscan self-monitor ledger -- a background loop samples the host and appends one
// fak.stallscan.v1 record per interval, so reading its tail costs one small file read and
// spawns NOTHING on this hot admission path (the discipline that keeps the anti-churn term
// from adding to the churn it measures).
//
// Fail-open and byte-identical when idle: no ledger (the self-monitor is not running), an
// unreadable/garbled tail, a reading older than the freshness bound (a drained burst must
// not keep gating), or a disabled threshold (FAK_CHURN_BURST_THRESHOLD=off) each yields the
// zero-value check -- a no-op fold that leaves the preflight untouched. A box that never
// runs `fak stallscan --watch` therefore behaves exactly as before this term existed.

// dispatchPreflightChurnState is dispatchPreflightChurn plus the ARMING verdict: not just
// what the churn signal said, but whether this host was measured at all.
//
// The two are separated because they answer different questions and are consumed by
// different readers. The ChurnCheck feeds the pure admission fold, which must abstain on
// anything it cannot trust. The Arming feeds the tick PAYLOAD, which must be able to tell
// an operator that the abstention happened -- and why -- because a silently-abstaining
// gate is indistinguishable from a gate that examined a calm host and passed it. That
// ambiguity is the ghost: the reaper reports health it never measured.
//
// Fail-open is preserved exactly: every non-armed state still yields the zero-value
// ChurnCheck, so a box with no self-monitor admits work byte-identically to before this
// term existed. Only the reporting changed.
func dispatchPreflightChurnState() (dispatchtick.ChurnCheck, stallscan.Arming) {
	threshold := dispatchChurnThreshold()
	enabled := threshold > 0
	fresh := dispatchChurnFreshness()

	read := dispatchReadChurnLedger()
	arming := stallscan.ClassifyArming(read, time.Now(), fresh, enabled)
	if !arming.Armed() {
		return dispatchtick.ChurnCheck{}, arming
	}
	return dispatchtick.ChurnCheck{
		Recent:        arming.SpawnBurst,
		WindowSeconds: arming.SpawnWindowSeconds,
		Threshold:     threshold,
		MinWorkers:    dispatchChurnMinWorkers(),
	}, arming
}

// dispatchReadChurnLedger is the impure half: it reads the self-monitor ledger tail and
// reports WHAT it managed to get, without deciding whether that is good enough. The
// Found/Parsed split is deliberate -- "a writer is running but emitting the wrong shape"
// and "no writer at all" call for different operator actions, so they must not collapse
// into one silent zero.
func dispatchReadChurnLedger() stallscan.LedgerRead {
	line := dispatchLastLine(dispatchStallLedgerPath())
	if line == "" {
		return stallscan.LedgerRead{} // no ledger / empty -> not found
	}
	var rec struct {
		TS     string `json:"ts"`
		Sample struct {
			SpawnBurst         int     `json:"spawn_burst"`
			SpawnWindowSeconds float64 `json:"spawn_window_seconds"`
		} `json:"sample"`
	}
	if err := json.Unmarshal([]byte(line), &rec); err != nil {
		return stallscan.LedgerRead{Found: true} // garbled tail
	}
	ts, err := time.Parse(time.RFC3339Nano, rec.TS)
	if err != nil {
		// A record whose stamp will not parse cannot be aged, so it can never be
		// shown to be fresh. Report it as garbled rather than as a zero-time
		// reading that would classify as astronomically stale.
		return stallscan.LedgerRead{Found: true}
	}
	return stallscan.LedgerRead{
		Found:              true,
		Parsed:             true,
		Timestamp:          ts,
		SpawnBurst:         rec.Sample.SpawnBurst,
		SpawnWindowSeconds: rec.Sample.SpawnWindowSeconds,
	}
}

// dispatchLastLine returns the last non-empty line of a file, or "" if the file is missing,
// empty, or unreadable. It reads only a bounded tail window so a large rolling ledger costs
// a small fixed read rather than an O(file) slurp on the admission path.
func dispatchLastLine(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || info.Size() == 0 {
		return ""
	}
	const tailCap = 64 << 10 // 64 KiB: far more than one fak.stallscan.v1 record
	start := int64(0)
	if info.Size() > tailCap {
		start = info.Size() - tailCap
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return ""
	}
	buf, err := io.ReadAll(f)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(buf), "\r\n"), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if s := strings.TrimSpace(lines[i]); s != "" {
			return s
		}
	}
	return ""
}
