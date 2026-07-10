package dispatchaudit

import (
	"bufio"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchconservation"
)

// shell.go is the THIN I/O boundary: it reads the on-disk dispatch artifacts
// (resolve-*.log, the .backend sidecars, progress.jsonl) into pure Worker
// records and hands them to Fold. No classification logic lives here.

var (
	// `# fak-spawn 20260629-235906 issue=1346 lane=docs backend=opencode argv0=...`
	reSpawnHeader = regexp.MustCompile(`^#\s*fak-spawn\b.*$`)
	reSpawnStamp  = regexp.MustCompile(`^#\s*fak-spawn\s+(\d{8}-\d{6})\b`)
	reIssue       = regexp.MustCompile(`\bissue=(\S+)`)
	reLane        = regexp.MustCompile(`\blane=(\S+)`)
	reHdrBackend  = regexp.MustCompile(`\bbackend=(\S+)`)

	// A self-reported created/shipped commit SHA in raw worker output. This is
	// quarantined as a claim; only the structured .commit sidecar below can set
	// Worker.CommitSHA and promote a worker to SHIPPED.
	reCommit = regexp.MustCompile("(?i)(?:commit created|✅ commit|shipped|committed)[^0-9a-f]*`?([0-9a-f]{7,40})`?")
	reSHA    = regexp.MustCompile(`^[0-9a-fA-F]{7,40}$`)

	// An RFC3339-ish leading timestamp on an opencode line:
	// `timestamp=2026-06-30T00:01:22.783Z level=ERROR ...`
	reTimestamp = regexp.MustCompile(`timestamp=(\S+)`)
	reLevelErr  = regexp.MustCompile(`level=ERROR\b`)

	// Provider cap / weekly-monthly limit banners (opencode + claude shapes).
	reCapBanner = regexp.MustCompile(`(?i)weekly/monthly limit|limit exhausted|usage limit reached|quota.{0,20}exceed|rate.?limit.{0,20}(week|month)`)

	// Generic provider stream/api error (a retry-storm symptom).
	reProviderErr = regexp.MustCompile(`(?i)stream error|AI_APICallError|APICallError|provider.{0,12}error|429|503`)
)

var pidAlive = processAlive

// logBaseRE matches the run id + timestamp at the tail of a resolve log name so
// the .backend sidecar can be paired by prefix.
var resolveLogRE = regexp.MustCompile(`^resolve-.*\.log$`)
var resolveIssueRE = regexp.MustCompile(`^resolve-(\d+)-`)

// ScanDir reads runsDir, parses every resolve-*.log into a Worker (pairing its
// .backend sidecar and folding in the shared progress ledger), and returns them
// sorted by log name for determinism. Windowless form of ScanDirSince: it scans
// every historical log, exactly as before the audit window existed (#3466).
func ScanDir(runsDir string) ([]Worker, error) {
	return ScanDirSince(runsDir, time.Time{})
}

// ScanDirSince is ScanDir windowed to logs spawned at/after since: an entry
// whose spawn stamp (parsed from the log NAME, mirroring
// dispatchconservation.CollectUnits) — or, for a stampless legacy name, whose
// mtime — falls before since is skipped BEFORE any open/parse or .commit/.pid
// sidecar read, so the scheduled audit stops paying O(total historical runs)
// on a never-reaped runs dir (#3466). A zero since scans everything
// (byte-identical to the legacy ScanDir behavior).
func ScanDirSince(runsDir string, since time.Time) ([]Worker, error) {
	entries, err := os.ReadDir(runsDir)
	if err != nil {
		return nil, err
	}

	// Index the sidecars and detect which logs have one. Windowed by the spawn
	// stamp shared with the sidecar's log name (never by sidecar mtime, which can
	// drift from the log's), so a log and its sidecar are windowed identically.
	sidecar := map[string]Backend{}
	hasSidecar := map[string]bool{}
	progress := loadProgress(filepath.Join(runsDir, "progress.jsonl"))

	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".backend") {
			base := strings.TrimSuffix(name, ".backend")
			if sidecarOutOfWindow(base, since) {
				continue
			}
			b, _ := os.ReadFile(filepath.Join(runsDir, name))
			sidecar[base] = NormalizeBackend(string(b))
			hasSidecar[base] = true
		}
	}

	var workers []Worker
	for _, e := range entries {
		if e.IsDir() || !resolveLogRE.MatchString(e.Name()) {
			continue
		}
		if !logInWindow(e, since) {
			continue
		}
		base := strings.TrimSuffix(e.Name(), ".log")
		w, err := parseLog(filepath.Join(runsDir, e.Name()))
		if err != nil {
			continue
		}
		if hasSidecar[base] {
			w.SidecarBackend = sidecar[base]
			w.SidecarMissing = false
		} else {
			w.SidecarMissing = true
		}
		if w.Issue == "" {
			if m := resolveIssueRE.FindStringSubmatch(e.Name()); m != nil {
				w.Issue = m[1]
			}
		}
		if p, ok := progress[w.Issue]; ok && w.Issue != "" {
			w.ProgressTicks = p.ticks
			w.ProgressMoved = p.moved
		}
		if sha, ok := readCommitSidecar(filepath.Join(runsDir, base+".commit")); ok {
			w.CommitSHA = sha
		}
		if pid, ok := readPID(filepath.Join(runsDir, base+".pid")); ok {
			w.PID = pid
			w.PIDAlive = pidAlive(pid)
		}
		workers = append(workers, w)
	}
	return workers, nil
}

// sigMaxReadBytes bounds the per-log text read for the signature fold (worker
// logs can be MB). Mirrors dispatch_log_audit.DEFAULTS["max_read_bytes"].
const sigMaxReadBytes = 2_000_000

// ScanDirSignatures reads runsDir, runs the log-signature detectors over every
// resolve-*.log's text (pairing its .backend sidecar, defaulting to claude for a
// legacy log with none — the Python tool's parity behavior), and returns the
// fileable candidate list. THIN I/O boundary: the classification is FoldSignatures.
// Windowless form of ScanDirSignaturesSince (zero since = scan everything, #3466).
func ScanDirSignatures(runsDir string, th SignatureThresholds) ([]SignatureFinding, error) {
	return ScanDirSignaturesSince(runsDir, th, time.Time{})
}

// ScanDirSignaturesSince is ScanDirSignatures windowed to logs spawned at/after
// since (same name-stamp-first, mtime-fallback rule as ScanDirSince), skipping
// out-of-window logs BEFORE the capped 2 MB text read. A zero since scans
// everything, byte-identical to the legacy behavior (#3466).
func ScanDirSignaturesSince(runsDir string, th SignatureThresholds, since time.Time) ([]SignatureFinding, error) {
	entries, err := os.ReadDir(runsDir)
	if err != nil {
		return nil, err
	}
	sidecar := map[string]Backend{}
	hasSidecar := map[string]bool{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".backend") {
			base := strings.TrimSuffix(name, ".backend")
			if sidecarOutOfWindow(base, since) {
				continue
			}
			b, _ := os.ReadFile(filepath.Join(runsDir, name))
			sidecar[base] = NormalizeBackend(string(b))
			hasSidecar[base] = true
		}
	}

	var logs []SigLog
	for _, e := range entries {
		if e.IsDir() || !resolveLogRE.MatchString(e.Name()) {
			continue
		}
		if !logInWindow(e, since) {
			continue
		}
		base := strings.TrimSuffix(e.Name(), ".log")
		text, err := readCappedLog(filepath.Join(runsDir, e.Name()))
		if err != nil {
			continue // a failing read is skipped, never fatal
		}
		backend := BackendClaude
		if hasSidecar[base] && sidecar[base] != BackendUnknown && sidecar[base] != "" {
			backend = sidecar[base]
		}
		logs = append(logs, SigLog{Name: e.Name(), Backend: backend, Text: text})
	}
	sort.Slice(logs, func(i, j int) bool { return logs[i].Name < logs[j].Name })
	return FoldSignatures(logs, th), nil
}

// logInWindow reports whether a resolve-log directory entry falls inside the
// audit window [since, now]. A zero since means no window: everything is in
// (the legacy scan-all behavior). The spawn stamp parsed from the log NAME is
// authoritative (reusing dispatchconservation.ParseLogStampUTC — the mtime
// moves on every write, but the unit was spent at spawn); a legacy name with
// no parseable stamp falls back to the file mtime, and an entry whose mtime
// cannot be read stays IN — the audit never silently drops evidence it cannot
// date (an unreadable file is skipped by the parse step anyway). (#3466)
func logInWindow(e os.DirEntry, since time.Time) bool {
	if since.IsZero() {
		return true
	}
	if stamp, ok := dispatchconservation.ParseLogStampUTC(e.Name()); ok {
		return !stamp.Before(since)
	}
	info, err := e.Info()
	if err != nil {
		return true
	}
	return !info.ModTime().Before(since)
}

// sidecarOutOfWindow reports whether a .backend sidecar's base name dates its
// run strictly before since. Name-stamp ONLY (no mtime fallback): the stamp is
// shared with the paired log's name, so a stamped log and its sidecar are
// always windowed identically, while a stampless sidecar is always indexed —
// it can never be dropped on an mtime guess its log did not make. (#3466)
func sidecarOutOfWindow(base string, since time.Time) bool {
	if since.IsZero() {
		return false
	}
	stamp, ok := dispatchconservation.ParseLogStampUTC(base + ".log")
	return ok && stamp.Before(since)
}

// readCappedLog reads at most sigMaxReadBytes of a log as text.
func readCappedLog(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, sigMaxReadBytes))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// parseLog folds one resolve-*.log into a Worker. PURE-adjacent: it only reads
// the one file and extracts structured fields; the verdict is Classify's job.
func parseLog(path string) (Worker, error) {
	f, err := os.Open(path)
	if err != nil {
		return Worker{}, err
	}
	defer f.Close()

	w := Worker{Log: filepath.Base(path)}
	if st, err := f.Stat(); err == nil {
		w.LogSizeKnown = true
		w.LogBytes = st.Size()
		w.LogMTime = st.ModTime()
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	nonBanner := 0
	lines := 0
	for sc.Scan() {
		line := sc.Text()
		lines++

		if reSpawnHeader.MatchString(line) {
			if m := reSpawnStamp.FindStringSubmatch(line); m != nil {
				// The spawn stamp is written in the host's local time (it is
				// the same stamp as the log file name), so parse it in Local
				// to stay comparable with the log's mtime.
				if t, err := time.ParseInLocation("20060102-150405", m[1], time.Local); err == nil {
					w.SpawnTime = t
				}
			}
			if m := reIssue.FindStringSubmatch(line); m != nil {
				w.Issue = m[1]
			}
			if m := reLane.FindStringSubmatch(line); m != nil {
				w.Lane = m[1]
			}
			if m := reHdrBackend.FindStringSubmatch(line); m != nil {
				w.HeaderBackend = NormalizeBackend(m[1])
			}
			continue
		}

		if !w.UntrustedCommitClaim {
			if reCommit.MatchString(line) {
				w.UntrustedCommitClaim = true
			}
		}

		isErr := false
		if reCapBanner.MatchString(line) {
			w.CapHit = true
			isErr = true
		}
		if reLevelErr.MatchString(line) || reProviderErr.MatchString(line) {
			isErr = true
		}
		if isErr {
			w.ErrorLines++
			if ts := parseTimestamp(line); !ts.IsZero() {
				if w.FirstError.IsZero() {
					w.FirstError = ts
				}
				w.LastError = ts
			}
			nonBanner++
		} else if !isBannerLine(line) && strings.TrimSpace(line) != "" {
			nonBanner++
		}
	}
	if err := sc.Err(); err != nil {
		return w, err
	}
	// A log is banner-only when nothing beyond the fak-guard startup banner and
	// blank lines appeared, and it carries no ship and no errors.
	w.BannerOnly = nonBanner == 0 && w.CommitSHA == "" && w.ErrorLines == 0 && lines > 0
	return w, nil
}

func readPID(path string) (int, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	return pid, err == nil && pid > 0
}

func readCommitSidecar(path string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	sha := strings.TrimSpace(string(b))
	return sha, reSHA.MatchString(sha)
}

// parseTimestamp extracts the `timestamp=...` RFC3339 value from an opencode line.
func parseTimestamp(line string) time.Time {
	m := reTimestamp.FindStringSubmatch(line)
	if m == nil {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05.000Z"} {
		if t, err := time.Parse(layout, m[1]); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}

// isBannerLine reports whether a line is part of the fak-guard startup banner
// (the #1275 banner-noop class) and therefore does NOT count as real work.
func isBannerLine(line string) bool {
	l := strings.TrimSpace(line)
	switch {
	case strings.HasPrefix(l, "fak guard"),
		strings.HasPrefix(l, "fak-turn"),
		strings.HasPrefix(l, "⚠"),
		strings.HasPrefix(l, "gateway"),
		strings.HasPrefix(l, "upstream"),
		strings.HasPrefix(l, "floor"),
		strings.HasPrefix(l, "wired via"),
		strings.HasPrefix(l, "metrics"),
		strings.HasPrefix(l, "cache value"),
		strings.HasPrefix(l, "audit log"),
		strings.HasPrefix(l, "gateway log"),
		strings.HasPrefix(l, "debug"),
		strings.HasPrefix(l, "every tool call"):
		return true
	}
	return false
}

type progRow struct {
	ticks int
	moved bool
}

// loadProgress folds progress.jsonl into a per-issue (tick count, did-it-move)
// summary. A no-op tick is a row that emitted but never advanced
// resolved_toward_target. Best-effort: a missing or malformed ledger yields an
// empty map, never an error (the progress signal is advisory).
func loadProgress(path string) map[string]progRow {
	out := map[string]progRow{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	// progress.jsonl rows are keyed by close_result/witnessed_numbers, not by a
	// single issue; we conservatively roll all rows into one synthetic bucket so
	// the parser stays robust to schema drift. The witnessed issue, when present,
	// keys the row.
	var lastResolved = -1
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		issue := firstWitnessedIssue(line)
		resolved := intField(line, "resolved_toward_target")
		row := out[issue]
		row.ticks++
		if lastResolved >= 0 && resolved > lastResolved {
			row.moved = true
		}
		if resolved > lastResolved {
			lastResolved = resolved
		}
		out[issue] = row
	}
	return out
}

var (
	reWitnessed = regexp.MustCompile(`"witnessed_numbers":\[(\d+)`)
	reResolved  = regexp.MustCompile(`"resolved_toward_target":(\d+)`)
)

func firstWitnessedIssue(line string) string {
	if m := reWitnessed.FindStringSubmatch(line); m != nil {
		return m[1]
	}
	return ""
}

func intField(line, _ string) int {
	if m := reResolved.FindStringSubmatch(line); m != nil {
		n := 0
		for _, r := range m[1] {
			n = n*10 + int(r-'0')
		}
		return n
	}
	return -1
}
