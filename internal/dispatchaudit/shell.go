package dispatchaudit

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// shell.go is the THIN I/O boundary: it reads the on-disk dispatch artifacts
// (resolve-*.log, the .backend sidecars, progress.jsonl) into pure Worker
// records and hands them to Fold. No classification logic lives here.

var (
	// `# fak-spawn 20260629-235906 issue=1346 lane=docs backend=opencode argv0=...`
	reSpawnHeader = regexp.MustCompile(`^#\s*fak-spawn\b.*$`)
	reIssue       = regexp.MustCompile(`\bissue=(\S+)`)
	reLane        = regexp.MustCompile(`\blane=(\S+)`)
	reHdrBackend  = regexp.MustCompile(`\bbackend=(\S+)`)

	// A created/shipped commit SHA. Matches the claude "Commit created: `b68ead49`"
	// shape and a bare "commit <sha>" line; 7-40 hex chars.
	reCommit = regexp.MustCompile("(?i)(?:commit created|✅ commit|shipped|committed)[^0-9a-f]*`?([0-9a-f]{7,40})`?")

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
// sorted by log name for determinism.
func ScanDir(runsDir string) ([]Worker, error) {
	entries, err := os.ReadDir(runsDir)
	if err != nil {
		return nil, err
	}

	// Index the sidecars and detect which logs have one.
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
		if pid, ok := readPID(filepath.Join(runsDir, base+".pid")); ok {
			w.PID = pid
			w.PIDAlive = pidAlive(pid)
		}
		workers = append(workers, w)
	}
	return workers, nil
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
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	nonBanner := 0
	lines := 0
	for sc.Scan() {
		line := sc.Text()
		lines++

		if reSpawnHeader.MatchString(line) {
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

		if w.CommitSHA == "" {
			if m := reCommit.FindStringSubmatch(line); m != nil {
				w.CommitSHA = m[1]
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
