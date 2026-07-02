package main

// resume_sweep_cli.go — `fak resume sweep`, the manifest-free discovery half of the
// cross-account resume layer: find EVERY recently-crashed session across ALL local
// ~/.claude* accounts and bucket each by the action it actually needs. Go port of
// tools/resume_sweep.py (the shell half; the classification core is the pure
// internal/resume/sweep leaf).
//
//	fak resume sweep                    # human summary (default 600m window)
//	fak resume sweep --window 1440      # last 24h
//	fak resume sweep --json             # machine record for a launcher to consume
//	fak resume sweep --bucket LIMIT_RESET_PASSED
//
// Why it exists: a manifest-bound watcher only classifies sessions its registry already
// lists, so a crash wave the manifest never recorded (a whole account's workers capping
// at once) is INVISIBLE to it. The sweep walks the transcripts on disk instead.
//
// It RESUMES NOTHING. Launching is a separate, gated step (`fak resume watchdog`,
// resume_resolver); this verb's job is to make the full actionable set visible and
// correctly bucketed.
//
// This shell does only the I/O the leaf must not: walk <home>/.claude*/projects/*/
// *.jsonl, parse each copy's records, census the live `claude --resume` processes
// (procguard — the same audited census `fak resume admit` uses), read the resume ledger
// for the already-resumed dedup, and (with --probe) consult the account-probe ledger for
// seat health. One deliberate divergence from the Python: --probe reads the RECORDED
// probe ledger (fresh within FLEET_PROBE_FRESH_MIN) instead of spawning an active
// account_probe run — a missing/stale verdict leaves seat_ok absent rather than false.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume/sweep"
)

// runResumeSweep discovers and buckets the recently-crashed sessions. Exit codes: 0 ok,
// 1 runtime error, 2 usage error.
func runResumeSweep(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume sweep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	window := fs.Float64("window", 600, "only sessions whose newest copy changed within N minutes")
	minRecords := fs.Int("min-records", 0, "only sessions with >= N transcript records — filters out the batch-spawned micro-stubs that cap before doing real work")
	probe := fs.Bool("probe", false, "annotate each row with seat_ok from the account-probe ledger (recorded verdict fresh within FLEET_PROBE_FRESH_MIN; absent when no fresh probe exists)")
	asJSON := fs.Bool("json", false, "emit the machine record")
	bucket := fs.String("bucket", "", "filter to one bucket (e.g. LIMIT_RESET_PASSED)")
	includeResumed := fs.Bool("include-resumed", false, "don't exclude sessions the ledger shows were already resumed in-window (default: exclude them so an active pass isn't re-flagged)")
	homeFlag := fs.String("home", "", "user home holding the .claude* account dirs (default: discovered)")
	regDir := fs.String("reg-dir", "", "registry dir holding resume_ledger.jsonl (default: $FLEET_REG_DIR, else <repo>/tools/_registry)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	home := *homeFlag
	if home == "" {
		if h, err := os.UserHomeDir(); err == nil {
			home = h
		}
	}
	if home == "" {
		fmt.Fprintln(stderr, "fak resume sweep: cannot resolve the user home (pass --home)")
		return 1
	}
	now := time.Now().UTC()

	// Group every on-disk transcript copy by session id, across all account dirs.
	paths, _ := filepath.Glob(filepath.Join(home, ".claude*", "projects", "*", "*.jsonl"))
	bySID := map[string][]string{}
	for _, p := range paths {
		sid := strings.TrimSuffix(filepath.Base(p), ".jsonl")
		if len(sid) != 36 {
			continue
		}
		bySID[sid] = append(bySID[sid], p)
	}

	live := liveResumeSIDs()
	resumed := map[string]bool{}
	if !*includeResumed {
		if f, err := os.Open(filepath.Join(resolveSweepRegDir(*regDir), "resume_ledger.jsonl")); err == nil {
			resumed = sweep.RecentlyResumed(f, *window, now)
			f.Close()
		}
	}

	cutoff := now.Add(-time.Duration(*window * float64(time.Minute)))
	cwdCandidates := sweepCwdCandidates(home)
	fallbackCwd, _ := os.Getwd()

	var rows []sweep.Row
	excluded := 0
	for sid, copyPaths := range bySID {
		newest := time.Time{}
		for _, p := range copyPaths {
			if fi, err := os.Stat(p); err == nil && fi.ModTime().After(newest) {
				newest = fi.ModTime()
			}
		}
		if newest.Before(cutoff) {
			continue
		}
		if resumed[sid] {
			excluded++
			continue
		}
		copies := make([]sweep.Copy, 0, len(copyPaths))
		for _, p := range copyPaths {
			copies = append(copies, loadSweepCopy(p))
		}
		r := sweep.Classify(sid, copies, live, now)
		if r.Bucket == sweep.BucketOther {
			continue
		}
		r.CWD = sweep.CwdForSlug(r.Project, cwdCandidates, fallbackCwd)
		if *probe {
			if fp := fleetaccounts.FreshProbeFromLedger(r.SupersetAccount, "", now, 0); fp != nil {
				ok := fp.Available
				r.SeatOK = &ok
			}
		}
		rows = append(rows, r)
	}
	sweep.Sort(rows)

	droppedStub := 0
	if *minRecords > 0 {
		kept := rows[:0]
		for _, r := range rows {
			if r.NRecords >= *minRecords {
				kept = append(kept, r)
			} else {
				droppedStub++
			}
		}
		rows = kept
	}
	if *bucket != "" {
		kept := rows[:0]
		for _, r := range rows {
			if r.Bucket == *bucket {
				kept = append(kept, r)
			}
		}
		rows = kept
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"ts":                        now.Format("2006-01-02T15:04:05Z"),
			"window_min":                *window,
			"count":                     len(rows),
			"excluded_recently_resumed": excluded,
			"dropped_below_min_records": droppedStub,
			"rows":                      rows,
		}, "fak resume sweep")
	}
	renderResumeSweep(stdout, rows, now, *window, *minRecords, excluded, droppedStub, *includeResumed)
	return 0
}

// resolveSweepRegDir honors FLEET_REG_DIR exactly as the fleet tools do, defaulting to
// <repo>/tools/_registry — the dir the launchers write the resume ledger into.
func resolveSweepRegDir(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	if v := strings.TrimSpace(os.Getenv("FLEET_REG_DIR")); v != "" {
		return v
	}
	cwd, _ := os.Getwd()
	return filepath.Join(findRepoRoot(cwd), "tools", "_registry")
}

var sweepUUIDRE = regexp.MustCompile(`[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`)

// liveResumeSIDs is the set of session ids a running claude process is currently driving,
// read from the same audited cross-platform process census `fak resume admit` counts with
// (one enumeration implementation, not a fork of the Python's PowerShell scrape).
func liveResumeSIDs() map[string]bool {
	out := map[string]bool{}
	procs, _ := procguard.CollectRelations()
	for _, p := range procs {
		low := strings.ToLower(p.Cmdline)
		if !strings.Contains(low, "claude") {
			continue
		}
		for _, m := range sweepUUIDRE.FindAllString(low, -1) {
			out[m] = true
		}
	}
	return out
}

// sweepCwdCandidates enumerates plausible session working directories for the lossy
// project-slug match (the slug collapses separators and hyphens alike, so it can only be
// matched by re-slugifying real dirs, never string-reversed).
func sweepCwdCandidates(home string) []string {
	var pats []string
	if runtime.GOOS == "windows" {
		pats = []string{`C:\work\*`, `C:\Users\*`, `C:\*`}
	} else {
		pats = []string{filepath.Join(home, "work", "*"), filepath.Join(home, "*"), "/work/*"}
	}
	var out []string
	for _, pat := range pats {
		matches, _ := filepath.Glob(pat)
		for _, m := range matches {
			if fi, err := os.Stat(m); err == nil && fi.IsDir() {
				out = append(out, m)
			}
		}
	}
	return out
}

// loadSweepCopy parses one transcript copy into the closed record facts the sweep leaf
// classifies. Malformed lines are skipped (a live writer can leave a torn tail); an
// unreadable file yields an empty copy, which classifies as OTHER and drops out.
func loadSweepCopy(path string) sweep.Copy {
	c := sweep.Copy{
		Path:    path,
		Account: filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(path)))),
		Project: filepath.Base(filepath.Dir(path)),
	}
	f, err := os.Open(path)
	if err != nil {
		return c
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20) // a single tool-result line can be large
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr struct {
			UUID       string          `json:"uuid"`
			Timestamp  string          `json:"timestamp"`
			Type       string          `json:"type"`
			Role       string          `json:"role"`
			IsAPIError bool            `json:"isApiErrorMessage"`
			Content    json.RawMessage `json:"content"`
			Message    *struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal(line, &jr) != nil {
			continue
		}
		rec := sweep.Record{
			UUID:      jr.UUID,
			Timestamp: jr.Timestamp,
			Role:      jr.Role,
			IsError:   jr.Type == "error" || jr.IsAPIError,
		}
		if jr.Message != nil {
			rec.Role = jr.Message.Role
			rec.Text = textBlocksJoined(jr.Message.Content)
		} else {
			rec.Text = textBlocksJoined(jr.Content)
		}
		c.Records = append(c.Records, rec)
	}
	return c
}

// textBlocksJoined extracts the human text of a message content field — a bare string or
// an array of typed blocks, keeping only the "text" blocks, newline-joined (the Python
// _text contract, distinct from resume_scan's space-joined transcriptText).
func textBlocksJoined(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) == nil {
		var parts []string
		for _, b := range blocks {
			if b.Type == "text" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// renderResumeSweep prints the human summary: bucket counts, the batch-cohort groups, the
// per-session rows, and the action hints — the same reading order the Python emitted.
func renderResumeSweep(w io.Writer, rows []sweep.Row, now time.Time, window float64,
	minRecords, excluded, droppedStub int, includeResumed bool) {
	counts := map[string]int{}
	for _, r := range rows {
		counts[r.Bucket]++
	}
	hdr := fmt.Sprintf("resume sweep %s  window=%dm", now.Format("2006-01-02T15:04:05Z"), int(window))
	if minRecords > 0 {
		hdr += fmt.Sprintf("  min_records=%d", minRecords)
	}
	var keys []string
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		hdr += fmt.Sprintf("  %s=%d", k, counts[k])
	}
	fmt.Fprintln(w, hdr)
	if excluded > 0 && !includeResumed {
		fmt.Fprintf(w, "  (excluded %d session(s) already resumed in-window per the ledger — pass --include-resumed to show them)\n", excluded)
	}
	if droppedStub > 0 {
		fmt.Fprintf(w, "  (dropped %d micro-stub session(s) below %d records — raise/lower --min-records to see them)\n", droppedStub, minRecords)
	}
	diverged := 0
	for _, r := range rows {
		if r.ProseDiverged {
			diverged++
		}
	}
	if diverged > 0 {
		fmt.Fprintf(w, "  (averted %d prose-only false-positive(s): the final assistant prose alone would have mis-bucketed these, but the error channel overruled it — shown as [prose≠err])\n", diverged)
	}
	// Grouped view: project x bucket x account, so a batch-spawned cohort reads as ONE line.
	type grpKey struct{ proj, bkt, acct string }
	grp := map[grpKey]int{}
	for _, r := range rows {
		grp[grpKey{r.Project, r.Bucket, r.SupersetAccount}]++
	}
	var bigGroups []grpKey
	for k, n := range grp {
		if n >= 5 {
			bigGroups = append(bigGroups, k)
		}
	}
	if len(bigGroups) > 0 {
		sort.Slice(bigGroups, func(i, j int) bool { return grp[bigGroups[i]] > grp[bigGroups[j]] })
		fmt.Fprintln(w, "  groups (project / bucket / account >=5):")
		for _, k := range bigGroups {
			fmt.Fprintf(w, "     %3d  %-22s %-18s %s\n", grp[k], k.proj, k.bkt, k.acct)
		}
	}
	for _, r := range rows {
		seat := ""
		if r.SeatOK != nil {
			seat = fmt.Sprintf(" seat_ok=%v", *r.SeatOK)
		}
		sup := ""
		if !r.IsSuperset {
			sup = " NON-SUPERSET!"
		}
		div := ""
		if r.ProseDiverged {
			div = " [prose≠err]"
		}
		reset := r.Reset
		if reset == "" {
			reset = "-"
		}
		fmt.Fprintf(w, "  %-18s %s %-22s proj=%-20s n=%-4d reset=%-28s%s%s%s\n",
			r.Bucket, shortID(r.SID), r.SupersetAccount, r.Project, r.NRecords, reset, seat, sup, div)
	}
	if len(rows) == 0 {
		fmt.Fprintln(w, "  (no recently-crashed sessions in window)")
	}
	// Action hints: what an operator (or launcher) should do with each bucket.
	np := counts[sweep.BucketLimitResetPassed] + counts[sweep.BucketAPIErr]
	if np > 0 {
		fmt.Fprintf(w, "\n%d resumable NOW (LIMIT_RESET_PASSED + API_ERR) — pin to the superset_account if healthy, else re-home; launch sequential.\n", np)
	}
	if nf := counts[sweep.BucketLimitResetFuture]; nf > 0 {
		fmt.Fprintf(w, "%d waiting on a usage reset — resume in place after the named window.\n", nf)
	}
	if na := counts[sweep.BucketAuth]; na > 0 {
		fmt.Fprintf(w, "%d AUTH-walled — need `claude /login` (a re-resume on the same account can't fix it).\n", na)
	}
}
