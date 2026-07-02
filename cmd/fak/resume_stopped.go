package main

// resume_stopped.go — `fak resume stopped`, the stopped-session triage across all local
// accounts: find the recently-STOPPED top-level Claude Code sessions and decide which are
// safe to resume headlessly, which must wait (account throttled / auth-walled), and which
// to leave alone (live / parked / done). Go port of tools/stopped_sessions.py (the shell
// half; the classification/decision core is the pure internal/resume/stopped leaf).
//
//	fak resume stopped                  # human triage (default 10h window)
//	fak resume stopped --window-h 24
//	fak resume stopped --json           # the full machine record
//
// Where `fak resume sweep` finds CRASHED sessions by their terminal error record, this
// verb triages every stopped session by HOW it stopped — the mid-tool deaths and quiet
// stops that carry no error banner at all — and folds per-ACCOUNT throttle state so a
// resumable session on a capped account defers instead of burning a doomed launch.
//
// This shell does only the I/O the leaf must not: enumerate the account dirs (skipping
// the ones policy tombstones — the same worker classification `fak accounts` uses), tail
// each transcript (last 512KB: the terminal turns are all the classifier reads), extract
// the per-record facts, and evaluate each throttle reset against the clock. It RESUMES
// NOTHING; the decisions feed a gated launcher.

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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/resume/stopped"
	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

var stoppedUUIDStemRE = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// runResumeStopped classifies and triages the recently-stopped sessions. Exit codes:
// 0 ok, 1 runtime error, 2 usage error.
func runResumeStopped(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume stopped", flag.ContinueOnError)
	fs.SetOutput(stderr)
	windowH := fs.Float64("window-h", 10, "only sessions whose transcript changed within N hours")
	asJSON := fs.Bool("json", false, "emit the full machine record (rows + decisions)")
	homeFlag := fs.String("home", "", "user home holding the .claude* account dirs (default: $FLEET_USER_HOME, else discovered)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	home := *homeFlag
	if home == "" {
		home = strings.TrimSpace(os.Getenv("FLEET_USER_HOME"))
	}
	if home == "" {
		if h, err := os.UserHomeDir(); err == nil {
			home = h
		}
	}
	if home == "" {
		fmt.Fprintln(stderr, "fak resume stopped: cannot resolve the user home (pass --home)")
		return 1
	}
	now := time.Now().UTC()

	// Account policy: only offered worker accounts enter the triage (the tombstoned /
	// excluded ones are exactly the seats a resume must not target).
	workerDirs := workerAccountDirs(home)

	var rows []stopped.Row
	for acctDir, acct := range workerDirs {
		proj := filepath.Join(acctDir, "projects")
		paths, _ := filepath.Glob(filepath.Join(proj, "*", "*.jsonl"))
		for _, path := range paths {
			stem := strings.TrimSuffix(filepath.Base(path), ".jsonl")
			if !stoppedUUIDStemRE.MatchString(stem) {
				continue // subagent/sidecar files — only top-level sessions triage
			}
			if strings.HasPrefix(filepath.Base(filepath.Dir(path)), "wf_") {
				continue // workflow transcript stores are not resumable sessions
			}
			fi, err := os.Stat(path)
			if err != nil {
				continue
			}
			ageMin := now.Sub(fi.ModTime().UTC()).Minutes()
			if ageMin > *windowH*60 {
				continue
			}
			recs := loadStoppedRecords(path)
			r := stopped.Classify(recs, ageMin, fi.Size()/1024,
				fi.ModTime().UTC().Format(time.RFC3339), stem, path)
			r.Account = acct
			r.Project = filepath.Base(filepath.Dir(path))
			rows = append(rows, r)
		}
	}

	// A reset window still blocks when it has not provably passed; an unparseable reset
	// is conservatively active (the Python throttle_is_active contract).
	throttleActive := func(reset string) bool {
		passed, ok := sessionsignals.ResetPassed(reset, now, now)
		return !ok || !passed
	}
	d := stopped.Decide(rows, throttleActive)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"now_utc":          now.Format(time.RFC3339),
			"window_h":         *windowH,
			"account_throttle": d.AccountThrottle,
			"counts":           d.Counts,
			"n_resume":         len(d.Resume),
			"n_defer":          len(d.Defer),
			"n_skip":           len(d.Skip),
			"resume":           d.Resume,
			"defer":            d.Defer,
			"rows":             d.Rows,
		}, "fak resume stopped")
	}
	renderResumeStopped(stdout, d, now, *windowH)
	return 0
}

// workerAccountDirs maps each offered worker account's config dir to its basename, using
// the same policy classification `fak accounts` runs (exclude tombstones, include_only
// allowlist) so this triage and the roster can never disagree about who is a worker.
func workerAccountDirs(home string) map[string]string {
	cwd, _ := os.Getwd()
	paths := fleetaccounts.ResolvePaths(filepath.Join(findRepoRoot(cwd), "tools"))
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	out := map[string]string{}
	for _, a := range fleetaccounts.AnnotatedRoster(home, paths.ConfigHome, pol, reg) {
		if a.Kind == fleetaccounts.KindWorker && a.Dir != "" {
			out[a.Dir] = a.Account
		}
	}
	return out
}

// loadStoppedRecords tails a transcript (last 512KB — the terminal turns are all the
// classifier reads) and extracts the closed per-record facts the stopped leaf needs. A
// torn first line or malformed row is skipped, never fatal.
func loadStoppedRecords(path string) []stopped.Record {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	const tailBytes = 512 * 1024
	if fi, err := f.Stat(); err == nil && fi.Size() > tailBytes {
		_, _ = f.Seek(fi.Size()-tailBytes, io.SeekStart)
	}
	var recs []stopped.Record
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr struct {
			Type      string `json:"type"`
			Timestamp string `json:"timestamp"`
			CWD       string `json:"cwd"`
			GitBranch string `json:"gitBranch"`
			Version   string `json:"version"`
			SessionID string `json:"sessionId"`
			Message   *struct {
				Role    string          `json:"role"`
				Model   string          `json:"model"`
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal(line, &jr) != nil {
			continue
		}
		rec := stopped.Record{
			Type: jr.Type, Timestamp: jr.Timestamp,
			CWD: jr.CWD, GitBranch: jr.GitBranch, Version: jr.Version, SessionID: jr.SessionID,
		}
		if m := jr.Message; m != nil {
			rec.Role = m.Role
			rec.Synthetic = m.Model == "<synthetic>"
			rec.Text, rec.ToolUseName, rec.HasToolResult = stoppedContentFacts(m.Content)
		}
		recs = append(recs, rec)
	}
	return recs
}

// stoppedContentFacts folds a message content field into the three facts the classifier
// needs: the human text (text blocks + tool_result payloads, space-joined — the Python
// text_of contract), the LAST tool_use block's name, and whether any tool_result block is
// present.
func stoppedContentFacts(raw json.RawMessage) (text, lastToolUse string, hasToolResult bool) {
	if len(raw) == 0 {
		return "", "", false
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, "", false
	}
	var blocks []struct {
		Type    string          `json:"type"`
		Text    string          `json:"text"`
		Name    string          `json:"name"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return "", "", false
	}
	var parts []string
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				parts = append(parts, b.Text)
			}
		case "tool_result":
			hasToolResult = true
			if t, _, _ := stoppedContentFacts(b.Content); t != "" {
				parts = append(parts, t)
			}
		case "tool_use":
			lastToolUse = b.Name
		}
	}
	return strings.Join(parts, " "), lastToolUse, hasToolResult
}

// renderResumeStopped prints the operator triage: the counts, the account throttles, and
// the three action buckets with the reason each deferred row is blocked.
func renderResumeStopped(w io.Writer, d stopped.Decisions, now time.Time, windowH float64) {
	fmt.Fprintf(w, "resume stopped %s  window=%.0fh  resume=%d defer=%d skip=%d\n",
		now.Format("2006-01-02T15:04:05Z"), windowH, len(d.Resume), len(d.Defer), len(d.Skip))
	if len(d.AccountThrottle) > 0 {
		fmt.Fprintln(w, "  account throttles (most-recent active banner per account):")
		for acct, thr := range d.AccountThrottle {
			fmt.Fprintf(w, "     %-24s resets %s  (seen %.0fm ago)\n", acct, thr.Reset, thr.AgeMin)
		}
	}
	section := func(title string, rows []stopped.Row) {
		if len(rows) == 0 {
			return
		}
		fmt.Fprintf(w, "  %s:\n", title)
		for _, r := range rows {
			blocked := ""
			if r.BlockedBy != "" {
				blocked = "  [" + r.BlockedBy + "]"
			}
			fmt.Fprintf(w, "     %-18s %s %-22s proj=%-20s age=%-6.0fm pending=%s%s\n",
				r.Disp, shortID(r.Session), r.Account, r.Project, r.AgeMin,
				orDash(r.PendingTool), blocked)
		}
	}
	section("RESUME (safe to resume headlessly now)", d.Resume)
	section("DEFER (blocked; resume after the named wall clears)", d.Defer)
	section("SKIP (live / parked / done — leave alone)", d.Skip)
	if len(d.Rows) == 0 {
		fmt.Fprintln(w, "  (no stopped sessions in window)")
	}
}
