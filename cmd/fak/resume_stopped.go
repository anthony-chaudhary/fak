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
	if len(argv) > 0 && argv[0] == "selfcheck" {
		return runReportSelfcheck(stdout, stderr, argv[1:], "resume stopped", stopped.TriageSelfcheck,
			"SELFCHECK OK -- decenter-the-human at resume-stopped: an auth/subscription wall "+
				"waits on a person; an account/session throttle clears on its own and the fleet waits.")
	}
	fs := flag.NewFlagSet("resume stopped", flag.ContinueOnError)
	fs.SetOutput(stderr)
	// --help documents the TWO independent axes each row carries (#3800): the flag dump
	// alone cannot teach that disp is single-axis and the dedup verdict rides its own fields.
	fs.Usage = func() {
		w := fs.Output()
		fmt.Fprint(w, `fak resume stopped [--window-h H] [--home DIR] [--json] — stopped-session triage

Each row carries TWO independent axes (#3800):
  disp          the single-axis stop-cause ONLY: STOPPED_LIMIT / STOPPED_AUTH /
                STOPPED_MIDTOOL / MIDTOOL_UNKNOWN / STOPPED_INTERRUPT / STOPPED_MIDTURN /
                STOPPED_DONE / STOPPED_QUIET / PARKED_WAIT / DONE / LIVE. When a terminal
                turn carries BOTH an auth wall and a current limit banner, auth wins (a
                login wall outlives any reset) and the outranked limit is retained on
                also_signals. A tool_use with no tool_result is reported as a CRASH
                (STOPPED_MIDTOOL) only with evidence the driver process is gone; the same
                tail is also what a driver still inside a SLOW tool call leaves, so with no
                such evidence the row is MIDTOOL_UNKNOWN and DEFERS rather than being
                resumed onto a transcript its original process may still be writing (#5386).
                This triage OBSERVES that evidence (#5440): a running process naming the
                session, or the driver pid a launcher recorded, read from the host process
                table. It claims live/gone only when one of those is actually witnessed —
                never from age, mtime or size — so an unreadable/incomplete process table or
                a session with no recorded driver pid still reads MIDTOOL_UNKNOWN and defers.
                The liveness field echoes which evidence decided the row, and --json carries
                the per-session reason under driver_liveness.
  dup_of_live   the dedup verdict: a stopped row whose (project, work-key) a LIVE
                sibling already owns is skipped, with the owning key on live_sibling —
                disp KEEPS the real stop-cause; a duplicate never masks WHY it stopped.

`)
		fs.PrintDefaults()
	}
	windowH := fs.Float64("window-h", 10, "only sessions whose transcript changed within N hours")
	asJSON := fs.Bool("json", false, "emit the full machine record (rows + decisions)")
	homeFlag := fs.String("home", "", "user home holding the .claude* account dirs (default: $FLEET_USER_HOME, else discovered)")
	if !parseFlags(fs, argv) {
		return 2
	}

	home := resolveFleetUserHome(*homeFlag, "FLEET_USER_HOME")
	if home == "" {
		fmt.Fprintln(stderr, "fak resume stopped: cannot resolve the user home (pass --home)")
		return 1
	}
	now := time.Now().UTC()
	rows, drivers, livenessWhy := scanStoppedRows(home, *windowH, now)
	d := stopped.Decide(rows, stoppedThrottleActive(now))

	if *asJSON {
		df := drivers.facts()
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"now_utc":  now.Format(time.RFC3339),
			"window_h": *windowH,
			// driver_liveness is the observability record for the evidence that decided every
			// mid-tool row: what the process table could see, what it could NOT examine, and
			// the per-session reason. It is emitted even when nothing was witnessable, so an
			// unobservable host is visible rather than indistinguishable from a quiet one.
			"driver_liveness": map[string]any{
				"readable":             df.readable,
				"scanned":              df.scanned,
				"self_seen":            df.selfSeen,
				"cmdline_not_examined": df.cmdlineUnread,
				"scan_error":           df.scanErr,
				"recorded_driver_pids": len(df.launchPIDs),
				// How many of those pids came from the session-start identity store rather than
				// the launch ledger — the first-generation population that had no handle on a
				// process at all before #5542.
				"identity_driver_pids": df.identityPIDs,
				"summary":              df.summary(),
				"reasons":              livenessWhy,
			},
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
	// Decenter the human in the human render: under FAK_RESUME_TRIAGE_GATE=enforce the
	// DEFER bucket splits into the rows that genuinely need a person (an auth wall) and
	// the throttle/limit rows that clear on their own — so an operator is not told to
	// babysit a wall that would have cleared without them. Default ("", "warn") keeps
	// the single DEFER section so the change can soak.
	renderResumeStopped(stdout, d, now, *windowH, stopped.TriageEnforced(os.Getenv("FAK_RESUME_TRIAGE_GATE")),
		drivers.facts().summary())
	return 0
}

// scanStoppedRows walks the offered worker accounts under home and classifies every
// top-level session transcript touched within the window. It returns the classified rows,
// the driver probe (so a caller that reports liveness evidence can read the same facts the
// classification used), and the per-session liveness reason.
//
// It is a shared helper rather than inline code because `fak resume dedup` (#3146) must run
// the EXACT classification `fak resume stopped` runs: the dedup actuator writes a tombstone
// that stops a relaunch, so a scan that drifted from the triage's could tombstone a session
// the triage would have resumed. One body, one verdict.
func scanStoppedRows(home string, windowH float64, now time.Time) ([]stopped.Row, *stoppedDriverProbe, map[string]string) {
	// Account policy: only offered worker accounts enter the triage (the tombstoned /
	// excluded ones are exactly the seats a resume must not target).
	workerDirs := workerAccountDirs(home)

	// Driver-liveness evidence for the mid-tool branch (#5440). Taken once, lazily, from the
	// host process table plus the durable launch record; see resume_stopped_liveness.go for
	// why only positive evidence ever produces a non-empty value.
	drivers := newStoppedDriverProbe()
	livenessWhy := map[string]string{}

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
			if ageMin > windowH*60 {
				continue
			}
			recs := loadStoppedRecords(path)
			// The mid-tool tail is ambiguous by construction, so hand the classifier the
			// observed driver-liveness instead of the LivenessUnknown the Classify wrapper
			// hard-codes (#5440). live/gone are asserted only when the process table actually
			// witnessed one; every unwitnessable shape stays LivenessUnknown and defers.
			live, why := drivers.facts().livenessFor(stem)
			livenessWhy[stem] = why
			r := stopped.ClassifyWithLiveness(recs, ageMin, fi.Size()/1024,
				fi.ModTime().UTC().Format(time.RFC3339), stem, path, live)
			r.Account = acct
			r.Project = filepath.Base(filepath.Dir(path))
			// Work-key for cross-session dedup: the authoritative /goal, /loop lane, or issue
			// number from the transcript's FIRST user turn (read from the head — the classifier
			// only saw the noisy tail). Empty when none is found; an empty key never dedups.
			r.WorkKey = firstTurnWorkKey(path)
			rows = append(rows, r)
		}
	}

	return rows, drivers, livenessWhy
}

// stoppedThrottleActive is the reset-window predicate stopped.Decide takes: a window still
// blocks when it has not provably passed, and an unparseable reset is conservatively active
// (the Python throttle_is_active contract).
func stoppedThrottleActive(now time.Time) func(string) bool {
	return func(reset string) bool {
		passed, ok := sessionsignals.ResetPassed(reset, now, now)
		return !ok || !passed
	}
}

// resolveFleetUserHome resolves the --home flag for the resume verbs, falling back to the OS
// user home. envKey, when non-empty, is consulted BETWEEN the two.
//
// The env step is a parameter rather than a constant because the two callers genuinely
// differ: `fak resume stopped` honours FLEET_USER_HOME, `fak resume sweep` never has. That
// asymmetry is preserved here verbatim — arming the env var for the sweep would silently
// change which account tree it walks on any host that exports it, which is a behaviour
// change that needs its own witness, not a refactor's side effect.
func resolveFleetUserHome(flagVal, envKey string) string {
	home := flagVal
	if home == "" && envKey != "" {
		home = strings.TrimSpace(os.Getenv(envKey))
	}
	if home == "" {
		if h, err := os.UserHomeDir(); err == nil {
			home = h
		}
	}
	return home
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

// workKeyGoalRE / workKeyLoopRE / workKeyIssueRE recognize the three authoritative work
// identities a headless session is launched with, most specific first. A /goal names a
// free-text objective (its args are the identity); a /loop or dos-dispatch-loop names a
// lane; an issue-resolution names the issue number. These are the launch contracts, so two
// sessions with the same one are doing the same work.
var (
	// A loop session's identity is its lane. It appears either as a slash-command arg block
	// (<command-args>--lane X</command-args>, tags newline-separated) or as a bare "--lane X"
	// in prose; both forms are covered. The lane is the last --lane token before the key is
	// taken (a re-fired loop turn restates it), so a plain scan for the flag suffices.
	workKeyLaneRE = regexp.MustCompile(`(?i)--lane[\s=]+([a-z0-9_-]+)`)
	// A loop launch is marked by the dispatch-loop skill name or a /loop command; without it,
	// a stray "--lane" mention in prose must NOT be read as a loop identity.
	workKeyLoopMarkerRE = regexp.MustCompile(`(?i)dos-dispatch-loop|/loop\b|/dispatch\b`)
	workKeyIssueRE      = regexp.MustCompile(`(?i)resolve\s+GitHub\s+issue\s+#?(\d+)|(?:^|\s)issue\s+#(\d+)|GH-(\d+)`)
	workKeyGoalRE       = regexp.MustCompile(`(?is)<command-name>\s*/goal\s*</command-name>.*?<command-args>\s*(.*?)\s*</command-args>`)
	workKeyWS           = regexp.MustCompile(`\s+`)
)

// deriveWorkKey folds one user-turn text into a canonical work identity, or "" when the
// text carries none. Pure and total so it is unit-testable without a transcript. Order is
// deliberate: an explicit /goal wins over a /loop lane wins over a bare issue mention, so
// an issue-resolution session that happens to discuss a lane is keyed by its issue.
func deriveWorkKey(text string) string {
	if m := workKeyGoalRE.FindStringSubmatch(text); m != nil {
		g := strings.ToLower(workKeyWS.ReplaceAllString(strings.TrimSpace(m[1]), " "))
		if len(g) > 80 {
			g = g[:80]
		}
		if g != "" {
			return "goal:" + g
		}
	}
	if m := workKeyIssueRE.FindStringSubmatch(text); m != nil {
		for _, g := range m[1:] {
			if g != "" {
				return "issue:#" + g
			}
		}
	}
	if workKeyLoopMarkerRE.MatchString(text) {
		if m := workKeyLaneRE.FindStringSubmatch(text); m != nil {
			return "loop:--lane " + strings.ToLower(m[1])
		}
	}
	return ""
}

// firstTurnWorkKey reads the HEAD of a transcript (the launch turns, where the /goal, /loop,
// or issue assignment lives — the classifier only saw the tail) and returns the work-key of
// the first user turn that carries one. Bounded read; a torn/oversized line is skipped, not
// fatal; "" when nothing matches (which never dedups — fail-open).
func firstTurnWorkKey(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(io.LimitReader(f, 512*1024))
	sc.Buffer(make([]byte, 0, 1<<20), 8<<20)
	scanned := 0
	for sc.Scan() && scanned < 40 {
		scanned++
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr struct {
			Type    string `json:"type"`
			Message *struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal(line, &jr) != nil {
			continue
		}
		if jr.Type != "user" || jr.Message == nil {
			continue
		}
		text, _, _ := stoppedContentFacts(jr.Message.Content)
		if key := deriveWorkKey(text); key != "" {
			return key
		}
	}
	return ""
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

// renderResumeStopped prints the operator triage: the counts, the driver-liveness evidence
// state, the account throttles, and the three action buckets with the reason each deferred
// row is blocked. driverNote states what the process table could and could not see, so a
// host where liveness is unobservable is visibly unobservable rather than silently
// deferring every mid-tool row (#5440).
func renderResumeStopped(w io.Writer, d stopped.Decisions, now time.Time, windowH float64, triage bool, driverNote string) {
	fmt.Fprintf(w, "resume stopped %s  window=%.0fh  resume=%d defer=%d skip=%d\n",
		now.Format("2006-01-02T15:04:05Z"), windowH, len(d.Resume), len(d.Defer), len(d.Skip))
	if driverNote != "" {
		fmt.Fprintf(w, "  %s\n", driverNote)
	}
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
	if triage {
		// Split the DEFER bucket: an auth/subscription wall genuinely needs a person;
		// a throttle/limit/structural wall clears on its own and the fleet waits behind
		// it, so it is not an operator page.
		need, wait := stopped.PartitionDefer(d)
		section("DEFER — NEEDS YOU (auth/subscription wall; a person must clear it)", need)
		section("DEFER — auto-clears (throttle/limit/structural; the fleet waits, no page)", wait)
	} else {
		section("DEFER (blocked; resume after the named wall clears)", d.Defer)
	}
	section("SKIP (live / parked / done / duplicate-of-live — leave alone)", d.Skip)
	if len(d.Rows) == 0 {
		fmt.Fprintln(w, "  (no stopped sessions in window)")
	}
}
