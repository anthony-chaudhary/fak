package main

// resume_dedup.go — `fak resume dedup`, the ACTUATOR half of the crashed-duplicate dedup
// (#3146). The classifier half already ships: stopped.Decide marks a stopped row DUP_LIVE
// when a LIVE sibling owns the same (project, work-key). But the live resume watchdog
// builds its relaunch plan from its own ledger, never from that verdict — so a crashed
// duplicate of a live session was relaunched over and over, re-running work already in
// flight, until someone appended a manual_override tombstone to resume_ledger.jsonl BY
// HAND (the 734355cc incident this issue retires). This verb writes that exact row:
//
//	fak resume dedup                 # dry-run: list every crashed duplicate, write nothing
//	fak resume dedup --apply         # append one manual_override tombstone per duplicate
//	fak resume dedup --json          # the machine record
//
// The tombstone is the SAME shape the watchdog already honors — resume_blocked()'s
// operator-settled check reads any history row whose manual_override is truthy — so no
// watchdog change is needed. Idempotent by the watchdog's own reading: a session the
// ledger already settles (any manual_override/consolidate row AFTER its last rearm,
// exactly resume_blocked's trim) is never re-written. Fail-open by inheritance: an empty
// work-key never dedups (that fence lives in stopped.Decide), so a session whose work
// cannot be identified — or that does genuinely distinct work — is never tombstoned.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume/stopped"
)

// dedupTombstoneAction stamps every row this verb writes, so they are queryable in the
// ledger and so the watchdog's is_resume_attempt_record never mistakes one for a spawned
// resume attempt (an action outside its allowlist is not an attempt).
const dedupTombstoneAction = "dedup_tombstone"

// dedupTombstoneRow is the ledger row `--apply` appends: the manual_override shape the
// watchdog's honor point reads as operator-settled, with the dedup evidence (work-key,
// live owner, real stop-cause) carried on the row so a later reader sees WHY the session
// was settled instead of re-deriving it. Phase "skipped" is deliberate: it is the one
// value in EVERY ledger reader's non-launch set (the Python watchdog's NON_LAUNCH_PHASES,
// launch_admission's _NON_LAUNCH_PHASES, and isNonLaunchPhase here), so a tombstone never
// counts as launch pressure against any rate window or spacing floor.
type dedupTombstoneRow struct {
	Ts             string `json:"ts"`
	Phase          string `json:"phase"`
	Session        string `json:"session"`
	Account        string `json:"account,omitempty"`
	Project        string `json:"project,omitempty"`
	Action         string `json:"action"`
	ManualOverride bool   `json:"manual_override"`
	Reason         string `json:"reason"`
	WorkKey        string `json:"work_key,omitempty"`
	LiveOwner      string `json:"live_owner,omitempty"`
	Disp           string `json:"disp,omitempty"`
}

// dedupCandidate is one crashed duplicate: the DUP_LIVE row joined to the live owner's
// session id (Decide keeps only the shared work-key) and to whether the ledger already
// settles it (the idempotency verdict).
type dedupCandidate struct {
	Session   string `json:"session"`
	Account   string `json:"account,omitempty"`
	Project   string `json:"project,omitempty"`
	WorkKey   string `json:"work_key"`
	LiveOwner string `json:"live_owner,omitempty"`
	Disp      string `json:"disp"`
	Settled   bool   `json:"settled"`
	Path      string `json:"path,omitempty"`
}

// runResumeDedup lists (and with --apply, tombstones) every stopped session that
// duplicates a live one. Exit codes: 0 ok, 1 runtime error, 2 usage error.
func runResumeDedup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume dedup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		w := fs.Output()
		fmt.Fprint(w, `fak resume dedup [--window-h H] [--home DIR] [--ledger FILE] [--apply] [--json]

Auto-tombstone crashed sessions that duplicate a LIVE one (#3146): runs the exact
`+"`fak resume stopped`"+` classification, and for each stopped row whose (project,
work-key) a live sibling already owns, appends the manual_override tombstone the
resume watchdog honors — so the duplicate is never relaunched into work in flight.

Dry-run by default: lists each duplicate with the live owner's session id and the
shared work-key, writes nothing. --apply appends exactly one tombstone per duplicate;
re-running is idempotent (a session the ledger already settles is not re-written).
A session with a distinct work-key, or none at all, is NEVER tombstoned (fail-open).

`)
		fs.PrintDefaults()
	}
	windowH := fs.Float64("window-h", 10, "only sessions whose transcript changed within N hours")
	homeFlag := fs.String("home", "", "user home holding the .claude* account dirs (default: $FLEET_USER_HOME, else discovered)")
	ledger := fs.String("ledger", "", "resume ledger JSONL path (default: <reg-dir>/resume_ledger.jsonl — the file the watchdog reads)")
	apply := fs.Bool("apply", false, "append the tombstones (default: dry-run, write nothing)")
	asJSON := fs.Bool("json", false, "emit the machine record (candidates + verdicts)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak resume dedup: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	home := resolveFleetUserHome(*homeFlag, "FLEET_USER_HOME")
	if home == "" {
		fmt.Fprintln(stderr, "fak resume dedup: cannot resolve the user home (pass --home)")
		return 1
	}
	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(resolveSweepRegDir(""), "resume_ledger.jsonl")
	}
	now := time.Now().UTC()

	rows, _, _ := scanStoppedRows(home, *windowH, now)
	d := stopped.Decide(rows, stoppedThrottleActive(now))
	cands := planResumeDedup(d, ledgerSettledSessions(ledgerPath))

	wrote := 0
	if *apply {
		var err error
		if wrote, err = appendDedupTombstones(ledgerPath, cands, now); err != nil {
			fmt.Fprintf(stderr, "fak resume dedup: write tombstones to %s: %v\n", ledgerPath, err)
			return 1
		}
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"now_utc":      now.Format(time.RFC3339),
			"window_h":     *windowH,
			"ledger":       ledgerPath,
			"apply":        *apply,
			"n_duplicates": len(cands),
			"n_written":    wrote,
			"candidates":   cands,
		}, "fak resume dedup")
	}
	renderResumeDedup(stdout, cands, ledgerPath, now, *windowH, *apply, wrote)
	return 0
}

// planResumeDedup folds the triage decisions into the tombstone plan: every Skip row
// Decide marked DUP_LIVE, joined to the live owner's session id and marked settled when
// the ledger already blocks it. Pure — the callers own all I/O — so the never-tombstone
// fences (distinct work, empty work-key, the row not actually duplicating a live one)
// are unit-testable without an account tree.
func planResumeDedup(d stopped.Decisions, settled map[string]bool) []dedupCandidate {
	// The live owners, keyed exactly as Decide keys its owned set — (project, work-key) —
	// so the owner join can never pair a duplicate with a sibling Decide did not dedup on.
	owner := map[string]string{}
	for _, r := range d.Rows {
		if r.Disp == stopped.DispLive && r.WorkKey != "" {
			if _, ok := owner[r.Project+"\x00"+r.WorkKey]; !ok {
				owner[r.Project+"\x00"+r.WorkKey] = r.Session
			}
		}
	}
	var out []dedupCandidate
	for _, r := range d.Skip {
		if !r.DupOfLive {
			continue
		}
		out = append(out, dedupCandidate{
			Session:   r.Session,
			Account:   r.Account,
			Project:   r.Project,
			WorkKey:   r.WorkKey,
			LiveOwner: owner[r.Project+"\x00"+r.WorkKey],
			Disp:      string(r.Disp),
			Settled:   settled[r.Session],
			Path:      r.Path,
		})
	}
	return out
}

// ledgerSettledSessions reads the resume ledger once and returns the sessions the
// watchdog's honor point ALREADY blocks as operator-settled: any manual_override or
// consolidate row, considered only AFTER the session's last rearm — the same trim
// resume_blocked() applies — so "already tombstoned" here and "blocked" there can never
// disagree. The rearm arm must run FIRST: a rearm row itself carries manual_override
// (see resumeRearmRow), and reading that flag before the phase would settle the very
// session the rearm just re-armed. A missing or unreadable ledger settles nothing.
func ledgerSettledSessions(path string) map[string]bool {
	settled := map[string]bool{}
	f, err := os.Open(path)
	if err != nil {
		return settled
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var row struct {
			Session        string `json:"session"`
			Phase          string `json:"phase"`
			Outcome        string `json:"outcome"`
			Action         string `json:"action"`
			ManualOverride bool   `json:"manual_override"`
		}
		if json.Unmarshal(line, &row) != nil || row.Session == "" {
			continue
		}
		if row.Phase == "rearm" || row.Outcome == "rearm" {
			delete(settled, row.Session)
			continue
		}
		if row.ManualOverride || strings.HasPrefix(row.Action, "consolidate") {
			settled[row.Session] = true
		}
	}
	return settled
}

// appendDedupTombstones appends one tombstone per unsettled candidate and returns how
// many it wrote. Unlike the watchdog's best-effort rwAppendLedger, a failed write here
// is a hard error: the whole point of --apply is the durable row, and a silent miss
// would report a relaunch as stopped when the watchdog can still fire it.
func appendDedupTombstones(path string, cands []dedupCandidate, now time.Time) (int, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return 0, err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	wrote := 0
	for _, c := range cands {
		if c.Settled {
			continue
		}
		row := dedupTombstoneRow{
			Ts:             now.UTC().Format("2006-01-02T15:04:05Z"),
			Phase:          "skipped",
			Session:        c.Session,
			Account:        c.Account,
			Project:        c.Project,
			Action:         dedupTombstoneAction,
			ManualOverride: true,
			Reason:         "duplicate of live session " + c.LiveOwner + " owning the same work (" + c.WorkKey + ")",
			WorkKey:        c.WorkKey,
			LiveOwner:      c.LiveOwner,
			Disp:           c.Disp,
		}
		if err := enc.Encode(row); err != nil {
			return wrote, err
		}
		wrote++
	}
	return wrote, nil
}

// renderResumeDedup prints the operator record: one line per duplicate with the verdict
// (TOMBSTONE planned / TOMBSTONED written / SETTLED already blocked), the live owner, and
// the shared work-key — each tombstone is a relaunch the watchdog will no longer fire,
// which is seat quota not burned on work already in flight.
func renderResumeDedup(w io.Writer, cands []dedupCandidate, ledgerPath string, now time.Time, windowH float64, applied bool, wrote int) {
	pending := 0
	for _, c := range cands {
		if !c.Settled {
			pending++
		}
	}
	fmt.Fprintf(w, "resume dedup %s  window=%.0fh  duplicates=%d (unsettled=%d already-settled=%d)\n",
		now.Format("2006-01-02T15:04:05Z"), windowH, len(cands), pending, len(cands)-pending)
	for _, c := range cands {
		verdict := "TOMBSTONE"
		if c.Settled {
			verdict = "SETTLED"
		} else if applied {
			verdict = "TOMBSTONED"
		}
		fmt.Fprintf(w, "  %-10s %s %-22s proj=%-20s disp=%-16s work=%s  live-owner=%s\n",
			verdict, shortID(c.Session), c.Account, c.Project, c.Disp, c.WorkKey, shortID(c.LiveOwner))
	}
	if len(cands) == 0 {
		fmt.Fprintln(w, "  (no stopped session duplicates a live one — nothing to tombstone)")
		return
	}
	if applied {
		fmt.Fprintf(w, "  wrote %d tombstone(s) to %s — the resume watchdog now refuses each relaunch (operator-settled)\n",
			wrote, ledgerPath)
	} else {
		fmt.Fprintf(w, "  dry-run: wrote nothing — re-run with --apply to append %d manual_override tombstone(s) to %s\n",
			pending, ledgerPath)
	}
}
