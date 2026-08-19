package main

// sessionjournal.go — `fak sessionjournal`, the crash-survivable session-registration
// journal and its monitoring/resume sidecar (epic #3784, C1 #3785). It writes boot-stamped
// lifecycle events (open/beat/close) to an append-only JSONL log and folds them — together
// with the existing guard_sessions.jsonl index (#3461) — into a boot-epoch verdict:
//
//	fak sessionjournal report              # classify recorded sessions LIVE/CRASHED/STALE/CLOSED
//	fak sessionjournal report --crashed    # just the resume-candidate set (with each cwd)
//	fak sessionjournal open  --id <sid>    # register a session at start (boot-stamped)
//	fak sessionjournal beat  --id <sid>    # heartbeat
//	fak sessionjournal close --id <sid>    # clean deregister at graceful exit
//
// The keystone is the boot epoch: a session that started BEFORE the machine's current boot
// and was never cleanly closed died in the reboot (a Windows-update reboot, a WindowsTerminal
// 0xc0000005 that killed every terminal at once). See
// docs/notes/CONCEPT-SESSION-CRASH-JOURNAL-BOOT-EPOCH-2026-07-09.md. This slice is
// detection only — driving the resume from the crashed set is C4 (#3788).

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func cmdSessionJournal(argv []string) { os.Exit(runSessionJournal(os.Stdout, os.Stderr, argv)) }

func runSessionJournal(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		switch argv[0] {
		case "report", "ls", "status":
			return runSessionJournalReport(stdout, stderr, argv[1:])
		case "open":
			return runSessionJournalWrite(stdout, stderr, sessionjournal.KindOpen, argv[1:])
		case "beat":
			return runSessionJournalWrite(stdout, stderr, sessionjournal.KindBeat, argv[1:])
		case "close":
			return runSessionJournalWrite(stdout, stderr, sessionjournal.KindClose, argv[1:])
		case "help":
			sessionJournalUsage(stdout)
			return 0
		default:
			fmt.Fprintf(stderr, "fak sessionjournal: unknown subcommand %q\n", argv[0])
			sessionJournalUsage(stderr)
			return 2
		}
	}
	// No subcommand (or leading flags) defaults to the report view.
	return runSessionJournalReport(stdout, stderr, argv)
}

func sessionJournalUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak sessionjournal [report|open|beat|close] [flags]")
	fmt.Fprintln(w, "  report   classify recorded sessions (LIVE/CRASHED/STALE/CLOSED) against the boot epoch")
	fmt.Fprintln(w, "  open     register a session at start (boot-stamped)   --id <sid> [--cwd --model --account ...]")
	fmt.Fprintln(w, "  beat     heartbeat                                    --id <sid>")
	fmt.Fprintln(w, "  close    clean deregister at graceful exit            --id <sid> [--reason ...]")
}

func sessionJournalHost() string {
	h, _ := os.Hostname()
	if h = strings.TrimSpace(h); h == "" {
		return "localhost"
	}
	return h
}

// runSessionJournalWrite appends one lifecycle event, stamping the current boot id and time.
func runSessionJournalWrite(stdout, stderr io.Writer, kind sessionjournal.Kind, argv []string) int {
	fs := flag.NewFlagSet("sessionjournal "+string(kind), flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("path", "", "journal path (default: $FAK_SESSION_JOURNAL, else <UserConfigDir>/fak/session-journal.jsonl)")
	id := fs.String("id", "", "session / trace id (required)")
	pid := fs.Int("pid", os.Getpid(), "process id")
	cwd := fs.String("cwd", "", "working dir (open; default: process cwd)")
	model := fs.String("model", "", "model (open)")
	agent := fs.String("agent", "", "wrapped agent (open)")
	account := fs.String("account", "", "config dir / account seat (open)")
	gateway := fs.String("gateway", "", "gateway addr (open)")
	startSHA := fs.String("start-sha", "", "git HEAD at start (open)")
	reason := fs.String("reason", "", "reason (close)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*id) == "" {
		fmt.Fprintln(stderr, "fak sessionjournal: --id is required")
		return 2
	}
	now := time.Now().UTC()
	bt, _ := sessionjournal.BootTime(now)
	ev := sessionjournal.Event{
		Kind:   kind,
		ID:     strings.TrimSpace(*id),
		TS:     now.Format(time.RFC3339),
		Boot:   sessionjournal.BootID(bt),
		PID:    *pid,
		Host:   sessionJournalHost(),
		Reason: strings.TrimSpace(*reason),
	}
	if kind == sessionjournal.KindOpen {
		c := strings.TrimSpace(*cwd)
		if c == "" {
			c, _ = os.Getwd()
		}
		ev.CWD = c
		ev.Model = strings.TrimSpace(*model)
		ev.Agent = strings.TrimSpace(*agent)
		ev.Account = strings.TrimSpace(*account)
		ev.Gateway = strings.TrimSpace(*gateway)
		ev.StartSHA = strings.TrimSpace(*startSHA)
		ev.Argv = fs.Args()
	}
	if err := sessionjournal.Append(*path, ev); err != nil {
		fmt.Fprintf(stderr, "fak sessionjournal: append failed: %v\n", err)
		return 1
	}
	resolved := *path
	if strings.TrimSpace(resolved) == "" {
		resolved = sessionjournal.DefaultPath()
	}
	fmt.Fprintf(stdout, "%s %s -> %s\n", kind, ev.ID, resolved)
	return 0
}

// runSessionJournalReport folds the lifecycle journal + guard_sessions.jsonl and classifies
// each session against the current boot epoch, surfacing the CRASHED (resume-candidate) set.
func runSessionJournalReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sessionjournal report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("path", "", "lifecycle journal path (default: DefaultPath)")
	regDir := fs.String("reg-dir", "", "registry dir holding guard_sessions.jsonl (default: $FLEET_REG_DIR, else the host registry)")
	noGuard := fs.Bool("no-guard-sessions", false, "do not fold in guard_sessions.jsonl")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	staleMin := fs.Int("stale-after-min", 15, "same-boot last-seen older than this (minutes) -> STALE (0 disables)")
	all := fs.Bool("all", false, "include CLOSED sessions")
	crashedOnly := fs.Bool("crashed", false, "only the CRASHED (resume-candidate) set")
	causes := fs.Bool("causes", false, "correlate CRASHED rows with Windows Event-1000 causes")
	causeSince := fs.Duration("cause-since", 24*time.Hour, "Windows event-log lookback for crash causes")
	bootOverride := fs.String("boot-time", "", "override the machine boot instant (RFC3339) — for folding a journal copied from a crashed box")
	if !parseFlags(fs, argv) {
		return 2
	}

	now := time.Now().UTC()
	bt, bootSrc := sessionjournal.BootTime(now)
	if s := strings.TrimSpace(*bootOverride); s != "" {
		if t, err := time.Parse(time.RFC3339, s); err == nil {
			bt, bootSrc = t.UTC(), "override"
		} else {
			fmt.Fprintf(stderr, "fak sessionjournal: bad --boot-time %q: %v\n", s, err)
			return 2
		}
	}

	// Source 1: the lifecycle journal (rich, has clean-close).
	jp := *path
	if strings.TrimSpace(jp) == "" {
		jp = sessionjournal.DefaultPath()
	}
	sessions := sessionjournal.FoldEvents(sessionjournal.LoadFile(jp))

	// Source 2: the existing guard_sessions.jsonl index as open-only records, so the report
	// is useful over fleet data recorded before this journal existed.
	if !*noGuard {
		rd := resolveSweepRegDir(*regDir)
		for _, r := range guardsessions.Load(rd) {
			sessions = mergeSession(sessions, guardRowToSession(r))
		}
	}

	cfg := sessionjournal.ClassifyConfig{
		Now:        now,
		BootTime:   bt,
		StaleAfter: time.Duration(*staleMin) * time.Minute,
		// PIDAlive stays nil in the foundation — the machine-wide-reboot test needs no live
		// process, and same-boot PID reconciliation is C2/C4. So a same-boot un-closed session
		// with no recent beat reports STALE (honest ambiguity), not a guessed CRASHED.
	}
	classified := sessionjournal.Classify(sessions, cfg)
	classified = filterClassified(classified, *all, *crashedOnly)
	sortClassified(classified)
	counts := sessionjournal.Counts(sessionjournal.Classify(sessions, cfg))
	var crashCauses []event1000Cause
	if *causes {
		records, reason := readEvent1000Records(*causeSince)
		if reason != "" {
			fmt.Fprintln(stderr, "fak sessionjournal report:", reason)
			return 1
		}
		crashCauses = matchEvent1000Crashes(classified, consoleFaultEventsFromWinRecords(records, now.UnixMilli()), event1000WindowMS)
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":      "fak.sessionjournal.report.v1",
			"boot_time":   bootTimeString(bt),
			"boot_source": bootSrc,
			"boot_id":     sessionjournal.BootID(bt),
			"journal":     jp,
			"counts": map[string]int{
				"live":    counts[sessionjournal.StatusLive],
				"crashed": counts[sessionjournal.StatusCrashed],
				"stale":   counts[sessionjournal.StatusStale],
				"closed":  counts[sessionjournal.StatusClosed],
			},
			"sessions": reportRows(classified),
			"causes":   crashCauses,
		}, "fak sessionjournal")
	}

	fmt.Fprintf(stdout, "boot: %s (%s)  |  live=%d crashed=%d stale=%d closed=%d\n",
		bootTimeString(bt), bootSrc,
		counts[sessionjournal.StatusLive], counts[sessionjournal.StatusCrashed],
		counts[sessionjournal.StatusStale], counts[sessionjournal.StatusClosed])
	if len(classified) == 0 {
		fmt.Fprintf(stdout, "no recorded sessions (journal: %s)\n", jp)
		return 0
	}
	renderSessionJournalTable(stdout, classified)
	for _, cause := range crashCauses {
		fmt.Fprintf(stdout, "CAUSE\t%s\t%s\t%s\t%s\n", cause.SessionID, cause.Cause, cause.Tool, cause.Detail)
	}
	if counts[sessionjournal.StatusCrashed] > 0 && !*crashedOnly {
		fmt.Fprintln(stdout, "\nCRASHED rows are resume candidates — each carries the cwd and remaining drive (budget/spend/generation) to relaunch from; DRIVE=- means no carried drive (legacy row).")
	}
	return 0
}

// sessionJournalReportRow is a classified session plus the resume_with_drive flag the C4
// relaunch pipeline (#3788) filters on. Classified is embedded (no json tag) so every field
// it carries — including the `drive` block E1 (#4129) adds to Session — flattens into the
// report envelope unchanged; this only ADDS the boolean.
type sessionJournalReportRow struct {
	sessionjournal.Classified
	ResumeWithDrive bool `json:"resume_with_drive"`
}

// reportRows projects the classified set into the JSON rows, stamping resume_with_drive on
// each. It is a CRASHED (resume-candidate) row that actually carries a drive-state — the one
// the pipeline can relaunch at the right remaining budget rather than a fresh full one.
func reportRows(cs []sessionjournal.Classified) []sessionJournalReportRow {
	rows := make([]sessionJournalReportRow, len(cs))
	for i, c := range cs {
		rows[i] = sessionJournalReportRow{
			Classified:      c,
			ResumeWithDrive: c.Status == sessionjournal.StatusCrashed && c.Drive != nil,
		}
	}
	return rows
}

func bootTimeString(bt time.Time) string {
	if bt.IsZero() {
		return "unknown"
	}
	return bt.Format(time.RFC3339)
}

// guardRowToSession maps a guard_sessions.jsonl row into a Session. The unique Handle is the
// id (a trace id can repeat as the default "guard"); guard rows carry no heartbeat, so
// last-seen is the start.
func guardRowToSession(r guardsessions.Row) sessionjournal.Session {
	var started time.Time
	if t, err := time.Parse(time.RFC3339, strings.TrimSpace(r.StartedAt)); err == nil {
		started = t.UTC()
	}
	id := strings.TrimSpace(r.Handle)
	if id == "" {
		id = strings.TrimSpace(r.TraceID)
	}
	return sessionjournal.Session{
		ID:        id,
		PID:       r.PID,
		CWD:       r.CWD,
		Agent:     r.Agent,
		StartedAt: started,
		LastSeen:  started,
	}
}

// mergeSession adds cand unless the id already folded from the richer lifecycle journal
// (which wins — it may carry a clean close the guard index lacks).
func mergeSession(existing []sessionjournal.Session, cand sessionjournal.Session) []sessionjournal.Session {
	for _, s := range existing {
		if s.ID == cand.ID {
			return existing
		}
	}
	return append(existing, cand)
}

func filterClassified(cs []sessionjournal.Classified, all, crashedOnly bool) []sessionjournal.Classified {
	out := cs[:0:0]
	for _, c := range cs {
		if crashedOnly && c.Status != sessionjournal.StatusCrashed {
			continue
		}
		if !all && !crashedOnly && c.Status == sessionjournal.StatusClosed {
			continue
		}
		out = append(out, c)
	}
	return out
}

// sortClassified puts the resume set first (CRASHED), then LIVE, STALE, CLOSED; newest start
// first within a status.
func sortClassified(cs []sessionjournal.Classified) {
	rank := map[sessionjournal.Status]int{
		sessionjournal.StatusCrashed: 0,
		sessionjournal.StatusLive:    1,
		sessionjournal.StatusStale:   2,
		sessionjournal.StatusClosed:  3,
	}
	sort.SliceStable(cs, func(i, j int) bool {
		if rank[cs[i].Status] != rank[cs[j].Status] {
			return rank[cs[i].Status] < rank[cs[j].Status]
		}
		return cs[i].StartedAt.After(cs[j].StartedAt)
	})
}

func renderSessionJournalTable(w io.Writer, cs []sessionjournal.Classified) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "STATUS\tREASON\tID\tAGENT\tPID\tSTARTED\tDRIVE\tCWD\n")
	for _, c := range cs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
			c.Status, c.Reason, orDash(c.ID), orDash(firstNonEmpty(c.Agent, c.Model)),
			c.PID, orDash(startedString(c.StartedAt)), orDash(driveSummary(c.Drive)), orDash(c.CWD))
	}
	_ = tw.Flush()
}

// microCentsPerDollar converts the DriveCarry spend axis (micro-cents; 1e6 per cent per
// internal/session.MicroCentsPerCent, ×100 cents) to dollars for the compact DRIVE cell.
const microCentsPerDollar = 100 * 1_000_000

// driveSummary renders a compact one-cell remaining-budget summary for the DRIVE column,
// e.g. "t=3 tok=12k $4.50 gen=2"; "" (rendered "-" by orDash) when no drive is carried. Only
// the set axes appear, so a carry with just turns reads "t=3" rather than a wall of zeros.
func driveSummary(d *sessionjournal.DriveCarry) string {
	if d == nil {
		return ""
	}
	var parts []string
	if d.TurnsLeft != 0 {
		parts = append(parts, fmt.Sprintf("t=%d", d.TurnsLeft))
	}
	if d.TokensLeft != 0 {
		parts = append(parts, "tok="+compactCount(d.TokensLeft))
	}
	if d.SpendMicroCentsLeft != 0 {
		parts = append(parts, fmt.Sprintf("$%.2f", float64(d.SpendMicroCentsLeft)/microCentsPerDollar))
	}
	if d.Generation != 0 {
		parts = append(parts, fmt.Sprintf("gen=%d", d.Generation))
	}
	return strings.Join(parts, " ")
}

// compactCount renders a token count compactly for the DRIVE cell: 12000 -> "12k",
// 3_500_000 -> "3.5m", small values verbatim.
func compactCount(n int64) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fm", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%dk", n/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func startedString(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.Format(time.RFC3339)
}
