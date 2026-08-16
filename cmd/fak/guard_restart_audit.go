package main

// guard_restart_audit.go — the restart-chain witness (#3057, the follow-up to
// #3055's --continue reattach).
//
// A `fak guard --restart-on-budget` session can hide any number of child
// relaunches, and each one is a continuity cliff: the wrapped agent either
// resumed the conversation the carryover seed was captured from, or it booted
// cold and silently lost the task. Before this rung the only evidence was two
// uncorrelated stderr lines plus a seed JSON in a temp dir — which is how a
// fleet accumulated 1,436 orphaned seed files with no record of whether
// continuity survived ANY of the restarts that wrote them.
//
// This file closes the loop in three pieces:
//
//  1. guardRestartHopFromEvent + guardEmitRestartHop — at each live restart the
//     supervision loop folds the whole hop (from/to trace, seed file + size,
//     handback mode, child session id, continuity status) into ONE correlated
//     record (schema journal.RestartChainSchema), appends it to the guard audit
//     journal as a RESTART_HOP row, and prints ONE stderr line carrying the same
//     fields — replacing the two disconnected lines.
//  2. `fak guard restart-audit` — the read-only scanner: joins RESTART_HOP rows
//     from the guard-audit journals against reset-*.json seed files on disk
//     (OS temp + any --seed-dir), reports every hop, and BACKFILLS the orphans —
//     a seed with no recorded hop is reported honestly as handback=ORPHANED /
//     status=loss, in red on a terminal. It never verifies chains (that is
//     `fak audit verify`'s job) and never silently caps: every unreadable file
//     becomes a note in the report.
//  3. guardRestartChainStatusAddendum — `fak session status <id>` appends the
//     locally-evidenced restart chain for that id, so the operator's existing
//     status verb answers "did my session restart, and did continuity survive?"
//     (The live gateway already exposes the lineage as SessionState.ContinuationID;
//     a /debug/vars block for `fak info` is deferred — the gateway lane is owned
//     by a concurrent change; this file keeps every new seam in cmd/fak.)

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// Handback modes — the CLOSED vocabulary for how a relaunched child was handed
// its continuity. "seed-prompt" is reserved for the #3056 headless/no-continue
// rung; today's emitter produces only "continue" and the honest "ORPHANED".
const (
	guardRestartHandbackContinue   = "continue"    // recognized agent relaunched with its resume flag (#3055)
	guardRestartHandbackSeedPrompt = "seed-prompt" // reserved: seed re-injected as a prompt (#3056)
	guardRestartHandbackOrphaned   = "ORPHANED"    // unrecognized agent relaunched cold; the seed sits unread
)

// guardRestartAuditSchema names the report `fak guard restart-audit --json`
// emits. Additive-only once shipped.
const guardRestartAuditSchema = "fak.guard.restart_audit.v1"

// guardApproxTokens is the seed-size gauge for the restart record: the standard
// ~4-bytes-per-token approximation, biased upward (ceiling) so a tiny non-empty
// seed never reports 0 tokens. The record wants a magnitude ("did 200 tokens or
// 20,000 ride along?"), not a tokenizer-exact count — the seed text is plain
// prose whose exact tokenization varies by provider anyway.
func guardApproxTokens(s string) int {
	if len(s) == 0 {
		return 0
	}
	return (len(s) + 3) / 4
}

// guardRestartHopFromEvent folds one live budget-restart event into the
// correlated hop record. hop is the 1-based restart ordinal within this guard
// session (the supervision loop's restarts counter, post-increment).
//
// Handback is decided the same way the relaunch itself is
// (guardRestartRelaunchCommand): a recognized agent gets its resume flag →
// "continue"; an unrecognized one relaunches cold → "ORPHANED". Status folds
// continuity: a hop with no continuation trace or no surviving seed at all is a
// "break" (the emit-time analogue of guardRestartLimitStatus's
// continuity=blocked); an ORPHANED handback is "inert" (the data exists,
// nothing consumes it); an engaged handback is "ok". Child is the session id
// the relaunched child serves under — the same continuation trace guard injects
// as FAK_SESSION_ID — recorded as its own field so a future handback that mints
// a distinct child id (#3056) stays additive.
func guardRestartHopFromEvent(ev guardBudgetRestartEvent, hop int, agentName string) journal.RestartHop {
	return guardRestartHopFromEventHandback(ev, hop, agentName, "")
}

// guardRestartHopFromEventHandback is guardRestartHopFromEvent with an explicit handback OVERRIDE:
// the live relaunch path passes the mode it ACTUALLY took, so the recorded hop matches the command
// that was launched. The #3056 seed-prompt handback needs this because a recognized child that
// would otherwise derive "continue" is instead relaunched with its seed injected as a prompt —
// handback "seed-prompt", status "ok" (the seed IS consumed into argv, not left inert). An empty
// override derives the handback the legacy way (recognized→continue, else ORPHANED).
func guardRestartHopFromEventHandback(ev guardBudgetRestartEvent, hop int, agentName, handbackOverride string) journal.RestartHop {
	handback := strings.TrimSpace(handbackOverride)
	if handback == "" {
		handback = guardRestartHandbackOrphaned
		if _, ok := guardContinueFlagForAgent(agentName); ok {
			handback = guardRestartHandbackContinue
		}
	}
	status := journal.RestartHopOK
	switch {
	case strings.TrimSpace(ev.ToTraceID) == "" ||
		(strings.TrimSpace(ev.SeedFile) == "" && strings.TrimSpace(ev.SeedText) == ""):
		status = journal.RestartHopBreak
	case handback == guardRestartHandbackOrphaned:
		status = journal.RestartHopInert
	}
	return journal.RestartHop{
		Schema:     journal.RestartChainSchema,
		Hop:        hop,
		FromTrace:  ev.FromTraceID,
		ToTrace:    ev.ToTraceID,
		SeedFile:   ev.SeedFile,
		SeedTokens: guardApproxTokens(ev.SeedText),
		Handback:   handback,
		Child:      ev.ToTraceID,
		Status:     status,
	}
}

// guardWireRetryHop builds the correlated RESTART_HOP record for a transient-wire-crash relaunch
// (#3514) so a supervisor-level wire retry folds into the SAME restart chain (and `fak guard
// restart-audit`) as a budget restart, rather than being an invisible relaunch. The relaunch is a
// --continue reattach under the SAME trace (the crashed session resumes in place — no new
// continuation trace is minted and no seed is written), so from/to/child are all guardTraceID,
// handback is "continue", and status is ok. It degrades to the ORPHANED/inert shape for an
// unrecognized agent for symmetry with guardRestartHopFromEventHandback, though the wire-retry arm
// only ever fires for a recognized agent (guardMaybeRetryTransientWireCrash gates on the resume flag).
func guardWireRetryHop(guardTraceID, agentName string, hop int) journal.RestartHop {
	return guardSameTraceRelaunchHop(guardTraceID, agentName, hop)
}

// guardSameTraceRelaunchHop is the shared body behind every RESTART_HOP whose relaunch
// reattaches IN PLACE: from/to/child are all guardTraceID because no continuation trace is
// minted and no seed is written. For a recognized agent the handback is "continue" and the
// hop is ok; for an agent fak cannot resume it degrades to the ORPHANED/inert shape. The
// wire-retry and crash-restart arms differ only in WHEN they fire, not in what they record.
func guardSameTraceRelaunchHop(guardTraceID, agentName string, hop int) journal.RestartHop {
	handback := guardRestartHandbackOrphaned
	status := journal.RestartHopInert
	if _, ok := guardContinueFlagForAgent(agentName); ok {
		handback = guardRestartHandbackContinue
		status = journal.RestartHopOK
	}
	return journal.RestartHop{
		Schema:    journal.RestartChainSchema,
		Hop:       hop,
		FromTrace: guardTraceID,
		ToTrace:   guardTraceID,
		Handback:  handback,
		Child:     guardTraceID,
		Status:    status,
	}
}

// guardRestartHopShrink reports whether the relaunch this hop records SHRANK the exhausted context
// window ("yes") or RE-INFLATED it ("no") — the fix-#2 no-shrink signal. Only the seed-prompt
// handback boots the child fresh on the bounded distilled seed (and strips --continue), so only it
// reduces the window; a --continue reattach, or an ORPHANED cold relaunch that leaves the seed
// unread, does not. Derived purely from the handback mode the RESTART_HOP row already carries, so no
// new durable field is needed: an operator watching restarts can spot a shrink=no hop as the one at
// risk of re-exhausting and looping.
func guardRestartHopShrink(hop journal.RestartHop) string {
	if hop.Handback == guardRestartHandbackSeedPrompt {
		return "yes"
	}
	return "no"
}

// guardRestartHopOneLiner renders the correlated stderr line for one live hop —
// the single line that replaces the old "context budget exhausted…" +
// "carryover seed written…" pair, carrying every axis of the record so a
// captured log alone can reconstruct the chain. The seed path is
// forward-slash-normalized for the same reason guardRestartLimitStatus's
// next_action is: byte-identical output on every OS.
func guardRestartHopOneLiner(hop journal.RestartHop) string {
	line := fmt.Sprintf("fak guard: restart #%d from=%s to=%s seed=%dtok handback=%s child=%s status=%s shrink=%s",
		hop.Hop, hop.FromTrace, hop.ToTrace, hop.SeedTokens, hop.Handback, hop.Child, hop.Status, guardRestartHopShrink(hop))
	if hop.SeedFile != "" {
		line += " seed_file=" + strings.ReplaceAll(filepath.ToSlash(hop.SeedFile), "\\", "/")
	}
	return line
}

// guardEmitRestartHop is the supervision loop's single call at each restart: the
// durable RESTART_HOP journal row plus the live one-liner. Both halves are
// nil-safe (a --no-audit session still gets the stderr line; a quiet restarter
// still gets the row), so the caller never guards it.
func guardEmitRestartHop(j *journal.Journal, stderr io.Writer, agentName, guardTraceID string, hop journal.RestartHop) {
	j.AppendRestartHop(agentName, guardTraceID, hop)
	if stderr != nil {
		fmt.Fprintln(stderr, guardRestartHopOneLiner(hop))
	}
}

// guardRestartAuditHop is one reported hop: the correlated record plus its
// provenance — where the evidence came from ("journal" = a recorded RESTART_HOP
// row; "backfill" = a seed file with NO recorded hop, the pre-#3057 orphans)
// and when it happened (row timestamp, or seed-file mtime for a backfill).
type guardRestartGiveUp struct {
	Agent   string `json:"agent,omitempty"`
	TraceID string `json:"trace_id,omitempty"`
	Reason  string `json:"reason"`
	Source  string `json:"source,omitempty"`
}
type guardRestartAuditHop struct {
	journal.RestartHop
	Agent      string `json:"agent,omitempty"`
	GuardTrace string `json:"guard_trace_id,omitempty"`
	Source     string `json:"source"` // "journal" | "backfill"
	TSUnixNano int64  `json:"ts_unix_nano,omitempty"`
	Journal    string `json:"journal,omitempty"` // journal file the row came from
}

// guardRestartAuditReport is the fak.guard.restart_audit.v1 document. Notes
// carry every degradation (unreadable journal, unparseable seed) so the scan
// never silently narrows its coverage.
type guardRestartAuditReport struct {
	Schema   string                 `json:"schema"`
	Journals int                    `json:"journals_scanned"`
	Seeds    int                    `json:"seeds_scanned"`
	Hops     []guardRestartAuditHop `json:"hops"`
	Counts   map[string]int         `json:"counts"`
	GiveUps  []guardRestartGiveUp   `json:"give_ups,omitempty"`
	Notes    []string               `json:"notes,omitempty"`
}

// guardRestartSeedDirs returns the seed directories to scan: every
// fak-guard-reset-* directory in the OS temp dir (writeGuardRestartSeedFile's
// default landing zone) when scanTemp is set, plus the operator's extra dir.
func guardRestartSeedDirs(extraSeedDir string, scanTemp bool) []string {
	var dirs []string
	if scanTemp {
		if tmp, err := filepath.Glob(filepath.Join(os.TempDir(), "fak-guard-reset-*")); err == nil {
			dirs = append(dirs, tmp...)
		}
	}
	if d := strings.TrimSpace(extraSeedDir); d != "" {
		dirs = append(dirs, d)
	}
	return dirs
}

// guardRestartAuditScan is the read-only join at the heart of the verb. It
// never verifies hash chains (`fak audit verify` owns tamper-evidence) and
// never caps: every hop row and every seed file found is either in Hops or
// explained in Notes.
func guardRestartAuditScan(journalDir string, seedDirs []string, trace string) guardRestartAuditReport {
	rep := guardRestartAuditReport{Schema: guardRestartAuditSchema, Counts: map[string]int{}}

	// 1. Recorded hops from the guard-audit journals, keyed from→to so seed
	// files can be matched off against them.
	recorded := map[string]bool{}
	if dir := strings.TrimSpace(journalDir); dir != "" {
		files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
		for _, f := range files {
			// Segment-aware (#6488): the scan promises every hop row, and the glob only
			// matches live *.jsonl files — the sealed <name>.jsonl.cut-<seq> archives are
			// reachable only through the segment read (so they are read once, not twice).
			rows, err := journal.ReadAllSegments(f)
			if err != nil {
				rep.Notes = append(rep.Notes, fmt.Sprintf("journal %s: %v", filepath.Base(f), err))
				continue
			}
			rows = journal.WithoutCutAnchors(rows)
			rep.Journals++
			for _, row := range rows {
				if row.Kind == "CHILD_CRASH" && row.Reason == guardCrashRestartExhaustedReason {
					rep.GiveUps = append(rep.GiveUps, guardRestartGiveUp{Agent: row.Tool, TraceID: row.TraceID, Reason: row.Reason, Source: f})
					continue
				}
				if row.Kind != journal.KindRestartHop {
					continue
				}
				if row.Restart == nil {
					rep.Notes = append(rep.Notes, fmt.Sprintf("journal %s: RESTART_HOP row seq=%d has no restart payload", filepath.Base(f), row.Seq))
					continue
				}
				hop := guardRestartAuditHop{
					RestartHop: *row.Restart,
					Agent:      row.Tool,
					GuardTrace: row.TraceID,
					Source:     "journal",
					TSUnixNano: row.TSUnixNano,
					Journal:    f,
				}
				recorded[hop.FromTrace+"\x00"+hop.ToTrace] = true
				rep.Hops = append(rep.Hops, hop)
			}
		}
	}

	// 2. Seed files on disk. A seed whose hop was recorded is already covered;
	// a seed with no record predates the RESTART_HOP rung (or its journal was
	// lost) — backfill it honestly: the handback is ORPHANED and the continuity
	// outcome is unknowable, so the status is loss, not a guess.
	for _, dir := range seedDirs {
		seeds, _ := filepath.Glob(filepath.Join(dir, "reset-*.json"))
		for _, sf := range seeds {
			raw, err := os.ReadFile(sf)
			if err != nil {
				rep.Notes = append(rep.Notes, fmt.Sprintf("seed %s: %v", filepath.Base(sf), err))
				continue
			}
			var ev guardBudgetRestartEvent
			if err := json.Unmarshal(raw, &ev); err != nil {
				rep.Notes = append(rep.Notes, fmt.Sprintf("seed %s: %v", filepath.Base(sf), err))
				continue
			}
			if ev.Schema != "" && ev.Schema != "fak.guard.budget_restart.v1" {
				rep.Notes = append(rep.Notes, fmt.Sprintf("seed %s: unknown schema %q", filepath.Base(sf), ev.Schema))
				continue
			}
			rep.Seeds++
			if recorded[ev.FromTraceID+"\x00"+ev.ToTraceID] {
				continue
			}
			var ts int64
			if st, err := os.Stat(sf); err == nil {
				ts = st.ModTime().UnixNano()
			}
			rep.Hops = append(rep.Hops, guardRestartAuditHop{
				RestartHop: journal.RestartHop{
					Schema:     journal.RestartChainSchema,
					FromTrace:  ev.FromTraceID,
					ToTrace:    ev.ToTraceID,
					SeedFile:   sf,
					SeedTokens: guardApproxTokens(ev.SeedText),
					Handback:   guardRestartHandbackOrphaned,
					Status:     journal.RestartHopLoss,
				},
				Source:     "backfill",
				TSUnixNano: ts,
			})
		}
	}

	// 3. Trace filter + deterministic order + counts.
	if t := strings.TrimSpace(trace); t != "" {
		kept := rep.Hops[:0]
		for _, h := range rep.Hops {
			if h.FromTrace == t || h.ToTrace == t || h.Child == t || h.GuardTrace == t {
				kept = append(kept, h)
			}
		}
		rep.Hops = kept
	}
	sort.SliceStable(rep.Hops, func(i, k int) bool {
		if rep.Hops[i].TSUnixNano != rep.Hops[k].TSUnixNano {
			return rep.Hops[i].TSUnixNano < rep.Hops[k].TSUnixNano
		}
		if rep.Hops[i].FromTrace != rep.Hops[k].FromTrace {
			return rep.Hops[i].FromTrace < rep.Hops[k].FromTrace
		}
		return rep.Hops[i].ToTrace < rep.Hops[k].ToTrace
	})
	sort.Strings(rep.Notes)
	for _, h := range rep.Hops {
		rep.Counts[h.Status]++
	}
	return rep
}

// guardRestartAuditHopLine renders one hop for the human report. A backfilled
// orphan (status=loss) is the line an operator must not skim past — with color
// on, the whole line goes red.
func guardRestartAuditHopLine(h guardRestartAuditHop, color bool) string {
	ts := ""
	if h.TSUnixNano != 0 {
		ts = time.Unix(0, h.TSUnixNano).UTC().Format(time.RFC3339) + " "
	}
	ordinal := ""
	if h.Hop > 0 {
		ordinal = fmt.Sprintf("hop#%d ", h.Hop)
	}
	line := fmt.Sprintf("%s%sfrom=%s to=%s seed=%dtok handback=%s child=%s status=%s [%s",
		ts, ordinal, h.FromTrace, h.ToTrace, h.SeedTokens, h.Handback, h.Child, h.Status, h.Source)
	switch {
	case h.Journal != "":
		line += " " + filepath.Base(h.Journal)
	case h.SeedFile != "":
		line += " " + filepath.Base(h.SeedFile)
	}
	line += "]"
	if color && h.Status == journal.RestartHopLoss {
		line = "\x1b[31m" + line + "\x1b[0m"
	}
	return line
}

// guardRestartChainSection renders the compact chain report shared by the verb's
// human mode and the `fak session status` addendum.
func guardRestartChainSection(w io.Writer, rep guardRestartAuditReport, color bool) {
	fmt.Fprintf(w, "restart-chain audit: %d hop(s) (ok=%d inert=%d break=%d loss=%d) across %d journal(s), %d seed(s)\n",
		len(rep.Hops), rep.Counts[journal.RestartHopOK], rep.Counts[journal.RestartHopInert],
		rep.Counts[journal.RestartHopBreak], rep.Counts[journal.RestartHopLoss], rep.Journals, rep.Seeds)
	for _, h := range rep.Hops {
		fmt.Fprintf(w, "  %s\n", guardRestartAuditHopLine(h, color))
	}
	for _, g := range rep.GiveUps {
		fmt.Fprintf(w, "  give-up: reason=%s agent=%s trace=%s source=%s\n", g.Reason, g.Agent, g.TraceID, g.Source)
	}
	for _, n := range rep.Notes {
		fmt.Fprintf(w, "  note: %s\n", n)
	}
}

// guardRestartChainStatusAddendum appends the locally-evidenced restart chain
// for one session id under `fak session status <id>` (human mode only). It is
// best-effort local evidence — the guard-audit journals and seed dirs on THIS
// host — and prints nothing when the id has no restart history, so every
// restart-free session's status output is byte-for-byte unchanged.
func guardRestartChainStatusAddendum(w io.Writer, traceID string) {
	rep := guardRestartAuditScan(guardAuditDir(findRepoRoot(".")), guardRestartSeedDirs("", true), traceID)
	if len(rep.Hops) == 0 {
		return
	}
	fmt.Fprintf(w, "restart chain (local evidence for %s):\n", traceID)
	for _, h := range rep.Hops {
		fmt.Fprintf(w, "  %s\n", guardRestartAuditHopLine(h, false))
	}
}

// runGuardRestartAudit is `fak guard restart-audit` — the read-only scan verb.
// Exit 0 always when the scan itself ran: an orphan is a reported fact, not a
// process failure (the operator gates on the report, not the exit code).
func runGuardRestartAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("guard restart-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	journalDir := fs.String("journal-dir", "", "guard-audit journal directory to scan for RESTART_HOP rows (default: .dispatch-runs/guard-audit under the repo root)")
	seedDir := fs.String("seed-dir", "", "extra carryover-seed directory to scan for reset-*.json (in addition to the OS temp dir)")
	scanTemp := fs.Bool("scan-temp", true, "scan the OS temp dir for fak-guard-reset-* seed directories (writeGuardRestartSeedFile's default landing zone)")
	trace := fs.String("trace", "", "only report hops touching this trace/session id (matches from/to/child/guard ids)")
	asJSON := fs.Bool("json", false, "emit the "+guardRestartAuditSchema+" report as JSON")
	colorMode := fs.String("color", "auto", "colorize backfilled ORPHANED/loss hops in the human report: auto|always|never")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak guard restart-audit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	dir := strings.TrimSpace(*journalDir)
	if dir == "" {
		dir = guardAuditDir(findRepoRoot("."))
	}
	rep := guardRestartAuditScan(dir, guardRestartSeedDirs(*seedDir, *scanTemp), *trace)
	if *asJSON {
		raw, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak guard restart-audit: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	color := false
	switch *colorMode {
	case "always":
		color = true
	case "never", "":
	default: // auto
		color = guardFdIsTerminal(int(os.Stdout.Fd())) && os.Getenv("NO_COLOR") == ""
	}
	guardRestartChainSection(stdout, rep, color)
	return 0
}
