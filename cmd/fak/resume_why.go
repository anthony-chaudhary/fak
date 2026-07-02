package main

// resume_why.go — `fak resume why <sid>`, the single-session NARRATIVE over the same
// folds `resume status` tables and `resume resolve` acts on. It exists because the death
// of a session is invisible at the moment it matters: the operator sees a process end in
// what looks like a random error, the transcript store holds the evidence, and every
// answer is scattered across three verbs and their flags. `why` is the zero-knob pane
// that assembles the story in one call:
//
//	fak resume why <sid>
//
// It locates the session across every ~/.claude* account (no --store to know), reads how
// it died from the transcript's own records, folds the resume journey from the durable
// ledger, reads the owner account's block/reset state from the roster, dry-runs the SAME
// resolve decision the launcher would take, and prints the story with the one command to
// run. Everything is read from artifacts — transcript, ledger, roster — never from a
// worker's self-report; the row fold is shared with `resume status` (foldStatusRow) and
// the account decision with `resume resolve` (buildResolveInput), so the narrative can
// never disagree with the verbs it explains.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resume/rehome"
)

// runResumeWhy tells one session's story: how it died, where it stands, what owns it,
// and what to do next. Returns the process exit code (0 ok, 1 not found / read error,
// 2 usage).
func runResumeWhy(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume why", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("home", "", "user home holding the .claude* account dirs (default: discovered)")
	ledger := fs.String("ledger", defaultResumeLedger(), "durable resume ledger JSONL (the record every launcher appends to)")
	maxAttempts := fs.Int("max-attempts", resume.DefaultMaxResumeAttempts, "give-up cap on automatic resumes of one session")
	asJSON := fs.Bool("json", false, "emit the underlying records as JSON instead of the narrative")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak resume why: need exactly one <session-id>")
		return 2
	}
	sid := fs.Arg(0)

	paths, homeDir := discoverFleetPaths(*home)
	owner := rehome.LocateOwner(sid, homeDir)
	if owner == nil {
		fmt.Fprintf(stderr, "fak resume why: no %s account holds session %s\n", filepath.Join(homeDir, ".claude*"), sid)
		return 1
	}
	transcript := filepath.Join(owner.ConfigDir, "projects", owner.Project, sid+".jsonl")
	f, err := os.Open(transcript)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume why: open %s: %v\n", transcript, err)
		return 1
	}
	tr := scanTranscriptForStatus(f)
	f.Close()

	ledgerPath := pathutil.ExpandTilde(*ledger)
	hist := loadResumeHistory(ledgerPath)[sid]
	admit := foldHostAdmission(ledgerPath)
	now := time.Now().Unix()
	row := foldStatusRow(sid, tr, hist, admit, *maxAttempts, now)

	// Dry-run the exact resolve decision the launcher would take, so the account plane —
	// owner blocked? reset when? wait or re-home? — is part of the story, not a surprise
	// the operator meets only when the launch fails.
	dec := rehome.Resolve(buildResolveInput(paths, homeDir, sid, "", true, true, false))

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":     "fak.resume-why.v1",
			"session":    sid,
			"transcript": transcript,
			"row":        row,
			"resolve":    dec,
			"real_turns": len(tr.turnTimes),
		}, "fak resume why")
	}
	renderResumeWhy(stdout, sid, transcript, tr, row, dec, now)
	return 0
}

// renderResumeWhy prints the story. Each line answers one operator question in plain
// language; the closing note says where the answers came from.
func renderResumeWhy(w io.Writer, sid, transcript string, tr statusTranscript, row statusRow, dec rehome.Decision, now int64) {
	fmt.Fprintf(w, "session %s — why it stopped, and what happens next\n\n", sid)

	goal := row.Goal
	if goal == "" {
		goal = "(no /goal recorded — see the transcript for its work)"
	}
	fmt.Fprintf(w, "  doing    %s\n", goal)
	fmt.Fprintf(w, "  ended    %s\n", crashStory(row))
	fmt.Fprintf(w, "  when     %s — idle %s, %d real model turn(s) on record\n",
		humanUnix(row.LastActivityUnix), humanIdle(row.IdleSeconds), len(tr.turnTimes))
	fmt.Fprintf(w, "  journey  %d resume attempt(s), %d new turn(s) since the last launch, state %s\n",
		row.Attempts, row.NewTurns, row.State)

	ownerLine := fmt.Sprintf("owner %s", dec.OwnerAccount)
	switch {
	case dec.OwnerAvailable:
		ownerLine += " — available"
	case dec.OwnerBlockReason != "":
		ownerLine += " — BLOCKED (" + dec.OwnerBlockReason + ")"
	default:
		ownerLine += " — BLOCKED"
	}
	if dec.DupCount > 1 {
		ownerLine += fmt.Sprintf("; transcript forked across %d accounts — the resolver disambiguates", dec.DupCount)
	}
	fmt.Fprintf(w, "  account  %s\n", ownerLine)
	if dec.Action == "WAIT_RESET" {
		fmt.Fprintf(w, "           frees up in ~%s (at %s) — the launcher WAITS for it rather than copying\n",
			compactDuration(dec.WaitSeconds), humanUnix(dec.ResetUnix))
	} else if !dec.OwnerAvailable && dec.Action == "REHOME" {
		fmt.Fprintf(w, "           a resume now would re-home the transcript onto %s\n", dec.PinAccount)
	}

	fmt.Fprintf(w, "  next     %s — %s\n", row.NextAction, row.NextReason)
	switch {
	case row.Command != "":
		fmt.Fprintf(w, "  run      %s\n", row.Command)
	case row.NextAction == resume.ActWaitReset || dec.Action == "WAIT_RESET":
		fmt.Fprintf(w, "  run      %s\n", resumeRunCommand(sid))
		fmt.Fprintln(w, "           (safe to start now: resolve -wait sleeps out the reset before pinning)")
	case row.NextAction == resume.ActLogin:
		fmt.Fprintf(w, "  run      CLAUDE_CONFIG_DIR=%q claude /login   # human step: clear the auth wall first\n", dec.OwnerConfigDir)
	}

	fmt.Fprintf(w, "\n  read from the transcript, resume ledger, and account roster — never a self-report.\n")
	fmt.Fprintf(w, "  transcript: %s\n", transcript)
}

// crashStory renders the crash class as the sentence an operator needed at the moment
// the session died — especially the interrupted class, whose whole failure mode is that
// nothing anywhere said what happened.
func crashStory(row statusRow) string {
	switch row.Crash {
	case resume.CrashRateLimit:
		return fmt.Sprintf("crashed on a %s — the provider refused and no turn ever followed", row.LimitReason)
	case resume.CrashOther:
		return "ended on a non-rate API error with no turn after it"
	case resume.CrashInterrupted:
		return "died mid-turn — the last record is an unanswered user turn; no refusal or error was\n" +
			"           written (a killed process, machine sleep, or an account wall that struck before\n" +
			"           the reply could land)"
	default:
		if row.Attempts > 0 {
			return "ended cleanly (its resume journey is the story — see journey below)"
		}
		return "ended cleanly — the last meaningful record is a real model turn"
	}
}

// humanUnix renders a unix instant in the operator's local clock, or "unknown".
func humanUnix(u int64) string {
	if u <= 0 {
		return "unknown"
	}
	return time.Unix(u, 0).Format("3:04pm Jan 2")
}
