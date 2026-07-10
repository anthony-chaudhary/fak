package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// resume_self.go — `fak resume self` (#3804), the WORKER-FACING half of the resume readout.
// `resume status` is the operator sweep: it walks the whole store and folds EVERY crashed
// session for a human triaging a batch. `resume self` answers the first-person question a
// single guarded worker asks about ITS OWN session — "was I resumed, did it take, will another
// attempt fire, or does a human own me now?" — by folding the same closed outcome folds over
// one session's ledger history plus its own transcript (resume.FoldSelfObservation). It shares
// the fold with the operator table and the fak_resume_history MCP tool, so the three can never
// tell a worker a different story than they tell a human.
//
// It is fail-closed: a session with no ledger history reads the honest floor (pending, nothing
// to recover), never a fabricated "took".

// runResumeSelf resolves the calling session, reads its ledger history and (when available)
// its transcript, folds the self-observation, and renders it. Exit 0 ok, 1 runtime error,
// 2 usage error.
func runResumeSelf(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume self", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	store := fs.String("store", "", "directory of Claude Code session transcripts (.jsonl) — used to witness this session's outcome and post-resume turns")
	session := fs.String("session", "", "the session id to self-observe (default: $CLAUDE_SESSION_ID, else the newest transcript in --store)")
	ledger := fs.String("ledger", defaultResumeLedger(), "durable resume ledger JSONL (the record every launcher appends to)")
	maxAttempts := fs.Int("max-attempts", 0, "give-up cap on automatic resumes (0 = the progress-earned budget the automatic relauncher rations to)")
	asJSON := fs.Bool("json", false, "emit the raw SelfObservation JSON instead of the human readout")
	if !parseFlags(fs, argv) {
		return 2
	}

	sid := strings.TrimSpace(*session)
	if sid == "" {
		sid = strings.TrimSpace(os.Getenv("CLAUDE_SESSION_ID"))
	}
	var storeDir string
	if *store != "" {
		storeDir = pathutil.ExpandTilde(*store)
		if sid == "" {
			sid = newestTranscriptSession(storeDir)
		}
	}
	if sid == "" {
		fmt.Fprintln(stderr, "fak resume self: need --session SID or --store DIR (a session to self-observe)")
		return 2
	}

	// Transcript facts are BEST-EFFORT: a worker whose transcript is gone (rotated, pruned)
	// still gets an honest ledger-only self-observation — outcome unknown, zero witnessed
	// post-resume turns — never a hard error. Only the ledger is load-bearing.
	var tr statusTranscript
	if storeDir != "" {
		if f, err := os.Open(filepath.Join(storeDir, sid+".jsonl")); err == nil {
			tr = scanTranscriptForStatus(f)
			f.Close()
		}
	}

	history := loadResumeHistory(pathutil.ExpandTilde(*ledger))
	hist := history[sid]

	outcome := resume.ClassifyOutcome(classifyTerminalSignal(tr.terminalText, tr.terminalFound))
	lastLaunch := resume.LastLaunchUnix(hist)
	newTurns := resume.NewTurnsAfter(tr.turnTimes, lastLaunch)

	obs := resume.FoldSelfObservation(resume.SelfFacts{
		Session:     sid,
		History:     hist,
		Outcome:     outcome,
		NewTurns:    newTurns,
		MaxAttempts: *maxAttempts,
	})

	if *asJSON {
		payload := map[string]any{
			"schema":      "fak.resume-self.v1",
			"ledger_path": *ledger,
			"observation": obs,
		}
		return encodeJSONOrFail(stdout, stderr, payload, "fak resume self")
	}
	renderResumeSelf(stdout, obs)
	return 0
}

// newestTranscriptSession returns the session id of the most-recently-modified .jsonl in dir,
// or "" when the directory has none/cannot be read. It is how a worker that omits --session
// gets "the session I am in" without threading an id through the environment.
func newestTranscriptSession(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	best, bestUnix := "", int64(-1)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if mt := info.ModTime().Unix(); mt > bestUnix {
			best, bestUnix = strings.TrimSuffix(e.Name(), ".jsonl"), mt
		}
	}
	return best
}

// renderResumeSelf writes the compact human self-readout. The hint line is the worker's
// takeaway; the rows above it are the evidence the hint rests on.
func renderResumeSelf(w io.Writer, o resume.SelfObservation) {
	fmt.Fprintf(w, "resume self — %s   state=%s", shortID(o.Session), o.State)
	if o.Outcome != "" && o.Outcome != resume.OutcomeUnknown {
		fmt.Fprintf(w, "   outcome=%s", o.Outcome)
	}
	fmt.Fprintln(w)

	if !o.HasHistory {
		fmt.Fprintf(w, "  %s\n", o.NextHint)
		return
	}

	fmt.Fprintf(w, "  attempts        : %d  (earned budget %d)\n", o.Attempts, o.EarnedBudget)
	fmt.Fprintf(w, "  new turns       : %d  since last launch\n", o.NewTurns)
	if o.RetryBlocked {
		fmt.Fprintf(w, "  retry           : BLOCKED — %s\n", o.RetryReason)
	} else {
		fmt.Fprintf(w, "  retry           : open — %s\n", o.RetryReason)
	}
	if o.OperatorSettled {
		fmt.Fprintln(w, "  operator settled: yes")
	}
	fmt.Fprintf(w, "  → %s\n", o.NextHint)
}
