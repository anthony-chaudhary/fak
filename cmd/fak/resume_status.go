package main

// resume_status.go — `fak resume status`, the PROVE-THE-RESUME-TOOK readout over a
// transcript store + the durable resume ledger (#1146). `fak resume scan` finds the
// sessions that crashed and never resumed; status answers the four questions an operator
// asks about a live crash batch, in one call, from one source: how many crashed, what was
// each doing (its /goal), how far into resume is each (pending / launched / took /
// re-stranded / gave-up / settled), and — the rung the ledger alone cannot answer — did
// each resume actually TAKE, proven from the transcript's own turns, never from the
// launcher's "launched" row.
//
//	fak resume status --store ~/.claude/projects/<project>
//	fak resume status --store DIR [--ledger FILE] [--max-attempts N] [--all] [--json]
//
// This is the wire half of internal/resume's outcome fold (ClassifyOutcome / RetryGate /
// FoldResumeState). It does only the I/O the pure leaf must not: walk the store, read the
// ledger JSONL into typed Attempts, read each transcript's terminal turn and classify its
// text against the wall vocabularies into a TerminalSignal, count the real turns that
// landed after the last launch, and render the fold. The leaf never sees transcript
// content; the goal string is shell-side display context only.

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
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// runResumeStatus walks the store, joins each session against the ledger, folds the
// resume state, and renders the per-session journey. Returns the process exit code
// (0 ok, 1 a runtime error, 2 a usage error).
func runResumeStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	store := fs.String("store", "", "directory of Claude Code session transcripts (.jsonl) to report on")
	ledger := fs.String("ledger", defaultResumeLedger(), "durable resume ledger JSONL (the record every launcher appends to)")
	maxAttempts := fs.Int("max-attempts", resume.DefaultMaxResumeAttempts, "give-up cap on automatic resumes of one session")
	all := fs.Bool("all", false, "also report sessions that ended cleanly and have no resume history")
	asJSON := fs.Bool("json", false, "emit the raw per-session rows as JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}
	if *store == "" {
		fmt.Fprintln(stderr, "fak resume status: need --store DIR (a directory of Claude Code .jsonl transcripts)")
		return 2
	}

	storeDir := pathutil.ExpandTilde(*store)
	entries, err := os.ReadDir(storeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume status: read store %q: %v\n", *store, err)
		return 1
	}
	history := loadResumeHistory(pathutil.ExpandTilde(*ledger))

	// The 529 burst wall is a per-SOURCE bound, so fold the host launch-admission verdict
	// ONCE and apply it to every fire-eligible session — an unconfigured host is permissive
	// (admit), so this adds no latency and no false holds unless a source policy is set.
	admit := foldHostAdmission(pathutil.ExpandTilde(*ledger))

	now := time.Now().Unix()
	var rows []statusRow
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		sid := strings.TrimSuffix(e.Name(), ".jsonl")
		f, err := os.Open(filepath.Join(storeDir, e.Name()))
		if err != nil {
			continue // a transcript we cannot read simply does not enter the report
		}
		tr := scanTranscriptForStatus(f)
		f.Close()

		hist := history[sid]
		idle := idleSince(now, tr.lastUnix)
		d := resume.Diagnose(tr.events, resume.Input{IdleSeconds: idle, TTL: resume.TTL5m})
		if d.Crash == resume.CrashNone && len(hist) == 0 && !*all {
			continue // neither crashed nor part of any resume journey
		}

		outcome := resume.ClassifyOutcome(classifyTerminalSignal(tr.terminalText, tr.terminalFound))
		attempts := resume.CountAttempts(hist)
		newTurns := resume.NewTurnsAfter(tr.turnTimes, resume.LastLaunchUnix(hist))
		settled := false
		for _, a := range hist {
			if a.ManualOverride || strings.HasPrefix(strings.ToLower(strings.TrimSpace(a.Action)), "consolidate") {
				settled = true
			}
		}
		gate := resume.RetryGate(hist, outcome, *maxAttempts)
		state := resume.FoldResumeState(resume.ResumeFacts{
			Attempts:        attempts,
			MaxAttempts:     *maxAttempts,
			OperatorSettled: settled,
			NewTurns:        newTurns,
			Outcome:         outcome,
		})
		next := resume.FoldNextAction(resume.NextInput{
			State:               state,
			Outcome:             outcome,
			Retry:               gate,
			LimitReason:         d.LimitReason,
			IdleSeconds:         idle,
			Admitted:            admit.Admit,
			AdmitReason:         admit.Reason,
			AdmitRetryAfterUnix: admit.RetryAfterUnix,
		})
		cmd := ""
		if next.Fire {
			cmd = resumeRunCommand(sid)
		}
		rows = append(rows, statusRow{
			SessionID:        sid,
			Crash:            d.Crash,
			LimitReason:      d.LimitReason,
			Goal:             tr.goal,
			Attempts:         attempts,
			NewTurns:         newTurns,
			Outcome:          outcome,
			State:            state,
			RetryBlocked:     gate.Blocked,
			RetryReason:      gate.Reason,
			IdleSeconds:      idle,
			LastActivityUnix: tr.lastUnix,
			NextAction:       next.Action,
			NextReason:       next.Reason,
			Fire:             next.Fire,
			Command:          cmd,
			RetryAfterUnix:   next.RetryAfterUnix,
		})
	}
	if len(rows) == 0 {
		fmt.Fprintf(stderr, "fak resume status: no reportable sessions in %q (try --all)\n", *store)
		return 1
	}

	// Fire-now first, then the sessions waiting on a reset / admission, then the walls a
	// human owns, then the quiet tail — the runbook order an agent reads top-down. Ties
	// break on session id so two runs render identically.
	sort.SliceStable(rows, func(i, j int) bool {
		ri, rj := actionRank(rows[i].NextAction), actionRank(rows[j].NextAction)
		if ri != rj {
			return ri < rj
		}
		return rows[i].SessionID < rows[j].SessionID
	})

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":       "fak.resume-status.v1",
			"store":        *store,
			"ledger_path":  *ledger,
			"host_admitted": admit.Admit,
			"admit_reason":  admit.Reason,
			"sessions":      rows,
		}, "fak resume status")
	}
	renderResumeStatus(stdout, *store, rows, admit)
	return 0
}

// foldHostAdmission computes the host's launch-admission verdict once for the whole batch:
// the per-SOURCE 529-burst bound the runbook applies to every fire-eligible session. An
// unconfigured or permissive source policy short-circuits to admit WITHOUT running the OS
// process census, so status stays a cheap readout on the common (no-policy) host; a
// configured policy folds the live census + launch ledger through resume.AdmitSource. A
// malformed policy is treated as permissive here — a status readout must never fail closed
// on operator config (the loud failure belongs to `fak resume admit`, the enforcing gate).
func foldHostAdmission(ledgerPath string) resume.SourceDecision {
	admitted := resume.SourceDecision{Admit: true, Reason: resume.ReasonSourceAdmitted, Summary: "no source policy — admit"}
	policies, err := resume.LoadSourcePolicy(defaultResumeSourcePolicy())
	if err != nil {
		return admitted
	}
	p := policies.Default
	if p.MaxLiveResumes == 0 && p.MaxLaunchesPerWindow == 0 && p.MinLaunchSpacingSeconds == 0 {
		return admitted // permissive: nothing to census
	}
	now := time.Now()
	return resume.AdmitSource(foldSourceSnapshot(ledgerPath, now), p, now)
}

// resumeRunCommand is the exact, copy-pasteable command that resumes one session: pin (and,
// if the owner is throttled, re-home) via `fak resume resolve` — which prints the
// CLAUDE_CONFIG_DIR to use — then `claude --resume`. The agent adds only genuine one-offs
// (a --cwd for a session born in another dir, or -p for a headless single turn).
func resumeRunCommand(sid string) string {
	return fmt.Sprintf(`CLAUDE_CONFIG_DIR="$(fak resume resolve %s)" claude --resume %s`, sid, sid)
}

// actionRank orders the runbook fire-first: run, then the deferrals (wait_reset,
// hold_admission), then the human-owned walls (login, gave_up), then the quiet done tail.
func actionRank(a resume.NextAction) int {
	switch a {
	case resume.ActRun:
		return 0
	case resume.ActWaitReset:
		return 1
	case resume.ActHoldAdmission:
		return 2
	case resume.ActLogin:
		return 3
	case resume.ActGaveUp:
		return 4
	default: // done
		return 5
	}
}

// statusRow is one session's resume journey: the leaf verdicts plus the shell-only
// context (which session, its /goal, when it last moved), and the folded runbook action —
// the one deterministic thing to DO about the session now, with its exact command.
type statusRow struct {
	SessionID        string             `json:"session_id"`
	Crash            resume.CrashKind   `json:"crash"`
	LimitReason      string             `json:"limit_reason,omitempty"`
	Goal             string             `json:"goal,omitempty"`
	Attempts         int                `json:"attempts"`
	NewTurns         int                `json:"new_turns_since_resume"`
	Outcome          resume.Outcome     `json:"outcome"`
	State            resume.ResumeState `json:"resume_state"`
	RetryBlocked     bool               `json:"retry_blocked"`
	RetryReason      string             `json:"retry_reason"`
	IdleSeconds      int64              `json:"idle_seconds"`
	LastActivityUnix int64              `json:"last_activity_unix,omitempty"`
	// NextAction is the folded runbook verdict (resume.FoldNextAction): the closed
	// next-action token, its reason, whether the agent should fire a resume now, and the
	// exact command when it should.
	NextAction     resume.NextAction `json:"next_action"`
	NextReason     string            `json:"next_reason"`
	Fire           bool              `json:"fire"`
	Command        string            `json:"command,omitempty"`
	RetryAfterUnix int64             `json:"retry_after_unix,omitempty"`
}

// idleSince is now-minus-last as a non-negative idle, -1 when the transcript carried no
// usable timestamp (the same "unknown" convention scan uses).
func idleSince(now, last int64) int64 {
	if last <= 0 {
		return -1
	}
	if idle := now - last; idle > 0 {
		return idle
	}
	return 0
}

// loadResumeHistory reads the durable resume ledger JSONL into per-session typed
// Attempts (oldest first, the order the gate expects). Best-effort: a malformed row is
// skipped; a missing/unreadable ledger yields no history — sessions then read as pending,
// which is the honest floor when the launch record is gone.
func loadResumeHistory(path string) map[string][]resume.Attempt {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	type lrec struct {
		Ts             string `json:"ts"`
		Session        string `json:"session"`
		Phase          string `json:"phase"`
		Action         string `json:"action"`
		ManualOverride bool   `json:"manual_override"`
	}
	out := make(map[string][]resume.Attempt)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r lrec
		if json.Unmarshal(line, &r) != nil || r.Session == "" {
			continue
		}
		out[r.Session] = append(out[r.Session], resume.Attempt{
			UnixSeconds:    parseTranscriptUnix(r.Ts),
			Phase:          r.Phase,
			Action:         r.Action,
			ManualOverride: r.ManualOverride,
		})
	}
	return out
}

// statusTranscript is everything one transcript contributes to the status fold: the
// closed events for the crash diagnosis, the real-turn timestamps for the new-turns
// count, the terminal user/assistant turn for the outcome, and the /goal for display.
type statusTranscript struct {
	events        []resume.Event
	turnTimes     []int64
	lastUnix      int64
	terminalFound bool
	terminalText  string
	goal          string
}

// goalCommandRe pulls the /goal condition out of the Stop-hook command record — the
// "what was it doing" field the epic wants without hand-scraping transcripts.
var goalCommandRe = regexp.MustCompile(`(?s)<command-name>/goal</command-name>.*?<command-args>(.*?)</command-args>`)

// scanTranscriptForStatus streams one transcript into the status facts. It mirrors
// scanTranscriptToEvents for the event stream (real turns sized by prompt usage, synthetic
// api-error turns classified against the closed limit vocabulary) and adds the two reads
// status needs: the TERMINAL user/assistant record's text (the last real turn decides the
// outcome — an earlier banner a later clean turn superseded must not), and each real
// turn's timestamp. Best-effort over real data: a malformed line is skipped, never fatal.
func scanTranscriptForStatus(r io.Reader) statusTranscript {
	type usage struct {
		InputTokens         int `json:"input_tokens"`
		CacheReadTokens     int `json:"cache_read_input_tokens"`
		CacheCreationTokens int `json:"cache_creation_input_tokens"`
	}
	type rec struct {
		Type       string `json:"type"`
		Timestamp  string `json:"timestamp"`
		IsAPIError bool   `json:"isApiErrorMessage"`
		Message    *struct {
			Role    string          `json:"role"`
			Model   string          `json:"model"`
			Content json.RawMessage `json:"content"`
			Usage   *usage          `json:"usage"`
		} `json:"message"`
	}
	var out statusTranscript
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20) // a single tool-result line can be large
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr rec
		if json.Unmarshal(line, &jr) != nil {
			continue
		}
		ts := parseTranscriptUnix(jr.Timestamp)
		if ts > out.lastUnix {
			out.lastUnix = ts
		}
		m := jr.Message
		role := jr.Type
		if m != nil && m.Role != "" {
			role = m.Role
		}
		if role == "user" || role == "assistant" {
			var text string
			if m != nil {
				text = transcriptText(m.Content)
			}
			// The LAST user/assistant record is the terminal turn, even when its text is
			// empty (an unreadable terminal turn is honestly unknown, not "clean").
			out.terminalFound = true
			out.terminalText = text
			if role == "user" {
				if g := goalCommandRe.FindStringSubmatch(text); g != nil {
					out.goal = strings.TrimSpace(g[1])
				}
			}
		}
		if m == nil {
			continue
		}
		if jr.IsAPIError {
			if reason, ok := classifyLimit(transcriptText(m.Content)); ok {
				out.events = append(out.events, resume.Event{Kind: resume.EventRateLimitError, LimitReason: reason})
			} else {
				out.events = append(out.events, resume.Event{Kind: resume.EventOtherError})
			}
			continue
		}
		if m.Role == "assistant" && m.Model != "" && m.Model != "<synthetic>" && m.Usage != nil {
			prompt := m.Usage.InputTokens + m.Usage.CacheReadTokens + m.Usage.CacheCreationTokens
			if prompt > 0 {
				out.events = append(out.events, resume.Event{Kind: resume.EventRealAssistant, PromptTokens: prompt})
				if ts > 0 {
					out.turnTimes = append(out.turnTimes, ts)
				}
			}
		}
	}
	// An empty-text terminal turn cannot be classified — report it as not found so the
	// outcome reads unknown (the conservative burn-once path), never "progressed".
	if strings.TrimSpace(out.terminalText) == "" {
		out.terminalFound = false
	}
	return out
}

// authWallRe is the auth/login/credit/access-wall vocabulary — the walls a re-resume
// cannot fix (port of the fleet tooling's AUTH_RE, the is_auth_error signal).
var authWallRe = regexp.MustCompile(`(?i)Login interrupted|please run /login|authentication_error|` +
	`invalid x-api-key|invalid authentication credentials|` +
	`API Error:\s*401|HTTP\s*401|401\s+(?:authentication required|unauthorized)|` +
	`OAuth token has expired|credit balance is too low|` +
	`organization has disabled Claude subscription access|` +
	`Use an Anthropic API key instead|Not logged in`)

// limitWallRe matches a usage-limit banner carrying a reset window ("…limit · resets
// 8pm…") — the recoverable wall (port of LIMIT_RE's load-bearing prefix).
var limitWallRe = regexp.MustCompile(`(?i)limit\s*[·:|.\-]?\s*resets?\s+`)

// classifyTerminalSignal maps the terminal turn's text onto the closed TerminalSignal
// facts the pure leaf classifies. Keyed off the TERMINAL record only — a session that
// merely discusses an auth wall or a 529 five turns back is not in that state.
func classifyTerminalSignal(text string, found bool) resume.TerminalSignal {
	if !found {
		return resume.TerminalSignal{}
	}
	low := strings.ToLower(text)
	return resume.TerminalSignal{
		Found:     true,
		AuthWall:  authWallRe.MatchString(text),
		LimitWall: limitWallRe.MatchString(text),
		TransientAPIError: strings.Contains(low, "overloaded") || strings.Contains(text, "529") ||
			(strings.Contains(low, "api error") && strings.Contains(low, "rate")),
	}
}

// renderResumeStatus prints the runbook: the per-session journey table keyed on the folded
// next-action, an action rollup, and a copy-pasteable command block for the fire-eligible
// sessions — the one-call answer to "which dead sessions do I resume, and how?".
func renderResumeStatus(w io.Writer, store string, rows []statusRow, admit resume.SourceDecision) {
	counts := map[resume.NextAction]int{}
	for _, r := range rows {
		counts[r.NextAction]++
	}
	fmt.Fprintf(w, "resume status — %d session(s) in %s\n", len(rows), store)
	fmt.Fprintf(w, "  next: %d run   %d wait-reset   %d hold-admission   %d login   %d gave-up   %d done\n",
		counts[resume.ActRun], counts[resume.ActWaitReset], counts[resume.ActHoldAdmission],
		counts[resume.ActLogin], counts[resume.ActGaveUp], counts[resume.ActDone])
	if !admit.Admit {
		fmt.Fprintf(w, "  host admission: REFUSED (%s) — %d run-eligible session(s) held\n", admit.Reason, counts[resume.ActHoldAdmission])
	}
	fmt.Fprintln(w)

	fmt.Fprintf(w, "%-10s %-13s %-12s %-13s %6s %6s %8s  %s\n",
		"session", "next", "state", "crash", "att", "new", "idle", "goal")
	for _, r := range rows {
		crash := string(r.Crash)
		if r.LimitReason != "" {
			crash = r.LimitReason
		}
		fmt.Fprintf(w, "%-10s %-13s %-12s %-13s %6d %6d %8s  %s\n",
			shortID(r.SessionID), r.NextAction, r.State, crash, r.Attempts, r.NewTurns,
			humanIdle(r.IdleSeconds), truncateGoal(r.Goal, 40))
	}

	// The runbook: the exact command for every session the fold says to fire now. An agent
	// bringing a dead batch back copies these; everything else carries its reason in-table.
	var fire []statusRow
	for _, r := range rows {
		if r.Fire && r.Command != "" {
			fire = append(fire, r)
		}
	}
	if len(fire) > 0 {
		fmt.Fprintf(w, "\nrun now (%d) — resolve pins/re-homes onto a healthy account, then resume:\n", len(fire))
		for _, r := range fire {
			fmt.Fprintf(w, "  %s\n", r.Command)
		}
	}
	fmt.Fprintln(w, "\n  next-action, new-turns and outcome are read from each transcript's own turns — the")
	fmt.Fprintln(w, "  ledger's \"launched\" row alone cannot tell a resume that took from one that no-op'd.")
}

// truncateGoal keeps the table scannable; the full goal rides the JSON.
func truncateGoal(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
