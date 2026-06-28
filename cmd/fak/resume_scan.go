package main

// resume_scan.go — `fak resume scan`, the DETECT-and-plan front door over a whole Claude
// Code transcript STORE. `fak resume plan` answers "what do I do with the cache for THIS
// session?"; scan answers the question that comes before it for an operator with a directory
// full of `.jsonl` sessions: "WHICH of my sessions crashed on a rate limit and never resumed
// — and what is each one's managed-cache restart?"
//
//	fak resume scan --store ~/.claude/projects/<project>
//	fak resume scan --store DIR --json
//
// This is the wire half of internal/resume.Diagnose (the pure verdict). It does only the I/O
// the leaf must not: walk the store, parse each transcript into the closed set of EVENT FACTS
// the leaf reasons over (a real model turn and its prompt size; a synthetic rate-limit refusal
// and its closed reason), derive idle from the last record's timestamp, and render the
// rate-limited crashes with their managed-cache restart strategy. The leaf never sees a byte
// of transcript content — this shell classifies the provider's refusal strings against the
// closed limit vocabulary and hands Diagnose only the typed events.
//
// CRITICAL FIX over the single-transcript path: a rate-limit refusal is recorded as a
// synthetic assistant turn ("model":"<synthetic>", isApiErrorMessage:true) carrying an
// all-zero usage block. scanTranscriptToEvents skips those when sizing the resident context,
// so a crashed session is sized from its last REAL model turn, never mis-sized to zero (which
// is exactly what would happen to the sessions this verb exists to find).

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// runResumeScan walks a Claude Code transcript store, diagnoses every session, and renders the
// rate-limited crashes with their managed-cache restart plan. Returns the process exit code
// (0 ok, 1 a runtime error, 2 a usage error).
func runResumeScan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume scan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	store := fs.String("store", "", "directory of Claude Code session transcripts (.jsonl) to scan")
	ttlStr := fs.String("ttl", "5m", "provider cache TTL tier the sessions used: 5m (default) or 1h")
	inputPrice := fs.Float64("input-price", 5, "model base input price per million tokens (default: Opus 4.8 = 5)")
	outputPrice := fs.Float64("output-price", 25, "model base output price per million tokens (default: Opus 4.8 = 25)")
	horizon := fs.Int("horizon", 0, "turns expected to remain after each restart (0 = default)")
	shedBudget := fs.Int("shed-budget", 0, "CUT target in tokens — what a managed restart keeps (0 = default ~48k)")
	all := fs.Bool("all", false, "also report sessions that ended cleanly or on a non-rate error")
	asJSON := fs.Bool("json", false, "emit the raw per-session diagnoses as JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}
	if *store == "" {
		fmt.Fprintln(stderr, "fak resume scan: need --store DIR (a directory of Claude Code .jsonl transcripts)")
		return 2
	}
	ttl, ok := parseResumeTTL(*ttlStr)
	if !ok {
		fmt.Fprintf(stderr, "fak resume scan: bad --ttl %q (want 5m or 1h)\n", *ttlStr)
		return 2
	}

	entries, err := os.ReadDir(*store)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume scan: read store %q: %v\n", *store, err)
		return 1
	}
	now := time.Now().Unix()
	var rows []scanRow
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		path := filepath.Join(*store, e.Name())
		f, err := os.Open(path)
		if err != nil {
			continue // a transcript we cannot read simply does not enter the scan
		}
		events, model, lastUnix, limitMsg := scanTranscriptToEvents(f)
		f.Close()

		idle := int64(-1)
		if lastUnix > 0 {
			if idle = now - lastUnix; idle < 0 {
				idle = 0
			}
		}
		d := resume.Diagnose(events, resume.Input{
			IdleSeconds:      idle,
			TTL:              ttl,
			Pricing:          resume.Pricing{InputPerMTokUSD: *inputPrice, OutputPerMTokUSD: *outputPrice},
			HorizonTurns:     *horizon,
			ShedBudgetTokens: *shedBudget,
		})
		rows = append(rows, scanRow{
			SessionID:    strings.TrimSuffix(e.Name(), ".jsonl"),
			Model:        model,
			IdleSeconds:  idle,
			LimitMessage: limitMsg,
			Diagnosis:    d,
		})
	}
	if len(rows) == 0 {
		fmt.Fprintf(stderr, "fak resume scan: no .jsonl transcripts in %q\n", *store)
		return 1
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rows, "fak resume scan")
	}
	renderScan(stdout, *store, rows, *all)
	return 0
}

// scanRow is one diagnosed session in the store: the leaf verdict plus the shell-only context
// (which session, what model, and the human refusal text — content the leaf never sees).
type scanRow struct {
	SessionID    string           `json:"session_id"`
	Model        string           `json:"model,omitempty"`
	IdleSeconds  int64            `json:"idle_seconds"`
	LimitMessage string           `json:"limit_message,omitempty"`
	Diagnosis    resume.Diagnosis `json:"diagnosis"`
}

// scanTranscriptToEvents streams a Claude Code transcript JSONL into the closed event facts
// resume.Diagnose reasons over. It emits ONLY real model turns and synthetic error turns (the
// records that decide the verdict); user/tool/bookkeeping lines are dropped. It also returns
// the model of the last real turn, the latest record timestamp (for idle), and the text of the
// most recent rate-limit refusal (for the operator-facing reset hint). Best-effort over real
// data: a malformed line is skipped, never fatal.
func scanTranscriptToEvents(r io.Reader) (events []resume.Event, model string, lastUnix int64, limitMsg string) {
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
		if t := parseTranscriptUnix(jr.Timestamp); t > lastUnix {
			lastUnix = t
		}
		m := jr.Message
		if m == nil {
			continue
		}
		// A synthetic api-error turn ends the session: classify its text against the closed
		// limit vocabulary. Its all-zero usage is deliberately NOT treated as a model turn.
		if jr.IsAPIError {
			text := transcriptText(m.Content)
			if reason, ok := classifyLimit(text); ok {
				events = append(events, resume.Event{Kind: resume.EventRateLimitError, LimitReason: reason})
				limitMsg = strings.TrimSpace(text)
			} else {
				events = append(events, resume.Event{Kind: resume.EventOtherError})
			}
			continue
		}
		// A real model turn: a non-synthetic assistant message carrying prompt usage. Its
		// prompt size is the resident context a resume would re-prefill.
		if m.Role == "assistant" && m.Model != "" && m.Model != "<synthetic>" && m.Usage != nil {
			prompt := m.Usage.InputTokens + m.Usage.CacheReadTokens + m.Usage.CacheCreationTokens
			if prompt > 0 {
				events = append(events, resume.Event{Kind: resume.EventRealAssistant, PromptTokens: prompt})
				model = m.Model
			}
		}
	}
	return events, model, lastUnix, limitMsg
}

// transcriptText extracts the human text from a Claude Code message content field, which is
// either a bare string or an array of typed blocks. Anything else yields "".
func transcriptText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) == nil {
		parts := make([]string, 0, len(blocks))
		for _, b := range blocks {
			if b.Text != "" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, " ")
	}
	return ""
}

// classifyLimit maps a provider refusal string onto exactly one closed rate-limit reason,
// or reports that it is not a rate limit at all. The order matters: the specific session and
// weekly caps are checked first, then a server-side "rate limited"/429 throttle, then the
// generic usage cap — so "(not your usage limit) · Rate limited" is correctly read as a rate
// throttle rather than a usage cap.
func classifyLimit(text string) (string, bool) {
	s := strings.ToLower(text)
	switch {
	case strings.Contains(s, "session limit"):
		return resume.LimitSession, true
	case strings.Contains(s, "weekly limit"):
		return resume.LimitWeekly, true
	case strings.Contains(s, "rate limit"), strings.Contains(s, "rate-limit"), strings.Contains(s, "429"):
		return resume.LimitRate, true
	case strings.Contains(s, "usage limit"):
		return resume.LimitUsage, true
	default:
		return "", false
	}
}

// renderScan prints the rate-limited crashes (and, with --all, the rest) as an aligned table:
// which session, why it died, how big its resident context is, the projected cache posture, and
// the managed-cache restart strategy that avoids a cold full re-prefill.
func renderScan(w io.Writer, store string, rows []scanRow, all bool) {
	var crashed, otherErr, clean int
	for _, r := range rows {
		switch {
		case r.Diagnosis.NeedsRestart:
			crashed++
		case r.Diagnosis.Crash == resume.CrashOther:
			otherErr++
		default:
			clean++
		}
	}
	fmt.Fprintf(w, "scanned %d session(s) in %s\n", len(rows), store)
	fmt.Fprintf(w, "  %d crashed on a rate limit and never resumed   %d other unclean end   %d clean\n\n", crashed, otherErr, clean)

	show := make([]scanRow, 0, len(rows))
	for _, r := range rows {
		if r.Diagnosis.NeedsRestart || (all && r.Diagnosis.Crash != resume.CrashNone) {
			show = append(show, r)
		}
	}
	if len(show) == 0 {
		fmt.Fprintln(w, "no rate-limited crashes to restart.")
		return
	}
	// Biggest resident context first: the most expensive cold re-prefill is the most worth managing.
	sort.SliceStable(show, func(i, j int) bool {
		return show[i].Diagnosis.ResidentTokens > show[j].Diagnosis.ResidentTokens
	})

	fmt.Fprintf(w, "%-10s %-13s %9s %8s %-7s %-22s %12s\n",
		"session", "limit", "resident", "idle", "posture", "managed restart", "saves/horizon")
	var totalSaved float64
	for _, r := range show {
		d := r.Diagnosis
		rec := pickStrategy(d.Plan, d.Plan.Recommended)
		restart := fmt.Sprintf("%s -> %d tok", d.Plan.Recommended, rec.PrefillTokens)
		totalSaved += d.Plan.RecommendedSavingsUSD
		fmt.Fprintf(w, "%-10s %-13s %9d %8s %-7s %-22s %12s\n",
			shortID(r.SessionID), limitLabel(d), d.ResidentTokens, humanIdle(r.IdleSeconds),
			d.Plan.Posture, restart, usd(d.Plan.RecommendedSavingsUSD))
	}

	fmt.Fprintf(w, "\nrestart %d session(s) with cache managed — projected horizon saving vs blind resume_full: %s\n", len(show), usd(totalSaved))
	if hint := firstResetHint(show); hint != "" {
		fmt.Fprintf(w, "  (these limits reset around: %s — restart AFTER that)\n", hint)
	}
	fmt.Fprintln(w, "  scan PLANS the managed restart; it does not relaunch. For each session, start a")
	fmt.Fprintln(w, "  NEW session seeded with the recommended cut/reset above — not a cold `claude --resume`")
	fmt.Fprintln(w, "  (which re-prefills the whole resident transcript and tends to re-hit the same limit).")
	fmt.Fprintln(w, "  (dollars are a projection over the resident-token count, not a witnessed bill)")
}

// pickStrategy returns the priced cost for one strategy in a report (the strategies are in the
// fixed order resume_full, cut, reset).
func pickStrategy(rep resume.Report, s resume.Strategy) resume.StrategyCost {
	for _, c := range rep.Strategies {
		if c.Strategy == s {
			return c
		}
	}
	return resume.StrategyCost{}
}

// limitLabel is the closed rate-limit reason for a crashed row, or the crash kind otherwise.
func limitLabel(d resume.Diagnosis) string {
	if d.Crash == resume.CrashRateLimit {
		return d.LimitReason
	}
	return string(d.Crash)
}

// shortID is the first 8 characters of a session UUID — enough to identify it in the table.
func shortID(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}

// humanIdle renders an idle duration compactly (unknown, seconds, minutes, hours, or days).
func humanIdle(s int64) string {
	if s < 0 {
		return "unknown"
	}
	return compactDuration(s)
}

// firstResetHint pulls the "resets <when>" tail from the first crashed row that carries one,
// so the operator sees when it is safe to restart. Content-side, display-only.
func firstResetHint(rows []scanRow) string {
	for _, r := range rows {
		if h := resetHint(r.LimitMessage); h != "" {
			return h
		}
	}
	return ""
}

// resetHint extracts the human "resets ..." phrase from a refusal message, e.g.
// "You've hit your session limit · resets 8pm (America/Los_Angeles)" -> "8pm (America/Los_Angeles)".
func resetHint(msg string) string {
	i := strings.Index(strings.ToLower(msg), "resets ")
	if i < 0 {
		return ""
	}
	return strings.TrimSpace(msg[i+len("resets "):])
}
