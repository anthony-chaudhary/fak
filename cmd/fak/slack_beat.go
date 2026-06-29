package main

// `fak slack beat` — the LIVENESS beat. The missing third leg of the Slack control surface.
//
// The cadence feeders post a card when there is something to say; `fak slack health` folds a
// closed verdict but only exits a code (or files a once-a-day GH issue via the watchdog). The
// gap: a QUIET channel is indistinguishable from a DEAD feeder. An operator scrolling a silent
// channel cannot tell "nothing happened today" from "the poster broke a week ago."
//
// This verb closes that gap. It runs the same health fold (`buildSurfaceReports` +
// `runAuthChecks` + `foldSlackHealth`) and posts ONE compact line to a status channel —
// UNCONDITIONALLY on its cadence, whether or not anything else posted. A green beat means the
// whole surface is alive; a ⚠️/🔴 beat names the down surfaces; NO beat at all means the
// scheduler running this verb itself died (the alarm of last resort, cf. #1430).
//
//	fak slack beat                 # post the beat to the default status channel
//	fak slack beat --channel C0…   # post it somewhere else
//	fak slack beat --dry-run       # render the line, resolve nothing-posts (fork-safe)
//	fak slack beat --json          # machine-readable beat + per-surface health
//
// It invents no transport: it posts through the same `internal/scoreboard` client `fak slack
// send` uses, and resolves the channel/token the same env-then-.env.slack.local way. Heartbeat
// grain, not a card — one line so the channel carries a steady, low-noise pulse.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// beatResult is the machine-readable verdict of one beat (the --json contract). It carries the
// folded health summary plus where the beat went, so a scheduled tick's log is self-explaining.
type beatResult struct {
	Schema   string         `json:"schema"`
	Verdict  healthVerdict  `json:"verdict"` // worst surface verdict: OK | STALE | AUTH_FAIL | INCOMPLETE
	OK       bool           `json:"ok"`      // every surface OK
	Total    int            `json:"total"`
	OKCount  int            `json:"ok_count"`
	Down     []string       `json:"down,omitempty"` // "name:VERDICT" for each non-OK surface
	Line     string         `json:"line"`           // the one-line beat body that was (or would be) posted
	Channel  string         `json:"channel"`
	Posted   bool           `json:"posted"`
	DryRun   bool           `json:"dry_run"`
	TS       string         `json:"ts,omitempty"`
	Skipped  string         `json:"skipped,omitempty"`
	Error    string         `json:"error,omitempty"`
	Surfaces []healthReport `json:"surfaces,omitempty"`
}

const beatSchema = "fak-slack-beat/1"

// beatGlyph maps the worst verdict to a leading glyph so the pulse reads at a glance in the
// channel and in a mobile push preview: green when all alive, ⚠️ for staleness/config drift,
// 🔴 for an auth wall (the loudest — the bot token itself is rejected).
func beatGlyph(worst healthVerdict, allOK bool) string {
	if allOK {
		return "✅"
	}
	if worst == verdictAuthFail {
		return "🔴"
	}
	return "⚠️"
}

// worstVerdict folds the per-surface verdicts into the single worst one for the beat headline,
// using the same severity order the human health table sorts by (auth-fail > stale > incomplete
// > ok). An empty set is OK (nothing configured is not an alarm here — the health verb owns that).
func worstVerdict(health []healthReport) (healthVerdict, bool) {
	worst := verdictOK
	worstRank := verdictRank(verdictOK)
	allOK := true
	for _, h := range health {
		if h.Verdict != verdictOK {
			allOK = false
		}
		if r := verdictRank(h.Verdict); r < worstRank {
			worstRank = r
			worst = h.Verdict
		}
	}
	return worst, allOK
}

// beatLine renders the compact one-line (occasionally two-line) beat body from the folded
// health. Green: a single "alive" line with the OK count and the freshest cadence age. Non-OK:
// the same headline plus a terse "down:" clause naming each broken surface and its mode, so the
// operator can act without opening a dashboard. Pure (now injected) so it is unit-testable.
func beatLine(health []healthReport, now time.Time) string {
	worst, allOK := worstVerdict(health)
	ok := 0
	var down []string
	freshest := time.Duration(-1)
	for _, h := range health {
		if h.Verdict == verdictOK {
			ok++
		} else {
			down = append(down, fmt.Sprintf("%s %s", h.Name, h.Verdict))
		}
		if h.LastPostAgeS >= 0 {
			age := time.Duration(h.LastPostAgeS) * time.Second
			if freshest < 0 || age < freshest {
				freshest = age
			}
		}
	}
	glyph := beatGlyph(worst, allOK)
	headline := fmt.Sprintf("%s *slack surfaces alive* — %d/%d OK", glyph, ok, len(health))
	if freshest >= 0 {
		headline += fmt.Sprintf(" · freshest feeder post %s ago", freshest.Round(time.Minute))
	}
	headline += " · " + now.UTC().Format("2006-01-02 15:04Z")
	if len(down) == 0 {
		return headline
	}
	sort.Strings(down)
	return headline + "\n_down:_ " + strings.Join(down, " · ")
}

// runSlackBeat is the `fak slack beat` handler. It always folds health (auth + a real
// conversations.history read per cadence surface), renders the beat, and posts it
// UNCONDITIONALLY (the liveness guarantee) unless --dry-run. Exit codes: 0 on a real post or a
// dry-run; 1 when a live post was attempted and failed or was skipped for a missing
// precondition (so a scheduled tick's result flags a misconfiguration, never a silent no-op).
func runSlackBeat(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack beat", flag.ContinueOnError)
	fs.SetOutput(stderr)
	channel := fs.String("channel", "", "target status channel id (default: $FAK_DISPATCH_CHANNEL, then $FAK_SCOREBOARD_CHANNEL, then .env.slack.local)")
	token := fs.String("token", "", "bot token (default: $FAK_SCOREBOARD_TOKEN, then .env.slack.local)")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (default https://slack.com/api/; for testing/proxying)")
	dryRun := fs.Bool("dry-run", false, "fold health and render the beat, but post nothing (fork-safe)")
	asJSON := fs.Bool("json", false, "emit the beat verdict (and per-surface health) as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	reports := buildSurfaceReports()
	runAuthChecks(reports, *apiBase)
	now := time.Now()
	health := foldSlackHealth(reports, *apiBase, now)

	worst, allOK := worstVerdict(health)
	line := beatLine(health, now)
	res := beatResult{
		Schema:   beatSchema,
		Verdict:  worst,
		OK:       allOK,
		Total:    len(health),
		Line:     line,
		DryRun:   *dryRun,
		Surfaces: health,
	}
	for _, h := range health {
		if h.Verdict == verdictOK {
			res.OKCount++
		} else {
			res.Down = append(res.Down, fmt.Sprintf("%s:%s", h.Name, h.Verdict))
		}
	}

	// Resolve where the beat goes: explicit flag, else the dispatch status channel, else the
	// scoreboard channel — the status surfaces an operator already watches.
	chan_ := *channel
	if chan_ == "" {
		if r := slackenv.Lookup("FAK_DISPATCH_CHANNEL"); r.Set() {
			chan_ = r.Value
		} else if r := slackenv.Lookup("FAK_SCOREBOARD_CHANNEL"); r.Set() {
			chan_ = r.Value
		}
	}
	res.Channel = chan_

	tok := *token
	if tok == "" {
		tok = scoreboard.ResolveToken()
	}

	switch {
	case chan_ == "":
		res.Skipped = "no status channel resolved (set --channel, FAK_DISPATCH_CHANNEL, or FAK_SCOREBOARD_CHANNEL)"
	case tok == "":
		res.Skipped = "no bot token resolved (set --token, FAK_SCOREBOARD_TOKEN, or add it to " + slackenv.EnvFileName + ")"
	case *dryRun:
		res.Skipped = "dry-run"
	default:
		var opts []scoreboard.Option
		if *apiBase != "" {
			opts = append(opts, scoreboard.WithAPIBase(*apiBase))
		}
		c, err := scoreboard.NewClient(tok, opts...)
		if err != nil {
			res.Error = err.Error()
			break
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		ts, err := c.Post(ctx, chan_, line, nil)
		if err != nil {
			res.Error = err.Error()
			break
		}
		res.Posted = true
		res.TS = ts
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak slack beat: encode json: %v\n", err)
			return 1
		}
	} else {
		renderBeat(stdout, res)
	}
	return beatExit(res)
}

// renderBeat prints the human summary: the beat line itself, then what happened to it.
func renderBeat(w io.Writer, res beatResult) {
	fmt.Fprintln(w, res.Line)
	switch {
	case res.Posted:
		fmt.Fprintf(w, "→ posted to %s (ts=%s)\n", res.Channel, res.TS)
	case res.DryRun:
		fmt.Fprintf(w, "→ dry-run: would post to %s\n", orUnset(res.Channel))
	case res.Skipped != "":
		fmt.Fprintf(w, "→ skipped — %s\n", res.Skipped)
	case res.Error != "":
		fmt.Fprintf(w, "→ FAILED — %s\n", res.Error)
	}
}

// beatExit returns 0 on a real post or a dry-run (both are success: the beat was rendered and
// either delivered or deliberately withheld), 1 otherwise (a live attempt that failed or was
// skipped for a missing precondition) so a scheduled tick flags a misconfiguration.
func beatExit(res beatResult) int {
	if res.Posted || res.DryRun {
		return 0
	}
	return 1
}
