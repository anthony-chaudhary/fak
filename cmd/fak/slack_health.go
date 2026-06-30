package main

// `fak slack health` — the WATCHDOG dual of the feeders.
//
// The cadence feeders (`scoreboard-feed.yml`, `bench-feed.yml`, …) POST a card on a
// schedule and fail OPEN: a missing token or channel renders to the step summary and
// exits 0. That is correct for forks, but it makes a misconfigured or broken feeder
// SILENT — green CI, nothing in the channel, until a human notices. This verb CONFIRMS
// the other half: per surface it folds resolution + auth.test (does the token work?) +
// a real `conversations.history` read (did a post actually land inside the feeder's
// intended cadence?) into one closed verdict, and exits non-zero on any non-OK so a
// scheduled job can gate on it (see `.github/workflows/slack-watchdog.yml`).
//
//	fak slack health           # per-surface OK | INCOMPLETE | AUTH_FAIL | STALE table
//	fak slack health --json    # machine-readable, for the watchdog workflow / a dashboard
//
// It reuses the same offline resolver (`buildSurfaceReports`) and auth probe
// (`runAuthChecks`) as `fak slack check`, and the same tracked `conversations.history`
// transport the chatrelay bridge uses (`internal/chatrelay`). No lab identifiers, no
// shell: the public side of the GPU-server/Slack boundary.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"sort"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/chatrelay"
)

// healthVerdict is the closed per-surface health classification. The set is deliberately
// small and total: every surface lands in exactly one, and only OK passes the gate.
type healthVerdict string

const (
	verdictOK         healthVerdict = "OK"         // ready, auth works, and (if it has a cadence) a recent post is witnessed
	verdictIncomplete healthVerdict = "INCOMPLETE" // token or channel unresolved — the feeder posts nowhere (config drift)
	verdictAuthFail   healthVerdict = "AUTH_FAIL"  // token resolved but auth.test rejected it (rotated/revoked bot)
	verdictStale      healthVerdict = "STALE"      // ready + auth OK but no recent post could be witnessed past the budget
	verdictDeferred   healthVerdict = "DEFERRED"   // an OPTIONAL surface with no channel yet — expected, NOT an alarm (no #channel exists)
)

// surfaceFreshnessBudget maps a surface to the staleness budget implied by the cadence of
// the scheduled feeder that pushes to it: a DAILY feeder gets 36h (24h cadence + 12h grace
// so one late/missed run is not a false alarm); a WEEKLY feeder gets 8 days. A surface with
// NO scheduled feeder (grafana, dispatch, marketing, chatrelay post on demand) has no entry
// and is never graded STALE — "quiet" is judged only where a feeder is supposed to be loud.
var surfaceFreshnessBudget = map[string]time.Duration{
	"scoreboard": 36 * time.Hour,     // scoreboard-feed.yml — daily
	"bench":      36 * time.Hour,     // bench-feed.yml — daily
	"blockers":   36 * time.Hour,     // blockers-feed.yml — daily
	"cachevalue": 36 * time.Hour,     // cachevalue-feed.yml — daily
	"dojo":       36 * time.Hour,     // dojo-feed.yml / dojo-rsi-feed.yml — daily
	"node-usage": 36 * time.Hour,     // node-usage-feed.yml — daily
	"steering":   36 * time.Hour,     // steering-guard.yml — daily
	"product":    8 * 24 * time.Hour, // product-feed.yml — weekly (Mon)
}

// historyProbeLimit is how many recent messages the staleness probe fetches. Any message
// counts as channel activity, so we take the newest ts across the small page (one is enough
// in practice; a few tolerate a trailing join/edit event without an extra round-trip).
const historyProbeLimit = 5

// historyReader is the narrow `conversations.history` capability the staleness probe needs.
// chatrelay.HTTPSlack is the live implementation; a test injects an in-memory fake.
type historyReader interface {
	History(ctx context.Context, channel, oldestTS string, limit int) ([]chatrelay.Message, error)
}

// newHealthHistoryReader builds the live reader for a surface's token. It is a package var
// so a test can substitute an in-memory reader with no network.
var newHealthHistoryReader = func(token, apiBase string) historyReader {
	return &chatrelay.HTTPSlack{Token: token, APIBase: apiBase}
}

// healthReport is one surface's folded health, the JSON contract the watchdog reads. The
// field names match the issue's acceptance criteria: {surface, ready, auth_ok,
// last_post_age_s, budget_s, verdict}.
type healthReport struct {
	Name         string        `json:"surface"`
	Ready        bool          `json:"ready"`
	AuthOK       bool          `json:"auth_ok"`
	Channel      string        `json:"channel,omitempty"`
	LastPostAgeS int64         `json:"last_post_age_s"` // -1 when not measured (incomplete / auth-fail / no cadence / unreadable)
	BudgetS      int64         `json:"budget_s"`        // 0 when the surface has no scheduled-feeder cadence
	Verdict      healthVerdict `json:"verdict"`
	Detail       string        `json:"detail,omitempty"`
}

// runSlackHealth is the `fak slack health` handler. It always probes auth (a health check
// without a liveness probe is not a health check) and, for every ready+authed surface with
// a cadence, witnesses a recent post via conversations.history.
func runSlackHealth(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack health", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the per-surface health report as JSON")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (default https://slack.com/api/; for testing/proxying)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	reports := buildSurfaceReports()
	runAuthChecks(reports, *apiBase)
	health := foldSlackHealth(reports, *apiBase, time.Now())

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(health); err != nil {
			fmt.Fprintf(stderr, "fak slack health: encode json: %v\n", err)
			return 1
		}
	} else {
		renderHealthReports(stdout, health)
	}
	return healthExit(health)
}

// foldSlackHealth classifies every resolved+auth-probed surface into one verdict. now is
// injected so the staleness arithmetic is deterministic under test. The history read only
// fires for a ready, authed surface that HAS a cadence — incomplete/auth-fail surfaces are
// decided before any network call, and on-demand surfaces are never probed.
func foldSlackHealth(reports []*surfaceReport, apiBase string, now time.Time) []healthReport {
	out := make([]healthReport, 0, len(reports))
	for _, r := range reports {
		hr := healthReport{
			Name:         r.Name,
			Ready:        r.Ready,
			Channel:      r.Channel,
			LastPostAgeS: -1,
		}
		if r.Auth != nil {
			hr.AuthOK = r.Auth.OK
		}
		budget, hasBudget := surfaceFreshnessBudget[r.Name]
		if hasBudget {
			hr.BudgetS = int64(budget / time.Second)
		}

		switch {
		case !r.Ready && r.Optional:
			// An optional surface with no dedicated channel yet (no #marketing / #chatrelay in
			// the workspace). Reporting it INCOMPLETE would be a permanent false alarm; DEFERRED
			// says "expected, nothing to fix" and is exempt from the gate. Wiring a channel later
			// makes it Ready and it promotes to OK automatically.
			hr.Verdict = verdictDeferred
			hr.Detail = "optional surface — no dedicated channel yet (wire FAK_" + "*_CHANNEL or a ChannelDefault to enable)"
		case !r.Ready:
			hr.Verdict = verdictIncomplete
			hr.Detail = "token or channel unresolved — the feeder would post nowhere"
		case r.Auth != nil && !r.Auth.OK:
			hr.Verdict = verdictAuthFail
			hr.Detail = "auth.test rejected the token: " + r.Auth.Err
		case !hasBudget:
			// Ready + authed, but on-demand (no scheduled feeder) — nothing to time against.
			hr.Verdict = verdictOK
			hr.Detail = "on-demand surface (no cadence) — auth OK"
		default:
			age, err := lastPostAge(r, apiBase, now)
			if err != nil {
				// We could not WITNESS a recent post (empty channel, bot not in channel,
				// bad channel id, API error). Operationally that is the same alarm a stale
				// feeder raises — no confirmed post — so it grades STALE with the true cause
				// in Detail and age left at -1 (unknown, not measured-as-old).
				hr.Verdict = verdictStale
				hr.Detail = "no recent post could be witnessed: " + err.Error()
				break
			}
			hr.LastPostAgeS = int64(age / time.Second)
			if age > budget {
				hr.Verdict = verdictStale
				hr.Detail = fmt.Sprintf("last post %s ago exceeds the %s budget", age.Round(time.Second), budget)
			} else {
				hr.Verdict = verdictOK
			}
		}
		out = append(out, hr)
	}
	return out
}

// lastPostAge reads the channel's most recent message via conversations.history and returns
// its age. It takes the newest ts across the small page (Slack returns newest-first, but we
// fold to the max so order can never fool us). An empty channel is an error (nothing to
// witness), which the caller maps to STALE.
func lastPostAge(r *surfaceReport, apiBase string, now time.Time) (time.Duration, error) {
	reader := newHealthHistoryReader(r.tokenValue, apiBase)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	msgs, err := reader.History(ctx, r.Channel, "", historyProbeLimit)
	if err != nil {
		return 0, err
	}
	var newest float64
	for _, m := range msgs {
		if ts := parseSlackTS(m.TS); ts > newest {
			newest = ts
		}
	}
	if newest == 0 {
		return 0, fmt.Errorf("channel %s has no messages", r.Channel)
	}
	last := time.Unix(0, int64(newest*float64(time.Second)))
	age := now.Sub(last)
	if age < 0 {
		age = 0 // clock skew: a future ts is fresh, not negative
	}
	return age, nil
}

// parseSlackTS parses a Slack message ts ("1719600000.000100") as epoch seconds. A
// malformed ts yields 0 (treated as "not a witnessable message").
func parseSlackTS(ts string) float64 {
	f, err := strconv.ParseFloat(ts, 64)
	if err != nil {
		return 0
	}
	return f
}

// healthExit returns 1 when ANY surface is non-OK, so a CI/watchdog job can gate on the
// exit code alone. The workflow that runs this fails OPEN by checking the token is present
// before treating a non-zero exit as actionable (a secret-less fork makes everything
// INCOMPLETE, which is expected, not a real alarm).
func healthExit(health []healthReport) int {
	for _, h := range health {
		// DEFERRED is an expected, non-actionable state (an optional surface with no channel
		// yet), so it never trips the gate — only a real problem (incomplete/auth-fail/stale)
		// does.
		if h.Verdict != verdictOK && h.Verdict != verdictDeferred {
			return 1
		}
	}
	return 0
}

// renderHealthReports prints the human table, worst-verdict-first within a stable order so
// the surfaces that need attention sit at the top.
func renderHealthReports(w io.Writer, health []healthReport) {
	var ok, incomplete, authFail, stale, deferred int
	for _, h := range health {
		switch h.Verdict {
		case verdictOK:
			ok++
		case verdictIncomplete:
			incomplete++
		case verdictAuthFail:
			authFail++
		case verdictStale:
			stale++
		case verdictDeferred:
			deferred++
		}
	}
	fmt.Fprintf(w, "fak slack health — %d surfaces; OK=%d STALE=%d AUTH_FAIL=%d INCOMPLETE=%d DEFERRED=%d\n\n",
		len(health), ok, stale, authFail, incomplete, deferred)

	ordered := make([]healthReport, len(health))
	copy(ordered, health)
	sort.SliceStable(ordered, func(i, j int) bool {
		return verdictRank(ordered[i].Verdict) < verdictRank(ordered[j].Verdict)
	})

	for _, h := range ordered {
		fmt.Fprintf(w, "● %-11s %-10s %s\n", h.Name, h.Verdict, h.Detail)
		if h.Verdict == verdictOK && h.BudgetS > 0 && h.LastPostAgeS >= 0 {
			fmt.Fprintf(w, "    last post %s ago (budget %s)\n",
				(time.Duration(h.LastPostAgeS) * time.Second).Round(time.Second),
				time.Duration(h.BudgetS)*time.Second)
		}
	}
}

// verdictRank orders the human table so the loudest alarms float to the top.
func verdictRank(v healthVerdict) int {
	switch v {
	case verdictAuthFail:
		return 0
	case verdictStale:
		return 1
	case verdictIncomplete:
		return 2
	case verdictOK:
		return 3
	default: // DEFERRED — expected, sorts last (below OK)
		return 4
	}
}
