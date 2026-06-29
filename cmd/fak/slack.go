package main

// `fak slack` — the one place to DEBUG and USE fak's Slack surface.
//
// fak posts to ~ten Slack channels (scoreboard, blockers, bench, dispatch, dojo,
// marketing, node-usage, product, steering) and bridges one (chatrelay). Each resolves a
// bot token and a channel id from an env var or a gitignored .env.slack.local, with
// surface-specific fallbacks. When a post silently fails the operator had no way to see
// WHICH token/channel a surface would use or WHETHER the token even works — the failure
// surfaced as a cryptic chat.postMessage error deep in a feed job.
//
//	fak slack check            # resolution report for every surface (offline)
//	fak slack check --auth     # + call auth.test per token: does it actually work?
//	fak slack check --json     # machine-readable, for a CI gate or a dashboard
//	fak slack send --channel C0ABC123 --text "deploy is green"   # ad-hoc message
//	echo "hi" | fak slack send --channel C0ABC123 --text -        # text from stdin
//
// It depends only on the tracked outbound transport (internal/scoreboard) and the shared
// resolver (internal/slackenv): no lab identifiers, no shell, public side of the
// GPU-server/Slack boundary.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blockerpost"
	"github.com/anthony-chaudhary/fak/internal/dojopost"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// scoreboardTokenKey is the shared workspace bot token every non-scoreboard surface falls
// back to (one bot serves the whole #scoreboard workspace).
const scoreboardTokenKey = "FAK_SCOREBOARD_TOKEN"

// slackSurface describes one configured Slack target: the env/file keys it resolves its
// bot token and channel from, and the public channel default (if any). The resolution
// methods below reproduce, in one tested place, the env-then-file-then-fallback order each
// internal/*post package documents — so `fak slack check` reports exactly what a live post
// would use.
type slackSurface struct {
	Name           string // display name
	Purpose        string // one-line: what it posts
	TokenEnv       string // its own token key; "" => no own token, uses the scoreboard token
	ChannelEnv     string // its channel key
	ChannelDefault string // public built-in channel default; "" => none (channel REQUIRED)
}

// slackSurfaces is the registry `fak slack check` walks. The channel defaults reference the
// PUBLIC, non-secret constants the post packages already expose (blockerpost.ChannelDefault,
// dojopost.ChannelDefault) and the steering default in this package — never a real id baked
// in here.
var slackSurfaces = []slackSurface{
	{"scoreboard", "scorecard / score / run-event status", "FAK_SCOREBOARD_TOKEN", "FAK_SCOREBOARD_CHANNEL", ""},
	{"product", "product direction / persona findings", "", "FAK_PRODUCT_CHANNEL", ""},
	{"blockers", "fleet blockers (status vs operator page)", "FAK_BLOCKERS_TOKEN", "FAK_BLOCKERS_CHANNEL", blockerpost.ChannelDefault},
	{"bench", "benchmark rollups / run-requests", "FAK_BENCH_TOKEN", "FAK_BENCH_CHANNEL", ""},
	{"dispatch", "background code-dispatch results", "FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", ""},
	{"dojo", "dojo rollups / trends", "FAK_DOJO_TOKEN", "FAK_DOJO_CHANNEL", dojopost.ChannelDefault},
	{"marketing", "marketing updates feed", "FAK_MARKETING_TOKEN", "FAK_MARKETING_CHANNEL", ""},
	{"node-usage", "compute-node usage snapshots", "FAK_NODE_USAGE_TOKEN", "FAK_NODE_USAGE_CHANNEL", ""},
	{"steering", "steering-guard surface", "", "FAK_STEERING_CHANNEL", steeringChannelDefault},
	{"chatrelay", "Slack <-> served-model chat bridge", "FAK_CHATRELAY_TOKEN", "FAK_CHATRELAY_CHANNEL", ""},
}

// resolvedField is a resolved value plus a human-readable source label.
type resolvedField struct {
	Value  string
	Source string
}

// token resolves the surface's bot token the way its package does: its own key (env then
// file) first, then a fall-back to the shared scoreboard token. The source label records
// which path won so `fak slack check` is self-explaining.
func (s slackSurface) token() resolvedField {
	if s.TokenEnv != "" {
		if r := slackenv.Lookup(s.TokenEnv); r.Set() {
			return resolvedField{r.Value, string(r.Source) + ":" + r.Key}
		}
	}
	if r := slackenv.Lookup(scoreboardTokenKey); r.Set() {
		return resolvedField{r.Value, "scoreboard-fallback (" + string(r.Source) + ":" + r.Key + ")"}
	}
	return resolvedField{}
}

// channel resolves the surface's channel id: its own key (env then file) first, then the
// public built-in default if it has one, else unset.
func (s slackSurface) channel() resolvedField {
	if r := slackenv.Lookup(s.ChannelEnv); r.Set() {
		return resolvedField{r.Value, string(r.Source) + ":" + r.Key}
	}
	if s.ChannelDefault != "" {
		return resolvedField{s.ChannelDefault, "built-in default"}
	}
	return resolvedField{}
}

// authReport is the auth.test outcome for a surface's token (only with --auth).
type authReport struct {
	OK   bool   `json:"ok"`
	Team string `json:"team,omitempty"`
	User string `json:"user,omitempty"`
	Err  string `json:"error,omitempty"`
}

// surfaceReport is one row of `fak slack check`, JSON-serializable for --json.
type surfaceReport struct {
	Name          string      `json:"name"`
	Purpose       string      `json:"purpose"`
	TokenSet      bool        `json:"token_set"`
	Token         string      `json:"token,omitempty"`          // redacted
	TokenSource   string      `json:"token_source,omitempty"`   //
	Channel       string      `json:"channel,omitempty"`        //
	ChannelSource string      `json:"channel_source,omitempty"` //
	Ready         bool        `json:"ready"`                    // token AND channel resolved
	Auth          *authReport `json:"auth,omitempty"`           //

	tokenValue string // raw token, for the auth probe; never serialized
}

// cmdSlack routes `fak slack <check|send>`; a bare `fak slack` runs the check report so the
// most common debug action takes zero extra typing.
func cmdSlack(argv []string) {
	if len(argv) == 0 {
		os.Exit(runSlackCheck(os.Stdout, os.Stderr, nil))
	}
	dispatchSubcommands("slack", "check | send", argv,
		subcommand{"check", runSlackCheck},
		subcommand{"send", runSlackSend},
	)
}

// runSlackCheck reports token/channel resolution for every surface, optionally verifying
// each token with auth.test. Exit 0 for a plain report; with --auth, exit 1 if any resolved
// token fails auth (so `fak slack check --auth` can gate CI).
func runSlackCheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	doAuth := fs.Bool("auth", false, "call Slack auth.test for each resolved token to verify it actually works")
	asJSON := fs.Bool("json", false, "emit the resolution report as JSON")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (default https://slack.com/api/; for testing/proxying)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	reports := buildSurfaceReports()
	if *doAuth {
		runAuthChecks(reports, *apiBase)
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(reports); err != nil {
			fmt.Fprintf(stderr, "fak slack check: encode json: %v\n", err)
			return 1
		}
	} else {
		renderSurfaceReports(stdout, reports, *doAuth)
	}
	return checkExit(reports, *doAuth)
}

// buildSurfaceReports resolves every surface offline (no network).
func buildSurfaceReports() []*surfaceReport {
	out := make([]*surfaceReport, 0, len(slackSurfaces))
	for _, s := range slackSurfaces {
		tok := s.token()
		ch := s.channel()
		rep := &surfaceReport{
			Name:          s.Name,
			Purpose:       s.Purpose,
			TokenSet:      tok.Value != "",
			tokenValue:    tok.Value,
			Channel:       ch.Value,
			ChannelSource: ch.Source,
			Ready:         tok.Value != "" && ch.Value != "",
		}
		if rep.TokenSet {
			rep.Token = redactToken(tok.Value)
			rep.TokenSource = tok.Source
		}
		out = append(out, rep)
	}
	return out
}

// runAuthChecks calls auth.test once per DISTINCT resolved token (many surfaces share the
// scoreboard token) and maps the verdict back onto every surface using that token.
func runAuthChecks(reports []*surfaceReport, apiBase string) {
	cache := map[string]*authReport{}
	for _, rep := range reports {
		if !rep.TokenSet {
			continue
		}
		if cached, ok := cache[rep.tokenValue]; ok {
			rep.Auth = cached
			continue
		}
		ar := probeAuth(rep.tokenValue, apiBase)
		cache[rep.tokenValue] = ar
		rep.Auth = ar
	}
}

// probeAuth runs a single auth.test against a token, returning a typed verdict (never an
// error — a failed probe IS the answer the report wants).
func probeAuth(token, apiBase string) *authReport {
	var opts []scoreboard.Option
	if apiBase != "" {
		opts = append(opts, scoreboard.WithAPIBase(apiBase))
	}
	c, err := scoreboard.NewClient(token, opts...)
	if err != nil {
		return &authReport{OK: false, Err: err.Error()}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	info, err := c.AuthTest(ctx)
	if err != nil {
		return &authReport{OK: false, Err: err.Error()}
	}
	return &authReport{OK: true, Team: info.Team, User: info.User}
}

// checkExit returns 1 only when --auth ran and a resolved token failed auth — an unset
// token is "incomplete", not "failed", and never trips the gate.
func checkExit(reports []*surfaceReport, doAuth bool) int {
	if !doAuth {
		return 0
	}
	for _, r := range reports {
		if r.Auth != nil && !r.Auth.OK {
			return 1
		}
	}
	return 0
}

// renderSurfaceReports prints the human table.
func renderSurfaceReports(w io.Writer, reports []*surfaceReport, auth bool) {
	fmt.Fprintf(w, "fak slack — %d surfaces; token/channel resolved from env or %s (walked up from cwd)\n\n",
		len(reports), slackenv.EnvFileName)
	for _, r := range reports {
		status := "READY"
		if !r.Ready {
			status = "incomplete"
		}
		fmt.Fprintf(w, "● %-11s %-10s %s\n", r.Name, status, r.Purpose)
		if r.TokenSet {
			fmt.Fprintf(w, "    token   %s  [%s]\n", r.Token, r.TokenSource)
		} else {
			fmt.Fprintf(w, "    token   (unset)\n")
		}
		if r.Channel != "" {
			fmt.Fprintf(w, "    channel %s  [%s]\n", r.Channel, r.ChannelSource)
		} else {
			fmt.Fprintf(w, "    channel (unset — pass --channel or set its env / %s)\n", slackenv.EnvFileName)
		}
		if auth && r.Auth != nil {
			if r.Auth.OK {
				fmt.Fprintf(w, "    auth    OK — %s as %s\n", slackOrDash(r.Auth.Team), slackOrDash(r.Auth.User))
			} else {
				fmt.Fprintf(w, "    auth    FAIL — %s\n", r.Auth.Err)
			}
		}
	}
}

// runSlackSend posts an ad-hoc message to any channel — the "just send something" path that
// needed a feeder subcommand before. Token defaults to the shared scoreboard token.
func runSlackSend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack send", flag.ContinueOnError)
	fs.SetOutput(stderr)
	channel := fs.String("channel", "", "target channel id (REQUIRED), e.g. C0ABC123")
	text := fs.String("text", "", "message text (REQUIRED); pass - to read the message from stdin")
	token := fs.String("token", "", "bot token (default: $FAK_SCOREBOARD_TOKEN, then .env.slack.local)")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (for testing/proxying)")
	dryRun := fs.Bool("dry-run", false, "print what would be sent and exit without posting")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	msg := *text
	if msg == "-" {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak slack send: read stdin: %v\n", err)
			return 2
		}
		msg = strings.TrimSpace(string(b))
	}

	tok := *token
	tokSource := "--token"
	if tok == "" {
		tok = scoreboard.ResolveToken()
		tokSource = scoreboardTokenKey + " / " + slackenv.EnvFileName
	}

	if *channel == "" {
		fmt.Fprintln(stderr, "fak slack send: --channel is required (e.g. --channel C0ABC123)")
		return 2
	}
	if msg == "" {
		fmt.Fprintln(stderr, "fak slack send: --text is required (or pipe a message via --text -)")
		return 2
	}

	if *dryRun {
		fmt.Fprintf(stdout, "fak slack send (dry-run):\n")
		fmt.Fprintf(stdout, "  channel : %s\n", *channel)
		fmt.Fprintf(stdout, "  token   : %s  [%s]\n", redactToken(tok), tokSource)
		fmt.Fprintf(stdout, "  text    : %s\n", msg)
		if tok == "" {
			fmt.Fprintln(stderr, "  (token is UNSET — set --token or "+scoreboardTokenKey+" before a live send)")
		}
		return 0
	}

	if tok == "" {
		fmt.Fprintln(stderr, "fak slack send: no bot token — set --token, "+scoreboardTokenKey+", or add it to "+slackenv.EnvFileName)
		return 2
	}

	var opts []scoreboard.Option
	if *apiBase != "" {
		opts = append(opts, scoreboard.WithAPIBase(*apiBase))
	}
	c, err := scoreboard.NewClient(tok, opts...)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack send: %v\n", err)
		return 1
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ts, err := c.Post(ctx, *channel, msg, nil)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack send: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak slack send: posted to %s (ts=%s)\n", *channel, ts)
	return 0
}

// redactToken shows only that a token is present plus its last 4 chars, never the secret.
// (cmd/fak's other redact helper lives in the chatrelay command file; this one is named
// distinctly so the two never collide.)
func redactToken(s string) string {
	if s == "" {
		return "(unset)"
	}
	if len(s) <= 4 {
		return "****"
	}
	return "****" + s[len(s)-4:]
}

// slackOrDash renders an empty string as "-" for the auth line.
func slackOrDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
