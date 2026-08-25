package main

// `fak slack` — the one place to DEBUG and USE fak's Slack surface.
//
// fak posts to Slack channels (scoreboard, blockers, bench, dispatch, dojo, marketing,
// news, node-usage, product, steering) and bridges one (chatrelay). Each resolves a
// bot token and a channel id from an env var or a gitignored .env.slack.local, with
// surface-specific fallbacks. When a post silently fails the operator had no way to see
// WHICH token/channel a surface would use or WHETHER the token even works — the failure
// surfaced as a cryptic chat.postMessage error deep in a feed job.
//
//	fak slack check            # resolution report for every surface (offline)
//	fak slack check --auth     # + verify token auth AND bounded channel read access
//	fak slack check --json     # machine-readable, for a CI gate or a dashboard
//	fak slack walk             # registry + refresh command map for every surface
//	fak slack refresh          # dry-run every locally refreshable feed
//	fak slack send --channel C0ABC123 --text "deploy is green"   # ad-hoc message
//	echo "hi" | fak slack send --channel C0ABC123 --text -        # text from stdin
//
// It depends only on the tracked outbound transport (internal/scoreboard) and the shared
// resolver (internal/slackenv): no lab identifiers, no shell, public side of the
// GPU-server/Slack boundary.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchpost"
	"github.com/anthony-chaudhary/fak/internal/blockerpost"
	"github.com/anthony-chaudhary/fak/internal/cachevaluepost"
	"github.com/anthony-chaudhary/fak/internal/dojopost"
	"github.com/anthony-chaudhary/fak/internal/grafanapost"
	"github.com/anthony-chaudhary/fak/internal/nodeusagepost"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackmeta"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
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
	Optional       bool   // true => no dedicated channel exists yet; INCOMPLETE here is EXPECTED, not a regression
}

// slackSurfaces is the registry `fak slack check` walks. The channel defaults reference the
// PUBLIC, non-secret constants the post packages already expose (blockerpost.ChannelDefault,
// dojopost.ChannelDefault) and the steering default in this package — never a real id baked
// in here. Surfaces marked Optional have no dedicated channel yet (no #marketing / #chatrelay
// in the workspace), so their INCOMPLETE state is expected and does NOT trip the health gate;
// wiring a channel later (set the var or add a ChannelDefault) promotes them automatically.
var slackSurfaces = []slackSurface{
	{"scoreboard", "scorecard / score / run-event status", "FAK_SCOREBOARD_TOKEN", "FAK_SCOREBOARD_CHANNEL", "", false},
	{"product", "product direction / persona findings", "", "FAK_PRODUCT_CHANNEL", scoreboard.CICDReportChannel, false},
	{"grafana", "grafana snapshots + dashboard/debug links", "FAK_GRAFANA_TOKEN", "FAK_GRAFANA_CHANNEL", grafanapost.ChannelDefault, false},
	{"alerts", "Prometheus/Alertmanager alerts (fak slack alert receiver)", "", alertsChannelEnv, grafanapost.ChannelDefault, false},
	{"blockers", "fleet blockers (status vs operator page)", "FAK_BLOCKERS_TOKEN", "FAK_BLOCKERS_CHANNEL", blockerpost.ChannelDefault, false},
	{"cachevalue", "cache-value P&L roll-up (WITNESSED kernel reuse trend)", "FAK_CACHEVALUE_TOKEN", "FAK_CACHEVALUE_CHANNEL", cachevaluepost.ChannelDefault, false},
	{"bench", "benchmark rollups / run-requests", "FAK_BENCH_TOKEN", "FAK_BENCH_CHANNEL", benchpost.ChannelDefault, false},
	{"dispatch", "background code-dispatch results", "FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "", false},
	{"dojo", "dojo rollups / trends", "FAK_DOJO_TOKEN", "FAK_DOJO_CHANNEL", dojopost.ChannelDefault, false},
	{"backlog", "issue triage + bottleneck digest", "", "FAK_BACKLOG_CHANNEL", scoreboard.CICDReportChannel, false},
	{"marketing", "marketing updates feed", "FAK_MARKETING_TOKEN", "FAK_MARKETING_CHANNEL", "", true},
	{"news", "external industry / SOTA / OSS research updates", "", "FAK_NEWS_CHANNEL", "", false},
	{"node-usage", "compute-node usage snapshots", "FAK_NODE_USAGE_TOKEN", "FAK_NODE_USAGE_CHANNEL", nodeusagepost.ChannelDefault, false},
	{"steering", "steering-guard surface", "", "FAK_STEERING_CHANNEL", steeringChannelDefault, false},
	{Name: "guard-sessions", Purpose: "one root thread per fak guard session", TokenEnv: guardSessionsTokenEnv, ChannelEnv: guardSessionsChannelEnv, ChannelDefault: guardSessionsChannelDefault},
	{"chatrelay", "Slack <-> served-model chat bridge", "FAK_CHATRELAY_TOKEN", "FAK_CHATRELAY_CHANNEL", "", true},
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
	if s.Name != "guard-sessions" {
		if r := slackenv.Lookup(scoreboardTokenKey); r.Set() {
			return resolvedField{r.Value, "scoreboard-fallback (" + string(r.Source) + ":" + r.Key + ")"}
		}
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

// channelAccessReport is the bounded conversations.history verdict for one configured
// channel. Reason is a stable operator token; Remediation turns Slack's terse API code into
// the exact next move without exposing the channel id or token.
type channelAccessReport struct {
	OK          bool   `json:"ok"`
	Reason      string `json:"reason,omitempty"`
	Remediation string `json:"remediation,omitempty"`
	Err         string `json:"error,omitempty"`
}

// surfaceReport is one row of `fak slack check`, JSON-serializable for --json.
type deliveryReport struct {
	Pending int    `json:"pending"`
	Dead    int    `json:"dead"`
	Reason  string `json:"reason,omitempty"`
}

type surfaceReport struct {
	Name          string               `json:"name"`
	Purpose       string               `json:"purpose"`
	TokenSet      bool                 `json:"token_set"`
	Token         string               `json:"token,omitempty"`          // redacted
	TokenSource   string               `json:"token_source,omitempty"`   //
	Channel       string               `json:"channel,omitempty"`        //
	ChannelSource string               `json:"channel_source,omitempty"` //
	Ready         bool                 `json:"ready"`                    // token AND channel resolved and no durable delivery failure
	Optional      bool                 `json:"optional,omitempty"`       // no dedicated channel yet — INCOMPLETE is expected, not a regression
	Delivery      *deliveryReport      `json:"delivery,omitempty"`       // durable outbox health for this producing surface
	Auth          *authReport          `json:"auth,omitempty"`           //
	ChannelAccess *channelAccessReport `json:"channel_access,omitempty"` // conversations.history read-back witness (with --auth)
	SignalNoise   slackmeta.Score      `json:"signal_noise"`             // S/N meta self-score

	tokenValue string // raw token, for the auth probe; never serialized
}

// cmdSlack routes `fak slack <check|health|beat|walk|refresh|send>`; a bare `fak slack` runs the check report so the
// most common debug action takes zero extra typing.
func cmdSlack(argv []string) {
	if len(argv) == 0 {
		os.Exit(runSlackCheck(os.Stdout, os.Stderr, nil))
	}
	dispatchSubcommands("slack", "check | health | beat | walk | refresh | send | shot | alert | fleet-status | trajectory | outbox", argv,
		subcommand{"check", runSlackCheck},
		subcommand{"health", runSlackHealth},
		subcommand{"beat", runSlackBeat},
		subcommand{"walk", runSlackWalk},
		subcommand{"refresh", runSlackRefresh},
		subcommand{"send", runSlackSend},
		subcommand{"shot", runSlackShot},
		subcommand{"alert", runSlackAlert},
		subcommand{"fleet-status", runSlackFleetStatus},
		subcommand{"trajectory", runSlackTrajectory},
		subcommand{"outbox", runSlackOutbox},
	)
}

// runSlackCheck reports token/channel resolution for every surface, optionally verifying
// each token with auth.test and every distinct token/channel pair with a one-row history
// probe. Exit 0 for a plain report; with --auth, exit 1 if auth or channel access fails (so
// `fak slack check --auth` cannot call an inaccessible surface ready).
func runSlackCheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	doAuth := fs.Bool("auth", false, "verify each resolved token and configured channel can actually be read")
	asJSON := fs.Bool("json", false, "emit the resolution report as JSON")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (default https://slack.com/api/; for testing/proxying)")
	if !parseFlags(fs, argv) {
		return 2
	}

	reports := buildSurfaceReports()
	applyOutboxDeliveryHealth(reports)
	if *doAuth {
		runAuthChecks(reports, *apiBase)
		runChannelAccessChecks(reports, *apiBase)
		for _, r := range reports {
			r.refreshSignalNoise()
		}
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

// applyOutboxDeliveryHealth folds durable producer failures into readiness. A configured
// token/channel is not READY while the outbox proves that this surface is dead-lettering.
var slackCheckOpenOutbox = openOutbox

func applyOutboxDeliveryHealth(reports []*surfaceReport) {
	ob, err := slackCheckOpenOutbox()
	if err != nil {
		return
	}
	stats, err := ob.CallStats(time.Now())
	if err != nil {
		return
	}
	var pending, dead int
	for _, source := range stats.Sources {
		if source.Source == guardSessionThreadSource || strings.HasPrefix(source.Source, guardSessionThreadSource+":") {
			pending += source.Pending
			dead += source.Dead
		}
	}
	if pending == 0 && dead == 0 {
		return
	}
	for _, report := range reports {
		if report.Name != "guard-sessions" {
			continue
		}
		report.Delivery = &deliveryReport{Pending: pending, Dead: dead}
		if dead > 0 {
			report.Delivery.Reason = "OUTBOX_DEAD_LETTER"
			report.Ready = false
		} else if pending >= 3 {
			report.Delivery.Reason = "OUTBOX_DELIVERY_STALLED"
			report.Ready = false
		}
	}
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
			Optional:      s.Optional,
		}
		if rep.TokenSet {
			rep.Token = redactToken(tok.Value)
			rep.TokenSource = tok.Source
		}
		rep.refreshSignalNoise()
		out = append(out, rep)
	}
	return out
}

func (r *surfaceReport) refreshSignalNoise() {
	signal := 1 + slackmeta.NonEmpty(r.Purpose, r.TokenSource, r.Channel, r.ChannelSource)
	noise := 1
	if !r.TokenSet {
		noise++
	}
	if r.Channel == "" {
		noise++
	}
	if r.Auth != nil {
		if r.Auth.OK {
			signal++
		} else {
			noise++
		}
	}
	if r.ChannelAccess != nil {
		if r.ChannelAccess.OK {
			signal++
		} else {
			noise++
		}
	}
	r.SignalNoise = slackmeta.New(signal, noise, "resolved Slack surface wiring vs config/auth/channel-access failures")
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
	c, err := scoreboardClient(token, apiBase)
	if err != nil {
		return &authReport{OK: false, Err: err.Error()}
	}
	info, err := slackCallWithTimeout(10*time.Second, c.AuthTest)
	if err != nil {
		return &authReport{OK: false, Err: err.Error()}
	}
	return &authReport{OK: true, Team: info.Team, User: info.User}
}

// runChannelAccessChecks probes each DISTINCT token/channel pair once, then maps the same
// verdict onto every surface sharing that pair. A valid auth.test is necessary but not
// sufficient: Slack returns channel_not_found for a stale id or a private channel the bot
// was never invited to, which is exactly the false-ready state this rung closes.
func runChannelAccessChecks(reports []*surfaceReport, apiBase string) {
	cache := map[string]*channelAccessReport{}
	for _, rep := range reports {
		if !rep.TokenSet || rep.Channel == "" {
			rep.Ready = false
			continue
		}
		if rep.Auth == nil || !rep.Auth.OK {
			rep.Ready = false
			continue
		}
		key := rep.tokenValue + "\x00" + rep.Channel
		access, ok := cache[key]
		if !ok {
			access = probeChannelAccess(rep.tokenValue, rep.Channel, apiBase)
			cache[key] = access
		}
		rep.ChannelAccess = access
		rep.Ready = access.OK
	}
}

func probeChannelAccess(token, channel, apiBase string) *channelAccessReport {
	client := slackWireClient(token, apiBase)
	if _, err := slackCallWithTimeout(10*time.Second, func(ctx context.Context) ([]slackwire.Message, error) {
		return client.History(ctx, channel, "", 1)
	}); err != nil {
		return classifyChannelAccessError(err)
	}
	return &channelAccessReport{OK: true, Reason: "CHANNEL_ACCESS_OK"}
}

func slackWireClient(token, apiBase string) *slackwire.Client {
	var opts []slackwire.Option
	if apiBase != "" {
		opts = append(opts, slackwire.WithAPIBase(apiBase))
	}
	return slackwire.New(token, opts...)
}

func slackCallWithTimeout[T any](timeout time.Duration, call func(context.Context) (T, error)) (T, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return call(ctx)
}

func classifyChannelAccessError(err error) *channelAccessReport {
	report := &channelAccessReport{OK: false, Reason: "CHANNEL_ACCESS_FAILED", Err: err.Error(), Remediation: "inspect the Slack API error and repair the surface channel wiring"}
	var apiErr *slackwire.APIError
	if !errors.As(err, &apiErr) {
		return report
	}
	switch apiErr.Code {
	case "channel_not_found":
		report.Reason = "CHANNEL_NOT_FOUND"
		report.Remediation = "verify the configured channel and invite the bot/app to it"
	case "not_in_channel":
		report.Reason = "BOT_NOT_IN_CHANNEL"
		report.Remediation = "invite the bot/app to the configured channel"
	case "missing_scope":
		report.Reason = "MISSING_HISTORY_SCOPE"
		report.Remediation = "grant the matching channel-history scope and reinstall the Slack app"
	case "invalid_auth", "not_authed", "token_revoked":
		report.Reason = "TOKEN_AUTH_FAILED"
		report.Remediation = "replace or reinstall the Slack bot token"
	}
	return report
}

// checkExit returns 1 when --auth ran and a resolved token or configured channel failed its
// live probe. An unset token/channel remains "incomplete", not "failed", and does not trip.
func checkExit(reports []*surfaceReport, doAuth bool) int {
	for _, r := range reports {
		if r.Delivery != nil && r.Delivery.Reason != "" {
			return 1
		}
	}
	if !doAuth {
		return 0
	}
	for _, r := range reports {
		if r.Auth != nil && !r.Auth.OK {
			return 1
		}
		if r.ChannelAccess != nil && !r.ChannelAccess.OK {
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
		if auth && r.ChannelAccess != nil {
			if r.ChannelAccess.OK {
				fmt.Fprintln(w, "    access  OK — bounded conversations.history read succeeded")
			} else {
				fmt.Fprintf(w, "    access  FAIL [%s] — %s\n", r.ChannelAccess.Reason, r.ChannelAccess.Err)
				fmt.Fprintf(w, "    fix     %s\n", r.ChannelAccess.Remediation)
			}
		}
		fmt.Fprintf(w, "    S/N     %s\n", r.SignalNoise.Line())
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
	durable := fs.Bool("durable", true, "enqueue through the durable outbox: the message survives crashes/429s/token drift and is delivered by this call's drain or a later one (default; pass --direct to bypass)")
	direct := fs.Bool("direct", false, "bypass the durable outbox: one fire-and-forget post (old behavior — a failed send is lost)")
	dryRun := fs.Bool("dry-run", false, "print what would be sent and exit without posting")
	if !parseFlags(fs, argv) {
		return 2
	}

	// Durable is the default (#2262): an ad-hoc send survives a crash/429/token drift
	// instead of being dropped. --direct restores the old fire-and-forget post for the
	// rare case a caller explicitly wants a single in-process attempt and a hard failure.
	useDurable := *durable && !*direct

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
		if useDurable {
			fmt.Fprintf(stdout, "  durable : would enqueue into %s then drain\n", resolveOutboxDir())
		} else {
			fmt.Fprintf(stdout, "  direct  : would post once, fire-and-forget (a failed send is lost)\n")
		}
		if tok == "" {
			fmt.Fprintln(stderr, "  (token is UNSET — set --token or "+scoreboardTokenKey+" before a live send)")
		}
		return 0
	}

	if useDurable {
		// Durability first: the row is on disk before any network is attempted, so a
		// crash, 429, or missing token delays the message instead of losing it. The
		// in-process drain is best-effort — a failure leaves the row for the next
		// drain (another --durable send, `fak slack outbox drain`, or the watchdog).
		ob, err := openOutbox()
		if err != nil {
			fmt.Fprintf(stderr, "fak slack send: outbox: %v\n", err)
			return 1
		}
		nonce, err := ob.Enqueue(slackoutbox.Row{Channel: *channel, Text: msg, Source: "slack-send"})
		if err != nil {
			fmt.Fprintf(stderr, "fak slack send: enqueue: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "fak slack send: enqueued durably (nonce=%s)\n", nonce)
		wire, werr := outboxWire(tok, *apiBase)
		if werr != nil {
			fmt.Fprintf(stdout, "  delivery deferred: %v — run `fak slack outbox drain` once configured\n", werr)
			return 0
		}
		rep, derr := ob.Drain(ctx(), wire, stdDrainOpts())
		switch {
		case derr == slackoutbox.ErrDrainBusy:
			fmt.Fprintln(stdout, "  delivery deferred: another drainer holds the lock")
		case derr != nil:
			fmt.Fprintf(stdout, "  delivery deferred: %v\n", derr)
		default:
			fmt.Fprintf(stdout, "  drained: posted %d  refused %d  failed %d  remaining %d\n",
				rep.Posted, rep.Refused, rep.Failed, rep.Remaining)
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
func redactToken(s string) string {
	return redactSecret(s)
}

// slackOrDash renders an empty string as "-" for the auth line.
func slackOrDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
