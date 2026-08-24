package main

// `fak slack outbox` — operate the durable Slack outbox (#2262, epic #2259).
//
// Producers (feeders, `fak slack send --durable`) ENQUEUE rows into a local JSONL
// spool and return once the row is on disk; this verb family is the operator's window
// into delivery:
//
//	fak slack outbox status          # pending/posted/dead/refused counts + ages
//	fak slack outbox status --json   # machine-readable, for the watchdog
//	fak slack outbox drain           # run one serialized drain pass now
//	fak slack outbox drain --dry-run # print the send plan, touch nothing
//	fak slack outbox retry --all     # re-arm every dead row (or --nonce <n>)
//	fak slack outbox dead            # list dead rows with their structured reasons
//	fak slack outbox compact         # fold old settled rows + heartbeats out of the spool
//	fak slack outbox compact --dry-run --json  # preview what a pass would drop
//	fak slack outbox reap            # delete ephemeral (bridge-channel) messages idle past their TTL
//	fak slack outbox reap --dry-run  # show what would be reaped, touch nothing
//	fak slack outbox limits          # effective retention windows + live occupancy vs them
//	fak slack outbox calls           # per-source Slack API-call spend vs saved (rate-limit gauge)
//	fak slack outbox calls --json    # machine-readable, for a before/after noise baseline
//
// The spool lives at $FAK_SLACK_OUTBOX_DIR (env or .env.slack.local), defaulting to
// .dispatch-runs/slack-outbox under the working directory — the same local-runs root
// the dispatch ledgers use. The drainer posts with the shared scoreboard workspace
// token (one bot serves every fak surface today); rows never carry a secret.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

const (
	// outboxDirEnv overrides where the outbox spool lives (env or .env.slack.local).
	outboxDirEnv = "FAK_SLACK_OUTBOX_DIR"
	// outboxDirDefault sits under the same local-runs root as the dispatch ledgers.
	outboxDirDefault = ".dispatch-runs/slack-outbox"
	// outboxStallBudget is how old the oldest pending row may grow before the health
	// rung grades the outbox STALLED — generous enough for a long 429 storm, tight
	// enough that a wedged drain pages within a workday.
	outboxStallBudget = 2 * time.Hour
	// ephemeralChannelsEnv lists the channel ids whose posted messages the reaper deletes
	// by default once idle past its TTL (the dgx-bridge allowlist). Comma- or
	// whitespace-separated channel ids; unset disables the channel-default reaper (rows
	// still opt in per message via DeleteAfterS). Env or .env.slack.local.
	ephemeralChannelsEnv = "FAK_SLACK_EPHEMERAL_CHANNELS"
)

// resolveEphemeralChannels folds FAK_SLACK_EPHEMERAL_CHANNELS into the set of channel ids
// whose messages auto-expire by default. Empty when the key is unset or blank.
func resolveEphemeralChannels() map[string]bool {
	r := slackenv.Lookup(ephemeralChannelsEnv)
	if !r.Set() {
		return nil
	}
	set := map[string]bool{}
	for _, ch := range strings.FieldsFunc(r.Value, func(c rune) bool { return c == ',' || c == ' ' || c == '\t' || c == '\n' }) {
		if ch != "" {
			set[ch] = true
		}
	}
	if len(set) == 0 {
		return nil
	}
	return set
}

// ephemeralPredicate builds the reaper's channel-default membership test from a set. It
// returns nil (the reaper's "no channel default" signal) for an empty set, so an unconfigured
// deployment never auto-deletes anything.
func ephemeralPredicate(set map[string]bool) func(string) bool {
	if len(set) == 0 {
		return nil
	}
	return func(channel string) bool { return set[channel] }
}

// stdDrainOpts is the drain configuration every fak drainer shares: the leak-fence root plus
// the ephemeral (dgx-bridge) reaper resolved from FAK_SLACK_EPHEMERAL_CHANNELS. Because it
// rides on every drainer — the guard session-card path, the watchdog, alerts, feeders, `fak
// slack outbox drain` — any drain in a bridge channel also clears that channel's messages
// once they go idle past the TTL, so keeping the channels clean needs no separate scheduler.
// It is a no-op when the allowlist is unset (nil predicate → the reaper touches nothing).
func stdDrainOpts() slackoutbox.DrainOpts {
	return slackoutbox.DrainOpts{Root: ".", ReapEphemeral: ephemeralPredicate(resolveEphemeralChannels())}
}

// resolveOutboxDir applies the documented resolution order for the spool directory.
func resolveOutboxDir() string {
	if r := slackenv.Lookup(outboxDirEnv); r.Set() {
		return r.Value
	}
	return filepath.FromSlash(outboxDirDefault)
}

// openOutbox opens the resolved spool directory.
func openOutbox() (*slackoutbox.Outbox, error) {
	return slackoutbox.Open(resolveOutboxDir())
}

// outboxWire builds the drain transport on the shared workspace token. token=""
// resolves the scoreboard token (the one bot every surface shares today).
func outboxWire(token, apiBase string) (*slackwire.Client, error) {
	if token == "" {
		token = scoreboard.ResolveToken()
	}
	if token == "" {
		return nil, fmt.Errorf("no bot token: set FAK_SCOREBOARD_TOKEN, or add it to %s", slackenv.EnvFileName)
	}
	var opts []slackwire.Option
	if apiBase != "" {
		opts = append(opts, slackwire.WithAPIBase(apiBase))
	}
	return slackwire.New(token, opts...), nil
}

// runSlackOutbox routes `fak slack outbox <status|drain|retry|dead|compact|limits|calls>`.
func runSlackOutbox(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return runSlackOutboxStatus(stdout, stderr, nil)
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "status":
		return runSlackOutboxStatus(stdout, stderr, rest)
	case "drain":
		return runSlackOutboxDrain(stdout, stderr, rest)
	case "retry":
		return runSlackOutboxRetry(stdout, stderr, rest)
	case "dead":
		return runSlackOutboxDead(stdout, stderr, rest)
	case "compact":
		return runSlackOutboxCompact(stdout, stderr, rest)
	case "reap":
		return runSlackOutboxReap(stdout, stderr, rest)
	case "limits":
		return runSlackOutboxLimits(stdout, stderr, rest)
	case "calls":
		return runSlackOutboxCalls(stdout, stderr, rest)
	default:
		fmt.Fprintf(stderr, "fak slack outbox: unknown subcommand %q (want status | drain | retry | dead | compact | reap | limits | calls)\n", sub)
		return 2
	}
}

func runSlackOutboxStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the status fold as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox status: %v\n", err)
		return 1
	}
	st, err := ob.Status(time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox status: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(st); err != nil {
			fmt.Fprintf(stderr, "fak slack outbox status: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "fak slack outbox — spool %s\n", ob.Dir())
	fmt.Fprintf(stdout, "  pending %d  posted %d  dead %d  refused %d  superseded %d  reaped %d  corrupt %d\n",
		st.Pending, st.Posted, st.Dead, st.Refused, st.Superseded, st.Reaped, st.Corrupt)
	fmt.Fprintf(stdout, "  oldest pending: %s   last drain: %s\n",
		ageOrDash(st.OldestPendingAgeS), ageOrDash(st.LastDrainAgeS))
	for _, d := range st.DeadRows {
		fmt.Fprintf(stdout, "  ● dead %s ch=%s src=%s attempts=%d — %s\n", d.Nonce, d.Channel, d.Source, d.Attempts, d.Reason)
	}
	return 0
}

// ageOrDash renders an age-in-seconds fold field (-1 = not applicable).
func ageOrDash(s int64) string {
	if s < 0 {
		return "-"
	}
	return (time.Duration(s) * time.Second).String()
}

func runSlackOutboxDrain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox drain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", false, "print the send plan (coalesced, per-channel FIFO) and exit without sending")
	token := fs.String("token", "", "bot token (default: $FAK_SCOREBOARD_TOKEN, then .env.slack.local)")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (for testing/proxying)")
	maxAttempts := fs.Int("max-attempts", 0, "dead-letter a row after this many failed sends (default 5)")
	if !parseFlags(fs, argv) {
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox drain: %v\n", err)
		return 1
	}
	if *dryRun {
		plan, _, err := ob.Plan()
		if err != nil {
			fmt.Fprintf(stderr, "fak slack outbox drain: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "fak slack outbox drain (dry-run): %d send(s) planned\n", len(plan))
		for _, p := range plan {
			extra := ""
			if len(p.Supersedes) > 0 {
				extra = fmt.Sprintf("  (coalesces %d older update(s))", len(p.Supersedes))
			}
			if p.NeedsProbe {
				extra += "  (nonce probe first)"
			}
			fmt.Fprintf(stdout, "  %-6s %s ch=%s attempts=%d%s\n", p.Action, p.Row.Nonce, p.Row.Channel, p.Attempts, extra)
		}
		return 0
	}
	wire, err := outboxWire(*token, *apiBase)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox drain: %v\n", err)
		return 2
	}
	dopts := stdDrainOpts()
	dopts.MaxAttempts = *maxAttempts
	rep, err := ob.Drain(ctx(), wire, dopts)
	if err == slackoutbox.ErrDrainBusy {
		fmt.Fprintln(stdout, "another drainer holds the lock — nothing to do")
		return 0
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox drain: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "drained: posted %d  updated %d  recovered %d  refused %d  superseded %d  failed %d  dead %d  reaped %d  remaining %d\n",
		rep.Posted, rep.Updated, rep.Recovered, rep.Refused, rep.Superseded, rep.Failed, rep.Dead, rep.Reaped, rep.Remaining)
	return 0
}

func runSlackOutboxRetry(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox retry", flag.ContinueOnError)
	fs.SetOutput(stderr)
	nonce := fs.String("nonce", "", "re-arm one dead row by nonce")
	all := fs.Bool("all", false, "re-arm every dead row")
	dryRun := fs.Bool("dry-run", false, "print which rows would re-arm without writing")
	if !parseFlags(fs, argv) {
		return 2
	}
	if (*nonce == "") == !*all {
		fmt.Fprintln(stderr, "fak slack outbox retry: pass exactly one of --nonce <n> or --all")
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox retry: %v\n", err)
		return 1
	}
	if *dryRun {
		dead, err := ob.Dead()
		if err != nil {
			fmt.Fprintf(stderr, "fak slack outbox retry: %v\n", err)
			return 1
		}
		n := 0
		for _, d := range dead {
			if *all || d.Nonce == *nonce {
				fmt.Fprintf(stdout, "would re-arm %s ch=%s — %s\n", d.Nonce, d.Channel, d.Reason)
				n++
			}
		}
		fmt.Fprintf(stdout, "fak slack outbox retry (dry-run): %d row(s)\n", n)
		return 0
	}
	armed, err := ob.Retry(*nonce)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox retry: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "re-armed %d row(s); run `fak slack outbox drain` to deliver\n", len(armed))
	return 0
}

func runSlackOutboxDead(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox dead", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit dead rows as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox dead: %v\n", err)
		return 1
	}
	dead, err := ob.Dead()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox dead: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(dead); err != nil {
			fmt.Fprintf(stderr, "fak slack outbox dead: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	if len(dead) == 0 {
		fmt.Fprintln(stdout, "no dead rows")
		return 0
	}
	for _, d := range dead {
		fmt.Fprintf(stdout, "● %s ch=%s src=%s attempts=%d enqueued=%s\n    %s\n",
			d.Nonce, d.Channel, d.Source, d.Attempts, d.EnqueuedAt, d.Reason)
	}
	return 0
}

// runSlackOutboxCompact folds old settled rows and the drain_pass heartbeat storm out of
// the spool. The retention windows default to the assertive package defaults; --retain
// overrides the SETTLED (superseded/refused) window and --retain-dead the DEAD window
// (an unretried dead row is dropped past it, default 14d). Posted rows keep their safe 48h
// so a deferred producer's PostedTS probe still resolves.
func runSlackOutboxCompact(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox compact", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", false, "report what a pass would drop without touching a file")
	asJSON := fs.Bool("json", false, "emit the compaction report as JSON")
	retain := fs.Duration("retain", 0, "settled-row (superseded/refused) retention window (default 1h)")
	retainDead := fs.Duration("retain-dead", 0, "dead-row retention window before an unretried dead row is dropped (default 336h)")
	maxPendingAge := fs.Duration("max-pending-age", 0, "drop undelivered rows older than this explicit age (default off; preview with --dry-run)")
	if !parseFlags(fs, argv) {
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox compact: %v\n", err)
		return 1
	}
	rep, err := ob.Compact(slackoutbox.CompactOpts{RetainSettled: *retain, RetainDead: *retainDead, MaxPendingAge: *maxPendingAge, DryRun: *dryRun})
	if err == slackoutbox.ErrDrainBusy {
		fmt.Fprintln(stdout, "another drainer holds the lock — nothing to do")
		return 0
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox compact: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak slack outbox compact: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "fak slack outbox — spool %s\n  %s\n", ob.Dir(), slackoutbox.CompactReportLine(rep))
	return 0
}

// runSlackOutboxReap deletes posted messages in the ephemeral (dgx-bridge) channels that have
// been idle past their TTL, so those channels stay clean. The channel allowlist comes from
// FAK_SLACK_EPHEMERAL_CHANNELS (supplemented by --channels); the drain path runs this same
// pass automatically, so this verb is the manual/one-off and --dry-run inspection surface. A
// row that opted in per message (DeleteAfterS) is reaped regardless of the allowlist.
func runSlackOutboxReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", false, "report what would be reaped without deleting or recording anything")
	asJSON := fs.Bool("json", false, "emit the reap report as JSON")
	ttl := fs.Duration("ttl", 0, "idle window a message survives before it is reaped (default 30m)")
	channels := fs.String("channels", "", "comma/space-separated channel ids to reap, on top of FAK_SLACK_EPHEMERAL_CHANNELS")
	token := fs.String("token", "", "bot token (default: $FAK_SCOREBOARD_TOKEN, then .env.slack.local)")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (for testing/proxying)")
	if !parseFlags(fs, argv) {
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox reap: %v\n", err)
		return 1
	}

	set := resolveEphemeralChannels()
	if *channels != "" {
		if set == nil {
			set = map[string]bool{}
		}
		for _, ch := range strings.FieldsFunc(*channels, func(c rune) bool { return c == ',' || c == ' ' || c == '\t' || c == '\n' }) {
			if ch != "" {
				set[ch] = true
			}
		}
	}
	opts := slackoutbox.ReapOpts{Ephemeral: ephemeralPredicate(set), TTL: *ttl, DryRun: *dryRun}

	// A dry run never touches the wire (it stops before the delete), so it needs no token.
	var wire slackoutbox.Wire
	if !*dryRun {
		w, werr := outboxWire(*token, *apiBase)
		if werr != nil {
			fmt.Fprintf(stderr, "fak slack outbox reap: %v\n", werr)
			return 2
		}
		wire = w
	}

	rep, err := ob.Reap(ctx(), wire, opts)
	if err == slackoutbox.ErrDrainBusy {
		fmt.Fprintln(stdout, "another drainer holds the lock — nothing to do")
		return 0
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox reap: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak slack outbox reap: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "fak slack outbox — spool %s\n  %s\n", ob.Dir(), slackoutbox.ReapReportLine(rep))
	if len(set) == 0 {
		fmt.Fprintf(stdout, "  (no ephemeral channels configured — set %s or pass --channels; only rows with an explicit delete_after_s are reaped)\n", ephemeralChannelsEnv)
	}
	return 0
}

// runSlackOutboxLimits prints the outbox's effective retention/compaction envelope and
// where the live spool currently sits against it — how many folded rows are terminal, how
// many are already past their window (droppable), and whether an automatic pass is due. It
// is the read-only companion to `compact`: it reuses the same predicates, so what it calls
// droppable is exactly what a pass would drop.
func runSlackOutboxLimits(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox limits", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the limits report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox limits: %v\n", err)
		return 1
	}
	lim, err := ob.Limits()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox limits: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(lim); err != nil {
			fmt.Fprintf(stderr, "fak slack outbox limits: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "fak slack outbox — spool %s\n  %s\n", ob.Dir(), slackoutbox.LimitsReportLine(lim))
	return 0
}

// runSlackOutboxCalls prints the per-source Slack API-call footprint the delivery has spent —
// the rate-limit gauge behind "the session cards are wasting our Slack limits". It folds the
// durable state log (no live API read): calls SENT (chat.postMessage + chat.update that
// reached the wire) and calls SAVED (edits coalesced away + fence refusals), attributed to the
// producing surface and sorted loudest-first. Run it, change a producer's cadence, run it
// again — the reduction is a number, not a vibe. The counts cover only the retained log
// (compaction folds settled rows out on their retention windows), reported as the window floor.
func runSlackOutboxCalls(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox calls", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the per-source call-volume fold as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox calls: %v\n", err)
		return 1
	}
	cs, err := ob.CallStats(time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox calls: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(cs); err != nil {
			fmt.Fprintf(stderr, "fak slack outbox calls: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "fak slack outbox — spool %s\n  %s\n", ob.Dir(), slackoutbox.CallStatsReportLine(cs))
	for _, sc := range cs.Sources {
		fmt.Fprintf(stdout, "  ● %-24s sent %d (post %d, update %d)  saved %d (coalesced %d, refused %d)",
			sc.Source, sc.Sent(), sc.Posts, sc.Updates, sc.Saved(), sc.Coalesced, sc.Refused)
		if sc.Dead > 0 {
			fmt.Fprintf(stdout, "  dead %d", sc.Dead)
		}
		if sc.Pending > 0 {
			fmt.Fprintf(stdout, "  pending %d", sc.Pending)
		}
		fmt.Fprintln(stdout)
	}
	return 0
}

// outboxHealthRung folds the outbox into one `fak slack health` row: dead rows or a
// stalled backlog are exactly the delivery failures the durable outbox exists to make
// LOUD — silence here would re-create the fire-and-forget hole (#2262).
func outboxHealthRung(now time.Time) healthReport {
	hr := healthReport{Name: "outbox", LastPostAgeS: -1, BudgetS: int64(outboxStallBudget / time.Second)}
	ob, err := openOutbox()
	if err != nil {
		hr.Verdict = verdictOutboxStalled
		hr.Detail = "outbox unreadable: " + err.Error()
		return hr
	}
	st, err := ob.Status(now)
	if err != nil {
		hr.Verdict = verdictOutboxStalled
		hr.Detail = "outbox unreadable: " + err.Error()
		return hr
	}
	hr.Ready = true
	hr.AuthOK = true // no token of its own; delivery auth surfaces per-surface above
	switch {
	case st.Dead > 0:
		hr.Verdict = verdictDeadRows
		hr.Detail = fmt.Sprintf("%d dead row(s) — first: %s (fak slack outbox dead)", st.Dead, st.DeadRows[0].Reason)
	case st.Pending > 0 && time.Duration(st.OldestPendingAgeS)*time.Second > outboxStallBudget:
		hr.Verdict = verdictOutboxStalled
		hr.Detail = fmt.Sprintf("oldest pending row is %s old (budget %s; last drain %s) — run fak slack outbox drain",
			ageOrDash(st.OldestPendingAgeS), outboxStallBudget, ageOrDash(st.LastDrainAgeS))
	default:
		hr.Verdict = verdictOK
		hr.Detail = fmt.Sprintf("pending %d, posted %d, refused %d (spool %s)", st.Pending, st.Posted, st.Refused, ob.Dir())
	}
	return hr
}
