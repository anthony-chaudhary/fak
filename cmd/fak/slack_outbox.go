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
)

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

// runSlackOutbox routes `fak slack outbox <status|drain|retry|dead>`.
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
	default:
		fmt.Fprintf(stderr, "fak slack outbox: unknown subcommand %q (want status | drain | retry | dead)\n", sub)
		return 2
	}
}

func runSlackOutboxStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the status fold as JSON")
	if err := fs.Parse(argv); err != nil {
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
	fmt.Fprintf(stdout, "  pending %d  posted %d  dead %d  refused %d  superseded %d  corrupt %d\n",
		st.Pending, st.Posted, st.Dead, st.Refused, st.Superseded, st.Corrupt)
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
	if err := fs.Parse(argv); err != nil {
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
	rep, err := ob.Drain(ctx(), wire, slackoutbox.DrainOpts{Root: ".", MaxAttempts: *maxAttempts})
	if err == slackoutbox.ErrDrainBusy {
		fmt.Fprintln(stdout, "another drainer holds the lock — nothing to do")
		return 0
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak slack outbox drain: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "drained: posted %d  updated %d  recovered %d  refused %d  superseded %d  failed %d  dead %d  remaining %d\n",
		rep.Posted, rep.Updated, rep.Recovered, rep.Refused, rep.Superseded, rep.Failed, rep.Dead, rep.Remaining)
	return 0
}

func runSlackOutboxRetry(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack outbox retry", flag.ContinueOnError)
	fs.SetOutput(stderr)
	nonce := fs.String("nonce", "", "re-arm one dead row by nonce")
	all := fs.Bool("all", false, "re-arm every dead row")
	dryRun := fs.Bool("dry-run", false, "print which rows would re-arm without writing")
	if err := fs.Parse(argv); err != nil {
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
	if err := fs.Parse(argv); err != nil {
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
