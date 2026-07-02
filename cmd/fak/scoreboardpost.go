package main

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

// scoreboardPostFlow is the shared dry-run / channel-resolve / token-resolve / post
// tail that the scoreboard, node-usage, and product `post` handlers all run once they
// have built their scoreboard.Update. The per-command differences (error prefix, which
// channel/token resolver to fall back to, the no-channel hint, and whether to use the
// dedupe-aware PostWithUpdate) are threaded through scoreboardPostOpts.
type scoreboardPostOpts struct {
	prefix        string // error/usage prefix, e.g. "fak scoreboard post"
	channelFlag   string // --channel override ("" => resolve)
	tokenFlag     string // --token override
	resolveChan   func() string
	resolveToken  func() string // optional fallback when tokenFlag == ""
	noChannelHint string        // shown when no channel could be resolved
	dryRun        bool
	// dedupe true posts via PostWithUpdate (skips an unchanged repost); false uses Post.
	dedupe bool
}

func scoreboardPostFlow(stdout, stderr io.Writer, up scoreboard.Update, o scoreboardPostOpts) int {
	if o.dryRun {
		fmt.Fprintln(stdout, up.Text())
		return 0
	}

	ch := o.channelFlag
	if ch == "" {
		ch = o.resolveChan()
	}
	if ch == "" {
		fmt.Fprintln(stderr, o.noChannelHint)
		return 2
	}

	tok := o.tokenFlag
	if tok == "" && o.resolveToken != nil {
		tok = o.resolveToken()
	}
	client, err := newScoreboardPostClient(tok)
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", o.prefix, err)
		return 2
	}

	// Durable tail (#2262): any spooled backlog for this channel drains FIRST (FIFO —
	// a fresh post must not jump messages that failed earlier), and a failed post is
	// ENQUEUED instead of lost. The feeders keep failing open (exit 0), but open now
	// means "delayed, on disk, alarmed by the health rung" rather than "gone".
	if left := drainOutboxBacklog(stdout, ch, tok); left > 0 {
		nonce, qerr := enqueueOutboxRow(ch, up, o.prefix)
		if qerr != nil {
			fmt.Fprintf(stderr, "%s: channel backlog pending and enqueue failed: %v\n", o.prefix, qerr)
			return 1
		}
		fmt.Fprintf(stdout, "channel has %d undelivered row(s); enqueued behind them (nonce=%s)\n", left, nonce)
		return 0
	}

	var ts string
	if o.dedupe {
		ts, err = client.PostWithUpdate(ctx(), ch, up, up.Text(), up.Blocks())
	} else {
		ts, err = client.Post(ctx(), ch, up.Text(), up.Blocks())
	}
	if err != nil {
		nonce, qerr := enqueueOutboxRow(ch, up, o.prefix)
		if qerr != nil {
			fmt.Fprintf(stderr, "%s: %v (and enqueue failed: %v)\n", o.prefix, err, qerr)
			return 1
		}
		fmt.Fprintf(stdout, "post failed (%v); enqueued durably (nonce=%s) — delivered by the next outbox drain\n", err, nonce)
		return 0
	}
	if o.dedupe && ts == "" {
		fmt.Fprintln(stdout, "skipped: no change from last post for this title")
		return 0
	}
	fmt.Fprintf(stdout, "posted to %s ts=%s\n", ch, ts)
	return 0
}

// newScoreboardPostClient builds the flow's Slack client. It is a package var so a
// test can bind it to an httptest server instead of the live API (the same seam idiom
// as newHealthHistoryReader).
var newScoreboardPostClient = func(tok string) (*scoreboard.Client, error) {
	return scoreboard.NewClient(tok)
}

// drainOutboxBacklog best-effort drains the outbox and returns how many rows are
// still owed for channel ch afterwards (0 on any outbox error — the direct-post path
// then proceeds exactly as before this leaf existed).
func drainOutboxBacklog(stdout io.Writer, ch, tok string) int {
	ob, err := openOutbox()
	if err != nil {
		return 0
	}
	if wire, werr := outboxWire(tok, ""); werr == nil {
		if rep, derr := ob.Drain(ctx(), wire, slackoutbox.DrainOpts{Root: "."}); derr == nil && rep.Posted+rep.Updated+rep.Recovered > 0 {
			fmt.Fprintf(stdout, "outbox: delivered %d spooled row(s)\n", rep.Posted+rep.Updated+rep.Recovered)
		}
	}
	plan, _, err := ob.Plan()
	if err != nil {
		return 0
	}
	left := 0
	for _, p := range plan {
		if p.Row.Channel == ch {
			left++
		}
	}
	return left
}

// enqueueOutboxRow spools one scoreboard-family card for a later drain.
func enqueueOutboxRow(ch string, up scoreboard.Update, source string) (string, error) {
	ob, err := openOutbox()
	if err != nil {
		return "", err
	}
	return ob.Enqueue(slackoutbox.Row{Channel: ch, Text: up.Text(), Blocks: up.Blocks(), Source: source})
}
