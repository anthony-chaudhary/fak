package main

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
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
	client, err := scoreboard.NewClient(tok)
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", o.prefix, err)
		return 2
	}

	var ts string
	if o.dedupe {
		ts, err = client.PostWithUpdate(ctx(), ch, up, up.Text(), up.Blocks())
	} else {
		ts, err = client.Post(ctx(), ch, up.Text(), up.Blocks())
	}
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", o.prefix, err)
		return 1
	}
	if o.dedupe && ts == "" {
		fmt.Fprintln(stdout, "skipped: no change from last post for this title")
		return 0
	}
	fmt.Fprintf(stdout, "posted to %s ts=%s\n", ch, ts)
	return 0
}
