package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/marketing"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// cmdMarketing posts completion-driven marketing to the #marketing Slack channel and
// generates the artifacts that back it. It is the human/auto surface over internal/marketing:
// every claim it emits cites a git-witnessed, ship-stamped commit (hooks.StampOf), so a
// marketing post is exactly as trustworthy as the commit behind it — an unwitnessed boast is
// refused at construction, and a CLAIMS.md [SIMULATED]/[STUB] feature is excluded.
//
//	fak marketing generate --since HEAD~20            # build a digest, print it (dry by default)
//	fak marketing generate --range a..b --json        # the gathered ships as JSON (for piping)
//	fak marketing post --since HEAD~20                 # generate + post to #marketing
//	fak marketing post --since HEAD~20 --dry-run       # render + print; do not post
//
// It targets the FAK_MARKETING_* workspace (token falls back to the scoreboard token; the
// channel is never a hard-coded default — a marketing post never silently lands in
// #scoreboard).
func cmdMarketing(argv []string) {
	dispatchSubcommands("marketing", "generate | post | tick", argv,
		subcommand{"generate", runMarketingGenerate},
		subcommand{"post", runMarketingPost},
		subcommand{"tick", runMarketingTick},
	)
}

// marketingPoster adapts the scoreboard chat.postMessage client to the marketing.Poster
// seam. It keys the dedupe-aware PostWithUpdate on the artifact's stable DedupeKey, so a
// re-tick over the same ships is collapsed by the transport too (the backstop behind the
// high-water-mark CAS). The marketing channel id is never a tracked default.
type marketingPoster struct {
	client  *scoreboard.Client
	channel string
}

func (p marketingPoster) PostArtifact(ctx context.Context, a marketing.Artifact) (string, error) {
	// Update{Title: DedupeKey} routes the post through scoreboard's per-title dedupe gate,
	// so an unchanged digest for the same ship-set is skipped rather than reposted.
	return p.client.PostWithUpdate(ctx, p.channel, scoreboard.Update{Title: a.DedupeKey}, a.Text(), a.Blocks())
}

// runMarketingTick handles `fak marketing tick`: the single idempotent entrypoint every
// trigger (serve bgloop, git hook, cron) funnels through. It reads the high-water mark,
// gathers genuinely-new witnessed ships, gates on emptiness, advances the mark via a
// compare-and-swap, and posts once. --dry-run renders without posting OR advancing.
func runMarketingTick(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak marketing tick", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repo root to read git/CLAIMS.md from")
	source := fs.String("source", "", "who fired: serve | hook | cron | cli (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	bootstrap := fs.String("bootstrap", "HEAD~20", "first-run range start when no mark exists (e.g. HEAD~20, or ALL for full history)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_MARKETING_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_MARKETING_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the artifact and print it; do not post or advance the mark")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	src := *source
	if src == "" {
		src = defaultSource()
	}
	opts := marketing.Opts{Root: *root, Source: src, Bootstrap: *bootstrap, DryRun: *dryRun}

	// Wire the poster only for a real (non-dry) tick that has a resolvable channel + token.
	if !*dryRun {
		ch := *channel
		if ch == "" {
			ch = marketing.ResolveChannel()
		}
		if ch == "" {
			fmt.Fprintln(stderr, "fak marketing tick: no channel: pass --channel, set FAK_MARKETING_CHANNEL, or add it to .env.slack.local")
			return 2
		}
		tok := *token
		if tok == "" {
			tok = marketing.ResolveToken()
		}
		client, err := scoreboard.NewClient(tok)
		if err != nil {
			fmt.Fprintf(stderr, "fak marketing tick: %v\n", err)
			return 2
		}
		opts.Poster = marketingPoster{client: client, channel: ch}
	}

	res, err := opts.Tick(ctx(), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing tick: %v\n", err)
		return 1
	}
	switch res.Status {
	case "dry-run":
		fmt.Fprintln(stdout, res.Artifact.Text())
	case "posted":
		fmt.Fprintf(stdout, "posted %d witnessed ship(s) ts=%s (mark %s->%s)\n", res.NewShips, res.PostedTS, shortMark(res.OldMark), shortMark(res.NewMark))
	default:
		fmt.Fprintf(stdout, "%s (%d new ship(s))\n", res.Status, res.NewShips)
	}
	return 0
}

// shortMark trims a mark sha for display; "" renders as "(none)".
func shortMark(sha string) string {
	if sha == "" {
		return "(none)"
	}
	if len(sha) > 8 {
		return sha[:8]
	}
	return sha
}

// marketingRange resolves the rev-range from --range (verbatim) or --since (<ref>..HEAD).
// "" means the whole HEAD history. --range wins if both are set.
func marketingRange(rangeFlag, sinceFlag string) string {
	if rangeFlag != "" {
		return rangeFlag
	}
	if sinceFlag != "" {
		return sinceFlag + "..HEAD"
	}
	return ""
}

// runMarketingGenerate handles `fak marketing generate`: gather the witnessed ships in the
// range, fold them into a digest Artifact, and print it (the default is the rendered card;
// --json emits the gathered ships for piping). It never posts — it is the preview/inspect
// surface, so it has no --dry-run (it is always dry).
func runMarketingGenerate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak marketing generate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	since := fs.String("since", "", "gather ships since this ref (<ref>..HEAD)")
	rangeFlag := fs.String("range", "", "gather ships in this rev-range (e.g. abc123..HEAD); wins over --since")
	root := fs.String("root", ".", "repo root to read git/CLAIMS.md from")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	dated := fs.Bool("dated", true, "stamp the digest title with the current week")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	col, err := marketing.Gather(*root, marketingRange(*rangeFlag, *since))
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing generate: %v\n", err)
		return 1
	}
	when := time.Time{}
	if *dated {
		when = time.Now()
	}
	art := col.DigestFrom(when)
	art.Source = resolveMarketingSource(*source)
	fmt.Fprintln(stdout, art.Text())
	return 0
}

// runMarketingPost handles `fak marketing post`: generate a digest from the range, then post
// it to #marketing via the shared scoreboard transport (or print it on --dry-run). Mirrors
// `fak bench post` / `fak product post` exactly, reusing slackPostTail.
func runMarketingPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak marketing post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	since := fs.String("since", "", "gather ships since this ref (<ref>..HEAD)")
	rangeFlag := fs.String("range", "", "gather ships in this rev-range (e.g. abc123..HEAD); wins over --since")
	root := fs.String("root", ".", "repo root to read git/CLAIMS.md from")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_MARKETING_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_MARKETING_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	col, err := marketing.Gather(*root, marketingRange(*rangeFlag, *since))
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing post: %v\n", err)
		return 1
	}
	art := col.DigestFrom(time.Now())
	art.Source = resolveMarketingSource(*source)

	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           art,
		channel:        *channel,
		token:          *token,
		dryRun:         *dryRun,
		label:          "fak marketing post",
		chanEnv:        "FAK_MARKETING_CHANNEL",
		resolveChannel: marketing.ResolveChannel,
		resolveToken:   marketing.ResolveToken,
	})
}

// resolveMarketingSource picks the post source: the flag, else the shared defaultSource
// ($FAK_SCOREBOARD_SOURCE or hostname).
func resolveMarketingSource(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return defaultSource()
}
