package main

import (
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/marketing"
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
	dispatchSubcommands("marketing", "generate | post", argv,
		subcommand{"generate", runMarketingGenerate},
		subcommand{"post", runMarketingPost},
	)
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
