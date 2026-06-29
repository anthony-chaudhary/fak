package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
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
	dispatchSubcommands("marketing", "generate | post | tick | aeo | epic | release", argv,
		subcommand{"generate", runMarketingGenerate},
		subcommand{"post", runMarketingPost},
		subcommand{"tick", runMarketingTick},
		subcommand{"aeo", runMarketingAEO},
		subcommand{"epic", runMarketingEpic},
		subcommand{"release", runMarketingRelease},
	)
}

// runMarketingEpic handles `fak marketing epic`: announce the ships that closed an epic,
// grouped under its title. The caller supplies the title and the range it scoped (the
// gh-poll / issue-close integration lives in the caller, keeping the tier-1 core gh-free).
func runMarketingEpic(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak marketing epic", flag.ContinueOnError)
	fs.SetOutput(stderr)
	title := fs.String("title", "", "the epic's human title (e.g. the GitHub issue title)")
	since := fs.String("since", "", "gather the epic's ships since this ref (<ref>..HEAD)")
	rangeFlag := fs.String("range", "", "gather the epic's ships in this rev-range; wins over --since")
	root := fs.String("root", ".", "repo root")
	source := fs.String("source", "", "who is posting (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_MARKETING_CHANNEL)")
	token := fs.String("token", "", "override bot token (default: $FAK_MARKETING_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render and print; do not post")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	col, err := marketing.Gather(*root, marketingRange(*rangeFlag, *since))
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing epic: %v\n", err)
		return 1
	}
	art := col.EpicFrom(*title)
	art.Source = resolveMarketingSource(*source)
	return slackPostTail(stdout, stderr, marketingPostSpec(art, *channel, *token, *dryRun, "fak marketing epic"))
}

// runMarketingRelease handles `fak marketing release`: announce the ships in a release. The
// caller supplies the version tag and an optional one-line lead (pulled from docs/releases/
// v*.md), and the range the release spanned.
func runMarketingRelease(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak marketing release", flag.ContinueOnError)
	fs.SetOutput(stderr)
	version := fs.String("version", "", "the release tag (e.g. v0.18.0)")
	lead := fs.String("lead", "", "an optional one-line summary (e.g. from docs/releases/v*.md)")
	since := fs.String("since", "", "gather the release's ships since this ref (<ref>..HEAD)")
	rangeFlag := fs.String("range", "", "gather the release's ships in this rev-range; wins over --since")
	root := fs.String("root", ".", "repo root")
	source := fs.String("source", "", "who is posting (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_MARKETING_CHANNEL)")
	token := fs.String("token", "", "override bot token (default: $FAK_MARKETING_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render and print; do not post")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *version == "" {
		fmt.Fprintln(stderr, "fak marketing release: --version is required (e.g. --version v0.18.0)")
		return 2
	}
	col, err := marketing.Gather(*root, marketingRange(*rangeFlag, *since))
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing release: %v\n", err)
		return 1
	}
	art := col.ReleaseFrom(*version, *lead)
	art.Source = resolveMarketingSource(*source)
	return slackPostTail(stdout, stderr, marketingPostSpec(art, *channel, *token, *dryRun, "fak marketing release"))
}

// marketingPostSpec builds the shared slackPostSpec for the epic/release post paths (the
// generate/post paths build it inline; this folds the common channel/token/resolver wiring).
func marketingPostSpec(art marketing.Artifact, channel, token string, dryRun bool, label string) slackPostSpec {
	return slackPostSpec{
		card:           art,
		channel:        channel,
		token:          token,
		dryRun:         dryRun,
		label:          label,
		chanEnv:        "FAK_MARKETING_CHANNEL",
		resolveChannel: marketing.ResolveChannel,
		resolveToken:   marketing.ResolveToken,
	}
}

// runMarketingAEO handles `fak marketing aeo --refresh`: regenerate the AEO/AgentEO recency
// surface from the witnessed ships — docs/marketing/updates.json (a schema.org ItemList answer
// engines ingest) and llms-updates.txt (the plain feed agents poll). With --inject it then
// runs tools/gen_structured_data.py to fence a bounded "What's new" block into llms.txt + keep
// the FAQ/SoftwareApplication JSON-LD in sync (Go produces the data; Python owns the in-place
// doc injection, so hand-written prose is never clobbered). With --score it runs the SEO/AEO
// and agent-readiness scorecards as a freshness REPORT (not a hard gate in v1).
func runMarketingAEO(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak marketing aeo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	since := fs.String("since", "HEAD~50", "gather ships since this ref (<ref>..HEAD)")
	rangeFlag := fs.String("range", "", "gather ships in this rev-range; wins over --since")
	root := fs.String("root", ".", "repo root to read git/CLAIMS.md and write the feed under")
	refresh := fs.Bool("refresh", true, "write docs/marketing/updates.json + llms-updates.txt")
	inject := fs.Bool("inject", false, "after refresh, run tools/gen_structured_data.py to fence the What's-new block into llms.txt")
	score := fs.Bool("score", false, "after refresh, run the SEO/AEO + agent-readiness scorecards as a freshness report")
	dryRun := fs.Bool("dry-run", false, "print what would be written; do not touch any file")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	col, err := marketing.Gather(*root, marketingRange(*rangeFlag, *since))
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing aeo: %v\n", err)
		return 1
	}
	now := time.Now()
	feed, err := marketing.UpdatesFeed(col.Ships, now)
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing aeo: render feed: %v\n", err)
		return 1
	}
	updatesTxt := marketing.LlmsUpdatesText(col.Ships, now)

	feedPath := filepath.Join(*root, "docs", "marketing", "updates.json")
	txtPath := filepath.Join(*root, "llms-updates.txt")

	if *dryRun || !*refresh {
		fmt.Fprintf(stdout, "would write %s (%d witnessed ships) and %s\n", feedPath, len(col.Ships), txtPath)
		if *dryRun {
			fmt.Fprintln(stdout, string(feed))
			return 0
		}
	}
	if *refresh {
		if err := os.MkdirAll(filepath.Dir(feedPath), 0o755); err != nil {
			fmt.Fprintf(stderr, "fak marketing aeo: %v\n", err)
			return 1
		}
		if err := os.WriteFile(feedPath, append(feed, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak marketing aeo: write feed: %v\n", err)
			return 1
		}
		if err := os.WriteFile(txtPath, []byte(updatesTxt), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak marketing aeo: write llms-updates.txt: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "wrote %s (%d witnessed ships)\nwrote %s\n", feedPath, len(col.Ships), txtPath)
	}

	if *inject {
		if rc := runPyTool(stdout, stderr, *root, "tools/gen_structured_data.py"); rc != 0 {
			return rc
		}
	}
	if *score {
		// Freshness REPORT, not a gate (v1): print the scorecards' verdict lines so a human
		// sees whether the refresh moved AEO/agent debt. A non-zero scorecard exit is its
		// ACTION verdict, not a tool failure, so it never fails this command.
		_ = runPyTool(stdout, stderr, *root, "tools/seo_aeo_scorecard.py")
		_ = runPyTool(stdout, stderr, *root, "tools/agent_readiness_scorecard.py")
	}
	return 0
}

// runPyTool runs a project python tool (python then python3) from root, streaming its output.
// A non-zero exit is reported but not treated as fatal by the caller for scorecards (their
// exit code is a verdict); for the generator a non-zero exit IS a real failure.
func runPyTool(stdout, stderr io.Writer, root, tool string) int {
	for _, bin := range []string{"python", "python3"} {
		cmd := exec.CommandContext(ctx(), bin, tool)
		cmd.Dir = root
		cmd.Stdout, cmd.Stderr = stdout, stderr
		if err := cmd.Run(); err != nil {
			if _, ok := err.(*exec.ExitError); ok {
				return 1 // ran but exited non-zero (the caller decides if that's fatal)
			}
			continue // interpreter not found — try the next
		}
		return 0
	}
	fmt.Fprintf(stderr, "fak marketing aeo: could not run %s (no python/python3)\n", tool)
	return 1
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
