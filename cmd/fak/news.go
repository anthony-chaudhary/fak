package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// cmdNews posts externally sourced industry/SOTA/OSS research notes to the #news
// surface in the scoreboard workspace. It is editorial by design: the command routes a
// verified note to Slack, but it does not invent or scrape the news itself.
func cmdNews(argv []string) {
	dispatchSubcommands("news", "post", argv,
		subcommand{"post", runNewsPost},
	)
}

func runNewsPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak news post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	title := fs.String("title", "", "short headline for the digest")
	notes := fs.String("notes", "", "digest body (mutually exclusive with --notes-file)")
	notesFile := fs.String("notes-file", "", "read digest body from this file (- for stdin)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_NEWS_CHANNEL / .env.slack.local)")
	token := fs.String("token", "", "override bot token (default: $FAK_SCOREBOARD_TOKEN / .env.slack.local)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	up, err := buildNewsUpdate(*title, *notes, *notesFile, *source)
	if err != nil {
		fmt.Fprintf(stderr, "fak news post: %v\n", err)
		return 2
	}

	return scoreboardPostFlow(stdout, stderr, up, scoreboardPostOpts{
		prefix:        "fak news post",
		channelFlag:   *channel,
		tokenFlag:     *token,
		resolveChan:   resolveNewsChannel,
		resolveToken:  scoreboard.ResolveToken,
		noChannelHint: "fak news post: no channel: pass --channel, set FAK_NEWS_CHANNEL, or add it to .env.slack.local",
		dryRun:        *dryRun,
		dedupe:        false,
	})
}

func buildNewsUpdate(title, notes, notesFile, source string) (scoreboard.Update, error) {
	title = strings.TrimSpace(title)
	if title == "" {
		return scoreboard.Update{}, fmt.Errorf("--title is required")
	}
	if notes != "" && notesFile != "" {
		return scoreboard.Update{}, fmt.Errorf("pass --notes or --notes-file, not both")
	}
	if notes == "" && notesFile == "" {
		return scoreboard.Update{}, fmt.Errorf("pass --notes or --notes-file")
	}
	body := notes
	if notesFile != "" {
		raw, err := readFromFile(notesFile)
		if err != nil {
			return scoreboard.Update{}, err
		}
		body = string(raw)
	}
	if strings.TrimSpace(body) == "" {
		return scoreboard.Update{}, fmt.Errorf("news body is empty")
	}
	src := source
	if src == "" {
		src = defaultSource()
	}
	return scoreboard.Update{Title: "news - " + title, Notes: body, Source: src}, nil
}

func resolveNewsChannel() string {
	if r := slackenv.Lookup("FAK_NEWS_CHANNEL"); r.Set() {
		return r.Value
	}
	return ""
}
