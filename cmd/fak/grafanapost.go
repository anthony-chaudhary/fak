package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/grafanapost"
)

// cmdGrafana posts fak #grafana-channel cards — exported Grafana snapshots and
// long-lived dashboard / debug links. It is reached as `fak grafana post` and is the
// outbound Grafana surface, the twin of `fak bench post` / `fak dojo post`.
//
//	fak grafana post --snapshot --title "p99 spike" --url <snapshot> \
//	    --dashboard "FAK Gateway Observability" --range "last 6h"   # an exported snapshot
//	fak grafana post --rollup all                                   # fold the link registry
//	fak grafana post --rollup public-demo                           # only long-lived demo links
//	fak grafana post --link fak-gateway-observability               # one dashboard link
//
// A snapshot post is a POST of a URL the operator already exported from Grafana; there
// is no inbound listener (pulling snapshots from a live Grafana is a follow-on). All
// modes default to the FAK_GRAFANA_* workspace (token falls back to the scoreboard
// token; channel falls back to the public built-in default) and honor --dry-run.
func cmdGrafana(argv []string) {
	dispatchSubcommands("grafana", "post", argv,
		subcommand{"post", runGrafanaPost},
	)
}

func runGrafanaPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak grafana post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	snapshot := fs.Bool("snapshot", false, "post an exported Grafana snapshot (needs --url; --title/--dashboard/--range/--expires annotate it)")
	rollup := fs.String("rollup", "", "fold the link registry: all | public-demo | debug | rollup")
	link := fs.String("link", "", "post a single registry link by its dashboard uid")
	// Snapshot fields.
	title := fs.String("title", "", "snapshot: headline")
	url := fs.String("url", "", "snapshot: the exported snapshot/share URL (required for a live --snapshot post)")
	dashboard := fs.String("dashboard", "", "snapshot: source dashboard name, e.g. \"FAK Gateway Observability\"")
	timeRange := fs.String("range", "", "snapshot: captured time window, e.g. \"last 6h\"")
	expires := fs.String("expires", "", "snapshot: when the snapshot link expires (e.g. \"in 7d\")")
	note := fs.String("note", "", "snapshot: optional note under the headline")
	// Shared.
	registry := fs.String("registry", "docs/grafana/links.json", "link registry path (rollup / link modes)")
	baseURL := fs.String("base-url", "", "override the Grafana base URL used to resolve uid-only links (default: the registry base_url)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_GRAFANA_CHANNEL / .env.slack.local / built-in default)")
	token := fs.String("token", "", "override bot token (default: $FAK_GRAFANA_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	// Exactly one mode: snapshot, rollup, or link.
	modes := 0
	if *snapshot {
		modes++
	}
	if *rollup != "" {
		modes++
	}
	if *link != "" {
		modes++
	}
	if modes != 1 {
		fmt.Fprintln(stderr, "fak grafana post: pass exactly one of --snapshot, --rollup <category>, or --link <uid>")
		return 2
	}

	var post grafanapost.Post
	switch {
	case *snapshot:
		if *url == "" && !*dryRun {
			fmt.Fprintln(stderr, "fak grafana post: --snapshot needs --url (the exported Grafana snapshot link)")
			return 2
		}
		post = grafanapost.SnapshotPost(grafanapost.Snapshot{
			Title:     *title,
			URL:       *url,
			Dashboard: *dashboard,
			TimeRange: *timeRange,
			Expires:   *expires,
			Note:      *note,
		})
	case *rollup != "":
		reg, err := loadGrafanaRegistry(*registry, *baseURL, stderr)
		if err != nil {
			return 2
		}
		post = grafanapost.LinksRollup(reg, *rollup)
	case *link != "":
		reg, err := loadGrafanaRegistry(*registry, *baseURL, stderr)
		if err != nil {
			return 2
		}
		l, ok := reg.Find(*link)
		if !ok {
			fmt.Fprintf(stderr, "fak grafana post: no registry link with uid %q (see %s)\n", *link, *registry)
			return 2
		}
		post = grafanapost.DashboardPost(l, reg.Base())
	}

	src := *source
	if src == "" {
		src = defaultSource()
	}
	post.Source = src

	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           post,
		channel:        *channel,
		token:          *token,
		dryRun:         *dryRun,
		label:          "fak grafana post",
		chanEnv:        "FAK_GRAFANA_CHANNEL",
		resolveChannel: grafanapost.ResolveChannel,
		resolveToken:   grafanapost.ResolveToken,
	})
}

// loadGrafanaRegistry loads the link registry and applies an optional --base-url
// override (so a uid-only link can resolve against a tailnet/public Grafana without
// editing the committed registry).
func loadGrafanaRegistry(path, baseURL string, stderr io.Writer) (*grafanapost.Registry, error) {
	reg, err := grafanapost.LoadRegistry(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak grafana post: load registry: %v\n", err)
		return nil, err
	}
	if baseURL != "" {
		reg.BaseURL = baseURL
	}
	return reg, nil
}
