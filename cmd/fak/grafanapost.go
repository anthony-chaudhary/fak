package main

import (
	"context"
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
//	fak grafana post --snapshot --create --grafana-url http://localhost:3000 \
//	    --uid fak-gateway-observability --expires-seconds 604800     # PULL a fresh snapshot via the API
//	fak grafana post --rollup all                                   # fold the link registry
//	fak grafana post --rollup public-demo                           # only long-lived demo links
//	fak grafana post --link fak-gateway-observability               # one dashboard link
//
// A plain snapshot post is a POST of a URL the operator already exported from Grafana.
// With --create it instead PULLS one: it calls the live Grafana snapshots API
// (--grafana-url + a FAK_GRAFANA_API_TOKEN) to create a snapshot for --uid, then posts
// the public URL Grafana returned — never a URL fak fabricated. All modes default to the
// FAK_GRAFANA_* workspace (Slack token falls back to the scoreboard token; channel falls
// back to the public built-in default) and honor --dry-run.
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
	// Inbound ingestion (--snapshot --create): pull a snapshot from a live Grafana via
	// its API instead of posting a hand-exported URL.
	create := fs.Bool("create", false, "snapshot: create the snapshot via the Grafana API for --uid (needs --grafana-url + FAK_GRAFANA_API_TOKEN); posts the URL Grafana returns")
	grafanaURL := fs.String("grafana-url", "", "snapshot --create: the live Grafana base URL, e.g. http://localhost:3000")
	uid := fs.String("uid", "", "snapshot --create: the dashboard uid to snapshot")
	expiresSeconds := fs.Int("expires-seconds", 0, "snapshot --create: snapshot TTL in seconds (0 = Grafana default, no expiry)")
	grafanaToken := fs.String("grafana-token", "", "snapshot --create: override the Grafana API token (default: $FAK_GRAFANA_API_TOKEN / .env.slack.local)")
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
		snap := grafanapost.Snapshot{
			Title:     *title,
			URL:       *url,
			Dashboard: *dashboard,
			TimeRange: *timeRange,
			Expires:   *expires,
			Note:      *note,
		}
		if *create {
			// Inbound: pull a fresh snapshot from the live Grafana. The URL we post is
			// whatever Grafana returned — never one fak built.
			res, err := createGrafanaSnapshot(*grafanaURL, *grafanaToken, *uid, *expiresSeconds, stderr)
			if err != nil {
				return 2
			}
			snap.URL = res.URL
			if snap.Title == "" {
				snap.Title = *uid
			}
		} else if *url == "" && !*dryRun {
			fmt.Fprintln(stderr, "fak grafana post: --snapshot needs --url (the exported Grafana snapshot link), or --create to pull one via the API")
			return 2
		}
		post = grafanapost.SnapshotPost(snap)
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

// createGrafanaSnapshot drives the inbound (--create) path: it resolves the Grafana API
// token (--grafana-token override, else FAK_GRAFANA_API_TOKEN / .env.slack.local), then
// calls the live Grafana snapshots API to create a snapshot for uid and returns the
// result Grafana minted. Validation errors and API failures are written to stderr and
// returned, so the caller posts nothing rather than an empty link.
func createGrafanaSnapshot(grafanaURL, tokenOverride, uid string, expiresSeconds int, stderr io.Writer) (grafanapost.SnapshotResult, error) {
	var zero grafanapost.SnapshotResult
	if grafanaURL == "" {
		fmt.Fprintln(stderr, "fak grafana post: --create needs --grafana-url (the live Grafana base URL)")
		return zero, fmt.Errorf("missing --grafana-url")
	}
	if uid == "" {
		fmt.Fprintln(stderr, "fak grafana post: --create needs --uid (the dashboard to snapshot)")
		return zero, fmt.Errorf("missing --uid")
	}
	tok := tokenOverride
	if tok == "" {
		tok = grafanapost.ResolveAPIToken()
	}
	if tok == "" {
		fmt.Fprintln(stderr, "fak grafana post: --create needs a Grafana API token (set FAK_GRAFANA_API_TOKEN or pass --grafana-token)")
		return zero, fmt.Errorf("missing Grafana API token")
	}
	client := grafanapost.NewClient(grafanaURL, tok)
	res, err := client.CreateSnapshotForDashboard(context.Background(), uid, expiresSeconds)
	if err != nil {
		fmt.Fprintf(stderr, "fak grafana post: %v\n", err)
		return zero, err
	}
	return res, nil
}
