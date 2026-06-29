package grafanapost

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// clearEnv blanks every Slack key the resolver consults and moves to a clean cwd with
// no .env.slack.local, so a test runs deterministically regardless of the dev box.
func clearEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"FAK_GRAFANA_TOKEN", "FAK_GRAFANA_CHANNEL",
		"FAK_SCOREBOARD_TOKEN", "FAK_SCOREBOARD_CHANNEL",
	} {
		t.Setenv(k, "")
	}
	t.Chdir(t.TempDir())
}

func TestResolveChannelDefaultsToPublicGrafanaChannel(t *testing.T) {
	clearEnv(t)
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel default = %q, want the public grafana channel %q", got, ChannelDefault)
	}

	// An explicit env wins over the built-in default.
	t.Setenv("FAK_GRAFANA_CHANNEL", "C0OVERRIDE")
	if got := ResolveChannel(); got != "C0OVERRIDE" {
		t.Fatalf("ResolveChannel env override = %q, want C0OVERRIDE", got)
	}
}

func TestResolveChannelDoesNotInheritScoreboardChannel(t *testing.T) {
	clearEnv(t)
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE")
	// The grafana surface owns its default; it must NOT pick up the scoreboard channel.
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel leaked the scoreboard channel: got %q, want %q", got, ChannelDefault)
	}
}

func TestResolveTokenFallsBackToScoreboard(t *testing.T) {
	clearEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-sb")
	if got := ResolveToken(); got != "bottok-sb" {
		t.Fatalf("ResolveToken should fall back to the scoreboard token, got %q", got)
	}

	// Its own token wins over the fallback.
	t.Setenv("FAK_GRAFANA_TOKEN", "bottok-grafana-own")
	if got := ResolveToken(); got != "bottok-grafana-own" {
		t.Fatalf("ResolveToken should prefer the grafana token, got %q", got)
	}
}

func TestLinkResolveURL(t *testing.T) {
	// An absolute url wins outright.
	abs := Link{URL: "https://grafana.example/d/abc", UID: "abc"}
	if got := abs.ResolveURL("http://localhost:3000"); got != "https://grafana.example/d/abc" {
		t.Fatalf("absolute url = %q", got)
	}
	// uid + base builds the /d/<uid> route; a trailing slash on base is trimmed.
	uid := Link{UID: "fleet-bottleneck"}
	if got := uid.ResolveURL("http://localhost:3000/"); got != "http://localhost:3000/d/fleet-bottleneck" {
		t.Fatalf("uid route = %q", got)
	}
	// Neither url nor (uid AND base) => empty, never fabricated.
	if got := (Link{UID: "x"}).ResolveURL(""); got != "" {
		t.Fatalf("no base should yield empty url, got %q", got)
	}
	if got := (Link{}).ResolveURL("http://localhost:3000"); got != "" {
		t.Fatalf("no uid/url should yield empty url, got %q", got)
	}
}

func TestLoadRegistry(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "links.json")
	body := `{
	  "schema": "fak-grafana-links/1",
	  "base_url": "http://localhost:3000",
	  "links": [
	    {"title": "Fleet Bottleneck", "uid": "fleet-bottleneck", "category": "debug", "lifetime": "stack-local"}
	  ]
	}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if r.Base() != "http://localhost:3000" || len(r.Links) != 1 {
		t.Fatalf("registry = %+v", r)
	}
	if l, ok := r.Find("FLEET-BOTTLENECK"); !ok || l.Title != "Fleet Bottleneck" {
		t.Fatalf("Find(case-insensitive) = %+v ok=%v", l, ok)
	}
	if _, ok := r.Find("nope"); ok {
		t.Fatalf("Find(missing) should be false")
	}
}

func TestRegistryBaseDefault(t *testing.T) {
	var nilReg *Registry
	if got := nilReg.Base(); got != DefaultBaseURL {
		t.Fatalf("nil registry Base() = %q, want %q", got, DefaultBaseURL)
	}
	if got := (&Registry{}).Base(); got != DefaultBaseURL {
		t.Fatalf("empty base_url Base() = %q, want %q", got, DefaultBaseURL)
	}
}

func TestSnapshotPostText(t *testing.T) {
	p := SnapshotPost(Snapshot{
		Title:     "p99 spike",
		URL:       "https://grafana.example/dashboard/snapshot/xyz",
		Dashboard: "FAK Gateway Observability",
		TimeRange: "last 6h",
		Expires:   "in 7d",
		Note:      "spiked during the deploy",
	})
	txt := p.Text()
	for _, want := range []string{
		"grafana snapshot — p99 spike",
		"FAK Gateway Observability",
		"last 6h",
		"expires in 7d",
		"spiked during the deploy",
		"https://grafana.example/dashboard/snapshot/xyz",
	} {
		if !strings.Contains(txt, want) {
			t.Fatalf("snapshot text missing %q:\n%s", want, txt)
		}
	}
}

func TestSnapshotPostNoURLIsHonest(t *testing.T) {
	p := SnapshotPost(Snapshot{Title: "no link yet"})
	txt := p.Text()
	if strings.Contains(txt, "http") {
		t.Fatalf("a snapshot with no URL must not imply a link: %s", txt)
	}
	if !strings.Contains(txt, "no snapshot URL") {
		t.Fatalf("missing the honest no-URL flag: %s", txt)
	}
}

func TestLinksRollupGroupsAndFilters(t *testing.T) {
	r := &Registry{
		BaseURL: "http://localhost:3000",
		Links: []Link{
			{Title: "Gateway Obs", UID: "fak-gateway-observability", Category: "debug", Lifetime: "stack-local"},
			{Title: "Public Demo", URL: "https://demo.example/d/pub", Category: "public-demo", Lifetime: "long-lived"},
			{Title: "Fleet Bottleneck", UID: "fleet-bottleneck", Category: "debug", Lifetime: "stack-local"},
		},
	}

	all := LinksRollup(r, "all").Text()
	if !strings.Contains(all, "3 link(s)") {
		t.Fatalf("rollup all should count 3 links:\n%s", all)
	}
	if !strings.Contains(all, "http://localhost:3000/d/fak-gateway-observability") {
		t.Fatalf("rollup should resolve uid links against the base:\n%s", all)
	}
	if !strings.Contains(all, "https://demo.example/d/pub") {
		t.Fatalf("rollup should carry absolute urls:\n%s", all)
	}
	// public-demo group must precede debug (rollupOrder).
	if strings.Index(all, "public-demo") > strings.Index(all, "*debug*") {
		t.Fatalf("public-demo should sort before debug:\n%s", all)
	}

	debug := LinksRollup(r, "debug").Text()
	if strings.Contains(debug, "Public Demo") {
		t.Fatalf("filtered rollup should exclude public-demo:\n%s", debug)
	}
	if !strings.Contains(debug, "2 link(s)") {
		t.Fatalf("debug filter should count 2:\n%s", debug)
	}
}

func TestLinksRollupEmptyIsHonest(t *testing.T) {
	p := LinksRollup(&Registry{Links: nil}, "all")
	txt := p.Text()
	if !strings.Contains(txt, "none registered") || !strings.Contains(txt, "docs/grafana/links.json") {
		t.Fatalf("empty rollup should be an honest empty-state:\n%s", txt)
	}

	// A filter that matches nothing is also an honest empty-state, not a blank card.
	r := &Registry{Links: []Link{{Title: "x", Category: "debug"}}}
	got := LinksRollup(r, "public-demo").Text()
	if !strings.Contains(got, "none registered") || !strings.Contains(got, "public-demo") {
		t.Fatalf("no-match filter should name the scope:\n%s", got)
	}
}

func TestDashboardPostResolvesURL(t *testing.T) {
	p := DashboardPost(Link{Title: "Gateway Obs", UID: "fak-gateway-observability", Lifetime: "stack-local", Description: "live gateway p50/p99"}, "http://localhost:3000")
	txt := p.Text()
	for _, want := range []string{
		"grafana dashboard — Gateway Obs",
		"stack-local",
		"live gateway p50/p99",
		"http://localhost:3000/d/fak-gateway-observability",
	} {
		if !strings.Contains(txt, want) {
			t.Fatalf("dashboard text missing %q:\n%s", want, txt)
		}
	}
}
