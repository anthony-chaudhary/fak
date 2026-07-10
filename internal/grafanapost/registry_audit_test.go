package grafanapost

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// registry_audit_test.go binds the COMMITTED #grafana surface — the real
// docs/grafana/links.json the scheduled `fak grafana post --rollup` feeder folds — to
// the real dashboards it points at. Every other test in this package renders a
// synthetic registry, so a uid rename or a stray `git add` to either links.json or a
// dashboard JSON would silently ship a dead base/d/<uid> link to the channel: exactly
// the fabricated-link failure the package's ResolveURL / snapshot paths are architected
// to refuse, but at the registry layer nothing guarded it until here.

// repoRoot is the repository root relative to this package dir (internal/grafanapost).
const repoRoot = "../.."

// committedRegistryPath is the real link registry the #grafana rollup posts.
var committedRegistryPath = filepath.Join(repoRoot, "docs", "grafana", "links.json")

// TestCommittedRegistryIsPostable audits the committed registry against reality: valid
// schema, every link resolves to a real URL (never a "(no url)" card), a recognized
// category (a typo would misgroup the card and break the --rollup <cat> filter), no
// duplicate uid, and — for each uid-based link — a source dashboard whose top-level uid
// matches, so base/d/<uid> is a live route rather than a 404 in Slack.
func TestCommittedRegistryIsPostable(t *testing.T) {
	reg, err := LoadRegistry(committedRegistryPath)
	if err != nil {
		t.Fatalf("load committed registry %s: %v", committedRegistryPath, err)
	}
	if reg.Schema != "fak-grafana-links/1" {
		t.Errorf("registry schema = %q, want fak-grafana-links/1", reg.Schema)
	}
	if len(reg.Links) == 0 {
		t.Fatal("committed registry has no links")
	}

	base := reg.Base()
	knownCategory := map[string]bool{
		CategoryPublicDemo: true,
		CategoryDebug:      true,
		CategoryRollup:     true,
	}
	seenUID := map[string]bool{}

	for _, l := range reg.Links {
		if strings.TrimSpace(l.Title) == "" {
			t.Errorf("link %+v has no title", l)
		}
		if !knownCategory[l.Category] {
			t.Errorf("link %q has unknown category %q (want public-demo|debug|rollup — a typo silently misgroups the card and breaks the --rollup <cat> filter)", l.Title, l.Category)
		}
		// Load-bearing: every committed link must resolve to a URL, so the rollup never
		// folds a "(no url)" placeholder line into a #grafana card.
		if got := l.ResolveURL(base); got == "" {
			t.Errorf("link %q resolves to no URL (uid=%q url=%q base=%q)", l.Title, l.UID, l.URL, base)
		}
		if uid := strings.ToLower(strings.TrimSpace(l.UID)); uid != "" {
			if seenUID[uid] {
				t.Errorf("duplicate uid %q in registry", l.UID)
			}
			seenUID[uid] = true
			assertSourceUID(t, l)
		}
	}
}

// assertSourceUID checks a uid-based link's source dashboard JSON exists and declares
// the same top-level uid the registry claims — the drift that would make base/d/<uid>
// a dead link in the posted card. A uid link with no source can't be cross-checked, so
// the missing source is itself flagged rather than passed over.
func assertSourceUID(t *testing.T, l Link) {
	t.Helper()
	src := strings.TrimSpace(l.Source)
	if src == "" {
		t.Errorf("link %q (uid %q) has no source dashboard, so its uid cannot be verified against a committed dashboard", l.Title, l.UID)
		return
	}
	path := filepath.Join(repoRoot, filepath.FromSlash(src))
	b, err := os.ReadFile(path)
	if err != nil {
		t.Errorf("link %q source %s: %v", l.Title, src, err)
		return
	}
	var dash struct {
		UID string `json:"uid"`
	}
	if err := json.Unmarshal(b, &dash); err != nil {
		t.Errorf("link %q source %s: parse: %v", l.Title, src, err)
		return
	}
	if !strings.EqualFold(strings.TrimSpace(dash.UID), strings.TrimSpace(l.UID)) {
		t.Errorf("link %q: registry uid %q != source %s top-level uid %q — base/d/<uid> would 404 in the posted card", l.Title, l.UID, src, dash.UID)
	}
}

// TestCacheValueRollupItemResolvesToGoalURL pins the specific #grafana item this audit
// targeted: the FAK Cache Value roll-up dashboard must resolve to exactly the URL a
// human clicks in the channel card, and the folded card must actually carry it.
func TestCacheValueRollupItemResolvesToGoalURL(t *testing.T) {
	reg, err := LoadRegistry(committedRegistryPath)
	if err != nil {
		t.Fatalf("load committed registry: %v", err)
	}
	l, ok := reg.Find("fak-cache-value-rollup")
	if !ok {
		t.Fatal("registry has no fak-cache-value-rollup link")
	}
	if l.Category != CategoryRollup {
		t.Errorf("cache-value-rollup category = %q, want %q", l.Category, CategoryRollup)
	}
	const wantURL = "http://localhost:3000/d/fak-cache-value-rollup"
	if got := l.ResolveURL(reg.Base()); got != wantURL {
		t.Errorf("cache-value-rollup URL = %q, want %q", got, wantURL)
	}
	// The card that folds this link into #grafana must carry the resolved URL itself,
	// not a "(no url)" placeholder.
	if card := DashboardPost(l, reg.Base()).Text(); !strings.Contains(card, wantURL) {
		t.Errorf("dashboard card for cache-value-rollup missing the resolved URL:\n%s", card)
	}
}
