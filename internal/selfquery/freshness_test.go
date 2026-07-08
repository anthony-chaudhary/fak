package selfquery

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNoteInfoAdmitsOnlyDatedNotes(t *testing.T) {
	cases := []struct {
		ref       string
		wantOK    bool
		wantDate  string
		wantTopic string
	}{
		// Date-prefix and date-suffix note styles both parse, and the same topic
		// under either style yields the same key (position-independent).
		{"docs/notes/2026-07-07-guard-oot-write-positive-containment.md", true, "2026-07-07", "guard-oot-write-positive-containment"},
		{"docs/notes/COMPRESS-DEFAULT-ON-WITNESS-2026-07-06.md", true, "2026-07-06", "compress-default-on-witness"},
		{"docs/notes/2026-06-25-widget-rollout.md", true, "2026-06-25", "widget-rollout"},
		{"docs/notes/widget-rollout-2026-07-06.md", true, "2026-07-06", "widget-rollout"},
		// Rejections: plain doc (not under docs/notes), undated note, non-md,
		// anchored ref still resolves its base.
		{"docs/gateway.md", false, "", ""},
		{"docs/notes/undated-design-note.md", false, "", ""},
		{"docs/notes/2026-07-07-topic.txt", false, "", ""},
		{"docs/notes/2026-07-07-anchored.md#section", true, "2026-07-07", "anchored"},
		{"", false, "", ""},
	}
	for _, tc := range cases {
		got, ok := noteInfo(tc.ref)
		if ok != tc.wantOK {
			t.Errorf("noteInfo(%q) ok=%v, want %v", tc.ref, ok, tc.wantOK)
			continue
		}
		if !ok {
			continue
		}
		if got.date != tc.wantDate || got.topic != tc.wantTopic {
			t.Errorf("noteInfo(%q) = {%q,%q}, want {%q,%q}", tc.ref, got.date, got.topic, tc.wantDate, tc.wantTopic)
		}
	}
}

// TestFreshnessByKeyLogic pins the three decision boundaries directly on
// hand-built cards: strict supersession, a lone note (no sibling), and an
// ambiguous same-date tie (no strict order → no rung).
func TestFreshnessByKeyLogic(t *testing.T) {
	doc := func(name, ref string) FeatureCard {
		return FeatureCard{Kind: "dev-doc", Name: name, DetailRef: ref, Source: "devindex"}
	}
	cards := []FeatureCard{
		// same topic "widget-rollout", cross-style, strictly ordered
		doc("doc:rollout-old", "docs/notes/2026-06-25-widget-rollout.md"),
		doc("doc:rollout-new", "docs/notes/widget-rollout-2026-07-06.md"),
		// lone topic — one dated note only
		doc("doc:sidecar", "docs/notes/widget-sidecar-2026-07-01.md"),
		// same topic "clash", same date — ambiguous, must stay unmarked
		doc("doc:clash-a", "docs/notes/2026-07-02-clash.md"),
		doc("doc:clash-b", "docs/notes/clash-2026-07-02.md"),
		// not a dated note
		doc("doc:plain", "docs/gateway.md"),
	}
	rungs := freshnessByKey(cards)

	want := map[string]string{
		cardKey(cards[0]): FreshnessSupersededPrefix + "doc:rollout-new",
		cardKey(cards[1]): FreshnessFresh,
	}
	for k, v := range want {
		if rungs[k] != v {
			t.Errorf("rung[%q] = %q, want %q", k, rungs[k], v)
		}
	}
	// Everything else must carry no rung: lone note, ambiguous tie pair, plain doc.
	for _, unmarked := range []FeatureCard{cards[2], cards[3], cards[4], cards[5]} {
		if r, ok := rungs[cardKey(unmarked)]; ok {
			t.Errorf("card %q got rung %q, want none", unmarked.Name, r)
		}
	}
}

func writeFreshnessRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"dos.toml": "[lanes.trees]\ndocs = [\"docs/**\"] # docs\n",
		"INDEX.md": `# INDEX
- [Widget rollout June](docs/notes/2026-06-25-widget-rollout.md) - widget rollout plan.
- [Widget rollout July](docs/notes/widget-rollout-2026-07-06.md) - widget rollout plan, revised.
- [Widget sidecar](docs/notes/widget-sidecar-2026-07-01.md) - lone widget sidecar note.
- [Gateway guide](docs/gateway.md) - widget gateway docs.
`,
		"docs/gateway.md":                         "# Gateway guide\nwidget gateway.\n",
		"docs/notes/2026-06-25-widget-rollout.md": "# Widget rollout June\n",
		"docs/notes/widget-rollout-2026-07-06.md": "# Widget rollout July\n",
		"docs/notes/widget-sidecar-2026-07-01.md": "# Widget sidecar\n",
	}
	for path, body := range files {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// TestQueryStampsSupersessionRung is the end-to-end proof of #3163's first
// checkable step: the query surface stamps SUPERSEDED_BY / FRESH derived from
// dated-note timestamps, without filtering or re-ordering.
func TestQueryStampsSupersessionRung(t *testing.T) {
	cat, err := Load(writeFreshnessRepo(t), Options{})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "widget rollout", Plane: PlaneDev})
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]FeatureCard{}
	for _, c := range resp.Cards {
		byName[c.Name] = c
	}

	older, ok := byName["doc:Widget rollout June"]
	if !ok {
		t.Fatalf("older widget-rollout note dropped from results (non-filtering violated): %v", sortedNames(resp.Cards))
	}
	newer, ok := byName["doc:Widget rollout July"]
	if !ok {
		t.Fatalf("newer widget-rollout note missing: %v", sortedNames(resp.Cards))
	}
	if want := FreshnessSupersededPrefix + "doc:Widget rollout July"; older.Freshness != want {
		t.Errorf("older note Freshness = %q, want %q", older.Freshness, want)
	}
	if newer.Freshness != FreshnessFresh {
		t.Errorf("newer note Freshness = %q, want FRESH", newer.Freshness)
	}
	// A lone note and a plain (non-notes) doc carry no rung even when they match.
	if lone, ok := byName["doc:Widget sidecar"]; ok && lone.Freshness != "" {
		t.Errorf("lone note Freshness = %q, want empty", lone.Freshness)
	}
	if plain, ok := byName["doc:Gateway guide"]; ok && plain.Freshness != "" {
		t.Errorf("plain doc Freshness = %q, want empty", plain.Freshness)
	}
}

// TestFreshnessDoesNotChangeRanking pins the advisory fence: annotating the
// result must not reorder it. The ranked Name order equals the order produced
// by the same rankCards call the query uses, before any freshness stamping.
func TestFreshnessDoesNotChangeRanking(t *testing.T) {
	root := writeFreshnessRepo(t)
	cat, err := Load(root, Options{})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "widget rollout", Plane: PlaneDev})
	if err != nil {
		t.Fatal(err)
	}
	// Recompute the ranking directly (no freshness applied) and compare order.
	want := sortedNames(rankCards(cat.Cards(PlaneDev), "widget rollout"))
	got := sortedNames(resp.Cards)
	if len(got) != len(want) {
		t.Fatalf("result length changed: got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ranking drifted at %d: got %v\nwant %v", i, got, want)
		}
	}
}
