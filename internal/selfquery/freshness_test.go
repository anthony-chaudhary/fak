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
	cat, err := Load(writeFreshnessRepo(t), Options{DevLoader: testDevLoader})
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
	cat, err := Load(root, Options{DevLoader: testDevLoader})
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

// TestCitedRepoPathAdmitsOnlyRepoFileRefs pins the precision-first admission for
// the STALE check: only a repo-relative FILE ref is a candidate. URLs, cap refs
// (no slash), bare names, and parent-escaping refs are never treated as files.
func TestCitedRepoPathAdmitsOnlyRepoFileRefs(t *testing.T) {
	cases := []struct {
		ref    string
		want   string
		wantOK bool
	}{
		{"docs/notes/2026-07-07-topic.md", "docs/notes/2026-07-07-topic.md", true},
		{"internal/selfquery/freshness.go#L10", "internal/selfquery/freshness.go", true}, // anchor stripped
		{"", "", false},
		{"https://example.com/x.md", "", false}, // URL
		{"http://example.com/x.md", "", false},  // URL
		{"doc:some-cap", "", false},             // cap ref, no slash
		{"driver:memq@v1", "", false},           // cap ref with version
		{"README.md", "", false},                // bare name, no separator
		{"../escape/x.md", "", false},           // parent-escaping
	}
	for _, tc := range cases {
		got, ok := citedRepoPath(tc.ref)
		if ok != tc.wantOK || got != tc.want {
			t.Errorf("citedRepoPath(%q) = (%q,%v), want (%q,%v)", tc.ref, got, ok, tc.want, tc.wantOK)
		}
	}
}

// TestStalenessByKeyMarksDeletedCitedFile pins the four staleness boundaries:
// a surviving file (fresh), a deleted file whose folder remains (STALE), refs
// that are not files (skipped), and the parent-gone precision fence (skipped).
func TestStalenessByKeyMarksDeletedCitedFile(t *testing.T) {
	root := t.TempDir()
	notes := filepath.Join(root, "docs", "notes")
	if err := os.MkdirAll(notes, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(notes, "present.md"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	cards := []FeatureCard{
		{Kind: "doc", Name: "present", DetailRef: "docs/notes/present.md"}, // exists -> fresh (no entry)
		{Kind: "doc", Name: "gone", DetailRef: "docs/notes/gone.md"},       // parent exists, file gone -> STALE
		{Kind: "doc", Name: "url", DetailRef: "https://example.com/x.md"},  // URL -> skip
		{Kind: "cap", Name: "cap", DetailRef: "doc:some-cap"},              // cap ref -> skip
		{Kind: "doc", Name: "nodir", DetailRef: "docs/never/here.md"},      // parent missing -> skip (precision)
	}
	got := stalenessByKey(cards, root)
	if len(got) != 1 || got[cardKey(cards[1])] != FreshnessStale {
		t.Fatalf("want exactly {gone:STALE}, got %v", got)
	}
	// Empty root disables the check entirely (nothing to resolve against).
	if n := len(stalenessByKey(cards, "")); n != 0 {
		t.Errorf("empty root should disable staleness, got %d entries", n)
	}
}

// TestQueryStampsStaleRungAfterDeletion is the end-to-end, read-time proof of the
// #3163 STALE follow-on and the user's "query, then re-check" axis: the index
// still lists a note, but the note's bytes are deleted AFTER Load. The very next
// Query() stamps it STALE (computed against the current tree, not calendar age),
// and STALE outranks the SUPERSEDED_BY it would otherwise carry.
func TestQueryStampsStaleRungAfterDeletion(t *testing.T) {
	root := writeFreshnessRepo(t)
	cat, err := Load(root, Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatal(err)
	}
	// Delete the older, superseded note's bytes; its INDEX.md card remains, and
	// its folder (docs/notes) survives because the newer note is still there.
	if err := os.Remove(filepath.Join(root, "docs", "notes", "2026-06-25-widget-rollout.md")); err != nil {
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
		t.Fatalf("deleted note dropped from results (staleness must annotate, not filter): %v", sortedNames(resp.Cards))
	}
	if older.Freshness != FreshnessStale {
		t.Errorf("deleted note Freshness = %q, want STALE (should outrank SUPERSEDED_BY)", older.Freshness)
	}
	if newer, ok := byName["doc:Widget rollout July"]; ok && newer.Freshness != FreshnessFresh {
		t.Errorf("surviving newer note Freshness = %q, want FRESH", newer.Freshness)
	}
}

// TestQueryDetailCardCarriesFreshnessRung pins the --detail half of #3163: a
// faulted detail card is the single card an agent reads in full, so it is the one
// place a missing rung does the most damage. Query() stamps it from the same
// precomputed rungs as the ranked list; without that, `--detail` would hand back
// an unhedged superseded card while the list above it was correctly marked.
func TestQueryDetailCardCarriesFreshnessRung(t *testing.T) {
	cat, err := Load(writeFreshnessRepo(t), Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{
		Query:  "widget rollout",
		Plane:  PlaneDev,
		Detail: "docs/notes/2026-06-25-widget-rollout.md",
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Detail == nil {
		t.Fatal("Detail requested but not returned")
	}
	if want := FreshnessSupersededPrefix + "doc:Widget rollout July"; resp.Detail.Card.Freshness != want {
		t.Errorf("detail card Freshness = %q, want %q", resp.Detail.Card.Freshness, want)
	}
}

// writeLoneNoteRepo lays down a repo whose index carries ONLY the older
// widget-rollout note. Its card has byte-identical Kind+Name+DetailRef — hence an
// identical cardKey — to the two-note repo's older card, but no sibling to be
// superseded by. Merging the two roots is therefore the exact key collision the
// root-qualified rung map must survive.
func writeLoneNoteRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"dos.toml": "[lanes.trees]\ndocs = [\"docs/**\"] # docs\n",
		"INDEX.md": `# INDEX
- [Widget rollout June](docs/notes/2026-06-25-widget-rollout.md) - widget rollout plan.
`,
		"docs/notes/2026-06-25-widget-rollout.md": "# Widget rollout June\n",
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

// TestMultiRootFreshnessStaysScopedPerRoot pins the cross-repo isolation invariant
// MultiCatalog.Query claims (#3163 on the #3435 fan-out): supersession is scoped
// WITHIN a root, so a note supersedes only its own checkout's siblings. Two roots
// contribute a card with the same cardKey; only the root that actually holds the
// newer sibling may carry SUPERSEDED_BY. Drop the root qualification from
// freshnessRungs/applyFreshnessMulti and both cards inherit the rung — a repo
// would be told its current note is superseded by a note it does not have.
func TestMultiRootFreshnessStaysScopedPerRoot(t *testing.T) {
	twoNote := writeFreshnessRepo(t) // June + July -> June is superseded
	loneNote := writeLoneNoteRepo(t) // June alone  -> nothing supersedes it
	mc, err := LoadMany([]string{twoNote, loneNote}, Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatalf("LoadMany: %v", err)
	}
	if len(mc.Skipped()) != 0 {
		t.Fatalf("unexpected skips: %v", mc.Skipped())
	}
	resp, err := mc.Query(Request{Query: "widget rollout", Plane: PlaneDev})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}

	// Both roots contribute a "doc:Widget rollout June" card. Compare by rung
	// rather than by root path so the assertion does not depend on how TempDir
	// paths are normalized on this host.
	var junes []FeatureCard
	var july FeatureCard
	for _, c := range resp.Cards {
		switch c.Name {
		case "doc:Widget rollout June":
			junes = append(junes, c)
		case "doc:Widget rollout July":
			july = c
		}
	}
	if len(junes) != 2 {
		t.Fatalf("want the June card from both roots, got %d: %v", len(junes), sortedNames(resp.Cards))
	}
	if junes[0].Root == "" || junes[1].Root == "" || junes[0].Root == junes[1].Root {
		t.Fatalf("June cards not attributed to two distinct roots: %q vs %q", junes[0].Root, junes[1].Root)
	}
	wantRung := FreshnessSupersededPrefix + "doc:Widget rollout July"
	marked, unmarked := 0, 0
	for _, c := range junes {
		switch c.Freshness {
		case wantRung:
			marked++
		case "":
			unmarked++
		default:
			t.Errorf("June card (root %q) Freshness = %q, want %q or empty", c.Root, c.Freshness, wantRung)
		}
	}
	if marked != 1 || unmarked != 1 {
		t.Errorf("supersession leaked across roots: %d marked / %d unmarked, want exactly 1 each", marked, unmarked)
	}
	// The newer note exists in one root only and must still be FRESH there.
	if july.Name == "" {
		t.Fatal("July card missing from the merged result")
	}
	if july.Freshness != FreshnessFresh {
		t.Errorf("July card Freshness = %q, want FRESH", july.Freshness)
	}
}
