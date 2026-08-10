package marketing

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// recentchanges_test.go — the guards behind docs/whats-new.md (#6040). The page's whole value
// is that it cannot overclaim and cannot rot silently, so these tests bind exactly that: scope
// grading (shipped vs plan vs landed-but-not-claimed), counts that reconcile with git, the
// round-trippable freshness anchor, byte-stable rendering, and the AEO front-matter budget.

func testShip(sha, subject, leaf string, day int, paths ...string) Ship {
	return Ship{
		SHA:     sha,
		Subject: subject,
		Leaf:    leaf,
		Date:    time.Date(2026, 8, day, 12, 0, 0, 0, time.UTC),
		Paths:   paths,
	}
}

func testOptions() RecentChangesOptions {
	anchor := time.Date(2026, 8, 9, 12, 0, 0, 0, time.UTC)
	return RecentChangesOptions{
		AnchorSHA:       "0123456789abcdef0123456789abcdef01234567",
		AnchorDate:      anchor,
		RangeSpec:       "aaaaaaa..0123456789abcdef0123456789abcdef01234567",
		RangeLabel:      RecentRangeLabel(7, "aaaaaaa..0123456789abcdef0123456789abcdef01234567", anchor),
		Version:         "v0.1.0",
		GeneratorModule: "internal/marketing@r22+g3b57441796",
		PerTheme:        2,
		Days:            7,
	}
}

// A leaf may live in exactly one theme: two homes would double-count a change and make the
// page's totals disagree with git.
func TestRecentThemesAssignEachLeafOnce(t *testing.T) {
	seen := map[string]string{}
	for _, th := range recentThemes {
		for _, leaf := range th.Leaves {
			if prev, dup := seen[leaf]; dup {
				t.Errorf("leaf %q claimed by both %q and %q", leaf, prev, th.Title)
			}
			seen[leaf] = th.Title
		}
	}
	if len(seen) == 0 {
		t.Fatal("no leaves mapped: the theme table is empty")
	}
}

// Scope grading is the honesty rung: an honesty-gate exclusion is downgraded (never dropped),
// a research-only tree is a plan even under a feat() subject, and code paths are shipped.
func TestBuildRecentChangesGradesScope(t *testing.T) {
	col := Collected{
		Ships: []Ship{
			testShip("aaa1111111111111111111111111111111111111", "feat(gateway): admit same-tick ready (#10) (fak gateway)", "gateway", 8, "internal/gateway/admit.go"),
			testShip("bbb2222222222222222222222222222222222222", "feat(docs): plan the next fold (fak docs)", "docs", 7, "docs/research/plan.md", "docs/notes/2026-08-07.md"),
		},
		Excluded: []ExcludedShip{{
			Ship:   testShip("ccc3333333333333333333333333333333333333", "feat(model): add speculative decode (fak model)", "model", 6, "internal/model/spec.go"),
			Reason: "CLAIMS.md: [SIMULATED]",
		}},
		Activity: Activity{Commits: 9, Ships: 3},
	}
	page := BuildRecentChanges(col, testOptions())

	if page.Ships != 3 {
		t.Errorf("Ships = %d, want 3 (excluded ships are counted, not dropped)", page.Ships)
	}
	if page.Commits != 9 {
		t.Errorf("Commits = %d, want 9", page.Commits)
	}
	got := map[string]string{}
	total := 0
	for _, g := range page.Groups {
		total += g.Total
		for _, it := range g.Items {
			got[it.SHA] = it.Scope
		}
	}
	if total != 3 {
		t.Errorf("group totals sum to %d, want 3 — the page must reconcile with git", total)
	}
	want := map[string]string{
		"aaa1111111111111111111111111111111111111": RecentScopeShipped,
		"bbb2222222222222222222222222222222222222": RecentScopeResearch,
		"ccc3333333333333333333333333333333333333": RecentScopeUnclaimed,
	}
	for sha, wantScope := range want {
		if got[sha] != wantScope {
			t.Errorf("scope(%s) = %q, want %q", sha[:7], got[sha], wantScope)
		}
	}
}

// An unmapped leaf lands in the catch-all rather than vanishing.
func TestBuildRecentChangesKeepsUnmappedLeaves(t *testing.T) {
	col := Collected{
		Ships:    []Ship{testShip("ddd4444444444444444444444444444444444444", "fix(zzunmapped): keep it (fak zzunmapped)", "zzunmapped", 8, "internal/zzunmapped/a.go")},
		Activity: Activity{Commits: 1},
	}
	page := BuildRecentChanges(col, testOptions())
	if len(page.Groups) != 1 || page.Groups[0].Title != recentCatchAll.Title {
		t.Fatalf("unmapped leaf did not land in the catch-all: %+v", page.Groups)
	}
}

// The subject a reader sees keeps its conventional-commit prefix but sheds the machine
// trailers rendered as their own fields.
func TestCleanRecentSubject(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"fix(gateway): treat same-tick ready as positive (#6064) (fak gateway)", "fix(gateway): treat same-tick ready as positive"},
		{"feat(headroom): add compare (fak headroom)", "feat(headroom): add compare"},
		{"feat(loopsteer): fix benign repeat loop scores for #5907 (fak sessionaudit)", "feat(loopsteer): fix benign repeat loop scores"},
		{"docs: route the front door", "docs: route the front door"},
	} {
		if got := cleanRecentSubject(tc.in); got != tc.want {
			t.Errorf("cleanRecentSubject(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestRecentSelectionShowsPlansAndHonestyHolds(t *testing.T) {
	day := time.Date(2026, 8, 9, 0, 0, 0, 0, time.UTC)
	items := []RecentItem{
		{SHA: "ship-a", Date: day, Type: "feat", Scope: RecentScopeShipped},
		{SHA: "ship-b", Date: day.Add(-time.Hour), Type: "feat", Scope: RecentScopeShipped},
		{SHA: "ship-c", Date: day.Add(-2 * time.Hour), Type: "fix", Scope: RecentScopeShipped},
		{SHA: "hold", Date: day.Add(-3 * time.Hour), Type: "feat", Scope: RecentScopeUnclaimed},
		{SHA: "plan", Date: day.Add(-4 * time.Hour), Type: "docs", Scope: RecentScopeResearch},
	}
	ranked := append([]RecentItem(nil), items...)
	// buildRecentGroup performs the production ranking before applying the bounded slice.
	g := buildRecentGroup(recentThemes[0], ranked, 3)
	if len(g.Items) != 3 {
		t.Fatalf("shown = %d, want 3", len(g.Items))
	}
	if !hasRecentScope(g.Items, RecentScopeResearch) || !hasRecentScope(g.Items, RecentScopeUnclaimed) {
		t.Fatalf("bounded selection hid a plan or honesty hold: %+v", g.Items)
	}
}

// A ship with no recorded paths is NOT graded as research: the stamp is the stronger witness.
func TestResearchOnlyRequiresPaths(t *testing.T) {
	if researchOnly(nil) {
		t.Error("empty path set graded as research/plan")
	}
	if !researchOnly([]string{"docs/notes/a.md", "docs\\research\\b.md"}) {
		t.Error("research-only tree not graded as research/plan")
	}
	if researchOnly([]string{"docs/notes/a.md", "internal/gateway/x.go"}) {
		t.Error("mixed tree graded as research/plan")
	}
}

// The anchor is the machine-checkable freshness boundary: it must round-trip every input the
// fold takes, or a regeneration cannot be compared byte-for-byte against the committed page.
func TestRecentChangesAnchorRoundTrips(t *testing.T) {
	page := BuildRecentChanges(Collected{Activity: Activity{Commits: 4}}, testOptions())
	md := page.Markdown()
	a, ok := ParseRecentChangesAnchor(md)
	if !ok {
		t.Fatalf("no anchor found in rendered page:\n%s", md)
	}
	if a.SHA != page.AnchorSHA || a.RangeSpec != page.RangeSpec || a.Version != page.Version || a.GeneratorModule != page.GeneratorModule {
		t.Errorf("anchor round-trip lost fields: %+v", a)
	}
	if a.PerTheme != page.PerTheme || a.Days != page.Days || a.Commits != page.Commits {
		t.Errorf("anchor round-trip lost counts: %+v", a)
	}
	if !a.Date.Equal(page.AnchorDate.Truncate(24 * time.Hour)) {
		t.Errorf("anchor date = %s, want %s", a.Date, page.AnchorDate)
	}
	opt := a.Options()
	if opt.RangeLabel != page.RangeLabel {
		t.Errorf("replayed label = %q, want %q", opt.RangeLabel, page.RangeLabel)
	}
	if BuildRecentChanges(Collected{Activity: Activity{Commits: 4}}, opt).Markdown() != md {
		t.Error("replaying the anchor's options did not reproduce the page")
	}
}

// A page with no anchor is a defect the checker must see, not a page it silently trusts.
func TestParseRecentChangesAnchorMissing(t *testing.T) {
	if _, ok := ParseRecentChangesAnchor("# What's new\n\nhand written\n"); ok {
		t.Error("hand-written page reported as anchored")
	}
}

func TestRecentModuleVersionFromLog(t *testing.T) {
	got, err := recentModuleVersionFromLog("internal/marketing", "anchor123", "abc1234567\ndef7654321\n")
	if err != nil {
		t.Fatal(err)
	}
	if want := "internal/marketing@r2+gabc1234567"; got != want {
		t.Fatalf("module version = %q, want %q", got, want)
	}
	if _, err := recentModuleVersionFromLog("internal/marketing", "anchor123", "\n"); err == nil {
		t.Fatal("empty module history was accepted")
	}
}

// Rendering is a pure function of the fold: two runs over the same range are byte-identical,
// which is what lets --check treat any difference as drift.
func TestRecentChangesMarkdownDeterministic(t *testing.T) {
	col := Collected{
		Ships: []Ship{
			testShip("aaa1111111111111111111111111111111111111", "feat(gateway): a (fak gateway)", "gateway", 8, "internal/gateway/a.go"),
			testShip("bbb2222222222222222222222222222222222222", "fix(model): b (fak model)", "model", 7, "internal/model/b.go"),
		},
		Activity: Activity{Commits: 5},
	}
	a := BuildRecentChanges(col, testOptions()).Markdown()
	b := BuildRecentChanges(col, testOptions()).Markdown()
	if a != b {
		t.Fatal("two folds of the same range rendered different bytes")
	}
	if strings.Contains(a, time.Now().Format("15:04")) {
		t.Error("page carries a wall-clock time; it must be stamped with the commit date only")
	}
}

// Truncation must be disclosed: a theme that shows fewer items than it counted says so and
// hands the reader the command that lists the rest.
func TestRecentChangesDisclosesTruncation(t *testing.T) {
	var ships []Ship
	for i := 0; i < 5; i++ {
		ships = append(ships, testShip(strings.Repeat(string(rune('a'+i)), 40), "feat(gateway): change (fak gateway)", "gateway", 8, "internal/gateway/a.go"))
	}
	page := BuildRecentChanges(Collected{Ships: ships, Activity: Activity{Commits: 5}}, testOptions())
	md := page.Markdown()
	if page.Groups[0].Total != 5 || len(page.Groups[0].Items) != 2 {
		t.Fatalf("counted %d / showed %d, want 5 / 2", page.Groups[0].Total, len(page.Groups[0].Items))
	}
	if !strings.Contains(md, "3 further change(s)") {
		t.Errorf("page hides its own truncation:\n%s", md)
	}
	if !strings.Contains(md, "git log --no-merges --oneline") {
		t.Error("page does not hand the reader the recipe for the unlisted changes")
	}
}

// The page is published, so it must satisfy the same AEO/SEO front-matter budget the
// scorecard enforces (title 15–70, description 70–160) and open with prose after the H1.
func TestRecentChangesFrontMatterFitsAEOBudget(t *testing.T) {
	md := BuildRecentChanges(Collected{Activity: Activity{Commits: 1}}, testOptions()).Markdown()
	if !strings.Contains(md, "> **In one breath:**") {
		t.Error("page does not open with the required one-breath explanation")
	}
	title := frontMatterField(md, "title")
	desc := frontMatterField(md, "description")
	if n := len(title); n < 15 || n > 70 {
		t.Errorf("title is %d chars (want 15..70): %q", n, title)
	}
	if n := len(desc); n < 70 || n > 160 {
		t.Errorf("description is %d chars (want 70..160): %q", n, desc)
	}
	lines := strings.Split(md, "\n")
	for i, l := range lines {
		if strings.HasPrefix(l, "# ") {
			next := ""
			for _, cand := range lines[i+1:] {
				if strings.TrimSpace(cand) != "" {
					next = strings.TrimSpace(cand)
					break
				}
			}
			if next == "" || strings.HasPrefix(next, "#") || strings.HasPrefix(next, "|") || strings.HasPrefix(next, "```") {
				t.Errorf("first line after the H1 is not prose: %q", next)
			}
			break
		}
	}
}

// The page routes onward instead of becoming another general entry point.
func TestRecentChangesRoutesToAuthorities(t *testing.T) {
	md := BuildRecentChanges(Collected{Activity: Activity{Commits: 1}}, testOptions()).Markdown()
	for _, want := range []string{"CLAIMS.md", "docs/releases/README.md", "research/README.md", "module@rev", RecentChangesVerb} {
		if !strings.Contains(md, want) {
			t.Errorf("page does not route to %q", want)
		}
	}
}

func TestPublishedRecentChangesRelativeLinksResolve(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	pagePath := filepath.Join(root, filepath.FromSlash(RecentChangesPath))
	b, err := os.ReadFile(pagePath)
	if err != nil {
		t.Fatalf("read generated page: %v", err)
	}
	if _, ok := ParseRecentChangesAnchor(string(b)); !ok {
		t.Fatal("generated page has no machine-readable freshness anchor")
	}
	linkRE := regexp.MustCompile(`\]\(([^)]+)\)`)
	for _, match := range linkRE.FindAllStringSubmatch(string(b), -1) {
		target := strings.TrimSpace(match[1])
		if target == "" || strings.HasPrefix(target, "#") || strings.Contains(target, "://") || strings.HasPrefix(target, "mailto:") {
			continue
		}
		target = strings.SplitN(target, "#", 2)[0]
		resolved := filepath.Clean(filepath.Join(filepath.Dir(pagePath), filepath.FromSlash(target)))
		if _, err := os.Stat(resolved); err != nil {
			t.Errorf("generated page link %q does not resolve to %s: %v", target, resolved, err)
		}
	}
}

func frontMatterField(md, key string) string {
	for _, l := range strings.Split(md, "\n") {
		if strings.HasPrefix(l, key+":") {
			v := strings.TrimSpace(strings.TrimPrefix(l, key+":"))
			return strings.Trim(v, `"`)
		}
	}
	return ""
}
