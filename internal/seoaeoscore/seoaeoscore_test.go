package seoaeoscore

// Golden-parity port of the former tools/seo_aeo_scorecard_test.py. Every case here mirrors a Python
// test one-for-one: the pure per-KPI checks + front-matter parser + grader + site-level fold
// on fixture strings (mostly no disk), then a tolerant live smoke that Build folds the real
// published surfaces. The two Python cases that drive gen_structured_data (gsd) directly are
// adapted: the pure gsd idempotency test is a different module (no Go port) and is omitted;
// the breadcrumb-from-index site check is exercised with an inlined valid BreadcrumbList block.

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- helpers ---------------------------------------------------------------

func writeFile(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func anyContains(ss []string, sub string) bool {
	for _, s := range ss {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func checksByName(site Site) map[string]SiteCheck {
	m := map[string]SiteCheck{}
	for _, c := range site.Checks {
		m[c.Name] = c
	}
	return m
}

func repoRootDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		return ""
	}
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Clean(dir)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

// --- front-matter parser ---------------------------------------------------

func TestFMQuotedOneliner(t *testing.T) {
	fm := parseFrontMatter("---\ntitle: \"Hello World\"\n---\n# body\n")
	if fm["title"] != "Hello World" {
		t.Fatalf("got %q", fm["title"])
	}
}

func TestFMBareOneliner(t *testing.T) {
	fm := parseFrontMatter("---\ntitle: Hello World\n---\nbody")
	if fm["title"] != "Hello World" {
		t.Fatalf("got %q", fm["title"])
	}
}

func TestFMFoldedBlockScalar(t *testing.T) {
	text := "---\ntitle: T\ndescription: >-\n  one line\n  two line\nslug: x\n---\nbody"
	fm := parseFrontMatter(text)
	if fm["description"] != "one line two line" {
		t.Fatalf("desc=%q", fm["description"])
	}
	if fm["title"] != "T" {
		t.Fatalf("title=%q", fm["title"])
	}
}

func TestFMNoneWhenAbsent(t *testing.T) {
	if got := parseFrontMatter("# no front matter\n\ntext"); len(got) != 0 {
		t.Fatalf("expected empty, got %v", got)
	}
}

// --- title / description KPIs ----------------------------------------------

func TestTitleMissingIsHardDefect(t *testing.T) {
	k := kpiTitle(map[string]string{})
	if k.score != 0 || !anyContains(k.defects, "no front-matter title") {
		t.Fatalf("%+v", k)
	}
}

func TestTitlePresentInBandIs100(t *testing.T) {
	k := kpiTitle(map[string]string{"title": "A clear page title for fak"})
	if k.score != 100 || len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

func TestTitleTooLongIsSoftNotHard(t *testing.T) {
	longTitle := "fak the agent kernel: a default-deny permission gate fused with an " +
		"addressable bit-exact KV cache for self-hosted AI agent fleets"
	if len([]rune(longTitle)) <= titleMax {
		t.Fatalf("fixture not longer than TITLE_MAX")
	}
	k := kpiTitle(map[string]string{"title": longTitle})
	if len(k.defects) != 0 || k.score >= 100 || len(k.soft) == 0 {
		t.Fatalf("%+v", k)
	}
}

func TestDescriptionMissingIsHardDefect(t *testing.T) {
	k := kpiDescription(map[string]string{})
	if k.score != 0 || !anyContains(k.defects, "no front-matter description") {
		t.Fatalf("%+v", k)
	}
}

func TestDescriptionInBandIs100(t *testing.T) {
	desc := "fak is an agent kernel that gates every tool call a model makes and " +
		"reuses its KV cache across turns for cheaper self-hosted agent fleets."
	if n := len(desc); n < 70 || n > 160 {
		t.Fatalf("fixture out of band: %d", n)
	}
	k := kpiDescription(map[string]string{"description": desc})
	if k.score != 100 || len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

func TestDescriptionThinIsSoft(t *testing.T) {
	k := kpiDescription(map[string]string{"description": "too short"})
	if len(k.defects) != 0 || len(k.soft) == 0 {
		t.Fatalf("%+v", k)
	}
}

// --- headings --------------------------------------------------------------

func TestHeadingsMissingH1IsHard(t *testing.T) {
	k := kpiHeadings("no title\n\n## section\n")
	if !anyContains(k.defects, "no H1") {
		t.Fatalf("%+v", k)
	}
}

func TestHeadingsCleanIsHigh(t *testing.T) {
	k := kpiHeadings("# Title\n\nintro.\n\n## A\n\ntext\n\n## B\n\ntext\n")
	if len(k.defects) != 0 || k.score < 90 {
		t.Fatalf("%+v", k)
	}
}

func TestHeadingsSkipLevelIsSoft(t *testing.T) {
	k := kpiHeadings("# Title\n\n### deep\n\ntext\n")
	if len(k.defects) != 0 || !anyContains(k.soft, "skips") {
		t.Fatalf("%+v", k)
	}
}

func TestHeadingsIgnoresFrontMatter(t *testing.T) {
	k := kpiHeadings("---\ntitle: \"T\"\n---\n# Real H1\n\ntext\n")
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

// --- links -----------------------------------------------------------------

func TestLinksDeadIsHard(t *testing.T) {
	root := t.TempDir()
	k := kpiLinks("[x](nope/missing.md)", root, "docs/index.md")
	if k.score >= 100 || !anyContains(k.defects, "missing.md") {
		t.Fatalf("%+v", k)
	}
}

func TestLinksIgnoreNetwork(t *testing.T) {
	root := t.TempDir()
	k := kpiLinks("[w](https://x.io) [a](#sec)", root, "docs/index.md")
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

// --- answerability (never hard-fails) --------------------------------------

func TestAnswerabilityNoHardDefects(t *testing.T) {
	k := kpiAnswerability("# T\n\n## only scaffolding\n\n| a | b |\n")
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

func TestAnswerabilityProseOpenerScoresWell(t *testing.T) {
	good := kpiAnswerability("# T\n\nfak is an agent kernel that gates tool calls.\n\n## A\n")
	bad := kpiAnswerability("# T\n\n## A\n\n```code```\n")
	if good.score <= bad.score {
		t.Fatalf("good=%+v bad=%+v", good, bad)
	}
}

// --- per-page fold + grader ------------------------------------------------

func TestScorePagePerfect(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/b.md", "x")
	desc := "A clear plain-language description of what this fak page covers, with " +
		"the primary keywords a searcher would use, kept inside the length band."
	body := "---\ntitle: \"A solid page title here for fak\"\ndescription: \"" + desc + "\"\n---\n" +
		"# Title\n\nfak is a thing that does a thing, explained in plain words.\n\n## A\n\n[x](b.md)\n"
	d := scorePage(body, "docs/a.md", root)
	if d.NDefects != 0 || d.Grade != "A" {
		t.Fatalf("%+v", d)
	}
}

func TestMissingPageIsWorst(t *testing.T) {
	d := missingPageEntry("docs/gone.md")
	if d.Score != 0.0 || d.Grade != "F" || d.NDefects != 1 {
		t.Fatalf("%+v", d)
	}
}

func TestGradeBands(t *testing.T) {
	if gradeLetter(95) != "A" || gradeLetter(40) != "F" {
		t.Fatalf("bands wrong")
	}
}

// --- excellence credit: the above-100 layer (v5) ---------------------------

func TestGradeLadderLegacyInvariantsHold(t *testing.T) {
	if gradeLetter(90) != "A" || gradeLetter(100.0) != "A" || gradeLetter(100.9) != "A" || gradeLetter(40) != "F" {
		t.Fatalf("legacy ladder disturbed")
	}
}

func TestGradeLadderAbove100(t *testing.T) {
	if gradeLetter(101) != "A+" || gradeLetter(102.9) != "A+" || gradeLetter(103) != "S" || gradeLetter(104) != "S" {
		t.Fatalf("above-100 ladder wrong")
	}
}

func spotlessHubBody() string {
	desc := "The fak kernel is a default-deny permission gate fused with a bit-exact " +
		"key-value cache for self-hosted agent fleets that reproduce every run."
	return "---\ntitle: \"The fak kernel permission gate guide\"\ndescription: \"" + desc + "\"\n---\n" +
		"The fak kernel is a default-deny permission gate that reproduces cache " +
		"bytes exactly for every agent it fronts.\n\n" +
		"# The fak kernel guide\n\n" +
		"## Overview\n\nThe kernel gates tools and replays the cache deterministically.\n\n" +
		"## Guides\n\n- [alpha](alpha.md)\n- [beta](beta.md)\n- [gamma](gamma.md)\n"
}

func makeHubTargets(t *testing.T, root string) {
	for _, name := range []string{"alpha.md", "beta.md", "gamma.md"} {
		writeFile(t, root, "docs/"+name, "# x\n\nprose line here.\n")
	}
}

func TestPageCreditSpotlessEarnsS(t *testing.T) {
	root := t.TempDir()
	makeHubTargets(t, root)
	d := scorePage(spotlessHubBody(), "docs/hub.md", root)
	for name, v := range d.KPIs {
		if v != 100 {
			t.Fatalf("not spotless: %s=%d", name, v)
		}
	}
	if d.Baseline != 100.0 {
		t.Fatalf("baseline=%v", d.Baseline)
	}
	if d.Credit != 3 {
		t.Fatalf("credit=%d detail=%v", d.Credit, d.CreditDetail)
	}
	if d.Score != 103.0 || d.Grade != "S" {
		t.Fatalf("score=%v grade=%s", d.Score, d.Grade)
	}
	kf, ok := d.CreditDetail["keyword_focus"].(map[string]any)
	if !ok {
		t.Fatalf("no keyword_focus: %v", d.CreditDetail)
	}
	shared, _ := kf["shared"].([]string)
	if len(shared) != 1 || shared[0] != "kernel" {
		t.Fatalf("shared=%v", kf["shared"])
	}
	ch, ok := d.CreditDetail["crawl_hub"].(map[string]any)
	if !ok || ch["links"] != 3 {
		t.Fatalf("crawl_hub=%v", d.CreditDetail["crawl_hub"])
	}
}

func TestPageCreditRequiresSpotless(t *testing.T) {
	root := t.TempDir()
	makeHubTargets(t, root)
	body := strings.Replace(spotlessHubBody(),
		"# The fak kernel guide\n", "# The fak kernel guide\n\n# Second H1 here\n", 1)
	d := scorePage(body, "docs/hub.md", root)
	if d.KPIs["headings"] >= 100 {
		t.Fatalf("headings=%d", d.KPIs["headings"])
	}
	if d.Credit != 0 || len(d.CreditDetail) != 0 {
		t.Fatalf("credit=%d detail=%v", d.Credit, d.CreditDetail)
	}
	if d.Score > 100.0 || (d.Grade != "A" && d.Grade != "B") {
		t.Fatalf("score=%v grade=%s", d.Score, d.Grade)
	}
}

func TestPageCreditKeywordFocusNeedsAllFourSlots(t *testing.T) {
	root := t.TempDir()
	makeHubTargets(t, root)
	body := strings.Replace(spotlessHubBody(),
		"The fak kernel is a default-deny permission gate fused with a bit-exact "+
			"key-value cache for self-hosted agent fleets that reproduce every run.",
		"A default-deny permission gate fused with a bit-exact key-value cache for "+
			"self-hosted agent fleets that reproduce every run without any drift today.", 1)
	d := scorePage(body, "docs/hub.md", root)
	for name, v := range d.KPIs {
		if v != 100 {
			t.Fatalf("not spotless: %s=%d", name, v)
		}
	}
	if _, ok := d.CreditDetail["keyword_focus"]; ok {
		t.Fatalf("keyword_focus should be absent: %v", d.CreditDetail)
	}
	if d.Credit != 1 || d.Grade != "A+" {
		t.Fatalf("credit=%d grade=%s", d.Credit, d.Grade)
	}
}

func TestSiteCreditSchemaRichness(t *testing.T) {
	orgs := []any{map[string]any{"@type": "Organization", "name": "fak", "url": "https://fak.example"}}
	credit, detail := siteCredit(orgs, "", 0, false)
	if credit != 2 || len(detail) != 1 || detail["schema_richness"] != 2 {
		t.Fatalf("credit=%d detail=%v", credit, detail)
	}
}

func TestSiteCreditOrgWithoutIdentityEarnsNothing(t *testing.T) {
	orgs := []any{map[string]any{"@type": "Organization", "name": "fak"}}
	credit, detail := siteCredit(orgs, "", 0, false)
	if credit != 0 || len(detail) != 0 {
		t.Fatalf("credit=%d detail=%v", credit, detail)
	}
}

func robotsForUAs(uas []string) string {
	var b strings.Builder
	for _, ua := range uas {
		b.WriteString("User-agent: " + ua + "\nAllow: /\n")
	}
	return b.String()
}

func TestSiteCreditCrawlerBreadthNeedsEveryBot(t *testing.T) {
	var all []string
	for ua := range aiCrawlerUAs {
		all = append(all, ua)
	}
	credit, detail := siteCredit(nil, robotsForUAs(all), 0, false)
	if credit != 2 || len(detail) != 1 || detail["crawler_breadth"] != 2 {
		t.Fatalf("credit=%d detail=%v", credit, detail)
	}
	if c, _ := siteCredit(nil, robotsForUAs(aiCrawlerRequired), 0, false); c != 0 {
		t.Fatalf("partial should earn 0, got %d", c)
	}
}

func TestSiteCreditFaqDepthNeedsDepthAndSync(t *testing.T) {
	credit, detail := siteCredit(nil, "", 2*minFAQQuestions, true)
	if credit != 1 || len(detail) != 1 || detail["faq_depth"] != 1 {
		t.Fatalf("credit=%d detail=%v", credit, detail)
	}
	if c, _ := siteCredit(nil, "", 2*minFAQQuestions, false); c != 0 {
		t.Fatalf("unsynced should earn 0, got %d", c)
	}
	if c, _ := siteCredit(nil, "", minFAQQuestions, true); c != 0 {
		t.Fatalf("shallow should earn 0, got %d", c)
	}
}

func TestSiteCreditAllRungsStackToFive(t *testing.T) {
	orgs := []any{map[string]any{"@type": "Organization", "name": "fak", "url": "https://fak.example"}}
	var all []string
	for ua := range aiCrawlerUAs {
		all = append(all, ua)
	}
	credit, detail := siteCredit(orgs, robotsForUAs(all), 2*minFAQQuestions, true)
	if credit != 5 {
		t.Fatalf("credit=%d detail=%v", credit, detail)
	}
	want := map[string]bool{"schema_richness": true, "crawler_breadth": true, "faq_depth": true}
	if len(detail) != len(want) {
		t.Fatalf("detail keys=%v", detail)
	}
	for k := range detail {
		if !want[k] {
			t.Fatalf("unexpected detail key %q", k)
		}
	}
}

func TestSiteChecksDirtyRepoEarnsNoCredit(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	site := siteChecks(root)
	if len(site.Defects) == 0 {
		t.Fatalf("expected hard defects")
	}
	if site.Credit != 0 || len(site.CreditDetail) != 0 {
		t.Fatalf("credit=%d detail=%v", site.Credit, site.CreditDetail)
	}
	if site.ScoreWithCredit != site.Score {
		t.Fatalf("with=%v score=%v", site.ScoreWithCredit, site.Score)
	}
}

func TestPayloadExcellenceCreditLiftsOverallAbove100(t *testing.T) {
	pages := []Page{{
		Path: "docs/a.md", Score: 103.0, Grade: "S", NDefects: 0, Credit: 3,
		Defects: []string{}, Soft: []string{}, KPIs: map[string]int{"title": 100, "description": 100},
	}}
	site := Site{
		Checks: nil, Score: 100.0, ScoreWithCredit: 105.0, Credit: 5, CreditDetail: map[string]int{},
		NOK: 13, NTotal: 13, Defects: []string{}, Soft: []string{}, PresentJSONLD: []string{"SoftwareApplication"},
	}
	p := buildPayload(".", pages, site, "core", nil)
	c := p.Corpus
	if c.OverallScore != 104.0 || c.Grade != "S" {
		t.Fatalf("overall=%v grade=%s", c.OverallScore, c.Grade)
	}
	if c.ExcellenceCredit != (ExcellenceCredit{Page: 3, Site: 5, Total: 8}) {
		t.Fatalf("credit=%+v", c.ExcellenceCredit)
	}
	if c.SiteScoreWithCredit != 105.0 {
		t.Fatalf("site_with_credit=%v", c.SiteScoreWithCredit)
	}
	if c.GradeDistribution["S"] != 1 {
		t.Fatalf("grades=%v", c.GradeDistribution)
	}
	if !p.OK {
		t.Fatalf("expected ok")
	}
	out := Render(p)
	if !strings.Contains(out, "credit +8") || !strings.Contains(out, "grade S") {
		t.Fatalf("render missing tokens:\n%s", out)
	}
}

// --- published-set enumeration ---------------------------------------------

func TestPublishedExcludesReleasesAndNonMD(t *testing.T) {
	root := t.TempDir()
	if published(root, "docs/releases/v0.1.0.md") {
		t.Fatalf("releases should not publish")
	}
	if !published(root, "docs/index.md") {
		t.Fatalf("index should publish")
	}
	if !published(root, "docs/explainers/x.md") {
		t.Fatalf("explainers should publish")
	}
	if published(root, "docs/showcase.html") {
		t.Fatalf("html should not publish")
	}
	if published(root, "README.md") {
		t.Fatalf("root README should not publish")
	}
}

// --- site-level checks -----------------------------------------------------

func TestSiteFlagsMissingJSONLDAndLLMSFull(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	site := siteChecks(root)
	names := map[string]bool{}
	for _, c := range site.Checks {
		if !c.OK && c.Hard {
			names[c.Name] = true
		}
	}
	for _, want := range []string{"jsonld_SoftwareApplication", "jsonld_FAQPage", "llms_full"} {
		if !names[want] {
			t.Fatalf("missing hard defect %q", want)
		}
	}
}

func TestSiteDetectsJSONLDWhenPresent(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/_includes/head-custom.html",
		"<script type=\"application/ld+json\">{\"@type\":\"SoftwareApplication\"}</script>\n"+
			"<script type=\"application/ld+json\">{\"@type\":\"WebSite\"}</script>\n")
	site := siteChecks(root)
	if !anyContains(site.PresentJSONLD, "SoftwareApplication") || !anyContains(site.PresentJSONLD, "WebSite") {
		t.Fatalf("present=%v", site.PresentJSONLD)
	}
	if !checksByName(site)["jsonld_SoftwareApplication"].OK {
		t.Fatalf("SoftwareApplication check not ok")
	}
}

func TestSiteLLMSFullFreshness(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	writeFile(t, root, "llms.txt", "Key facts: the substrate")
	writeFile(t, root, "llms-full.txt", "# corpus\n\n## Index\n\nKey facts: the substrate\n\n--- more docs ---")
	by := checksByName(siteChecks(root))
	if !by["llms_txt"].OK {
		t.Fatalf("llms_txt=%+v", by["llms_txt"])
	}
	if !by["llms_full"].OK {
		t.Fatalf("llms_full=%+v", by["llms_full"])
	}
}

func TestSiteLLMSFullStaleByContent(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	writeFile(t, root, "llms.txt", "Key facts: the NEW substrate line")
	writeFile(t, root, "llms-full.txt", "old corpus without the new line")
	by := checksByName(siteChecks(root))
	if by["llms_full"].OK {
		t.Fatalf("expected stale llms_full")
	}
}

// --- payload + compare -----------------------------------------------------

func TestPayloadCleanIsOK(t *testing.T) {
	pages := []Page{{
		Path: "docs/a.md", Score: 100.0, Grade: "A", NDefects: 0,
		Defects: []string{}, Soft: []string{}, KPIs: map[string]int{"title": 100, "description": 100},
	}}
	site := Site{Score: 100.0, NOK: 13, NTotal: 13, Defects: []string{}, Soft: []string{}, PresentJSONLD: []string{"SoftwareApplication"}}
	p := buildPayload(".", pages, site, "core", nil)
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("%+v", p)
	}
}

func TestPayloadCountsSEODebt(t *testing.T) {
	pages := []Page{{
		Path: "docs/a.md", Score: 40.0, Grade: "F", NDefects: 2,
		Defects: []string{"x", "y"}, Soft: []string{}, KPIs: map[string]int{"title": 0, "description": 0},
	}}
	site := Site{Score: 60.0, NOK: 8, NTotal: 13, Defects: []string{"no JSON-LD FAQPage"}, Soft: []string{}, PresentJSONLD: []string{}}
	p := buildPayload(".", pages, site, "core", nil)
	if p.OK || p.Corpus.SEODebt != 3 {
		t.Fatalf("%+v", p.Corpus)
	}
	if p.Corpus.SEODebtInPages != 2 || p.Corpus.SEODebtInSite != 1 {
		t.Fatalf("%+v", p.Corpus)
	}
	if p.Corpus.MetaCoveragePct != 0.0 {
		t.Fatalf("meta=%v", p.Corpus.MetaCoveragePct)
	}
}

func TestCompareReports10x(t *testing.T) {
	base := map[string]any{"corpus": map[string]any{
		"seo_debt": 27, "overall_score": 53.0, "meta_coverage_pct": 25.0,
		"site_checks_ok": "7/13", "present_jsonld": []any{}}}
	cur := Payload{Corpus: Corpus{
		SEODebt: 2, OverallScore: 95.0, MetaCoveragePct: 100.0,
		SiteChecksOK: "13/13", PresentJSONLD: []string{"SoftwareApplication", "FAQPage"}}}
	out := Compare(cur, base)
	if !strings.Contains(out, ">=10x") {
		t.Fatalf("no 10x verdict:\n%s", out)
	}
}

func TestCompareNotYet10x(t *testing.T) {
	base := map[string]any{"corpus": map[string]any{"seo_debt": 27, "overall_score": 53.0}}
	cur := Payload{Corpus: Corpus{SEODebt: 10, OverallScore: 70.0}}
	out := Compare(cur, base)
	if !strings.Contains(out, "not yet 10x") {
		t.Fatalf("expected not-yet verdict:\n%s", out)
	}
}

// --- live smoke ------------------------------------------------------------

func TestLiveCollectCore(t *testing.T) {
	root := repoRootDir(t)
	if root == "" {
		t.Skip("repo root not found")
	}
	if _, err := os.Stat(filepath.Join(root, "docs", "index.md")); err != nil {
		t.Skip("not in the repo tree")
	}
	p := Build(root, "core")
	if p.Schema != SCHEMA {
		t.Fatalf("schema=%q", p.Schema)
	}
	abs := filepath.Clean(root)
	if p.Corpus.NPages != len(enumeratePages(abs, "core")) {
		t.Fatalf("n_pages=%d", p.Corpus.NPages)
	}
	if len(p.Pages) == 0 {
		t.Fatalf("no pages")
	}
}

func TestLiveCollectPublishedIsSuperset(t *testing.T) {
	root := repoRootDir(t)
	if root == "" {
		t.Skip("repo root not found")
	}
	if _, err := os.Stat(filepath.Join(root, "docs", "index.md")); err != nil {
		t.Skip("not in the repo tree")
	}
	p := Build(root, "published")
	if p.Scope != "published" {
		t.Fatalf("scope=%q", p.Scope)
	}
	abs := filepath.Clean(root)
	if p.Corpus.NPages < len(enumeratePages(abs, "core")) {
		t.Fatalf("published %d < core", p.Corpus.NPages)
	}
}

// --- adversarial / anti-gaming tests ---------------------------------------

func TestDegenerateTitleIsHardDefect(t *testing.T) {
	k := kpiTitle(map[string]string{"title": strings.Repeat("x", 100)})
	if k.score != 0 || !anyContains(k.defects, "degenerate") {
		t.Fatalf("%+v", k)
	}
}

func TestDegenerateDescriptionIsHardDefect(t *testing.T) {
	k := kpiDescription(map[string]string{"description": strings.Repeat(".", 120)})
	if k.score != 0 || !anyContains(k.defects, "degenerate") {
		t.Fatalf("%+v", k)
	}
}

func TestRealShortTitleNotDegenerate(t *testing.T) {
	k := kpiTitle(map[string]string{"title": "fak FAQ"})
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

// A localized title is a real title. Word counting must not be Latin-only, or every
// non-English entry page scores as filler and the only "fix" is to stuff Latin words
// into it — exactly the gaming this scorecard exists to refuse.
func TestLocalizedTitleNotDegenerate(t *testing.T) {
	for _, title := range []string{
		"fak — быстрый старт (10 минут до локальной модели)",
		"fak — البداية السريعة (نموذج محلي في نحو ١٠ دقائق)",
		"fak 快速开始（10 分钟跑起一个受管治的本地模型）",
	} {
		if k := kpiTitle(map[string]string{"title": title}); len(k.defects) != 0 {
			t.Fatalf("localized title flagged: %q -> %+v", title, k)
		}
	}
}

// …and the widened word class must still catch real filler in those same scripts.
func TestRepeatedLocalizedWordStillDegenerate(t *testing.T) {
	for _, title := range []string{"старт старт старт", "快速开始 快速开始"} {
		k := kpiTitle(map[string]string{"title": title})
		if k.score != 0 || !anyContains(k.defects, "degenerate") {
			t.Fatalf("filler title accepted: %q -> %+v", title, k)
		}
	}
}

func TestMalformedJSONLDNotCounted(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/_includes/head-custom.html",
		"<script type=\"application/ld+json\">{ \"@type\": \"SoftwareApplication\" BROKEN }</script>\n")
	site := siteChecks(root)
	if anyContains(site.PresentJSONLD, "SoftwareApplication") {
		t.Fatalf("phantom type: %v", site.PresentJSONLD)
	}
	if checksByName(site)["jsonld_valid"].OK {
		t.Fatalf("jsonld_valid should trip")
	}
}

func TestValidJSONLDArrayAndGraph(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/_includes/head-custom.html",
		"<script type=\"application/ld+json\">{\"@graph\":[{\"@type\":\"WebSite\"},{\"@type\":\"Organization\"}]}</script>")
	site := siteChecks(root)
	if !anyContains(site.PresentJSONLD, "WebSite") || !anyContains(site.PresentJSONLD, "Organization") {
		t.Fatalf("present=%v", site.PresentJSONLD)
	}
}

func TestSiteChecksCountsBreadcrumbFromIndexPage(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/_includes/head-custom.html",
		"<script type=\"application/ld+json\">{\"@type\":\"SoftwareApplication\"}</script>"+
			"<script type=\"application/ld+json\">{\"@type\":\"WebSite\"}</script>")
	// A valid BreadcrumbList block on the index page (Python sources this from
	// gen_structured_data.render_breadcrumb_block(); inlined here as an equivalent).
	writeFile(t, root, "docs/index.md",
		"<script type=\"application/ld+json\">"+
			"{\"@context\":\"https://schema.org\",\"@type\":\"BreadcrumbList\",\"itemListElement\":["+
			"{\"@type\":\"ListItem\",\"position\":1,\"name\":\"Home\",\"item\":\"https://example.com/\"},"+
			"{\"@type\":\"ListItem\",\"position\":2,\"name\":\"Docs\",\"item\":\"https://example.com/docs/\"}]}"+
			"</script>\n")
	by := checksByName(siteChecks(root))
	if !by["jsonld_BreadcrumbList"].OK {
		t.Fatalf("jsonld_BreadcrumbList=%+v", by["jsonld_BreadcrumbList"])
	}
	if !by["jsonld_BreadcrumbList"].Hard {
		t.Fatalf("BreadcrumbList should be hard")
	}
	if !by["breadcrumb_jsonld_shape"].OK {
		t.Fatalf("breadcrumb_jsonld_shape=%+v", by["breadcrumb_jsonld_shape"])
	}
}

func TestBreadcrumbShapeRejectsUnorderedRelativeItems(t *testing.T) {
	values := []any{map[string]any{
		"@type": "BreadcrumbList",
		"itemListElement": []any{
			map[string]any{"@type": "ListItem", "position": 2, "name": "Docs", "item": "/fak/"},
			map[string]any{"@type": "ListItem", "position": 1, "name": "Home", "item": "https://example.com/"},
		},
	}}
	ok, detail := breadcrumbShapeOK(values)
	if ok || !strings.Contains(detail, "invalid") {
		t.Fatalf("ok=%v detail=%q", ok, detail)
	}
}

func TestFAQJSONLDSyncDetectsStaleSchema(t *testing.T) {
	root := t.TempDir()
	var visible strings.Builder
	visible.WriteString("# FAQ\n\n")
	for i := 0; i < 6; i++ {
		visible.WriteString("## What is question " + itoa(i) + "?\n\nThis is a visible answer with enough prose.\n\n")
	}
	staleSchema := "<script type=\"application/ld+json\">" +
		"{\"@type\":\"FAQPage\",\"mainEntity\":[{\"@type\":\"Question\",\"name\":\"What is question 0?\"," +
		"\"acceptedAnswer\":{\"@type\":\"Answer\",\"text\":\"This answer is long enough.\"}}]}" +
		"</script>\n"
	writeFile(t, root, "docs/FAQ.md", staleSchema+visible.String())
	by := checksByName(siteChecks(root))
	if !by["faq_structured"].OK {
		t.Fatalf("faq_structured=%+v", by["faq_structured"])
	}
	if by["faq_jsonld_sync"].OK {
		t.Fatalf("faq_jsonld_sync should be stale")
	}
}

func TestLLMSFullSourcesDetectsMissingSource(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/a.md", "# A")
	writeFile(t, root, "llms.txt", "Key facts: x\n\n- [A](docs/a.md)")
	writeFile(t, root, "llms-full.txt", "# corpus\n\nKey facts: x\n\n- [A](docs/a.md)\n")
	by := checksByName(siteChecks(root))
	if !by["llms_full"].OK {
		t.Fatalf("llms_full=%+v", by["llms_full"])
	}
	if by["llms_full_sources"].OK {
		t.Fatalf("llms_full_sources should miss a source")
	}
}

func TestFAQNonQuestionH2NotCounted(t *testing.T) {
	root := t.TempDir()
	var b strings.Builder
	b.WriteString("# FAQ\n\n")
	for i := 0; i < 8; i++ {
		b.WriteString("## Section " + itoa(i) + "\n\ntext\n")
	}
	writeFile(t, root, "docs/FAQ.md", b.String())
	by := checksByName(siteChecks(root))
	if by["faq_structured"].OK {
		t.Fatalf("faq_structured should be false for non-question H2s")
	}
}

func TestDiscoveryExcludesEvidenceSubtrees(t *testing.T) {
	root := t.TempDir()
	if !discovery(root, "docs/fak/server-quickstart.md") {
		t.Fatalf("fak/ should be discovery")
	}
	for _, rel := range []string{"docs/proofs/policy.md", "docs/benchmarks/TURN-TAX-RESULTS.md", "docs/notes/x.md", "docs/releases/v0.1.0.md"} {
		if discovery(root, rel) {
			t.Fatalf("%s should not be discovery", rel)
		}
	}
}

func TestReachableAndOrphanDetection(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "README.md", "see [a](docs/a.md)")
	writeFile(t, root, "docs/a.md", "# A\n\n[b](b.md)")
	writeFile(t, root, "docs/b.md", "# B")
	writeFile(t, root, "docs/orphan.md", "# Orphan, linked from nowhere")
	reach := reachablePublished(root)
	if !reach["docs/a.md"] || !reach["docs/b.md"] {
		t.Fatalf("reach=%v", reach)
	}
	orphans := discoveryOrphans(root)
	if !anyContains(orphans, "docs/orphan.md") {
		t.Fatalf("orphans=%v", orphans)
	}
	if anyContains(orphans, "docs/a.md") {
		t.Fatalf("a.md should not be an orphan: %v", orphans)
	}
}

func TestLinksSkipFencedCode(t *testing.T) {
	root := t.TempDir()
	body := "see real prose\n\n```\nrm docs/does-not-exist.md\n```\n"
	k := kpiLinks(body, root, "docs/index.md")
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

// --- SUCCESS KPIs: presence -> success -------------------------------------

func TestKPIWeightsSumToOne(t *testing.T) {
	sum := 0.0
	names := map[string]bool{}
	for _, kw := range kpiWeightsOrder {
		sum += kw.w
		names[kw.name] = true
	}
	if math.Abs(sum-1.0) >= 1e-9 {
		t.Fatalf("weights sum=%v", sum)
	}
	if !names["links_crawlable"] || !names["alt_text"] {
		t.Fatalf("missing weighted KPI: %v", names)
	}
}

func TestSchemaIsV5(t *testing.T) {
	if !strings.HasSuffix(SCHEMA, "/5") {
		t.Fatalf("schema=%q", SCHEMA)
	}
}

func TestHeadingsIgnoresCodeFenceHashes(t *testing.T) {
	text := "# Real Title\n\nintro prose here.\n\n## Section\n\n" +
		"```bash\n# install deps\n### step three\nrun it\n```\n\nmore prose.\n"
	k := kpiHeadings(text)
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
	if !strings.Contains(k.detail, "1 H1") {
		t.Fatalf("detail=%q", k.detail)
	}
}

func TestLinksCrawlableNonPublishedIsHard(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/releases/v1.md", "# v1")
	k := kpiLinksCrawlable("see [notes](releases/v1.md)", root, "docs/index.md")
	if k.score >= 100 || !anyContains(k.defects, "crawl-404") {
		t.Fatalf("%+v", k)
	}
}

func TestLinksCrawlablePublishedIsOK(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/guide.md", "# guide")
	k := kpiLinksCrawlable("see [g](guide.md)", root, "docs/index.md")
	if len(k.defects) != 0 || k.score != 100 {
		t.Fatalf("%+v", k)
	}
}

func TestLinksCrawlableDeadOnDiskNotDoubleCounted(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	k := kpiLinksCrawlable("see [x](gone.md)", root, "docs/index.md")
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

func TestLinksCrawlableDirectoryIsSoft(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	writeFile(t, root, "examples/.keep", "")
	k := kpiLinksCrawlable("see [ex](../examples)", root, "docs/index.md")
	if len(k.defects) != 0 || !anyContains(k.soft, "directory link") {
		t.Fatalf("%+v", k)
	}
}

func TestMetaDistinctFlagsDuplicateTitle(t *testing.T) {
	pages := []Page{
		{Path: "docs/a.md", Meta: PageMeta{Title: "Same Title", Description: "desc a"}, Defects: []string{}, NDefects: 0},
		{Path: "docs/b.md", Meta: PageMeta{Title: "Same Title", Description: "desc b"}, Defects: []string{}, NDefects: 0},
	}
	if added := applyCorpusMetaDistinct(pages); added != 2 {
		t.Fatalf("added=%d", added)
	}
	for _, p := range pages {
		if !anyContains(p.Defects, "meta_distinct: title") || p.NDefects != 1 {
			t.Fatalf("%+v", p)
		}
	}
}

func TestMetaDistinctUniqueIsClean(t *testing.T) {
	pages := []Page{
		{Path: "docs/a.md", Meta: PageMeta{Title: "Title A", Description: "desc a"}, Defects: []string{}, NDefects: 0},
		{Path: "docs/b.md", Meta: PageMeta{Title: "Title B", Description: "desc b"}, Defects: []string{}, NDefects: 0},
	}
	if added := applyCorpusMetaDistinct(pages); added != 0 {
		t.Fatalf("added=%d", added)
	}
}

func TestCitationLinksDeadSelfRepoIsHard(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/index.md",
		"see [code](https://github.com/anthony-chaudhary/fak/blob/main/internal/gone.go)")
	cit := citationLinkAudit(root)
	if !anyContains(cit.DeadSelf, "internal/gone.go") {
		t.Fatalf("dead_self=%v", cit.DeadSelf)
	}
	if checksByName(siteChecks(root))["citation_links"].OK {
		t.Fatalf("citation_links should be false")
	}
}

func TestCitationLinksLiveSelfRepoIsOK(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "real.go", "package x")
	writeFile(t, root, "docs/index.md",
		"see [code](https://github.com/anthony-chaudhary/fak/blob/main/real.go)")
	cit := citationLinkAudit(root)
	if len(cit.DeadSelf) != 0 {
		t.Fatalf("dead_self=%v", cit.DeadSelf)
	}
}

func TestCitationSelfRepoIgnoresHTMLAttributeJunk(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "hero.mp4", "x")
	writeFile(t, root, "docs/index.md",
		"<video src=\"https://github.com/anthony-chaudhary/fak/raw/main/hero.mp4\">full</video>")
	cit := citationLinkAudit(root)
	if len(cit.DeadSelf) != 0 {
		t.Fatalf("dead_self=%v", cit.DeadSelf)
	}
}

func TestLLMSFullNavigableIsHard(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	writeFile(t, root, "llms.txt", "Key facts: x")
	writeFile(t, root, "llms-full.txt", "Key facts: x\n\nsee [p](policy-guide.md) for more.")
	by := checksByName(siteChecks(root))
	if by["llms_full_navigable"].OK {
		t.Fatalf("llms_full_navigable should be false")
	}
	if !by["llms_full_navigable"].Hard {
		t.Fatalf("llms_full_navigable should be hard")
	}
}

func TestScorePageCarriesCrawlableAndMeta(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/.keep", "")
	d := scorePage("---\ntitle: T\n---\n# T\n\nfak is a thing.\n", "docs/a.md", root)
	if _, ok := d.KPIs["links_crawlable"]; !ok {
		t.Fatalf("no links_crawlable: %v", d.KPIs)
	}
	if _, ok := d.KPIs["alt_text"]; !ok {
		t.Fatalf("no alt_text: %v", d.KPIs)
	}
	if d.Meta.Title != "T" {
		t.Fatalf("meta=%+v", d.Meta)
	}
}

// --- alt_text KPI ----------------------------------------------------------

func TestAltTextMissingIsHard(t *testing.T) {
	k := kpiAltText("# T\n\n![](../visuals/diagram.svg)\n")
	if k.score >= 100 || !anyContains(k.defects, "no alt text") {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextPresentIsClean(t *testing.T) {
	k := kpiAltText("# T\n\n![A labelled KV-cache residency diagram](x.svg)\n")
	if len(k.defects) != 0 || len(k.soft) != 0 || k.score != 100 {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextNoImagesIs100(t *testing.T) {
	k := kpiAltText("# T\n\njust prose, no images at all.\n")
	if k.score != 100 || !strings.Contains(k.detail, "no images") {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextIgnoresCodeExample(t *testing.T) {
	inline := kpiAltText("# T\n\nembed it directly (`![](visuals/x.svg)`).\n")
	if len(inline.defects) != 0 {
		t.Fatalf("inline=%+v", inline)
	}
	fenced := kpiAltText("# T\n\n```md\n![](visuals/x.svg)\n```\n")
	if len(fenced.defects) != 0 {
		t.Fatalf("fenced=%+v", fenced)
	}
}

func TestAltTextHTMLImgWithoutAltIsHard(t *testing.T) {
	k := kpiAltText("# T\n\n<img src=\"hero.png\" width=\"600\">\n")
	if k.score >= 100 || !anyContains(k.defects, "<img>") {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextHTMLImgWithAltIsClean(t *testing.T) {
	k := kpiAltText("# T\n\n<img src=\"hero.png\" alt=\"the fak control pane in action\">\n")
	if len(k.defects) != 0 || k.score != 100 {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextFillerIsSoftNotHard(t *testing.T) {
	k := kpiAltText("# T\n\n![image](x.svg)\n")
	if len(k.defects) != 0 || !anyContains(k.soft, "filler") {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextReferenceStyleMissingIsHard(t *testing.T) {
	k := kpiAltText("# T\n\n![][hero]\n\n[hero]: ../visuals/h.svg\n")
	if k.score >= 100 || !anyContains(k.defects, "no alt text") {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextReferenceStylePresentIsClean(t *testing.T) {
	k := kpiAltText("# T\n\n![a labelled throughput chart][hero]\n\n[hero]: h.svg\n")
	if len(k.defects) != 0 {
		t.Fatalf("%+v", k)
	}
}

func TestAltTextDataAltDoesNotSatisfy(t *testing.T) {
	k := kpiAltText("# T\n\n<img src=\"x.png\" data-alt=\"not real alt\">\n")
	if k.score >= 100 || !anyContains(k.defects, "no alt text") {
		t.Fatalf("%+v", k)
	}
}

// --- ai_crawlers site check (AEO) ------------------------------------------

func TestAICrawlersBareWildcardIsDefect(t *testing.T) {
	ok, detail := aiCrawlersOK("User-agent: *\nAllow: /\nSitemap: https://x/y.xml\n")
	if ok || !strings.Contains(detail, "does not explicitly welcome") {
		t.Fatalf("ok=%v detail=%q", ok, detail)
	}
}

func TestAICrawlersExplicitAllowlistPasses(t *testing.T) {
	robots := "User-agent: *\nAllow: /\n"
	for _, ua := range aiCrawlerRequired {
		robots += "\nUser-agent: " + ua + "\nAllow: /\n"
	}
	if ok, detail := aiCrawlersOK(robots); !ok {
		t.Fatalf("ok=%v detail=%q", ok, detail)
	}
}

func TestAICrawlersDisallowedBotIsDefect(t *testing.T) {
	robots := "User-agent: *\nAllow: /\n" +
		"\nUser-agent: GPTBot\nDisallow: /\n" +
		"\nUser-agent: ClaudeBot\nAllow: /\n" +
		"\nUser-agent: PerplexityBot\nAllow: /\n" +
		"\nUser-agent: Google-Extended\nAllow: /\n"
	ok, detail := aiCrawlersOK(robots)
	if ok || !strings.Contains(detail, "Disallow") || !strings.Contains(detail, "GPTBot") {
		t.Fatalf("ok=%v detail=%q", ok, detail)
	}
}

func TestAICrawlersWiredIntoSiteChecks(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "docs/robots.txt", "User-agent: *\nAllow: /\nSitemap: https://x/y.xml\n")
	by := checksByName(siteChecks(root))
	c, ok := by["ai_crawlers"]
	if !ok {
		t.Fatalf("no ai_crawlers check")
	}
	if c.OK || !c.Hard {
		t.Fatalf("%+v", c)
	}
}

func TestAICrawlersBlocksWildcardDisallow(t *testing.T) {
	for _, d := range []string{"Disallow: /*", "Disallow: *"} {
		robots := "User-agent: *\nAllow: /\n"
		for _, ua := range aiCrawlerRequired {
			robots += "\nUser-agent: " + ua + "\n" + d + "\n"
		}
		ok, detail := aiCrawlersOK(robots)
		if ok || !strings.Contains(detail, "Disallow") {
			t.Fatalf("d=%q ok=%v detail=%q", d, ok, detail)
		}
	}
}

func TestAICrawlersPartialDisallowStillWelcomes(t *testing.T) {
	robots := "User-agent: *\nAllow: /\n"
	for _, ua := range aiCrawlerRequired {
		robots += "\nUser-agent: " + ua + "\nAllow: /\nDisallow: /private/\n"
	}
	if ok, detail := aiCrawlersOK(robots); !ok {
		t.Fatalf("ok=%v detail=%q", ok, detail)
	}
}

func TestAICrawlersTrailingGlobalDisallowNotMisattributed(t *testing.T) {
	robots := "User-agent: *\nAllow: /\n"
	for _, ua := range aiCrawlerRequired {
		robots += "\nUser-agent: " + ua + "\nAllow: /\n"
	}
	robots += "\nDisallow: /\n"
	if ok, detail := aiCrawlersOK(robots); !ok {
		t.Fatalf("ok=%v detail=%q", ok, detail)
	}
}
