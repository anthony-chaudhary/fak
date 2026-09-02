// Package seoaeoscore is the Go port of the former tools/seo_aeo_scorecard.py: the SEO/AEO
// discoverability measuring stick that makes "more discoverable" provable.
//
// tools/docs_scorecard.py answers "are the docs CORRECT for a reader who already found
// us". This tool answers the orthogonal question: will a reader — or an answer engine —
// find us at all, and cite us correctly when they do. Same discipline (deterministic,
// content-only, no model, no network), aimed at the discoverability surface instead of
// the prose.
//
// The headline metric is an integer driven toward zero: seo-debt — the count of concrete,
// re-derivable discoverability defects (a published page with no meta description, a
// missing JSON-LD type, a stale llms-full.txt, a dead link, a missing social card).
//
// This is a faithful, behavior-preserving port: Build / Render / Markdown / Compare mirror
// collect / render / render_markdown / render_compare one-for-one, so the control-pane fold
// and the human surfaces are unchanged. Read-only by construction.
package seoaeoscore

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/walkfiles"
)

// SCHEMA is the payload schema tag (matches the python SCHEMA constant).
const SCHEMA = "fak-seo-aeo-scorecard/5"

// Repo-root-relative inputs (best-effort; a missing one degrades a check, never errors).
const (
	configRel       = "docs/_config.yml"
	robotsRel       = "docs/robots.txt"
	headIncludeRel  = "docs/_includes/head-custom.html"
	indexRel        = "docs/index.md"
	showcaseRel     = "docs/showcase.html"
	llmsRel         = "llms.txt"
	llmsFullRel     = "llms-full.txt"
	faqRel          = "docs/FAQ.md"
	titleMin        = 15
	titleMax        = 70
	descMin         = 70
	descMax         = 160
	minFAQQuestions = 6
)

// frontDoors are where a reader/crawler enters; reachability is a link-BFS from these.
var frontDoors = []string{"README.md", "llms.txt", "docs/index.md", "INDEX.md", "START-HERE.md"}

// evidenceDirs: deep-technical proof docs excluded from the --scope core discovery surface.
var evidenceDirs = map[string]bool{"proofs": true, "benchmarks": true, "notes": true}

// jsonldTypesHard / jsonldTypesSoft: expected JSON-LD @types for AEO.
var (
	jsonldTypesHard = []string{"SoftwareApplication", "FAQPage", "WebSite", "BreadcrumbList"}
	jsonldTypesSoft = []string{"Organization"}
)

// kpiWeight is one per-page KPI weight, in the python KPI_WEIGHTS insertion order.
type kpiWeight struct {
	name string
	w    float64
}

var kpiWeightsOrder = []kpiWeight{
	{"title", 0.21},
	{"description", 0.25},
	{"headings", 0.12},
	{"links", 0.16},
	{"links_crawlable", 0.12},
	{"answerability", 0.08},
	{"alt_text", 0.06},
}

// nonpublishedDirs: dirs under docs/ Jekyll does not publish as reader pages.
var nonpublishedDirs = map[string]bool{
	"benchmark": true, "benchmarking": true, "planning": true, "testing": true,
	"releases": true, "stable-releases": true, "launch": true, "archive": true,
	"_includes": true, "_layouts": true, "_data": true, "_site": true,
}

var altFiller = map[string]bool{
	"image": true, "img": true, "picture": true, "photo": true, "screenshot": true,
	"graphic": true, "figure": true, "icon": true, "logo": true, "diagram": true,
	"chart": true, "svg": true, "png": true, "jpg": true,
}

var aiCrawlerUAs = map[string]bool{
	"GPTBot": true, "OAI-SearchBot": true, "ChatGPT-User": true,
	"ClaudeBot": true, "anthropic-ai": true, "Claude-SearchBot": true,
	"PerplexityBot": true, "Perplexity-User": true,
	"Google-Extended": true, "Applebot-Extended": true, "CCBot": true,
}

var aiCrawlerRequired = []string{"GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended"}

var stopwords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "of": true, "for": true,
	"to": true, "in": true, "on": true, "with": true, "is": true, "are": true, "be": true,
	"this": true, "that": true, "it": true, "as": true, "at": true, "by": true, "from": true,
	"your": true, "you": true, "our": true, "we": true, "how": true, "what": true, "why": true,
	"when": true, "where": true, "who": true, "which": true, "its": true, "into": true,
	"not": true, "no": true,
}

var (
	reLink        = regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
	reH           = regexp.MustCompile(`(?m)^(#{1,6})\s+\S`)
	reH2          = regexp.MustCompile(`(?m)^##\s+(.+)$`)
	reFence       = regexp.MustCompile("(?s)```.*?```")
	reJSONLDBlock = regexp.MustCompile(`(?is)<script[^>]*type=["']application/ld\+json["'][^>]*>(.*?)</script>`)
	reQuestion    = regexp.MustCompile(`(?i)^(how|what|why|when|where|who|which|is|are|can|does|do|should|will)\b`)
	reSelfRepo    = regexp.MustCompile(`https?://(?:github\.com|raw\.githubusercontent\.com)/anthony-chaudhary/fak/` +
		`(?:(?:blob|tree|raw)/)?(?:main|master|HEAD)/([^)\]\s"'<>` + "`" + `#?]+)`)
	reMDImg      = regexp.MustCompile(`!\[([^\]]*)\]\(([^)]+)\)`)
	reMDRefImg   = regexp.MustCompile(`!\[([^\]]*)\]\[[^\]]*\]`)
	reHTMLImg    = regexp.MustCompile(`(?i)<img\b[^>]*>`)
	reHTMLAlt    = regexp.MustCompile(`(?i)(?:^|[\s"'])alt\s*=\s*["']([^"']*)["']`)
	reInlineCode = regexp.MustCompile("`[^`\\n]*`")
	reToken      = regexp.MustCompile(`[A-Za-z][A-Za-z'-]{3,}`)
	reWordApos   = regexp.MustCompile(`\p{L}[\p{L}\p{M}'’-]+`)
	reFMKey      = regexp.MustCompile(`^(title|description)\s*:\s*(.*)$`)
	reExclude    = regexp.MustCompile(`(?m)^exclude:\s*\n((?:[ \t]+-.*\n?|[ \t]*#.*\n?)+)`)
	reImage      = regexp.MustCompile(`image:\s*"?([^"\n]+)`)
	reMainPath   = regexp.MustCompile(`/main/(.+)$`)
	reURLLine    = regexp.MustCompile(`(?m)^url:\s*\S`)
	reDisallow   = regexp.MustCompile(`(?i)^disallow:\s*(?:/\*?|\*)\s*$`)
	reUserAgent  = regexp.MustCompile(`(?i)^user-agent:\s*(\S+)`)
	reSource     = regexp.MustCompile("(?m)^> Source: `([^`]+)`\\s*$")
	reIsAre      = regexp.MustCompile(`\b(is|are)\b\s+\w`)
	reHasLetter  = regexp.MustCompile(`[A-Za-z]`)
	reKeyFacts   = regexp.MustCompile(`(?i)key facts`)
)

// ---------------------------------------------------------------------------
// Payload types (mirror the python dict structure + JSON keys).
// ---------------------------------------------------------------------------

type PageMeta struct {
	Title       string `json:"title"`
	Description string `json:"description"`
}

type Page struct {
	Path         string            `json:"path"`
	Score        float64           `json:"score"`
	Baseline     float64           `json:"baseline"`
	Credit       int               `json:"credit"`
	CreditDetail map[string]any    `json:"credit_detail"`
	Grade        string            `json:"grade"`
	KPIs         map[string]int    `json:"kpis"`
	KPIDetail    map[string]string `json:"kpi_detail"`
	Meta         PageMeta          `json:"meta"`
	Defects      []string          `json:"defects"`
	Soft         []string          `json:"soft"`
	NDefects     int               `json:"n_defects"`
}

type SiteCheck struct {
	Name   string  `json:"name"`
	OK     bool    `json:"ok"`
	Detail string  `json:"detail"`
	Defect *string `json:"defect"`
	Hard   bool    `json:"hard"`
}

type Citation struct {
	DeadMap            []string `json:"dead_map"`
	DeadSelf           []string `json:"dead_self"`
	LLMSFullUnresolved int      `json:"llms_full_unresolved"`
}

type LLMSFullSources struct {
	Targets []string `json:"targets"`
	Sources []string `json:"sources"`
	Missing []string `json:"missing"`
}

type Site struct {
	Checks          []SiteCheck     `json:"checks"`
	Score           float64         `json:"score"`
	ScoreWithCredit float64         `json:"score_with_credit"`
	Credit          int             `json:"credit"`
	CreditDetail    map[string]int  `json:"credit_detail"`
	NOK             int             `json:"n_ok"`
	NTotal          int             `json:"n_total"`
	Defects         []string        `json:"defects"`
	Soft            []string        `json:"soft"`
	PresentJSONLD   []string        `json:"present_jsonld"`
	Citation        Citation        `json:"citation"`
	LLMSFullSources LLMSFullSources `json:"llms_full_sources"`
}

type WorstRow struct {
	Path     string  `json:"path"`
	Score    float64 `json:"score"`
	Grade    string  `json:"grade"`
	NDefects int     `json:"n_defects"`
}

type ExcellenceCredit struct {
	Page  int `json:"page"`
	Site  int `json:"site"`
	Total int `json:"total"`
}

type Corpus struct {
	NPages              int              `json:"n_pages"`
	OverallScore        float64          `json:"overall_score"`
	Grade               string           `json:"grade"`
	PageMeanScore       float64          `json:"page_mean_score"`
	SiteScore           float64          `json:"site_score"`
	SiteScoreWithCredit float64          `json:"site_score_with_credit"`
	ExcellenceCredit    ExcellenceCredit `json:"excellence_credit"`
	SiteChecksOK        string           `json:"site_checks_ok"`
	MetaCoveragePct     float64          `json:"meta_coverage_pct"`
	MedianScore         float64          `json:"median_score"`
	MinScore            float64          `json:"min_score"`
	GradeDistribution   map[string]int   `json:"grade_distribution"`
	SEODebt             int              `json:"seo_debt"`
	SEODebtInPages      int              `json:"seo_debt_in_pages"`
	SEODebtInSite       int              `json:"seo_debt_in_site"`
	Crawl404            int              `json:"crawl_404"`
	MetaDuplicates      int              `json:"meta_duplicates"`
	CitationDead        int              `json:"citation_dead"`
	LLMSFullUnresolved  int              `json:"llms_full_unresolved"`
	PresentJSONLD       []string         `json:"present_jsonld"`
	DiscoveryOrphans    int              `json:"discovery_orphans"`
	OrphanPages         []string         `json:"orphan_pages"`
	Worst               []WorstRow       `json:"worst"`
}

type Payload struct {
	Schema     string `json:"schema"`
	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"`
	Finding    string `json:"finding"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
	Workspace  string `json:"workspace"`
	Scope      string `json:"scope"`
	Corpus     Corpus `json:"corpus"`
	Site       Site   `json:"site"`
	Pages      []Page `json:"pages"`
}

// kpi is the internal per-KPI result (not serialized directly).
type kpi struct {
	name    string
	score   int
	detail  string
	defects []string
	soft    []string
}

// ---------------------------------------------------------------------------
// Build (the exported entry; mirrors collect()).
// ---------------------------------------------------------------------------

// Build scores the workspace at root under scope ("core"|"published") and returns the payload.
func Build(root, scope string) Payload {
	abs, err := filepath.Abs(root)
	if err != nil {
		abs = root
	}
	abs = filepath.Clean(abs)
	return collect(abs, scope)
}

func collect(rootAbs, scope string) Payload {
	rels := enumeratePages(rootAbs, scope)
	pages := make([]Page, 0, len(rels))
	for _, rel := range rels {
		abs := rjoin(rootAbs, rel)
		if !existsPath(abs) {
			pages = append(pages, missingPageEntry(rel))
			continue
		}
		text, ok := readText(abs)
		if !ok {
			d := missingPageEntry(rel)
			d.Defects = []string{fmt.Sprintf("unreadable: %s", rel)}
			pages = append(pages, d)
			continue
		}
		pages = append(pages, scorePage(text, rel, rootAbs))
	}
	applyCorpusMetaDistinct(pages)
	site := siteChecks(rootAbs)
	orphans := discoveryOrphans(rootAbs)
	return buildPayload(rootAbs, pages, site, scope, orphans)
}

// ---------------------------------------------------------------------------
// I/O + path helpers.
// ---------------------------------------------------------------------------

func readText(abs string) (string, bool) {
	b, err := os.ReadFile(abs)
	if err != nil {
		return "", false
	}
	s := string(b)
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return s, true
}

func safeRead(abs string) string { s, _ := readText(abs); return s }

func rjoin(rootAbs, rel string) string { return filepath.Join(rootAbs, filepath.FromSlash(rel)) }

func existsPath(abs string) bool { _, err := os.Stat(abs); return err == nil }

func isDirPath(abs string) bool { fi, err := os.Stat(abs); return err == nil && fi.IsDir() }

// rootRealCache memoizes the canonical (symlink- and case-resolved) form of a
// workspace root, so relPosix mirrors python's `root.resolve()` without re-stat'ing
// the root on every call.
var rootRealCache sync.Map // rootAbs -> realRoot

func realRootOf(rootAbs string) string {
	if v, ok := rootRealCache.Load(rootAbs); ok {
		return v.(string)
	}
	rr, err := filepath.EvalSymlinks(rootAbs)
	if err != nil {
		rr = rootAbs
	}
	rootRealCache.Store(rootAbs, rr)
	return rr
}

// relPosix mirrors python's `abs.resolve().relative_to(root.resolve()).as_posix()`:
// it canonicalizes symlinks AND on-disk case (so two differently-cased links to one
// file collapse to a single visited key, as CPython's resolve() does on Windows),
// then returns the forward-slashed repo-relative path. ok=false means abs is outside
// root (python's ValueError branch). Falls back to a lexical Rel when a path cannot
// be resolved (e.g. it does not exist).
func relPosix(rootAbs, abs string) (string, bool) {
	base, target := rootAbs, abs
	if ra, err := filepath.EvalSymlinks(abs); err == nil {
		base, target = realRootOf(rootAbs), ra
	}
	r, err := filepath.Rel(base, target)
	if err != nil {
		return "", false
	}
	r = filepath.ToSlash(r)
	if r == ".." || strings.HasPrefix(r, "../") {
		return "", false
	}
	return r, true
}

// ---------------------------------------------------------------------------
// Small pure helpers.
// ---------------------------------------------------------------------------

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	lines := strings.Split(s, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func splitOnce(s, sep string) string {
	if i := strings.Index(s, sep); i >= 0 {
		return s[:i]
	}
	return s
}

func hasSchemePrefix(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") ||
		strings.HasPrefix(s, "mailto:") || strings.HasPrefix(s, "#") || strings.HasPrefix(s, "tel:")
}

func hasPrefixAny(s string, prefixes ...string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

func hasSuffixAny(s string, suffixes ...string) bool {
	for _, p := range suffixes {
		if strings.HasSuffix(s, p) {
			return true
		}
	}
	return false
}

func firstNStr(s []string, n int) []string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func truncRunes(s string, n int) string {
	r := []rune(s)
	if len(r) > n {
		return string(r[:n])
	}
	return s
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func stripFrontMatter(text string) string {
	if !strings.HasPrefix(text, "---") {
		return text
	}
	idx := strings.Index(text[3:], "\n---")
	if idx == -1 {
		return text
	}
	end := 3 + idx
	return text[end+4:]
}

func clampScore(score float64) int { return mathx.ClampScore(score) }

func round1(x float64) float64 {
	f, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 1, 64), 64)
	return f
}

// ftoa mirrors python str(float): a float always carries a decimal point.
func ftoa(x float64) string {
	s := strconv.FormatFloat(x, 'f', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

func itoa(n int) string { return strconv.Itoa(n) }

func sortedStrings(m map[string]bool) []string {
	out := maputil.SortedKeys(m)
	return out
}

func sortedUnique(s []string) []string {
	m := map[string]bool{}
	for _, x := range s {
		m[x] = true
	}
	return sortedStrings(m)
}

// ss makes a non-nil slice so JSON emits [] not null.
func ss(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// ---------------------------------------------------------------------------
// Front-matter + tokenization.
// ---------------------------------------------------------------------------

func unquote(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == s[len(s)-1] && (s[0] == '"' || s[0] == '\'') {
		return s[1 : len(s)-1]
	}
	return s
}

func parseFrontMatter(text string) map[string]string {
	out := map[string]string{}
	if !strings.HasPrefix(text, "---") {
		return out
	}
	idx := strings.Index(text[3:], "\n---")
	if idx == -1 {
		return out
	}
	block := text[3 : 3+idx]
	lines := splitLines(block)
	i := 0
	for i < len(lines) {
		m := reFMKey.FindStringSubmatch(lines[i])
		if m == nil {
			i++
			continue
		}
		key := m[1]
		rest := strings.TrimSpace(m[2])
		switch rest {
		case ">", "|", ">-", "|-", ">+", "|+":
			var parts []string
			i++
			for i < len(lines) && (strings.HasPrefix(lines[i], " ") || strings.HasPrefix(lines[i], "\t") || strings.TrimSpace(lines[i]) == "") {
				parts = append(parts, strings.TrimSpace(lines[i]))
				i++
			}
			var nonempty []string
			for _, p := range parts {
				if p != "" {
					nonempty = append(nonempty, p)
				}
			}
			out[key] = strings.TrimSpace(strings.Join(nonempty, " "))
			continue
		}
		out[key] = unquote(rest)
		i++
	}
	return out
}

// degenerate reports whether a title/description is filler — a single repeated word,
// one character, or no word long enough to read. Word counting is script-agnostic on
// purpose (reWordApos spans every Unicode letter): a Latin-only class saw a correct
// Cyrillic, Arabic, or CJK title as a single word and called the whole page defective.
func degenerate(s string) bool {
	words := reWordApos.FindAllString(s, -1)
	distinct := map[string]bool{}
	hasReal := false
	for _, w := range words {
		distinct[strings.ToLower(w)] = true
		if utf8.RuneCountInString(w) >= 3 {
			hasReal = true
		}
	}
	stripped := strings.ReplaceAll(strings.TrimSpace(s), " ", "")
	chars := map[rune]bool{}
	for _, r := range stripped {
		chars[r] = true
	}
	singleChar := len(chars) <= 1
	return singleChar || !hasReal || len(distinct) < 2
}

func degenerateAlt(alt string) bool {
	words := reWordApos.FindAllString(alt, -1)
	if len(words) == 0 {
		return true
	}
	return len(words) == 1 && altFiller[strings.ToLower(words[0])]
}

func tokens(s string) map[string]bool {
	out := map[string]bool{}
	for _, m := range reToken.FindAllString(s, -1) {
		w := strings.ToLower(m)
		if !stopwords[w] {
			out[w] = true
		}
	}
	return out
}

func firstH1Text(text string) string {
	for _, raw := range splitLines(stripFrontMatter(text)) {
		line := strings.TrimSpace(raw)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimSpace(line[2:])
		}
	}
	return ""
}

func ledeTokens(text string) map[string]bool {
	body := reFence.ReplaceAllString(stripFrontMatter(text), " ")
	toks := map[string]bool{}
	for _, raw := range firstNStr(splitLines(body), 25) {
		line := strings.TrimSpace(raw)
		if line == "" || hasPrefixAny(line, "#", "|", "```", ">", "<") {
			continue
		}
		for t := range tokens(line) {
			toks[t] = true
		}
	}
	return toks
}

func isQuestion(heading string) bool {
	h := strings.TrimSpace(heading)
	return strings.HasSuffix(h, "?") || reQuestion.MatchString(h)
}

func hasProseOpener(text string) bool {
	body := stripFrontMatter(text)
	seenH1 := false
	for _, raw := range splitLines(body) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "# ") {
			seenH1 = true
			continue
		}
		if !seenH1 {
			if reHasLetter.MatchString(line) && !hasPrefixAny(line, "#", "```", "|", "<", "!", "-", "*", ">") {
				if utf8.RuneCountInString(line) > 40 || hasSuffixAny(line, ".", ":", "!") {
					return true
				}
			}
			continue
		}
		if hasPrefixAny(line, "#", "```", "|") {
			return false
		}
		if hasPrefixAny(line, "<", "!", "-", "*", ">", "1.") {
			continue
		}
		if reHasLetter.MatchString(line) && (utf8.RuneCountInString(line) > 40 || hasSuffixAny(line, ".", ":", "!")) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// robots.txt + citation + llms-full audits.
// ---------------------------------------------------------------------------

func robotsGroups(robots string) map[string][]string {
	groups := map[string][]string{}
	var pending []string
	afterDirective := false
	for _, raw := range splitLines(robots) {
		line := strings.TrimSpace(splitOnce(raw, "#"))
		if line == "" {
			pending = nil
			afterDirective = false
			continue
		}
		if m := reUserAgent.FindStringSubmatch(line); m != nil {
			if afterDirective {
				pending = nil
				afterDirective = false
			}
			ua := m[1]
			pending = append(pending, ua)
			if _, ok := groups[ua]; !ok {
				groups[ua] = []string{}
			}
			continue
		}
		afterDirective = true
		for _, ua := range pending {
			groups[ua] = append(groups[ua], line)
		}
	}
	return groups
}

func aiCrawlersOK(robots string) (bool, string) {
	if strings.TrimSpace(robots) == "" {
		return false, "robots.txt missing — no welcome for answer-engine crawlers"
	}
	groups := robotsGroups(robots)
	var blocked []string
	for ua, lines := range groups {
		if !(aiCrawlerUAs[ua] || ua == "*") {
			continue
		}
		for _, line := range lines {
			if reDisallow.MatchString(line) {
				blocked = append(blocked, ua)
				break
			}
		}
	}
	sort.Strings(blocked)
	if len(blocked) > 0 {
		return false, fmt.Sprintf("robots.txt Disallows answer-engine crawler(s): %s (delists the project from that engine's citations)", strings.Join(blocked, ", "))
	}
	var missing []string
	for _, ua := range aiCrawlerRequired {
		if _, ok := groups[ua]; !ok {
			missing = append(missing, ua)
		}
	}
	if len(missing) > 0 {
		return false, fmt.Sprintf("robots.txt does not explicitly welcome %d answer-engine crawler(s): %s (name + Allow each for AEO)", len(missing), strings.Join(missing, ", "))
	}
	required := map[string]bool{}
	for _, ua := range aiCrawlerRequired {
		required[ua] = true
	}
	var bonus []string
	for ua := range groups {
		if aiCrawlerUAs[ua] && !required[ua] {
			bonus = append(bonus, ua)
		}
	}
	sort.Strings(bonus)
	extra := ""
	if len(bonus) > 0 {
		extra = fmt.Sprintf(" (+%d more named)", len(bonus))
	}
	return true, fmt.Sprintf("robots.txt explicitly welcomes all %d major answer-engine crawlers%s", len(aiCrawlerRequired), extra)
}

func llmsFullSourceAudit(rootAbs string) LLMSFullSources {
	llms := safeRead(rjoin(rootAbs, llmsRel))
	llmsFull := safeRead(rjoin(rootAbs, llmsFullRel))
	var targets []string
	seen := map[string]bool{}
	for _, m := range reLink.FindAllStringSubmatch(llms, -1) {
		target := strings.TrimSpace(m[2])
		if hasSchemePrefix(target) {
			continue
		}
		rel := strings.TrimSpace(splitOnce(splitOnce(target, "#"), "?"))
		if !strings.HasSuffix(rel, ".md") || seen[rel] {
			continue
		}
		seen[rel] = true
		targets = append(targets, strings.ReplaceAll(rel, "\\", "/"))
	}
	sourcesSet := map[string]bool{}
	for _, m := range reSource.FindAllStringSubmatch(llmsFull, -1) {
		sourcesSet[m[1]] = true
	}
	var missing []string
	for _, t := range targets {
		if !sourcesSet[t] {
			missing = append(missing, t)
		}
	}
	return LLMSFullSources{Targets: ss(targets), Sources: sortedStrings(sourcesSet), Missing: ss(missing)}
}

func citationLinkAudit(rootAbs string) Citation {
	localDead := func(text string) []string {
		var out []string
		for _, m := range reLink.FindAllStringSubmatch(text, -1) {
			u := strings.TrimSpace(m[2])
			if hasSchemePrefix(u) {
				continue
			}
			pp := strings.TrimSpace(splitOnce(splitOnce(u, "#"), "?"))
			if pp == "" {
				continue
			}
			if !existsPath(resolveLink(rootAbs, rootAbs, pp)) {
				out = append(out, pp)
			}
		}
		return out
	}
	deadMap := sortedUnique(localDead(safeRead(rjoin(rootAbs, llmsRel))))
	deadSelf := map[string]bool{}
	for _, rel := range enumeratePages(rootAbs, "published") {
		txt := reFence.ReplaceAllString(safeRead(rjoin(rootAbs, rel)), " ")
		for _, m := range reSelfRepo.FindAllStringSubmatch(txt, -1) {
			p := strings.TrimRight(m[1], "/")
			if !existsPath(rjoin(rootAbs, p)) {
				deadSelf[rel+" -> "+p] = true
			}
		}
	}
	llmsFull := safeRead(rjoin(rootAbs, llmsFullRel))
	return Citation{
		DeadMap:            ss(deadMap),
		DeadSelf:           sortedStrings(deadSelf),
		LLMSFullUnresolved: len(localDead(llmsFull)),
	}
}

// ---------------------------------------------------------------------------
// Site-level checks.
// ---------------------------------------------------------------------------

func siteChecks(rootAbs string) Site {
	config := safeRead(rjoin(rootAbs, configRel))
	robots := safeRead(rjoin(rootAbs, robotsRel))
	head := safeRead(rjoin(rootAbs, headIncludeRel))
	index := safeRead(rjoin(rootAbs, indexRel))
	showcase := safeRead(rjoin(rootAbs, showcaseRel))
	llms := safeRead(rjoin(rootAbs, llmsRel))
	faq := safeRead(rjoin(rootAbs, faqRel))

	var checks []SiteCheck
	add := func(name string, ok, hard bool, detailOK, detailBad string) {
		c := SiteCheck{Name: name, OK: ok, Hard: hard}
		if ok {
			c.Detail = detailOK
		} else {
			c.Detail = detailBad
			d := detailBad
			c.Defect = &d
		}
		checks = append(checks, c)
	}

	robotsOK := robots != "" && strings.Contains(robots, "Allow: /") && strings.Contains(robots, "Sitemap:") && !strings.Contains(robots, "Disallow: /\n")
	add("robots", robotsOK, true, "robots.txt allows crawl + names sitemap", "robots.txt missing / blocks crawl / names no Sitemap:")

	aiOK, aiDetail := aiCrawlersOK(robots)
	add("ai_crawlers", aiOK, true, aiDetail, aiDetail)

	add("sitemap_plugin", strings.Contains(config, "jekyll-sitemap"), true, "jekyll-sitemap enabled (auto /sitemap.xml)", "jekyll-sitemap not in docs/_config.yml plugins")
	add("seo_tag_plugin", strings.Contains(config, "jekyll-seo-tag"), true, "jekyll-seo-tag enabled (canonical/OG/Twitter)", "jekyll-seo-tag not in docs/_config.yml plugins")

	add("canonical_url", reURLLine.MatchString(config), true, "site url set (canonical + absolute sitemap URLs resolve)", "no url: in docs/_config.yml (canonical/sitemap URLs break)")

	ogMatch := reImage.FindStringSubmatch(config)
	ogFileOK := false
	if ogMatch != nil {
		url := ogMatch[1]
		if m := reMainPath.FindStringSubmatch(url); m != nil {
			ogFileOK = existsPath(rjoin(rootAbs, m[1]))
		} else {
			ogFileOK = existsPath(rjoin(rootAbs, strings.TrimLeft(url, "/")))
		}
	}
	add("og_image", ogMatch != nil && ogFileOK, true, "Open Graph social image declared and present", "no og:image default in _config.yml or the image file is missing")

	blob := strings.Join([]string{head, index, showcase, faq}, "\n")
	var jsonldValues []any
	presentTypes := map[string]bool{}
	invalidBlocks := 0
	for _, m := range reJSONLDBlock.FindAllStringSubmatch(blob, -1) {
		body := strings.TrimSpace(m[1])
		var data any
		if err := json.Unmarshal([]byte(body), &data); err != nil {
			invalidBlocks++
			continue
		}
		jsonldValues = append(jsonldValues, data)
		for t := range collectJSONLDTypes(data) {
			presentTypes[t] = true
		}
	}
	add("jsonld_valid", invalidBlocks == 0, true, "every JSON-LD block parses as valid JSON", fmt.Sprintf("%d invalid JSON-LD block(s) — answer engines reject malformed JSON-LD", invalidBlocks))
	for _, t := range jsonldTypesHard {
		add("jsonld_"+t, presentTypes[t], true, fmt.Sprintf("JSON-LD %s present", t), fmt.Sprintf("no JSON-LD %s (answer engines can't identify/cite the project)", t))
	}
	for _, t := range jsonldTypesSoft {
		add("jsonld_"+t, presentTypes[t], false, fmt.Sprintf("JSON-LD %s present (bonus)", t), fmt.Sprintf("no JSON-LD %s (optional, a citation bonus)", t))
	}

	breadcrumbOK, breadcrumbDetail := breadcrumbShapeOK(jsonldValues)
	add("breadcrumb_jsonld_shape", breadcrumbOK, true, breadcrumbDetail, breadcrumbDetail)

	add("llms_txt", llms != "" && reKeyFacts.MatchString(llms), true, "llms.txt present with a Key-facts block", "llms.txt missing or has no 'Key facts' anchor block")

	llmsFull := safeRead(rjoin(rootAbs, llmsFullRel))
	if llmsFull == "" {
		add("llms_full", false, true, "", "llms-full.txt missing (no one-fetch corpus; run tools/gen_llms_full.py)")
	} else {
		fresh := strings.TrimSpace(llms) == "" || strings.Contains(llmsFull, strings.TrimSpace(llms))
		add("llms_full", fresh, true, "llms-full.txt present and fresh (inlines current llms.txt)", "llms-full.txt is STALE (does not contain current llms.txt; re-run tools/gen_llms_full.py)")
	}
	llmsSources := llmsFullSourceAudit(rootAbs)
	missingSources := llmsSources.Missing
	ell := ""
	if len(missingSources) > 3 {
		ell = "..."
	}
	add("llms_full_sources", len(missingSources) == 0, true,
		fmt.Sprintf("llms-full.txt includes all %d llms.txt source documents", len(llmsSources.Targets)),
		fmt.Sprintf("llms-full.txt misses %d llms.txt source document(s): %s%s", len(missingSources), strings.Join(firstNStr(missingSources, 3), ", "), ell))

	q := 0
	for _, h := range reH2.FindAllStringSubmatch(faq, -1) {
		if isQuestion(h[1]) {
			q++
		}
	}
	add("faq_structured", q >= minFAQQuestions, true,
		fmt.Sprintf("FAQ.md has %d question sections (seeds FAQPage)", q),
		fmt.Sprintf("FAQ.md missing or thin (%d question H2s; need >= %d)", q, minFAQQuestions))
	faqSyncOK, faqSyncDetail := faqJSONLDSyncOK(jsonldValues, faq)
	add("faq_jsonld_sync", faqSyncOK, true, faqSyncDetail, faqSyncDetail)

	cit := citationLinkAudit(rootAbs)
	nDead := len(cit.DeadMap) + len(cit.DeadSelf)
	add("citation_links", nDead == 0, true,
		"every llms.txt-map + self-repo github link resolves to a live target",
		fmt.Sprintf("%d citation link(s) dead — a stale link sends an answer engine/reader to a 404 (see corpus.citation)", nDead))
	add("llms_full_navigable", cit.LLMSFullUnresolved == 0, true,
		"llms-full.txt inlined links all resolve in the flat corpus",
		fmt.Sprintf("%d inlined link(s) in llms-full.txt don't resolve in the flat one-fetch corpus (fix = gen_llms_full.py rewrites inlined relative links to absolute, then regenerate)", cit.LLMSFullUnresolved))

	var hardDefects, soft []string
	nOK := 0
	for _, c := range checks {
		if c.OK {
			nOK++
			continue
		}
		if c.Hard {
			hardDefects = append(hardDefects, *c.Defect)
		} else {
			soft = append(soft, *c.Defect)
		}
	}
	score := round1(100 * float64(nOK) / float64(maxInt(1, len(checks))))
	siteCred := 0
	creditDetail := map[string]int{}
	if len(hardDefects) == 0 {
		siteCred, creditDetail = siteCredit(jsonldValues, robots, q, faqSyncOK)
	}
	return Site{
		Checks:          checks,
		Score:           score,
		ScoreWithCredit: round1(score + float64(siteCred)),
		Credit:          siteCred,
		CreditDetail:    creditDetail,
		NOK:             nOK,
		NTotal:          len(checks),
		Defects:         ss(hardDefects),
		Soft:            ss(soft),
		PresentJSONLD:   sortedStrings(presentTypes),
		Citation:        cit,
		LLMSFullSources: llmsSources,
	}
}

// ---------------------------------------------------------------------------
// Published-set enumeration.
// ---------------------------------------------------------------------------

func configExcludes(rootAbs string) (map[string]bool, map[string]bool) {
	cfg := safeRead(rjoin(rootAbs, configRel))
	dirs := map[string]bool{}
	files := map[string]bool{}
	if m := reExclude.FindStringSubmatch(cfg); m != nil {
		for _, line := range strings.Split(m[1], "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "-") {
				continue
			}
			entry := strings.Trim(strings.Trim(strings.TrimSpace(line[1:]), "\""), "'")
			if entry == "" || strings.Contains(entry, "*") {
				continue
			}
			if strings.HasSuffix(entry, "/") {
				dirs[strings.TrimRight(entry, "/")] = true
			} else if strings.HasSuffix(entry, ".md") {
				files[entry] = true
			}
		}
	}
	return dirs, files
}

func published(rootAbs, rel string) bool {
	if strings.ToLower(path.Ext(rel)) != ".md" {
		return false
	}
	parts := strings.Split(rel, "/")
	if len(parts) == 0 || parts[0] != "docs" {
		return false
	}
	seg := parts[1:]
	cfgDirs, cfgFiles := configExcludes(rootAbs)
	for _, s := range seg {
		if nonpublishedDirs[s] || cfgDirs[s] {
			return false
		}
	}
	return !cfgFiles[strings.Join(seg, "/")]
}

func discovery(rootAbs, rel string) bool {
	if !published(rootAbs, rel) {
		return false
	}
	for _, s := range strings.Split(rel, "/")[1:] {
		if evidenceDirs[s] {
			return false
		}
	}
	return true
}

func reachablePublished(rootAbs string) map[string]bool {
	var queue []string
	for _, s := range frontDoors {
		abs := rjoin(rootAbs, s)
		if existsPath(abs) {
			queue = append(queue, abs)
		}
	}
	visited := map[string]bool{}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		rel, ok := relPosix(rootAbs, cur)
		if !ok {
			continue
		}
		if visited[rel] {
			continue
		}
		visited[rel] = true
		txt, ok := readText(cur)
		if !ok {
			continue
		}
		for _, m := range reLink.FindAllStringSubmatch(txt, -1) {
			t := strings.TrimSpace(m[2])
			if hasSchemePrefix(t) {
				continue
			}
			pp := splitOnce(splitOnce(t, "#"), "?")
			if !strings.HasSuffix(pp, ".md") {
				continue
			}
			var nxt string
			if strings.HasPrefix(pp, "/") {
				nxt = rjoin(rootAbs, strings.TrimLeft(pp, "/"))
			} else {
				nxt = filepath.Join(filepath.Dir(cur), filepath.FromSlash(pp))
			}
			if existsPath(nxt) {
				queue = append(queue, nxt)
			}
		}
	}
	return visited
}

func docsMarkdownRels(rootAbs string) []string {
	var rels []string
	docsDir := filepath.Join(rootAbs, "docs")
	_ = walkfiles.Files(docsDir, func(p string, d fs.DirEntry) error {
		if !strings.HasSuffix(d.Name(), ".md") {
			return nil
		}
		if rel, ok := relPosix(rootAbs, p); ok {
			rels = append(rels, rel)
		}
		return nil
	})
	sort.Strings(rels)
	return rels
}

func discoveryOrphans(rootAbs string) []string {
	reach := reachablePublished(rootAbs)
	var out []string
	for _, rel := range docsMarkdownRels(rootAbs) {
		if discovery(rootAbs, rel) && !reach[rel] {
			out = append(out, rel)
		}
	}
	sort.Strings(out)
	return out
}

func enumeratePages(rootAbs, scope string) []string {
	switch scope {
	case "core":
		reach := reachablePublished(rootAbs)
		var out []string
		for r := range reach {
			if discovery(rootAbs, r) {
				out = append(out, r)
			}
		}
		sort.Strings(out)
		return out
	case "published":
		var out []string
		for _, rel := range docsMarkdownRels(rootAbs) {
			if published(rootAbs, rel) {
				out = append(out, rel)
			}
		}
		return out
	}
	return nil
}

// ---------------------------------------------------------------------------
// Per-page fold + corpus meta-distinct.
// ---------------------------------------------------------------------------

func gradeLetter(score float64) string {
	switch {
	case score >= 103:
		return "S"
	case score >= 101:
		return "A+"
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

func scorePage(text, docRel, rootAbs string) Page {
	fm := parseFrontMatter(text)
	kpis := []kpi{
		kpiTitle(fm),
		kpiDescription(fm),
		kpiHeadings(text),
		kpiLinks(text, rootAbs, docRel),
		kpiLinksCrawlable(text, rootAbs, docRel),
		kpiAnswerability(text),
		kpiAltText(text),
	}
	byName := map[string]kpi{}
	for _, k := range kpis {
		byName[k.name] = k
	}
	baseline := 0.0
	for _, kw := range kpiWeightsOrder {
		baseline += kw.w * float64(byName[kw.name].score)
	}
	baseline = round1(baseline)
	spotless := true
	for _, kw := range kpiWeightsOrder {
		if byName[kw.name].score != 100 {
			spotless = false
			break
		}
	}
	credit := 0
	creditDetail := map[string]any{}
	if spotless {
		credit, creditDetail = pageCredit(text, docRel, rootAbs, fm)
	}
	score := round1(baseline + float64(credit))
	defects := []string{}
	soft := []string{}
	for _, k := range kpis {
		for _, d := range k.defects {
			defects = append(defects, k.name+": "+d)
		}
	}
	for _, k := range kpis {
		for _, s := range k.soft {
			soft = append(soft, k.name+": "+s)
		}
	}
	kpisMap := map[string]int{}
	kpiDetail := map[string]string{}
	for _, k := range kpis {
		kpisMap[k.name] = k.score
		kpiDetail[k.name] = k.detail
	}
	return Page{
		Path:         docRel,
		Score:        score,
		Baseline:     baseline,
		Credit:       credit,
		CreditDetail: creditDetail,
		Grade:        gradeLetter(score),
		KPIs:         kpisMap,
		KPIDetail:    kpiDetail,
		Meta:         PageMeta{Title: strings.TrimSpace(fm["title"]), Description: strings.TrimSpace(fm["description"])},
		Defects:      defects,
		Soft:         soft,
		NDefects:     len(defects),
	}
}

func missingPageEntry(docRel string) Page {
	kpisMap := map[string]int{}
	for _, kw := range kpiWeightsOrder {
		kpisMap[kw.name] = 0
	}
	return Page{
		Path:         docRel,
		Score:        0.0,
		Baseline:     0.0,
		Credit:       0,
		CreditDetail: map[string]any{},
		Grade:        "F",
		KPIs:         kpisMap,
		KPIDetail:    map[string]string{},
		Meta:         PageMeta{},
		Defects:      []string{fmt.Sprintf("missing: core page %s does not exist on disk", docRel)},
		Soft:         []string{},
		NDefects:     1,
	}
}

func applyCorpusMetaDistinct(pages []Page) int {
	added := 0
	for _, field := range []string{"title", "description"} {
		groups := map[string][]int{}
		var order []string
		for i := range pages {
			var v string
			if field == "title" {
				v = pages[i].Meta.Title
			} else {
				v = pages[i].Meta.Description
			}
			v = strings.ToLower(strings.TrimSpace(v))
			if v == "" {
				continue
			}
			if _, ok := groups[v]; !ok {
				order = append(order, v)
			}
			groups[v] = append(groups[v], i)
		}
		for _, val := range order {
			idxs := groups[val]
			if len(idxs) < 2 {
				continue
			}
			paths := make([]string, len(idxs))
			for j, i := range idxs {
				paths[j] = pages[i].Path
			}
			for _, i := range idxs {
				var peers []string
				for _, p := range paths {
					if p != pages[i].Path {
						peers = append(peers, p)
					}
				}
				shown := strings.Join(firstNStr(peers, 3), ", ")
				if len(peers) > 3 {
					shown += "…"
				}
				pages[i].Defects = append(pages[i].Defects,
					fmt.Sprintf("meta_distinct: %s is not unique — duplicates %s (search can't tell the pages apart)", field, shown))
				pages[i].NDefects = len(pages[i].Defects)
				added++
			}
		}
	}
	return added
}

// ---------------------------------------------------------------------------
// build_payload.
// ---------------------------------------------------------------------------

func buildPayload(workspace string, pages []Page, site Site, scope string, orphans []string) Payload {
	orphans = ss(orphans)
	n := len(pages)
	var sumScore float64
	pageDefects := 0
	for _, d := range pages {
		sumScore += d.Score
		pageDefects += d.NDefects
	}
	siteDefects := len(site.Defects)
	totalDefects := pageDefects + siteDefects
	meanScore := round1(sumScore / float64(maxInt(1, n)))

	grades := map[string]int{"S": 0, "A+": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
	for _, d := range pages {
		grades[d.Grade]++
	}

	worstPages := append([]Page(nil), pages...)
	sort.SliceStable(worstPages, func(a, b int) bool {
		if worstPages[a].Score != worstPages[b].Score {
			return worstPages[a].Score < worstPages[b].Score
		}
		return worstPages[a].NDefects > worstPages[b].NDefects
	})
	worstPages = firstNPage(worstPages, 8)
	worst := make([]WorstRow, 0, len(worstPages))
	for _, d := range worstPages {
		worst = append(worst, WorstRow{Path: d.Path, Score: d.Score, Grade: d.Grade, NDefects: d.NDefects})
	}

	fullMeta := 0
	for _, d := range pages {
		if d.KPIs["title"] > 0 && d.KPIs["description"] > 0 {
			fullMeta++
		}
	}
	metaPct := round1(100 * float64(fullMeta) / float64(maxInt(1, n)))

	pageCreditTotal := 0
	for _, d := range pages {
		pageCreditTotal += d.Credit
	}
	siteCred := site.Credit
	siteHeadline := round1(site.Score + float64(siteCred))
	overall := round1(0.5*meanScore + 0.5*siteHeadline)

	crawl404 := 0
	metaDuplicates := 0
	for _, d := range pages {
		for _, x := range d.Defects {
			if strings.Contains(x, "links_crawlable: crawl-404") {
				crawl404++
			}
			if strings.HasPrefix(x, "meta_distinct:") {
				metaDuplicates++
			}
		}
	}
	citationDead := len(site.Citation.DeadMap) + len(site.Citation.DeadSelf)

	var medianScore, minScore float64
	if n > 0 {
		sc := make([]float64, n)
		for i, d := range pages {
			sc[i] = d.Score
		}
		sort.Float64s(sc)
		medianScore = round1(sc[n/2])
		minScore = round1(sc[0])
	}

	corpus := Corpus{
		NPages:              n,
		OverallScore:        overall,
		Grade:               gradeLetter(overall),
		PageMeanScore:       meanScore,
		SiteScore:           site.Score,
		SiteScoreWithCredit: siteHeadline,
		ExcellenceCredit:    ExcellenceCredit{Page: pageCreditTotal, Site: siteCred, Total: pageCreditTotal + siteCred},
		SiteChecksOK:        fmt.Sprintf("%d/%d", site.NOK, site.NTotal),
		MetaCoveragePct:     metaPct,
		MedianScore:         medianScore,
		MinScore:            minScore,
		GradeDistribution:   grades,
		SEODebt:             totalDefects,
		SEODebtInPages:      pageDefects,
		SEODebtInSite:       siteDefects,
		Crawl404:            crawl404,
		MetaDuplicates:      metaDuplicates,
		CitationDead:        citationDead,
		LLMSFullUnresolved:  site.Citation.LLMSFullUnresolved,
		PresentJSONLD:       site.PresentJSONLD,
		DiscoveryOrphans:    len(orphans),
		OrphanPages:         orphans,
		Worst:               worst,
	}

	var ok bool
	var verdict, finding, reason, nextAction string
	if totalDefects == 0 {
		ok, verdict, finding = true, "OK", "discoverable"
		reason = fmt.Sprintf("discoverability clean: %d pages, overall %s (grade %s), meta coverage %s%%, site %d/%d, zero seo-debt",
			n, ftoa(overall), gradeLetter(overall), ftoa(metaPct), site.NOK, site.NTotal)
		nextAction = "no required edit; re-run after the next docs/site change"
	} else {
		ok, verdict, finding = false, "ACTION", "seo_debt"
		reason = fmt.Sprintf("%d unit(s) of seo-debt across %d pages (%d in-page + %d site); overall %s, meta coverage %s%%, site %d/%d",
			totalDefects, n, pageDefects, siteDefects, ftoa(overall), ftoa(metaPct), site.NOK, site.NTotal)
		nextAction = "retire seo-debt worst-first (corpus.worst + site defects): add missing " +
			"front-matter title/description, JSON-LD types, llms-full.txt; repoint a " +
			"crawl-404 link to a published page or an absolute URL; dedup a non-unique " +
			"title/description; fix a dead citation link; re-run to prove the drop"
	}

	return Payload{
		Schema:     SCHEMA,
		OK:         ok,
		Verdict:    verdict,
		Finding:    finding,
		Reason:     reason,
		NextAction: nextAction,
		Workspace:  workspace,
		Scope:      scope,
		Corpus:     corpus,
		Site:       site,
		Pages:      pages,
	}
}

func firstNPage(s []Page, n int) []Page {
	if len(s) > n {
		return s[:n]
	}
	return s
}
