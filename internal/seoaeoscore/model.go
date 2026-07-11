package seoaeoscore

import "regexp"

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
	reWordApos   = regexp.MustCompile(`[A-Za-z][A-Za-z'’-]+`)
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
