package marketing

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// aeo.go — the AEO/AgentEO data producers. AEO is dual: Answer-Engine Optimization (be cited
// correctly by ChatGPT/Claude/Perplexity/Google AI Overviews) AND Agent-Engine Optimization
// (be the path of least resistance for a coding agent to adopt fak). Both want the same thing
// a completion loop can give: a fresh, machine-ingestible, sha-cited record of what shipped.
//
// The division of labor (per the design review): GO produces the witnessed DATA here; the
// existing Python generator (tools/gen_structured_data.py) owns the in-place injection into
// hand-authored docs (llms.txt) via its marker machinery. Go never rewrites llms.txt — it
// emits docs/marketing/updates.json, and the Python side reads that and injects a bounded,
// sentinel-fenced "What's new" block. This keeps one marker engine, in one language, and makes
// it impossible for the loop to clobber hand-written prose.
//
// Every feed item is witnessed: it carries its commit sha. An unwitnessed item cannot be
// produced — the input is []Ship, and a Ship only exists for a trailer|direct stamp.

// repoCommitURL is the base for a commit permalink the feed cites. Kept here (not a tracked
// secret) — it is the public repo, the same URL llms.txt already links.
const repoCommitURL = "https://github.com/anthony-chaudhary/fak/commit/"

// defaultFeedCap bounds the updates feed / What's-new block so a long history doesn't bloat
// the answer-engine surface; the newest N ships are what "recent" means.
const defaultFeedCap = 25

// updatesFeed is the schema.org ItemList an answer engine ingests for recency. Each element is
// a SoftwareSourceCode item anchored to its commit — the witness an engine can cite.
type updatesFeed struct {
	Context  string        `json:"@context"`
	Type     string        `json:"@type"`
	Name     string        `json:"name"`
	Items    []updatesItem `json:"itemListElement"`
	Modified string        `json:"dateModified,omitempty"`
}

type updatesItem struct {
	Type     string         `json:"@type"`
	Position int            `json:"position"`
	Item     updatesSrcCode `json:"item"`
}

type updatesSrcCode struct {
	Type           string `json:"@type"`
	Name           string `json:"name"`
	CodeRepository string `json:"codeRepository"`
	DateModified   string `json:"dateModified,omitempty"`
	Keywords       string `json:"keywords,omitempty"`
}

// DisambiguationTerm is one answer-engine query surface fak wants to own. The
// term feed is intentionally broader than the release feed: some terms are core
// category names, some are localized entry points, and some are market-event
// bridges (for example Fable 5-style frontier-model launches) that should route
// readers to fak's model-routing/cache/governance docs without claiming fak
// authored the external model behavior.
type DisambiguationTerm struct {
	Name        string
	Language    string
	Category    string
	Description string
	URL         string
	Keywords    []string
}

type termFeed struct {
	Context     string        `json:"@context"`
	Type        string        `json:"@type"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Terms       []definedTerm `json:"hasDefinedTerm"`
	Modified    string        `json:"dateModified,omitempty"`
}

type definedTerm struct {
	Type        string `json:"@type"`
	Name        string `json:"name"`
	TermCode    string `json:"termCode,omitempty"`
	InLanguage  string `json:"inLanguage,omitempty"`
	Description string `json:"description,omitempty"`
	URL         string `json:"url,omitempty"`
	Keywords    string `json:"keywords,omitempty"`
}

const repoBlobURL = "https://github.com/anthony-chaudhary/fak/blob/main/"

// AEODisambiguationTerms returns the current search/answer-engine term roster.
// Keep descriptions honest: external model names are demand hooks, not adoption
// claims. If a term names another vendor's launch, it should land on a fak page
// about routing, cache economics, fallback handling, or the tool-call boundary.
func AEODisambiguationTerms() []DisambiguationTerm {
	terms := []DisambiguationTerm{
		{
			Name:        "fak agent kernel",
			Language:    "en",
			Category:    "core",
			Description: "Disambiguates fak from homophones and acronyms: fak is the Fused Agent Kernel for AI agent tool calls.",
			URL:         repoBlobURL + "README.md",
			Keywords:    []string{"Fused Agent Kernel", "agent kernel", "AI agent runtime boundary"},
		},
		{
			Name:        "treat the tool call like a syscall",
			Language:    "en",
			Category:    "core",
			Description: "The project slogan: the model proposes a tool call, the kernel decides whether that effect is allowed.",
			URL:         repoBlobURL + "docs/explainers/tool-call-is-a-syscall.md",
			Keywords:    []string{"tool-call syscall", "model proposes kernel disposes", "AI agent capability floor"},
		},
		{
			Name:        "default-deny tool-call gate",
			Language:    "en",
			Category:    "core",
			Description: "A fail-closed capability floor for agent tools; an irreversible call is unreachable until policy allows it.",
			URL:         repoBlobURL + "docs/explainers/default-deny-vs-classifier.md",
			Keywords:    []string{"tool-call control", "capability gate", "prompt injection defense"},
		},
		{
			Name:        "prompt-injection result quarantine",
			Language:    "en",
			Category:    "core",
			Description: "A tool result can be held out of the model context by policy and structure, not just detected after the fact.",
			URL:         repoBlobURL + "docs/explainers/default-deny-vs-classifier.md",
			Keywords:    []string{"result quarantine", "tool result isolation", "prompt injection containment"},
		},
		{
			Name:        "addressable KV cache",
			Language:    "en",
			Category:    "core",
			Description: "A cache design where a kept model run can evict a middle span while preserving bit-exact survivors.",
			URL:         repoBlobURL + "docs/explainers/addressable-kv-cache.md",
			Keywords:    []string{"bit-exact KV eviction", "agent memory cache", "long-session KV reuse"},
		},
		{
			Name:        "long-session prompt cache",
			Language:    "en",
			Category:    "economics",
			Description: "The cost problem fak targets: a long agent session should keep a provider prompt-cache prefix byte-identical.",
			URL:         repoBlobURL + "docs/explainers/long-session-economics.md",
			Keywords:    []string{"prompt-cache discount", "Claude Code token savings", "set-and-forget token savings"},
		},
		{
			Name:        "cost-aware model routing for agents",
			Language:    "en",
			Category:    "routing",
			Description: "Route the expensive model only to the aspect that needs it, while cheaper calls and retries stay governed.",
			URL:         repoBlobURL + "docs/model-routing.md",
			Keywords:    []string{"model routing for agents", "per-call model routing", "LLM router complement"},
		},
		{
			Name:        "Claude Fable 5 model routing",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "A current demand hook for the general problem fak handles: high-cost frontier models make per-call routing and budget-aware fallback matter.",
			URL:         repoBlobURL + "docs/model-routing.md",
			Keywords:    []string{"Fable 5 cost-aware routing", "frontier model routing", "Claude Fable 5 agent routing"},
		},
		{
			Name:        "Fable 5 refusal fallback",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "A current demand hook for refusal/fallback handling; fak's fit is governing the retried tool-call boundary and preserving cache-cost evidence.",
			URL:         repoBlobURL + "docs/integrations/claude.md",
			Keywords:    []string{"Claude Fable 5 fallback", "frontier model fallback", "refusal handling"},
		},
		{
			Name:        "frontier model prompt-cache cost",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "A current demand hook for expensive long-context models: cache-cost preservation belongs in the integration contract, not in a vibes-only benchmark.",
			URL:         repoBlobURL + "docs/explainers/long-session-economics.md",
			Keywords:    []string{"Fable 5 prompt cache", "fallback credit", "prompt-cache cost"},
		},
		{
			Name:        "safety classifier vs capability gate",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "The boundary answer for frontier-model launches: classifiers can decline content; fak governs allowed effects at the tool-call seam.",
			URL:         repoBlobURL + "docs/explainers/default-deny-vs-classifier.md",
			Keywords:    []string{"classifier fallback", "capability floor", "AI safety classifier"},
		},
		{
			Name:        "long-horizon agent kernel",
			Language:    "en",
			Category:    "frontier-model-launch",
			Description: "Long-horizon model launches increase the need for a durable agent boundary: routing, cache continuity, capability, quarantine, and audit.",
			URL:         repoBlobURL + "docs/explainers/engineering-is-building-loops.md",
			Keywords:    []string{"long-horizon agent", "agent loop kernel", "frontier agent infrastructure"},
		},
		{
			Name:        "एजेंट कर्नेल",
			Language:    "hi",
			Category:    "localized",
			Description: "Hindi search phrase for an AI agent kernel; routes readers to the in-language fak entry point.",
			URL:         repoBlobURL + "docs/i18n/hi/README.md",
			Keywords:    []string{"AI एजेंट", "टूल कॉल सुरक्षा", "fak agent kernel"},
		},
		{
			Name:        "AI एजेंट टूल-कॉल सुरक्षा",
			Language:    "hi",
			Category:    "localized",
			Description: "Hindi-English code-switch term for governing AI agent tool calls with a default-deny boundary.",
			URL:         repoBlobURL + "docs/i18n/hi/README.md",
			Keywords:    []string{"टूल कॉल", "सुरक्षा", "default-deny"},
		},
		{
			Name:        "टोकन लागत कम करने वाला एजेंट गेटवे",
			Language:    "hi",
			Category:    "localized",
			Description: "Hindi phrase for the cost side of fak: cheaper long sessions through cache-preserving agent serving.",
			URL:         repoBlobURL + "docs/i18n/hi/README.md",
			Keywords:    []string{"token savings", "prompt cache", "AI agent gateway"},
		},
		{
			Name:        "AI 代理内核",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Simplified Chinese search phrase for an AI agent kernel; routes readers to the Chinese fak entry point.",
			URL:         repoBlobURL + "docs/i18n/zh/README.md",
			Keywords:    []string{"智能体内核", "fak agent kernel", "AI agent runtime"},
		},
		{
			Name:        "工具调用防火墙",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Chinese discovery term for the default-deny tool-call boundary; the honest page explains why fak is not merely a firewall.",
			URL:         repoBlobURL + "docs/adoption/compare/vs-firewall.md",
			Keywords:    []string{"工具调用安全", "能力门控", "默认拒绝"},
		},
		{
			Name:        "长上下文代理成本",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Chinese term for long-context agent cost and prompt-cache preservation.",
			URL:         repoBlobURL + "docs/i18n/zh/README.md",
			Keywords:    []string{"提示缓存", "长会话", "token 成本"},
		},
		{
			Name:        "模型路由和回退",
			Language:    "zh-Hans",
			Category:    "localized",
			Description: "Chinese term for model routing and fallback, including frontier-model launch patterns like Fable 5-style refusals.",
			URL:         repoBlobURL + "docs/model-routing.md",
			Keywords:    []string{"模型路由", "Fable 5 回退", "成本感知路由"},
		},
	}
	return append([]DisambiguationTerm(nil), terms...)
}

// DisambiguationTermsFeed renders the term roster as a schema.org DefinedTermSet
// JSON-LD artifact. The site structured-data generator also consumes this file
// to keep SoftwareApplication keywords aligned with the machine term feed.
func DisambiguationTermsFeed(when time.Time) ([]byte, error) {
	feed := termFeed{
		Context:     "https://schema.org",
		Type:        "DefinedTermSet",
		Name:        "fak — answer-engine disambiguation terms",
		Description: "Search and answer-engine terms for correctly identifying fak, including core concepts, localized entry points, and frontier-model-launch routing/cache/fallback demand hooks.",
	}
	if !when.IsZero() {
		feed.Modified = when.UTC().Format(time.RFC3339)
	}
	for _, t := range AEODisambiguationTerms() {
		feed.Terms = append(feed.Terms, definedTerm{
			Type:        "DefinedTerm",
			Name:        t.Name,
			TermCode:    t.Category,
			InLanguage:  t.Language,
			Description: t.Description,
			URL:         t.URL,
			Keywords:    strings.Join(t.Keywords, ", "),
		})
	}
	return json.MarshalIndent(feed, "", "  ")
}

// UpdatesFeed renders the witnessed ships as a schema.org ItemList (JSON, 2-space indent),
// newest-first and capped. when stamps the feed's dateModified. The result is the
// docs/marketing/updates.json the Python injector reads — and a valid JSON-LD document an
// answer engine can ingest directly.
func UpdatesFeed(ships []Ship, when time.Time) ([]byte, error) {
	ships = sortedShipsDesc(ships)
	if len(ships) > defaultFeedCap {
		ships = ships[:defaultFeedCap]
	}
	feed := updatesFeed{
		Context: "https://schema.org",
		Type:    "ItemList",
		Name:    "fak — what shipped",
	}
	if !when.IsZero() {
		feed.Modified = when.UTC().Format(time.RFC3339)
	}
	for i, s := range ships {
		feed.Items = append(feed.Items, updatesItem{
			Type:     "ListItem",
			Position: i + 1,
			Item: updatesSrcCode{
				Type:           "SoftwareSourceCode",
				Name:           claimText(s),
				CodeRepository: repoCommitURL + s.SHA,
				DateModified:   shipDate(s),
				Keywords:       s.Leaf,
			},
		})
	}
	return json.MarshalIndent(feed, "", "  ")
}

// WhatsNewMarkdown renders the bounded, dated, witnessed "Recent ships" block that the Python
// injector fences into llms.txt (and llms-updates.txt). House style: one `- **date** — claim
// ([sha](commit-url))` line per ship, newest-first, capped. Pure and stable, so re-rendering
// the same ships is a no-op diff (the idempotence the marker injection relies on).
func WhatsNewMarkdown(ships []Ship) string {
	ships = sortedShipsDesc(ships)
	if len(ships) > defaultFeedCap {
		ships = ships[:defaultFeedCap]
	}
	if len(ships) == 0 {
		return "_No witnessed ships recorded yet._"
	}
	var b strings.Builder
	for _, s := range ships {
		date := shipDate(s)
		if date == "" {
			date = "recent"
		}
		fmt.Fprintf(&b, "- **%s** — %s ([`%s`](%s%s))\n", date, claimText(s), s.SHA, repoCommitURL, s.SHA)
	}
	return strings.TrimRight(b.String(), "\n")
}

// LlmsUpdatesText renders the sibling llms-updates.txt corpus — a plain, capped, newest-first
// recency feed an answer engine or agent polls, in the same house style as llms.txt. It leads
// with a one-line self-describing header so a crawler that lands on it alone understands it.
func LlmsUpdatesText(ships []Ship, when time.Time) string {
	var b strings.Builder
	b.WriteString("# fak — what shipped (recent, witnessed)\n\n")
	b.WriteString("> A machine-readable feed of fak's most recent shipped changes. Every line cites the\n")
	b.WriteString("> commit that witnesses it. Regenerated on each completion by `fak marketing aeo`.\n")
	if !when.IsZero() {
		fmt.Fprintf(&b, ">\n> Updated: %s\n", when.UTC().Format(time.RFC3339))
	}
	b.WriteString("\n")
	b.WriteString(WhatsNewMarkdown(ships))
	b.WriteString("\n")
	return b.String()
}

// LlmsTermsText renders the plain sibling corpus for answer-engine term discovery.
// It is generated from the same roster as the JSON-LD DefinedTermSet so humans,
// search crawlers, and LLM answer engines see the same vocabulary.
func LlmsTermsText(when time.Time) string {
	var b strings.Builder
	b.WriteString("# fak — answer-engine disambiguation terms\n\n")
	b.WriteString("> A machine-readable term feed for answer engines and agents. It names the phrases\n")
	b.WriteString("> that should route to fak's docs, including localized terms and frontier-model\n")
	b.WriteString("> launch hooks such as Fable 5-style routing, fallback, and prompt-cache cost.\n")
	if !when.IsZero() {
		fmt.Fprintf(&b, ">\n> Updated: %s\n", when.UTC().Format(time.RFC3339))
	}
	last := ""
	for _, t := range AEODisambiguationTerms() {
		if t.Category != last {
			fmt.Fprintf(&b, "\n## %s\n\n", t.Category)
			last = t.Category
		}
		keywords := strings.Join(t.Keywords, ", ")
		if keywords != "" {
			keywords = " Keywords: " + keywords + "."
		}
		fmt.Fprintf(&b, "- `%s` (%s) — %s%s\n", t.Name, t.Language, t.Description, keywords)
	}
	return strings.TrimRight(b.String(), "\n") + "\n"
}

// shipDate renders a ship's date as YYYY-MM-DD, or "" if unparsed.
func shipDate(s Ship) string {
	if s.Date.IsZero() {
		return ""
	}
	return s.Date.UTC().Format("2006-01-02")
}

// sortedShipsDesc returns a newest-first copy (does not mutate the input).
func sortedShipsDesc(ships []Ship) []Ship {
	out := append([]Ship(nil), ships...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].Date.After(out[j-1].Date); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
