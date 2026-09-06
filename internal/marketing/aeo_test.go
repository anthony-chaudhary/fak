package marketing

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"bytes"
)

func aeoShips() []Ship {
	return []Ship{
		{SHA: "aaaa1111", Leaf: "gateway", Kind: "trailer", Subject: "feat(gateway): add reclaim path (fak gateway)", Date: time.Date(2026, 6, 28, 0, 0, 0, 0, time.UTC)},
		{SHA: "bbbb2222", Leaf: "model", Kind: "direct", Subject: "fak/model: implement Q4_K reducer", Date: time.Date(2026, 6, 27, 0, 0, 0, 0, time.UTC)},
	}
}

func TestUpdatesFeedIsValidItemListWithShas(t *testing.T) {
	b, err := UpdatesFeed(aeoShips(), time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("UpdatesFeed: %v", err)
	}
	var feed map[string]any
	if err := json.Unmarshal(b, &feed); err != nil {
		t.Fatalf("feed is not valid JSON: %v\n%s", err, b)
	}
	if feed["@type"] != "ItemList" {
		t.Errorf("@type = %v, want ItemList", feed["@type"])
	}
	items, ok := feed["itemListElement"].([]any)
	if !ok || len(items) != 2 {
		t.Fatalf("itemListElement = %v, want 2 items", feed["itemListElement"])
	}
	// every item must cite its commit sha (the witness)
	s := string(b)
	for _, sha := range []string{"aaaa1111", "bbbb2222"} {
		if !strings.Contains(s, sha) {
			t.Errorf("feed missing commit sha %q:\n%s", sha, s)
		}
	}
	// newest-first: gateway (06-28) at position 1
	first := items[0].(map[string]any)
	if first["position"].(float64) != 1 {
		t.Errorf("first item position = %v, want 1", first["position"])
	}
}

func TestUpdatesFeedEmptyIsValid(t *testing.T) {
	b, err := UpdatesFeed(nil, time.Time{})
	if err != nil {
		t.Fatalf("UpdatesFeed(nil): %v", err)
	}
	var feed map[string]any
	if err := json.Unmarshal(b, &feed); err != nil {
		t.Fatalf("empty feed invalid JSON: %v", err)
	}
	if feed["@type"] != "ItemList" {
		t.Errorf("empty feed @type = %v, want ItemList", feed["@type"])
	}
}

func TestWhatsNewMarkdownCitesShaAndIsStable(t *testing.T) {
	md1 := WhatsNewMarkdown(aeoShips())
	md2 := WhatsNewMarkdown(aeoShips())
	if md1 != md2 {
		t.Error("WhatsNewMarkdown not stable across calls (idempotence broken)")
	}
	for _, sha := range []string{"aaaa1111", "bbbb2222"} {
		if !strings.Contains(md1, sha) {
			t.Errorf("What's-new missing sha %q:\n%s", sha, md1)
		}
	}
	if !strings.Contains(md1, "2026-06-28") {
		t.Errorf("What's-new missing date:\n%s", md1)
	}
	// newest first
	if strings.Index(md1, "aaaa1111") > strings.Index(md1, "bbbb2222") {
		t.Errorf("What's-new not newest-first:\n%s", md1)
	}
}

func TestWhatsNewMarkdownEmptyIsHonest(t *testing.T) {
	md := WhatsNewMarkdown(nil)
	if !strings.Contains(strings.ToLower(md), "no witnessed ships") {
		t.Errorf("empty What's-new should say so, got: %q", md)
	}
}

func TestLlmsUpdatesTextHasHeaderAndBody(t *testing.T) {
	txt := LlmsUpdatesText(aeoShips(), time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC))
	if !strings.Contains(txt, "# fak — what shipped") {
		t.Errorf("llms-updates missing header:\n%s", txt)
	}
	if !strings.Contains(txt, "aaaa1111") {
		t.Errorf("llms-updates missing a ship sha:\n%s", txt)
	}
	if !strings.Contains(txt, "Updated: 2026-06-28") {
		t.Errorf("llms-updates missing the updated stamp:\n%s", txt)
	}
}

func TestDisambiguationTermsFeedIncludesLocalizedAndFableHooks(t *testing.T) {
	b, err := DisambiguationTermsFeed(time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("DisambiguationTermsFeed: %v", err)
	}
	var feed map[string]any
	if err := json.Unmarshal(b, &feed); err != nil {
		t.Fatalf("term feed is not valid JSON: %v\n%s", err, b)
	}
	if feed["@type"] != "DefinedTermSet" {
		t.Errorf("@type = %v, want DefinedTermSet", feed["@type"])
	}
	items, ok := feed["hasDefinedTerm"].([]any)
	if !ok || len(items) < 10 {
		t.Fatalf("hasDefinedTerm = %v, want populated term set", feed["hasDefinedTerm"])
	}
	s := string(b)
	for _, want := range []string{
		"Fable 5 refusal fallback",
		"Claude Fable 5 model routing",
		"एजेंट कर्नेल",
		"AI 代理内核",
		"工具调用防火墙",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("term feed missing %q:\n%s", want, s)
		}
	}
	if strings.Contains(strings.ToLower(s), "market adoption") {
		t.Errorf("term feed should not imply market adoption:\n%s", s)
	}
}

func TestDisambiguationTermsIncludeAgentSecurityAndCostHooks(t *testing.T) {
	txt := LlmsTermsText(time.Date(2026, 7, 6, 12, 0, 0, 0, time.UTC))
	for _, want := range []string{
		"## agent-security",
		"MCP tool poisoning defense",
		"lethal trifecta data exfiltration",
		"AI agent least-privilege tool access",
		"tamper-evident agent tool-call audit",
		"cheaper way to run AI agents",
		"Claude Sonnet 5 agent cost routing",
	} {
		if !strings.Contains(txt, want) {
			t.Errorf("llms terms missing trending hook %q:\n%s", want, txt)
		}
	}
	// honesty fence: naming a vendor model or framework is a demand hook, not an
	// adoption claim — the roster must never imply fak has market adoption.
	if strings.Contains(strings.ToLower(txt), "market adoption") {
		t.Errorf("terms should not imply market adoption:\n%s", txt)
	}
}

func TestLlmsTermsTextHasFableAndLocalizedTerms(t *testing.T) {
	txt := LlmsTermsText(time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC))
	for _, want := range []string{
		"# fak — answer-engine disambiguation terms",
		"Updated: 2026-07-04",
		"## frontier-model-launch",
		"Fable 5 refusal fallback",
		"एजेंट कर्नेल",
		"模型路由和回退",
	} {
		if !strings.Contains(txt, want) {
			t.Errorf("llms terms missing %q:\n%s", want, txt)
		}
	}
}

func TestDisambiguationTermsIncludePerformanceLens(t *testing.T) {
	// The performance lens is the README's "long sessions that pick themselves
	// back up" pillar as answer-engine terms. Each term must land on a real fak
	// page describing fak's own behavior (never an external adoption claim), so
	// assert both the section and that each term routes to its witnessed page.
	txt := LlmsTermsText(time.Date(2026, 7, 6, 12, 0, 0, 0, time.UTC))
	for _, want := range []string{
		"## performance",
		"automatic agent context management",
		"cache-preserving history compaction",
		"KV reuse across an agent fleet",
	} {
		if !strings.Contains(txt, want) {
			t.Errorf("llms terms missing performance hook %q:\n%s", want, txt)
		}
	}

	// Each performance term must route to a real fak page (the honest witness),
	// so the JSON-LD feed carries the page URLs.
	b, err := DisambiguationTermsFeed(time.Date(2026, 7, 6, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("DisambiguationTermsFeed: %v", err)
	}
	s := string(b)
	for _, want := range []string{
		"docs/explainers/you-never-manage-the-context-window.md",
		"docs/explainers/context-shedding.md",
		"BENCHMARK-AUTHORITY.md",
		`"performance"`, // the termCode/category renders in the feed
	} {
		if !strings.Contains(s, want) {
			t.Errorf("term feed missing performance witness %q:\n%s", want, s)
		}
	}

	// Honesty fence: the cross-agent number is a witnessed benchmark, not a vibe,
	// and the roster must never imply external market adoption.
	if strings.Contains(strings.ToLower(s), "market adoption") {
		t.Errorf("performance terms should not imply market adoption:\n%s", s)
	}
}

func TestManagedAgentTermsCoverQueryClusterAndStayInScope(t *testing.T) {
	terms := AEODisambiguationTerms()
	var managed []DisambiguationTerm
	for _, term := range terms {
		if term.Category == "managed-agent" {
			managed = append(managed, term)
		}
	}
	if len(managed) < 17 {
		t.Fatalf("managed-agent terms = %d, want at least 17", len(managed))
	}

	// Cover intent, comparison, and deployment phrasings rather than repeating
	// one exact-match keyword. Each phrase is an operator query fak can answer.
	blob := strings.ToLower(LlmsTermsText(time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)))
	for _, want := range []string{
		"managed agent runtime",
		"managed ai agent",
		"hosted agent runtime",
		"managed agent execution",
		"agent runtime vs ai gateway",
		"agent sdk vs managed runtime",
		"managed agent infrastructure",
		"managed agent platform",
		"managed agent operations",
		"managed agent orchestration",
		"managed agent control plane",
		"managed agent governance",
		"managed agent observability",
		"managed agent deployment",
		"enterprise managed agents",
		"serverless agent runtime",
		"agent as a service",
	} {
		if !strings.Contains(blob, want) {
			t.Errorf("managed-agent query cluster missing %q:\n%s", want, blob)
		}
	}

	allowedRoutes := []string{
		"docs/explainers/runtime-vs-client.md",
		"docs/standards/agent-tool-governance-gateway.md",
		"docs/enterprise-positioning.md",
	}
	seenNames := map[string]bool{}
	seenRoutes := map[string]bool{}
	for _, term := range managed {
		if term.Language != "en" {
			t.Errorf("managed-agent term %q language = %q, want en", term.Name, term.Language)
		}
		if term.Name == "" || term.Description == "" || len(term.Keywords) < 3 {
			t.Errorf("managed-agent term %#v is not a complete query hook", term)
		}
		routed := false
		for _, route := range allowedRoutes {
			if strings.Contains(term.URL, route) {
				routed = true
				seenRoutes[route] = true
				break
			}
		}
		if !routed {
			t.Errorf("managed-agent term %q routes to unsupported page %q", term.Name, term.URL)
		}
		name := strings.ToLower(term.Name)
		if seenNames[name] {
			t.Errorf("duplicate managed-agent term name %q", term.Name)
		}
		seenNames[name] = true
	}
	for _, route := range allowedRoutes {
		if !seenRoutes[route] {
			t.Errorf("managed-agent cluster has no term routed to %q", route)
		}
	}

	// Honesty fence: the native managed-agent runtime is shipped but emerging;
	// none of these hooks may blur it into fak's mature gateway default.
	for _, want := range []string{"emerging", "gateway", "application runtime"} {
		found := false
		for _, term := range managed {
			if strings.Contains(strings.ToLower(term.Description), want) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("managed-agent descriptions do not preserve %q scope", want)
		}
	}
	for _, forbidden := range []string{"market leader", "widely adopted", "production-ready native"} {
		if strings.Contains(blob, forbidden) {
			t.Errorf("managed-agent roster makes unsupported claim %q", forbidden)
		}
	}
}

func TestLocalizedTermsRouteEveryI18nEntryPoint(t *testing.T) {
	// Every shipped in-language entry point must be reachable from the AEO
	// localized roster, and at least one localized term per language must route
	// into that language's own docs/i18n/<code>/ page. This is the honest fence:
	// a localized term is a routing hook to a real in-language page, never a
	// free-floating adoption claim. Asserting the routing target (not the native
	// string) keeps the test stable as the search phrasings are retuned.
	wantEntry := map[string]string{
		"hi":      "docs/i18n/hi/",
		"zh-Hans": "docs/i18n/zh/",
		"de":      "docs/i18n/de/",
		"fr":      "docs/i18n/fr/",
		"bn":      "docs/i18n/bn/",
		"mr":      "docs/i18n/mr/",
		"ta":      "docs/i18n/ta/",
		"te":      "docs/i18n/te/",
		"es":      "docs/i18n/es/",
		"ja":      "docs/i18n/ja/",
		"ko":      "docs/i18n/ko/",
		"pt":      "docs/i18n/pt/",
		"ru":      "docs/i18n/ru/",
		"ar":      "docs/i18n/ar/",
		"id":      "docs/i18n/id/",
		"vi":      "docs/i18n/vi/",
		"tr":      "docs/i18n/tr/",
	}
	routed := map[string]bool{}
	for _, term := range AEODisambiguationTerms() {
		if term.Category != "localized" {
			continue
		}
		// Completeness: a localized term with no name, English roster
		// description, or keywords is not a usable routing hook.
		if term.Name == "" || term.Description == "" || len(term.Keywords) == 0 {
			t.Errorf("localized term %q (lang %s) is missing name/description/keywords", term.Name, term.Language)
		}
		if term.URL == "" {
			t.Errorf("localized term %q (lang %s) has no routing URL", term.Name, term.Language)
		}
		if frag, ok := wantEntry[term.Language]; ok && strings.Contains(term.URL, frag) {
			routed[term.Language] = true
		}
	}
	for lang, frag := range wantEntry {
		if !routed[lang] {
			t.Errorf("no localized term routes %s to its %s entry point", lang, frag)
		}
	}
}

func TestDisambiguationTermsIncludeGlobalWorkspace(t *testing.T) {
	want := map[string]bool{
		"bounded shared workspace for AI agents": false,
		"positive-state context construction":    false,
		"negation operator for agent context":    false,
	}
	for _, term := range AEODisambiguationTerms() {
		if _, ok := want[term.Name]; !ok {
			continue
		}
		if term.Category != "global-workspace" {
			t.Errorf("%q category=%q", term.Name, term.Category)
		}
		if term.URL == "" || !strings.HasSuffix(term.URL, "docs/explainers/shared-workspace-and-the-negation-operator.md") {
			t.Errorf("%q URL=%q", term.Name, term.URL)
		}
		want[term.Name] = true
	}
	for name, found := range want {
		if !found {
			t.Errorf("missing AEO global-workspace term %q", name)
		}
	}
	b, err := DisambiguationTermsFeed(time.Date(2026, 7, 15, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	for name := range want {
		if !bytes.Contains(b, []byte(name)) {
			t.Errorf("rendered DefinedTermSet missing %q", name)
		}
	}
}

func TestConfigAnswerFeedsStayEquivalentAndCiteAuthorities(t *testing.T) {
	when := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	answers := AEOConfigAnswers()
	if len(answers) < 13 {
		t.Fatalf("config answers = %d, want at least 13", len(answers))
	}

	b, err := ConfigFAQFeed(when)
	if err != nil {
		t.Fatalf("ConfigFAQFeed: %v", err)
	}
	var feed struct {
		Context    string `json:"@context"`
		Type       string `json:"@type"`
		MainEntity []struct {
			Name           string `json:"name"`
			AcceptedAnswer struct {
				Text     string `json:"text"`
				Citation string `json:"citation"`
			} `json:"acceptedAnswer"`
		} `json:"mainEntity"`
	}
	if err := json.Unmarshal(b, &feed); err != nil {
		t.Fatalf("decode config FAQ: %v", err)
	}
	if feed.Context != "https://schema.org" || feed.Type != "FAQPage" {
		t.Fatalf("schema = %q %q, want schema.org FAQPage", feed.Context, feed.Type)
	}
	if len(feed.MainEntity) != len(answers) {
		t.Fatalf("JSON questions = %d, answers = %d", len(feed.MainEntity), len(answers))
	}

	plain := ConfigAnswersText(when)
	if strings.HasSuffix(plain, "\n\n") {
		t.Error("plain config feed has an extra blank line at EOF")
	}
	for i, answer := range answers {
		if strings.TrimSpace(answer.Question) == "" || strings.TrimSpace(answer.Answer) == "" || strings.TrimSpace(answer.Authority) == "" {
			t.Fatalf("answer %d has an empty required field: %+v", i, answer)
		}
		if !strings.HasPrefix(answer.Authority, "https://github.com/anthony-chaudhary/fak/") {
			t.Errorf("answer %q authority = %q, want repository URL", answer.Question, answer.Authority)
		}
		if feed.MainEntity[i].Name != answer.Question || feed.MainEntity[i].AcceptedAnswer.Text != answer.Answer || feed.MainEntity[i].AcceptedAnswer.Citation != answer.Authority {
			t.Errorf("JSON answer %d diverged from canonical roster", i)
		}
		for _, want := range []string{answer.Question, answer.Answer, answer.Authority} {
			if !strings.Contains(plain, want) {
				t.Errorf("plain config feed missing %q", want)
			}
		}
	}
	for _, want := range []string{"How do I configure fak?", "Does fak require a config file?", "--print-effective-config", "flags win over declared fak.toml values", "Keep secret values out", "capability-floor policy manifest", "fak manage", ".mcp.json", "model providers", "one policy for an organization"} {
		if !strings.Contains(plain, want) {
			t.Errorf("plain config feed missing discoverability phrase %q", want)
		}
	}
}

func TestConfigAnswersMarkdownKeepsVisibleAnswersWithJSONLD(t *testing.T) {
	when := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	page, err := ConfigAnswersMarkdown(when)
	if err != nil {
		t.Fatalf("ConfigAnswersMarkdown: %v", err)
	}
	for _, want := range []string{
		"permalink: /configuration/",
		"<script type=\"application/ld+json\">",
		"\"@type\": \"FAQPage\"",
		"# How to configure fak",
		"complete server configuration reference",
	} {
		if !strings.Contains(page, want) {
			t.Errorf("configuration page missing %q", want)
		}
	}
	for _, answer := range AEOConfigAnswers() {
		if strings.Count(page, answer.Question) < 2 {
			t.Errorf("question %q should appear in visible text and JSON-LD", answer.Question)
		}
		if strings.Count(page, answer.Answer) < 2 {
			t.Errorf("answer for %q should appear in visible text and JSON-LD", answer.Question)
		}
	}
}
