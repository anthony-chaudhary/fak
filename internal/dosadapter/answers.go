// Package dosadapter provides answer corpus retrieval and index fallback
// loading for the DOS trust substrate, ensuring agents can query "how do I X?"
// even when running in environments without a local repository docs/ tree.
package dosadapter

import (
	_ "embed"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

//go:embed answers_index.json
var embeddedAnswersIndex []byte

var wordRegex = regexp.MustCompile(`[a-z0-9]+`)

// stopWords mirrors the DOS answer corpus stop words filter.
var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "the": {}, "to": {}, "of": {}, "for": {},
	"in": {}, "on": {}, "is": {}, "it": {}, "my": {}, "me": {}, "i": {},
	"how": {}, "do": {}, "does": {}, "can": {}, "what": {}, "why": {},
	"with": {}, "that": {}, "this": {}, "you": {}, "your": {}, "they": {},
	"them": {}, "their": {}, "or": {}, "not": {}, "no": {}, "as": {},
	"at": {}, "be": {},
}

// AnswerRow represents a single indexed answer corpus page.
type AnswerRow struct {
	Slug     string   `json:"slug"`
	Question string   `json:"question"`
	Answer   string   `json:"answer"`
	Commands []string `json:"commands"`
	Path     string   `json:"path"`
	URL      string   `json:"url"`
	Queries  []string `json:"queries"`
	Score    float64  `json:"score,omitempty"`
}

// AnswerResponse wraps the search results envelope matching the MCP tool schema.
type AnswerResponse struct {
	Query   string      `json:"query"`
	Results []AnswerRow `json:"results"`
	Count   int         `json:"count"`
	Note    string      `json:"note,omitempty"`
}

var (
	rowsRegistryMu sync.RWMutex
	rowsRegistry   = make(map[string][]AnswerRow)
)

// parseTokens extracts non-stop lowercase alphanumeric tokens of length > 1.
func parseTokens(text string) map[string]struct{} {
	tokens := make(map[string]struct{})
	matches := wordRegex.FindAllString(strings.ToLower(text), -1)
	for _, m := range matches {
		if len(m) <= 1 {
			continue
		}
		if _, isStop := stopWords[m]; isStop {
			continue
		}
		tokens[m] = struct{}{}
	}
	return tokens
}

// sequenceRatio computes difflib-like sequence similarity ratio in [0, 1].
// ratio = 2 * M / (len(a) + len(b)) where M is length of longest common subsequence.
func sequenceRatio(a, b string) float64 {
	a = strings.TrimSpace(strings.ToLower(a))
	b = strings.TrimSpace(strings.ToLower(b))
	if a == "" && b == "" {
		return 1.0
	}
	if a == "" || b == "" {
		return 0.0
	}
	if a == b {
		return 1.0
	}

	la := len(a)
	lb := len(b)
	// Dynamic programming for longest common subsequence length
	dp := make([]int, lb+1)
	for i := 1; i <= la; i++ {
		prev := 0
		for j := 1; j <= lb; j++ {
			temp := dp[j]
			if a[i-1] == b[j-1] {
				dp[j] = prev + 1
			} else if dp[j-1] > dp[j] {
				dp[j] = dp[j-1]
			}
			prev = temp
		}
	}
	lcsLen := dp[lb]
	return 2.0 * float64(lcsLen) / float64(la+lb)
}

// parseIndexBytes decodes JSON array or JSONL rows into AnswerRow records.
func parseIndexBytes(data []byte) []AnswerRow {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return nil
	}

	// Try parsing as JSON array
	if strings.HasPrefix(trimmed, "[") {
		var rows []AnswerRow
		if err := json.Unmarshal([]byte(trimmed), &rows); err == nil {
			return rows
		}
	}

	// Fallback to line-delimited JSON (JSONL)
	var rows []AnswerRow
	lines := strings.Split(trimmed, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row AnswerRow
		if err := json.Unmarshal([]byte(line), &row); err == nil {
			rows = append(rows, row)
		}
	}
	return rows
}

// LoadAnswers loads answer rows from the workspace docs tree, falling back to
// the precompiled bundled embedded index if the docs tree is absent.
func LoadAnswers(workspace string) []AnswerRow {
	storeKey := workspace
	rowsRegistryMu.RLock()
	if existing, ok := rowsRegistry[storeKey]; ok {
		rowsRegistryMu.RUnlock()
		return existing
	}
	rowsRegistryMu.RUnlock()

	var rows []AnswerRow

	if workspace != "" {
		candidates := []string{
			filepath.Join(workspace, "docs", "answers", "index.jsonl"),
			filepath.Join(workspace, "docs", "answers", "answers_index.json"),
			filepath.Join(workspace, "src", "dos", "data", "answers_index.json"),
			filepath.Join(workspace, "src", "dos_mcp", "data", "answers_index.json"),
		}
		for _, cand := range candidates {
			if data, err := os.ReadFile(cand); err == nil {
				if parsed := parseIndexBytes(data); len(parsed) > 0 {
					rows = parsed
					break
				}
			}
		}
	}

	// Fallback to precompiled bundled embedded index
	if len(rows) == 0 && len(embeddedAnswersIndex) > 0 {
		rows = parseIndexBytes(embeddedAnswersIndex)
	}

	rowsRegistryMu.Lock()
	rowsRegistry[storeKey] = rows
	rowsRegistryMu.Unlock()

	return rows
}

// haystack returns the searchable text of an AnswerRow: question + queries + answer.
func haystack(r AnswerRow) string {
	parts := []string{r.Question}
	parts = append(parts, r.Queries...)
	parts = append(parts, r.Answer)
	return strings.Join(parts, " ")
}

// ScoreAnswer evaluates deterministic relevance in [0, 1] for a query against a row.
// Overlap contributes 75%, best sequence ratio contributes 25%.
func ScoreAnswer(query string, r AnswerRow) float64 {
	qTokens := parseTokens(query)
	if len(qTokens) == 0 {
		return 0.0
	}
	hTokens := parseTokens(haystack(r))
	if len(hTokens) == 0 {
		return 0.0
	}

	overlapCount := 0
	for t := range qTokens {
		if _, ok := hTokens[t]; ok {
			overlapCount++
		}
	}
	overlap := float64(overlapCount) / float64(len(qTokens))

	candidates := append([]string{r.Question}, r.Queries...)
	bestRatio := 0.0
	for _, c := range candidates {
		if c == "" {
			continue
		}
		ratio := sequenceRatio(query, c)
		if ratio > bestRatio {
			bestRatio = ratio
		}
	}

	score := 0.75*overlap + 0.25*bestRatio
	return math.Round(score*10000) / 10000
}

// SearchAnswers scores and ranks the answer corpus, falling back to precompiled
// bundled assets when local documentation trees are omitted.
func SearchAnswers(workspace, query string, k int) AnswerResponse {
	trimmedQuery := strings.TrimSpace(query)
	if k <= 0 {
		k = 3
	}

	out := AnswerResponse{
		Query:   query,
		Results: []AnswerRow{},
		Count:   0,
	}

	if trimmedQuery == "" {
		out.Note = "query is empty"
		return out
	}

	rows := LoadAnswers(workspace)
	if len(rows) == 0 {
		out.Note = "the answer corpus index is not available here (an installed wheel ships no docs/ tree) — fetch the corpus at https://github.com/anthony-chaudhary/dos-kernel/blob/master/docs/answers/README.md"
		return out
	}

	type scoredRow struct {
		score float64
		row   AnswerRow
	}

	var scored []scoredRow
	for _, r := range rows {
		s := ScoreAnswer(trimmedQuery, r)
		scored = append(scored, scoredRow{score: s, row: r})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var results []AnswerRow
	for _, item := range scored {
		if len(results) >= k {
			break
		}
		if item.score <= 0.0 && len(results) > 0 {
			break
		}
		if item.score > 0.0 {
			r := item.row
			r.Score = item.score
			results = append(results, r)
		}
	}

	out.Results = results
	out.Count = len(results)
	if len(results) == 0 {
		out.Note = "no answer page scored above zero for this query; try rephrasing, or browse the corpus at docs/answers/README.md"
	}
	return out
}
