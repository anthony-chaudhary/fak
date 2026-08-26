// Package studybench measures bounded retrieval quality over study records.
package studybench

import (
	"encoding/json"
	"sort"
	"strings"
	"time"
	"unicode"
)

// Query is one falsifiable retrieval expectation.
type Query struct {
	Kind        string   `json:"kind"`
	Text        string   `json:"text"`
	ExpectedIDs []string `json:"expected_ids"`
}

// Record is the minimum corpus surface needed by the retrieval benchmark.
type Record struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// MethodResult reports quality, context cost, and observed latency for one method.
type MethodResult struct {
	Method               string  `json:"method"`
	RecallAtK            float64 `json:"recall_at_k"`
	IrrelevantResultRate float64 `json:"irrelevant_result_rate"`
	ReturnedBytes        int     `json:"returned_bytes"`
	ColdLatencyNS        int64   `json:"cold_latency_ns"`
	WarmLatencyNS        int64   `json:"warm_latency_ns"`
	BuildLatencyNS       int64   `json:"build_latency_ns"`
	BuildBytes           int     `json:"build_bytes"`
}

// IndexCriteria states when richer retrieval machinery earns adoption.
type IndexCriteria struct {
	PromotionEvidence      string `json:"promotion_evidence"`
	DemotionEvidence       string `json:"demotion_or_retirement_evidence"`
	InvalidatingAssumption string `json:"invalidating_assumption"`
}

// Report is the machine-readable study retrieval benchmark result.
type Report struct {
	Schema        string         `json:"schema"`
	K             int            `json:"k"`
	Queries       []Query        `json:"queries"`
	Methods       []MethodResult `json:"methods"`
	IndexCriteria IndexCriteria  `json:"optional_index_criteria"`
}

// RepresentativeQueries spans the six lineage and decision views required by #8612.
var RepresentativeQueries = []Query{
	{Kind: "source", Text: "original paper revision", ExpectedIDs: []string{"source-paper"}},
	{Kind: "mechanism", Text: "prefix cache reuse mechanism", ExpectedIDs: []string{"mechanism-cache"}},
	{Kind: "candidate", Text: "vector index candidate", ExpectedIDs: []string{"candidate-vector"}},
	{Kind: "disposition", Text: "candidate rejected latency", ExpectedIDs: []string{"disposition-reject"}},
	{Kind: "contradiction", Text: "contradicts cache speedup", ExpectedIDs: []string{"contradiction-cache"}},
	{Kind: "issue_lineage", Text: "issue 8606 parent 8605", ExpectedIDs: []string{"lineage-8606"}},
}

// FixtureRecords is a small committed corpus that makes the benchmark offline and repeatable.
var FixtureRecords = []Record{
	{ID: "source-paper", Text: "source original paper revision 7 with immutable path and date"},
	{ID: "mechanism-cache", Text: "mechanism prefix cache reuse avoids repeated prefill work"},
	{ID: "candidate-vector", Text: "candidate vector index for semantic retrieval remains optional"},
	{ID: "disposition-reject", Text: "disposition candidate rejected because cold latency exceeded baseline"},
	{ID: "contradiction-cache", Text: "contradiction evidence contradicts cache speedup under short prompts"},
	{ID: "lineage-8606", Text: "issue 8606 child has parent 8605 and migration lineage"},
	{ID: "irrelevant-routing", Text: "model routing policy and GPU scheduling observations"},
	{ID: "irrelevant-security", Text: "capability security floor denies destructive tool calls"},
}

type hit struct {
	id, text string
	score    int
}
type search func(string, int) []hit

// Run executes lexical-alpha and grep/full-document baselines. Latencies are observations;
// rankings and all quality/context counters are deterministic for the committed inputs.
func Run(records []Record, queries []Query, k int) Report {
	if k < 1 {
		k = 1
	}
	buildStart := time.Now()
	index, buildBytes := buildIndex(records)
	buildNS := time.Since(buildStart).Nanoseconds()
	lexical := func(q string, k int) []hit { return lexicalSearch(records, index, q, k) }
	grep := func(q string, k int) []hit { return grepSearch(records, q, k) }
	methods := []MethodResult{
		measure("lexical_alpha", lexical, queries, k, buildNS, buildBytes),
		measure("grep_full_document", grep, queries, k, 0, 0),
	}
	return Report{
		Schema: "study-retrieval-benchmark/1", K: k, Queries: append([]Query(nil), queries...), Methods: methods,
		IndexCriteria: IndexCriteria{
			PromotionEvidence:      "Promote an optional FTS/vector/graph index only when the same fixed query set has recall@k >= lexical alpha, irrelevant-result rate <= lexical alpha, returned bytes < lexical alpha, and cold+warm latency plus build cost is lower over the declared amortization window.",
			DemotionEvidence:       "Demote or retire it when two consecutive runs miss any promoted bound or corpus reconciliation changes an expected source/candidate ID.",
			InvalidatingAssumption: "The fixture token vocabulary may not represent the migrated corpus; rerun on reconciled migration fixtures before promotion.",
		},
	}
}

// RunJSON emits the report in a stable machine-readable shape.
func RunJSON(records []Record, queries []Query, k int) ([]byte, error) {
	return json.MarshalIndent(Run(records, queries, k), "", "  ")
}

func buildIndex(records []Record) (map[string][]int, int) {
	idx := make(map[string][]int)
	bytes := 0
	for i, r := range records {
		seen := map[string]bool{}
		for _, token := range tokens(r.Text) {
			if !seen[token] {
				idx[token] = append(idx[token], i)
				bytes += len(token) + 8
				seen[token] = true
			}
		}
	}
	return idx, bytes
}

func lexicalSearch(records []Record, idx map[string][]int, query string, k int) []hit {
	scores := map[int]int{}
	for _, token := range tokens(query) {
		for _, i := range idx[token] {
			scores[i]++
		}
	}
	hits := make([]hit, 0, len(scores))
	for i, score := range scores {
		hits = append(hits, hit{records[i].ID, records[i].Text, score})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].id < hits[j].id
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

func grepSearch(records []Record, query string, k int) []hit {
	var hits []hit
	for _, r := range records {
		score := 0
		lower := strings.ToLower(r.Text)
		for _, token := range tokens(query) {
			if strings.Contains(lower, token) {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, hit{r.ID, r.Text, score})
		}
	}
	// A grep baseline loads every matching document in corpus order; k limits quality scoring,
	// not bytes loaded, so its bounded-context cost remains visible.
	return hits
}

func measure(name string, searcher search, queries []Query, k int, buildNS int64, buildBytes int) MethodResult {
	start := time.Now()
	first := searcher(queries[0].Text, k)
	cold := time.Since(start).Nanoseconds()
	_ = first
	start = time.Now()
	_ = searcher(queries[0].Text, k)
	warm := time.Since(start).Nanoseconds()
	var expected, found, returned, irrelevant int
	for _, q := range queries {
		hits := searcher(q.Text, k)
		limit := len(hits)
		if limit > k {
			limit = k
		}
		wanted := map[string]bool{}
		for _, id := range q.ExpectedIDs {
			wanted[id] = true
		}
		expected += len(wanted)
		for i, h := range hits {
			returned += len(h.text)
			if i < limit && wanted[h.id] {
				found++
			} else {
				irrelevant++
			}
		}
	}
	recall := 1.0
	if expected > 0 {
		recall = float64(found) / float64(expected)
	}
	rate := 0.0
	if found+irrelevant > 0 {
		rate = float64(irrelevant) / float64(found+irrelevant)
	}
	return MethodResult{name, recall, rate, returned, cold, warm, buildNS, buildBytes}
}

func tokens(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
}
