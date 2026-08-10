package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
)

const provenanceFoldSchema = "fak-microcontext-provenance-fold/1"

type foldFact struct {
	SourceID   string   `json:"source_id"`
	SourceHash string   `json:"source_hash"`
	Status     string   `json:"status"`
	Cluster    string   `json:"cluster"`
	Tags       []string `json:"tags"`
	Score      int      `json:"score"`
	Claim      string   `json:"claim"`
	Outlier    bool     `json:"outlier"`
}
type foldCandidate struct {
	SourceID string `json:"source_id"`
	Score    int    `json:"score"`
}
type foldCluster struct {
	Name      string   `json:"name"`
	Count     int      `json:"count"`
	Exemplars []string `json:"exemplars"`
	Outliers  []string `json:"outliers"`
}
type foldSummary struct {
	Coverage     int             `json:"coverage"`
	StatusCounts map[string]int  `json:"status_counts"`
	Tags         []string        `json:"tags"`
	TopK         []foldCandidate `json:"top_k"`
	Clusters     []foldCluster   `json:"clusters"`
}
type foldNode struct {
	Hash     string      `json:"hash"`
	Level    int         `json:"level"`
	Children []string    `json:"children,omitempty"`
	SourceID string      `json:"source_id,omitempty"`
	Summary  foldSummary `json:"summary"`
}
type foldRun struct {
	FanIn              int         `json:"fan_in"`
	Levels             int         `json:"levels"`
	Nodes              int         `json:"nodes"`
	MaxInput           int         `json:"max_input"`
	MaxOutputCitations int         `json:"max_output_citations"`
	RootHash           string      `json:"root_hash"`
	ResultHash         string      `json:"result_hash"`
	Root               foldSummary `json:"root"`
	Tree               []foldNode  `json:"tree,omitempty"`
}
type provenanceFoldReport struct {
	Schema                   string   `json:"schema"`
	Mode                     string   `json:"mode"`
	Sources                  int      `json:"sources"`
	Baseline                 foldRun  `json:"baseline"`
	ReorderedResultHash      string   `json:"reordered_result_hash"`
	AlternateFanInResultHash string   `json:"alternate_fan_in_result_hash"`
	Mutated                  foldRun  `json:"mutated"`
	MutationSourceID         string   `json:"mutation_source_id"`
	RecomputedNodes          int      `json:"recomputed_nodes"`
	ExpectedAncestorPath     int      `json:"expected_ancestor_path"`
	UnaffectedNodesReused    int      `json:"unaffected_nodes_reused"`
	FinalCitations           int      `json:"final_citations"`
	CitationsResolved        bool     `json:"citations_resolved"`
	MinorityOutlierPreserved bool     `json:"minority_outlier_preserved"`
	UncertaintyPreserved     bool     `json:"uncertainty_preserved"`
	NegativeAuditSampled     int      `json:"negative_audit_sampled"`
	NegativeAuditPassed      bool     `json:"negative_audit_passed"`
	Notes                    []string `json:"notes"`
}

func hashJSON(v any) string {
	b, _ := json.Marshal(v)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
func makeFoldFacts(n int) []foldFact {
	facts := make([]foldFact, n)
	for i := 0; i < n; i++ {
		f := foldFact{SourceID: fmt.Sprintf("issue-%05d", i), Status: "excluded", Cluster: "routine", Tags: []string{"triaged"}, Score: i % 97, Claim: "routine maintenance"}
		if i%41 == 0 {
			f.Status = "kept"
			f.Cluster = "auth"
			f.Tags = append(f.Tags, "authentication")
			f.Claim = "authentication behavior needs review"
		}
		if i == 777 {
			f.Status = "kept"
			f.Cluster = "security-dissent"
			f.Tags = []string{"security", "minority"}
			f.Score = 1000
			f.Claim = "single security dissent must survive"
			f.Outlier = true
		}
		if i == 778 {
			f.Status = "abstain"
			f.Cluster = "uncertain"
			f.Tags = []string{"uncertain"}
			f.Claim = "missing relation prevents classification"
			f.Outlier = true
		}
		f.SourceHash = hashJSON(struct{ ID, Status, Claim string }{f.SourceID, f.Status, f.Claim})
		facts[i] = f
	}
	return facts
}
func leafSummary(f foldFact) foldSummary {
	c := foldCluster{Name: f.Cluster, Count: 1, Exemplars: []string{f.SourceID}}
	if f.Outlier {
		c.Outliers = []string{f.SourceID}
	}
	return foldSummary{Coverage: 1, StatusCounts: map[string]int{f.Status: 1}, Tags: append([]string(nil), f.Tags...), TopK: []foldCandidate{{f.SourceID, f.Score}}, Clusters: []foldCluster{c}}
}
func mergeSummaries(xs []foldSummary) foldSummary {
	r := foldSummary{StatusCounts: map[string]int{}}
	tags := map[string]bool{}
	cm := map[string]foldCluster{}
	var cand []foldCandidate
	for _, x := range xs {
		r.Coverage += x.Coverage
		for k, v := range x.StatusCounts {
			r.StatusCounts[k] += v
		}
		for _, t := range x.Tags {
			tags[t] = true
		}
		cand = append(cand, x.TopK...)
		for _, c := range x.Clusters {
			m := cm[c.Name]
			m.Name = c.Name
			m.Count += c.Count
			m.Exemplars = append(m.Exemplars, c.Exemplars...)
			m.Outliers = append(m.Outliers, c.Outliers...)
			cm[c.Name] = m
		}
	}
	for t := range tags {
		r.Tags = append(r.Tags, t)
	}
	sort.Strings(r.Tags)
	sort.Slice(cand, func(i, j int) bool {
		if cand[i].Score != cand[j].Score {
			return cand[i].Score > cand[j].Score
		}
		return cand[i].SourceID < cand[j].SourceID
	})
	seen := map[string]bool{}
	for _, c := range cand {
		if !seen[c.SourceID] {
			r.TopK = append(r.TopK, c)
			seen[c.SourceID] = true
			if len(r.TopK) == 5 {
				break
			}
		}
	}
	names := make([]string, 0, len(cm))
	for n := range cm {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		c := cm[n]
		c.Exemplars = uniqueBounded(c.Exemplars, 2)
		c.Outliers = uniqueBounded(c.Outliers, 2)
		r.Clusters = append(r.Clusters, c)
	}
	return r
}
func uniqueBounded(xs []string, n int) []string {
	sort.Strings(xs)
	r := xs[:0]
	for _, x := range xs {
		if len(r) == 0 || r[len(r)-1] != x {
			r = append(r, x)
			if len(r) == n {
				break
			}
		}
	}
	return append([]string(nil), r...)
}
func citationCount(s foldSummary) int {
	n := len(s.TopK)
	for _, c := range s.Clusters {
		n += len(c.Exemplars) + len(c.Outliers)
	}
	return n
}
func buildFold(facts []foldFact, fanIn int, includeTree bool) (foldRun, map[string]foldNode) {
	ordered := append([]foldFact(nil), facts...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].SourceID < ordered[j].SourceID })
	cache := map[string]foldNode{}
	level := make([]foldNode, 0, len(ordered))
	run := foldRun{FanIn: fanIn}
	for _, f := range ordered {
		s := leafSummary(f)
		n := foldNode{Level: 0, SourceID: f.SourceID, Summary: s}
		n.Hash = hashJSON(struct {
			SourceHash string
			Summary    foldSummary
		}{f.SourceHash, s})
		cache[n.Hash] = n
		level = append(level, n)
		if citationCount(s) > run.MaxOutputCitations {
			run.MaxOutputCitations = citationCount(s)
		}
	}
	for len(level) > 1 {
		run.Levels++
		next := make([]foldNode, 0, (len(level)+fanIn-1)/fanIn)
		for i := 0; i < len(level); i += fanIn {
			j := i + fanIn
			if j > len(level) {
				j = len(level)
			}
			children := level[i:j]
			sums := make([]foldSummary, len(children))
			hashes := make([]string, len(children))
			for k, c := range children {
				sums[k] = c.Summary
				hashes[k] = c.Hash
			}
			s := mergeSummaries(sums)
			n := foldNode{Level: run.Levels, Children: hashes, Summary: s}
			n.Hash = hashJSON(struct {
				Children []string
				Summary  foldSummary
			}{hashes, s})
			cache[n.Hash] = n
			next = append(next, n)
			run.Nodes++
			if len(children) > run.MaxInput {
				run.MaxInput = len(children)
			}
			if citationCount(s) > run.MaxOutputCitations {
				run.MaxOutputCitations = citationCount(s)
			}
		}
		level = next
	}
	run.Root = level[0].Summary
	run.RootHash = level[0].Hash
	run.ResultHash = hashJSON(run.Root)
	if includeTree {
		keys := make([]string, 0, len(cache))
		for k := range cache {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			run.Tree = append(run.Tree, cache[k])
		}
	}
	return run, cache
}
func allCitations(s foldSummary) []string {
	set := map[string]bool{}
	for _, x := range s.TopK {
		set[x.SourceID] = true
	}
	for _, c := range s.Clusters {
		for _, x := range c.Exemplars {
			set[x] = true
		}
		for _, x := range c.Outliers {
			set[x] = true
		}
	}
	r := make([]string, 0, len(set))
	for x := range set {
		r = append(r, x)
	}
	sort.Strings(r)
	return r
}
func hasCitation(s foldSummary, id string) bool {
	for _, x := range allCitations(s) {
		if x == id {
			return true
		}
	}
	return false
}
func verifyFoldReport(r provenanceFoldReport) error {
	if r.Schema != provenanceFoldSchema || r.Sources != 1000 {
		return errors.New("schema/source mismatch")
	}
	if r.Baseline.MaxInput > r.Baseline.FanIn || r.Baseline.MaxOutputCitations > 64 {
		return errors.New("reducer bound violated")
	}
	if r.Baseline.ResultHash != r.ReorderedResultHash || r.Baseline.ResultHash != r.AlternateFanInResultHash {
		return errors.New("algebraic determinism violated")
	}
	if r.RecomputedNodes != r.ExpectedAncestorPath || r.UnaffectedNodesReused < 990 {
		return errors.New("incremental invalidation widened")
	}
	if !r.CitationsResolved || !r.MinorityOutlierPreserved || !r.UncertaintyPreserved || !r.NegativeAuditPassed {
		return errors.New("provenance or dissent lost")
	}
	if r.Baseline.Root.Coverage != r.Sources {
		return errors.New("coverage mismatch")
	}
	return nil
}
func runProvenanceFoldSelfcheck(output string) error {
	facts := makeFoldFacts(1000)
	base, bc := buildFold(facts, 8, true)
	rev := append([]foldFact(nil), facts...)
	for i, j := 0, len(rev)-1; i < j; i, j = i+1, j-1 {
		rev[i], rev[j] = rev[j], rev[i]
	}
	reorder, _ := buildFold(rev, 8, false)
	alt, _ := buildFold(facts, 13, false)
	mut := append([]foldFact(nil), facts...)
	mut[321].Claim = "mutated source claim"
	mut[321].SourceHash = hashJSON(mut[321])
	changed, _ := buildFold(mut, 8, true)
	cc := map[string]bool{}
	for _, n := range changed.Tree {
		cc[n.Hash] = true
	}
	reused := 0
	for h := range bc {
		if cc[h] {
			reused++
		}
	}
	recomputed := len(changed.Tree) - reused
	registry := map[string]foldFact{}
	for _, f := range facts {
		registry[f.SourceID] = f
	}
	cites := allCitations(base.Root)
	resolved := true
	for _, id := range cites {
		if _, ok := registry[id]; !ok {
			resolved = false
		}
	}
	audit := 0
	passed := true
	for i := 0; i < len(facts) && audit < 25; i += 37 {
		audit++
		if facts[i].Status == "excluded" && hasCitation(base.Root, facts[i].SourceID) {
			passed = false
		}
	}
	r := provenanceFoldReport{Schema: provenanceFoldSchema, Mode: "fixture-backed typed hierarchical reducers", Sources: len(facts), Baseline: base, ReorderedResultHash: reorder.ResultHash, AlternateFanInResultHash: alt.ResultHash, Mutated: changed, MutationSourceID: mut[321].SourceID, RecomputedNodes: recomputed, ExpectedAncestorPath: base.Levels + 1, UnaffectedNodesReused: reused, FinalCitations: len(cites), CitationsResolved: resolved, MinorityOutlierPreserved: hasCitation(base.Root, "issue-00777"), UncertaintyPreserved: base.Root.StatusCounts["abstain"] == 1 && hasCitation(base.Root, "issue-00778"), NegativeAuditSampled: audit, NegativeAuditPassed: passed, Notes: []string{"counts, sets, and stable top-k are deterministic under reorder and fan-in changes", "semantic clusters retain bounded exemplars and named outliers; they are not claimed lossless", "every reducer consumes only bounded child summaries, never the original corpus", "one changed leaf invalidates only its content-addressed ancestor path"}}
	if err := verifyFoldReport(r); err != nil {
		return fmt.Errorf("%w: recomputed=%d expected=%d reused=%d", err, recomputed, r.ExpectedAncestorPath, reused)
	}
	b, _ := json.MarshalIndent(r, "", "  ")
	if output != "" {
		if err := os.WriteFile(output, append(b, '\n'), 0644); err != nil {
			return err
		}
	}
	fmt.Println(string(b))
	return nil
}
func verifyProvenanceFoldArtifact(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r provenanceFoldReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	return verifyFoldReport(r)
}
