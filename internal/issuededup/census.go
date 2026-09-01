package issuededup

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/simhash"
)

// CensusDefaultThreshold is the pairwise cosine floor the retrospective backlog
// census links two issues at. It is deliberately looser than the write-time
// DefaultThreshold (0.80): the census only *proposes* clusters for a human to
// confirm before closing (the confirm-before-closing-as-dup discipline stands),
// so it favors recall — surfacing a body-similar twin whose title diverged, the
// #2401/#2417 shape the title-only Jaccard census missed — and leans on the
// ranked per-pair evidence to let the reviewer discard a weak match. Tuned
// against the fixture pair in census_test.go: the body-similar twins cluster
// well above it while an unrelated same-template pair stays below.
const CensusDefaultThreshold = 0.62

// CensusDefaultTopK bounds the neighbor fan-out per issue before union-find, so
// the census stays O(n·k) over the backlog instead of O(n²) all-pairs — the
// "simhash TopK + union-find" shape. A dup cluster is small, so a dozen
// neighbors is ample headroom over any real twin set.
const CensusDefaultTopK = 12

// CensusMaxCluster is the size above which a connected component is treated as a
// shared-template FAMILY rather than a duplicate SET, and suppressed from the
// proposals (its size is still reported, never silently dropped).
//
// Body-aware matching is recall-first by design (it links a body-similar twin
// whose title diverged), but on a real backlog a producer's epic sub-issues all
// carry the same long templated body — distinct work ("Dimension E" vs
// "Dimension H") that is near-identical on the body axis (0.92) yet is NOT a
// duplicate of each other. Single-linkage union-find over that dense template
// clique collapses the whole family into ONE component, which both proposes an
// absurd "close 400 issues as a dup of one" and BURIES the genuine small twin
// pairs inside the blob. A real accidental-duplicate set is small (two or a few
// filings of the same work); a component larger than this cap is almost always a
// template family, so the census refuses to propose it and points the reviewer
// at the shared epic/template instead. The simhash title+body signal alone
// cannot separate a same-epic sibling from a true twin — this size guard is the
// conservative backstop that keeps the report usable.
const CensusMaxCluster = 6

// The exact-title pre-pass is intentionally much narrower than the fuzzy
// census. A dominant-prefix extension must share at least this much of the
// shorter normalized body and add a material suffix. The absolute floors keep
// a common one-line template from becoming exact evidence.
const (
	censusExactBodyMinChars       = 256
	censusSupersetPrefixMinChars  = 512
	censusSupersetPrefixMinRatio  = 0.80
	censusSupersetSuffixMinChars  = 128
	MatchedOnExactBodySuperset    = "exact-title+body-superset"
	CensusReasonCosineThreshold   = "cosine-threshold"
	CensusReasonExactBodySuperset = "exact-title-body-superset"
)

// pathRe matches a slash-bearing, extensioned path token in an issue body
// (internal/gateway/foo.go, cmd/fak/x.go, tools/y.py, docs/z.md) so two issues
// that name the same file are cited as sharing it. A bare `./internal/gateway`
// test path has no extension and is intentionally not matched; shared paths are
// bonus evidence, never the matcher.
var pathRe = regexp.MustCompile(`(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+`)

// PairEvidence is the auditable justification for linking two issues — never a
// bare verdict. It carries the similarity on BOTH axes (title-only and
// title+body) so the report shows both scores rather than silently changing the
// metric, the axis the link matched on, the file paths and labels the two
// issues share, and a short prose excerpt from each so a reviewer sees *what*
// overlapped, not only a number.
type PairEvidence struct {
	A          int     `json:"a"`
	B          int     `json:"b"`
	Similarity float64 `json:"similarity"`
	TitleScore float64 `json:"title_score"`
	BodyScore  float64 `json:"body_score"`
	MatchedOn  string  `json:"matched_on"`
	Reason     string  `json:"reason"`
	// Exact-title/body-superset evidence is deliberately numeric and therefore
	// falsifiable from the two normalized bodies. These fields stay zero for a
	// cosine link.
	CommonPrefixChars int      `json:"common_prefix_chars,omitempty"`
	ShorterBodyChars  int      `json:"shorter_body_chars,omitempty"`
	LongerBodyChars   int      `json:"longer_body_chars,omitempty"`
	SharedPaths       []string `json:"shared_paths,omitempty"`
	SharedLabels      []string `json:"shared_labels,omitempty"`
	ExcerptA          string   `json:"excerpt_a,omitempty"`
	ExcerptB          string   `json:"excerpt_b,omitempty"`
}

// Cluster is a connected component of dup-suspected issues plus the ranked
// pairwise evidence that connects them. Keep is the proposed canonical issue to
// retain (the lowest number — the oldest filing); CloseAsDup lists the rest,
// proposed to close as duplicates of Keep *after human confirmation*. The
// census never writes to GitHub.
type Cluster struct {
	ID         int            `json:"id"`
	Members    []int          `json:"members"`
	Keep       int            `json:"keep"`
	CloseAsDup []int          `json:"close_as_dup"`
	TopScore   float64        `json:"top_score"`
	Pairs      []PairEvidence `json:"pairs"`
}

// CensusReport is the whole ranked cluster report the issue gardener consumes,
// as markdown or JSON. Clusters are ranked by their strongest pair first.
type CensusReport struct {
	Threshold float64   `json:"threshold"`
	TopK      int       `json:"topk"`
	Issues    int       `json:"issues"`
	Clusters  []Cluster `json:"clusters"`
	// SuppressedFamilies/SuppressedIssues account for the components larger than
	// CensusMaxCluster that were withheld as shared-template families rather than
	// proposed as duplicate sets — reported so the cap is never a silent drop.
	SuppressedFamilies int `json:"suppressed_families"`
	SuppressedIssues   int `json:"suppressed_issues"`
}

// censusRow is one issue's precomputed vectors and evidence metadata.
type censusRow struct {
	title     string
	normTitle string
	body      string
	titleV    simhash.Vector
	combV     simhash.Vector
	paths     map[string]bool
	labels    map[string]bool
	excerpt   string
}

// Census builds the body-aware duplicate-cluster report over a read-only
// backlog. It embeds each issue on the title and the title+body axes, finds the
// top-k combined-vector neighbors of each, links a pair whose best axis clears
// threshold, unions the links into clusters, and ranks the clusters by their
// strongest pair. threshold <= 0 uses CensusDefaultThreshold; topK <= 0 uses
// CensusDefaultTopK. It performs no I/O and no gh write — cached reads in,
// ranked proposals out. A repeated issue number keeps its last row (simhash
// Add-replaces), so overlapping paged reads never double-count one issue.
func Census(issues []BacklogIssue, threshold float64, topK int) CensusReport {
	if threshold <= 0 {
		threshold = CensusDefaultThreshold
	}
	if topK <= 0 {
		topK = CensusDefaultTopK
	}
	rows, order, comb := buildCensusIndex(issues)
	groups := newCensusGroups(order)
	pairs := findCensusPairs(rows, order, comb, groups, threshold, topK)
	clusters, suppressedFamilies, suppressedIssues := buildCensusClusters(order, pairs, groups)

	return CensusReport{
		Threshold:          threshold,
		TopK:               topK,
		Issues:             len(order),
		Clusters:           clusters,
		SuppressedFamilies: suppressedFamilies,
		SuppressedIssues:   suppressedIssues,
	}
}

func buildCensusIndex(issues []BacklogIssue) (map[int]*censusRow, []int, simhash.Index) {
	rows := make(map[int]*censusRow, len(issues))
	order := make([]int, 0, len(issues))
	var comb simhash.Index
	for _, is := range issues {
		title := strings.TrimSpace(is.Title)
		if is.Number <= 0 || title == "" {
			continue
		}
		if _, seen := rows[is.Number]; !seen {
			order = append(order, is.Number)
		}
		nb := normalizeBody(is.Body)
		r := &censusRow{
			title:     title,
			normTitle: normalizeCensusTitle(title),
			body:      nb,
			titleV:    simhash.Embed(title),
			combV:     simhash.Embed(title + "\n" + nb),
			paths:     tokenSet(pathRe.FindAllString(is.Body, -1)),
			labels:    labelSet(is.Labels),
			excerpt:   excerpt(nb),
		}
		rows[is.Number] = r
		comb.Add(strconv.Itoa(is.Number), r.combV, "")
	}
	return rows, order, comb
}

type censusGroups map[int]int

func newCensusGroups(order []int) censusGroups {
	parent := make(censusGroups, len(order))
	for _, n := range order {
		parent[n] = n
	}
	return parent
}

func (g censusGroups) find(x int) int {
	for g[x] != x {
		g[x] = g[g[x]]
		x = g[x]
	}
	return x
}

func (g censusGroups) union(a, b int) {
	ra, rb := g.find(a), g.find(b)
	if ra != rb {
		g[ra] = rb
	}
}

func findCensusPairs(rows map[int]*censusRow, order []int, comb simhash.Index, groups censusGroups, threshold float64, topK int) []PairEvidence {
	seen := make(map[[2]int]bool)
	exactMembers := make(map[int]bool)
	pairs := make([]PairEvidence, 0)

	// Strong evidence must not enter through the approximate top-k path: a
	// dense template family can push a real superset outside top-k, or absorb it
	// into an oversized component that the family guard suppresses. Group exact
	// normalized titles first, then test the bounded body predicate. Members of
	// an exact component are withheld from fuzzy unioning so that component
	// remains independently reviewable rather than disappearing into a family.
	numbers := append([]int(nil), order...)
	sort.Ints(numbers)
	pairs = appendExactCensusPairs(pairs, rows, numbers, seen, exactMembers, groups)
	return appendFuzzyCensusPairs(pairs, rows, order, comb, seen, exactMembers, groups, threshold, topK)
}

func appendExactCensusPairs(pairs []PairEvidence, rows map[int]*censusRow, numbers []int, seen map[[2]int]bool, exactMembers map[int]bool, groups censusGroups) []PairEvidence {
	for i, n := range numbers {
		r := rows[n]
		for _, other := range numbers[i+1:] {
			o := rows[other]
			if r.normTitle != o.normTitle {
				continue
			}
			prefix, shorter, longer, ok := exactBodySupersetEvidence(r.body, o.body)
			if !ok {
				continue
			}
			key := [2]int{n, other}
			seen[key] = true
			exactMembers[n], exactMembers[other] = true, true
			groups.union(n, other)
			pairs = append(pairs, PairEvidence{
				A: n, B: other, Similarity: 1,
				TitleScore: simhash.Cosine(r.titleV, o.titleV), BodyScore: simhash.Cosine(r.combV, o.combV),
				MatchedOn: MatchedOnExactBodySuperset, Reason: CensusReasonExactBodySuperset,
				CommonPrefixChars: prefix, ShorterBodyChars: shorter, LongerBodyChars: longer,
				SharedPaths: sharedKeys(r.paths, o.paths), SharedLabels: sharedKeys(r.labels, o.labels),
				ExcerptA: r.excerpt, ExcerptB: o.excerpt,
			})
		}
	}
	return pairs
}

func appendFuzzyCensusPairs(pairs []PairEvidence, rows map[int]*censusRow, order []int, comb simhash.Index, seen map[[2]int]bool, exactMembers map[int]bool, groups censusGroups, threshold float64, topK int) []PairEvidence {
	for _, n := range order {
		if exactMembers[n] {
			continue
		}
		r := rows[n]
		for _, mt := range comb.TopK(r.combV, topK+1) {
			other, err := strconv.Atoi(mt.ID)
			if err != nil || other == n || exactMembers[other] {
				continue
			}
			key := [2]int{min(n, other), max(n, other)}
			if seen[key] {
				continue
			}
			o := rows[other]
			titleScore := simhash.Cosine(r.titleV, o.titleV)
			bodyScore := mt.Score // Cosine(r.combV, o.combV): the title+body axis.
			sim, axis := bodyScore, MatchedOnTitleBody
			if titleScore > sim {
				sim, axis = titleScore, MatchedOnTitle
			}
			if sim < threshold {
				continue
			}
			seen[key] = true
			groups.union(n, other)
			lo, hi := rows[key[0]], rows[key[1]]
			pairs = append(pairs, PairEvidence{
				A:            key[0],
				B:            key[1],
				Similarity:   sim,
				TitleScore:   titleScore,
				BodyScore:    bodyScore,
				MatchedOn:    axis,
				Reason:       CensusReasonCosineThreshold,
				SharedPaths:  sharedKeys(lo.paths, hi.paths),
				SharedLabels: sharedKeys(lo.labels, hi.labels),
				ExcerptA:     lo.excerpt,
				ExcerptB:     hi.excerpt,
			})
		}
	}
	return pairs
}

func buildCensusClusters(order []int, pairs []PairEvidence, groups censusGroups) ([]Cluster, int, int) {
	members := make(map[int][]int)
	for _, n := range order {
		root := groups.find(n)
		members[root] = append(members[root], n)
	}
	clusters := make([]Cluster, 0)
	suppressedFamilies, suppressedIssues := 0, 0
	for root, ms := range members {
		if len(ms) < 2 {
			continue
		}
		hasExactEvidence := false
		for _, p := range pairs {
			if p.Reason == CensusReasonExactBodySuperset && groups.find(p.A) == root {
				hasExactEvidence = true
				break
			}
		}
		if len(ms) > CensusMaxCluster && !hasExactEvidence {
			// A component this large is a shared-template family, not a duplicate
			// set — withhold it from the proposals but account for it so the cap
			// is transparent, never a silent drop.
			suppressedFamilies++
			suppressedIssues += len(ms)
			continue
		}
		sort.Ints(ms)
		cp := make([]PairEvidence, 0)
		top := 0.0
		for _, p := range pairs {
			if groups.find(p.A) != root {
				continue
			}
			cp = append(cp, p)
			if p.Similarity > top {
				top = p.Similarity
			}
		}
		sort.Slice(cp, func(i, j int) bool {
			if cp[i].Similarity != cp[j].Similarity {
				return cp[i].Similarity > cp[j].Similarity
			}
			if cp[i].A != cp[j].A {
				return cp[i].A < cp[j].A
			}
			return cp[i].B < cp[j].B
		})
		clusters = append(clusters, Cluster{
			Members:    ms,
			Keep:       ms[0],
			CloseAsDup: append([]int(nil), ms[1:]...),
			TopScore:   top,
			Pairs:      cp,
		})
	}
	sort.Slice(clusters, func(i, j int) bool {
		if clusters[i].TopScore != clusters[j].TopScore {
			return clusters[i].TopScore > clusters[j].TopScore
		}
		return clusters[i].Members[0] < clusters[j].Members[0]
	})
	for i := range clusters {
		clusters[i].ID = i
	}
	return clusters, suppressedFamilies, suppressedIssues
}

// RenderCensus renders a census report as the markdown the issue gardener reads:
// a header line with the run parameters, then one section per proposed cluster —
// the canonical issue to keep, the ones proposed to close as duplicates, and the
// ranked per-pair evidence (similarity on both axes, the axis it matched on, the
// shared labels/paths, and an excerpt from each issue). Every proposal names its
// evidence; the census never writes to GitHub, so the report is advisory only.
func RenderCensus(rep CensusReport) string {
	var b strings.Builder
	b.WriteString("# Backlog duplicate census\n\n")
	fmt.Fprintf(&b, "_%d open issue(s) · threshold %.2f · top-k %d · %d cluster(s)_\n\n",
		rep.Issues, rep.Threshold, rep.TopK, len(rep.Clusters))
	if rep.SuppressedFamilies > 0 {
		fmt.Fprintf(&b, "_Suppressed %d template-family component(s) covering %d issue(s) larger than the %d-issue dup cap — likely shared-template epic siblings, not duplicate sets; review the shared epic/template directly._\n\n",
			rep.SuppressedFamilies, rep.SuppressedIssues, CensusMaxCluster)
	}
	if len(rep.Clusters) == 0 {
		b.WriteString("No duplicate clusters found. Advisory only — the census never writes to GitHub.\n")
		return b.String()
	}
	b.WriteString("Advisory — confirm before closing as dup. The census never writes to GitHub.\n")
	for _, c := range rep.Clusters {
		fmt.Fprintf(&b, "\n## Cluster %d — keep #%d, close %d as dup (top similarity %.2f)\n",
			c.ID, c.Keep, len(c.CloseAsDup), c.TopScore)
		fmt.Fprintf(&b, "Proposed: keep **#%d**", c.Keep)
		if len(c.CloseAsDup) > 0 {
			parts := make([]string, len(c.CloseAsDup))
			for i, n := range c.CloseAsDup {
				parts[i] = "#" + strconv.Itoa(n)
			}
			fmt.Fprintf(&b, ", close %s as duplicate(s)", strings.Join(parts, ", "))
		}
		b.WriteString(".\nEvidence:\n")
		for _, p := range c.Pairs {
			fmt.Fprintf(&b, "- #%d ↔ #%d — similarity %.2f on %s (title %.2f / body %.2f; reason %s)\n",
				p.A, p.B, p.Similarity, p.MatchedOn, p.TitleScore, p.BodyScore, p.Reason)
			if p.Reason == CensusReasonExactBodySuperset {
				fmt.Fprintf(&b, "  - exact normalized title; common body prefix: %d/%d shorter chars; longer body: %d chars\n",
					p.CommonPrefixChars, p.ShorterBodyChars, p.LongerBodyChars)
			}
			if len(p.SharedLabels) > 0 {
				fmt.Fprintf(&b, "  - shared labels: %s\n", strings.Join(p.SharedLabels, ", "))
			}
			if len(p.SharedPaths) > 0 {
				fmt.Fprintf(&b, "  - shared paths: %s\n", strings.Join(p.SharedPaths, ", "))
			}
			if p.ExcerptA != "" {
				fmt.Fprintf(&b, "  - #%d: %s\n", p.A, p.ExcerptA)
			}
			if p.ExcerptB != "" {
				fmt.Fprintf(&b, "  - #%d: %s\n", p.B, p.ExcerptB)
			}
		}
	}
	return b.String()
}

// normalizeCensusTitle is the exact-title identity used only by the census
// pre-pass. It folds case and whitespace without rewriting punctuation or
// vocabulary, keeping the predicate materially narrower than semantic title
// matching.
func normalizeCensusTitle(title string) string {
	return strings.ToLower(strings.Join(strings.Fields(title), " "))
}

// exactBodySupersetEvidence recognizes two high-confidence shapes after exact
// normalized-title equality: literal containment (including identical bodies),
// or a dominant identical prefix followed by a materially longer suffix. The
// latter admits a duplicate filing that lightly edits its final section before
// appending a work breakdown, while rejecting equal-length template siblings.
func exactBodySupersetEvidence(a, b string) (prefix, shorter, longer int, ok bool) {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	short, long := a, b
	if len(short) > len(long) {
		short, long = long, short
	}
	shorter, longer = len(short), len(long)
	if shorter < censusExactBodyMinChars {
		return 0, shorter, longer, false
	}
	if strings.HasPrefix(long, short) {
		return shorter, shorter, longer, true
	}
	for prefix < shorter && short[prefix] == long[prefix] {
		prefix++
	}
	if prefix < censusSupersetPrefixMinChars || longer-shorter < censusSupersetSuffixMinChars {
		return prefix, shorter, longer, false
	}
	return prefix, shorter, longer, float64(prefix)/float64(shorter) >= censusSupersetPrefixMinRatio
}

// labelSet folds an issue's gh labels into a name set for the shared-label
// evidence. Empty/blank names are dropped.
func labelSet(labels []Label) map[string]bool {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]bool, len(labels))
	for _, l := range labels {
		if name := strings.TrimSpace(l.Name); name != "" {
			out[name] = true
		}
	}
	return out
}

// tokenSet folds a slice of tokens into a presence set.
func tokenSet(tokens []string) map[string]bool {
	if len(tokens) == 0 {
		return nil
	}
	out := make(map[string]bool, len(tokens))
	for _, t := range tokens {
		out[t] = true
	}
	return out
}

// sharedKeys returns the sorted intersection of two presence sets — the shared
// paths or labels a pair cites as evidence.
func sharedKeys(a, b map[string]bool) []string {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	var out []string
	for k := range a {
		if b[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// excerpt is the first substantive prose line of a normalized body, truncated
// for a compact report cell. normalizeBody has already dropped the marker
// comment and the template section headings, so the first line is the issue's
// own problem statement — the evidence a reviewer reads, not a bare score.
func excerpt(normalizedBody string) string {
	line := normalizedBody
	if i := strings.IndexByte(line, '\n'); i >= 0 {
		line = line[:i]
	}
	line = strings.TrimSpace(line)
	const max = 140
	if r := []rune(line); len(r) > max {
		return strings.TrimSpace(string(r[:max])) + "…"
	}
	return line
}
