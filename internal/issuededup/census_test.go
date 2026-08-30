package issuededup

import (
	"fmt"
	"strings"
	"testing"
)

// censusBacklog is a fixture backlog for the retrospective census. #3001/#3002
// are the #2401/#2417 shape: DIFFERENT titles (one verb-led feat, one prose
// restatement) whose BODIES describe the same lazy-shard loader change — the
// body-only twin the title-Jaccard census was blind to. #3100 (a docs gallery
// issue) and #3200 (a dispatch backoff fix) are unrelated work carrying the same
// issue-template skeleton, so the census must leave them apart on body prose,
// not cluster them on the shared headings normalizeBody strips.
var censusBacklog = []BacklogIssue{
	{
		Number: 3001,
		Title:  "feat(loader): stream safetensors shards lazily to cut peak RSS",
		Labels: []Label{{Name: "enhancement"}, {Name: "loader"}},
		Body: "<!-- fak-idea-scout-key: loader-lazy-shards -->\n" +
			"## Current state\n" +
			"The loader reads every safetensors shard fully resident before the first forward pass, so peak resident memory is the whole checkpoint even though decode only touches a few tensors at a time.\n" +
			"## In scope\n" +
			"Stream each shard lazily with an mmap-backed reader and materialize a tensor only when a layer first touches it, so peak RSS tracks the working set instead of the full checkpoint size.\n" +
			"## Done condition\n" +
			"Loading a multi-shard checkpoint holds only the touched tensors resident, witnessed by the loader RSS test.\n" +
			"## Likely files\n" +
			"internal/loader/shard.go\n" +
			"## Witness\n" +
			"go test ./internal/loader -run TestLazyShardRSS\n",
	},
	{
		Number: 3002,
		Title:  "reduce model load memory: don't hold every tensor resident at startup",
		Labels: []Label{{Name: "enhancement"}, {Name: "loader"}},
		Body: "<!-- fak-dogfood-key: loader-resident-memory -->\n" +
			"## Current state\n" +
			"At load the loader pulls every safetensors shard fully into resident memory before the first forward pass, so peak RSS is the entire checkpoint size even though only a handful of tensors are touched per step.\n" +
			"## In scope\n" +
			"Read shards lazily behind an mmap reader and materialize a tensor only the first time a layer touches it, so peak resident memory tracks the working set rather than the whole checkpoint.\n" +
			"## Done condition\n" +
			"A multi-shard checkpoint keeps only touched tensors resident during load, witnessed by the loader memory test.\n" +
			"## Likely files\n" +
			"internal/loader/shard.go\n" +
			"## Witness\n" +
			"go test ./internal/loader -run TestResidentSet\n",
	},
	{
		Number: 3100,
		Title:  "docs(gpu): lay out the benchmark gallery two columns wide",
		Labels: []Label{{Name: "documentation"}},
		Body: "<!-- fak-dogfood-key: gpu-gallery-layout -->\n" +
			"## Current state\n" +
			"The benchmark gallery renders one figure per row so a reader scrolls a long way to compare two GPU families side by side.\n" +
			"## In scope\n" +
			"Lay the gallery figures out two columns wide on wide viewports so paired families sit next to each other.\n" +
			"## Done condition\n" +
			"The gallery renders two columns on a wide viewport, checked by the docs lint.\n" +
			"## Witness\n" +
			"make docs-lint\n",
	},
	{
		Number: 3200,
		Title:  "fix(dispatch): retry a gh rate-limit response with capped backoff",
		Labels: []Label{{Name: "bug"}, {Name: "dispatch"}},
		Body: "<!-- fak-dogfood-key: dispatch-gh-backoff -->\n" +
			"## Current state\n" +
			"A gh call that returns a rate-limit error aborts the dispatch tick immediately, dropping the rest of the wave instead of waiting out the reset.\n" +
			"## In scope\n" +
			"Retry a rate-limited gh call with exponential backoff capped at the reset window, then continue the wave.\n" +
			"## Done condition\n" +
			"A rate-limited call is retried and the wave completes, witnessed by the dispatch backoff test.\n" +
			"## Witness\n" +
			"go test ./internal/dispatchtick -run TestRateLimitBackoff\n",
	},
}

// clusterOf returns the cluster containing issue n, or nil.
func clusterOf(rep CensusReport, n int) *Cluster {
	for i := range rep.Clusters {
		for _, m := range rep.Clusters[i].Members {
			if m == n {
				return &rep.Clusters[i]
			}
		}
	}
	return nil
}

// TestCensusClustersBodySimilarPair — the done condition's first half: the
// title-divergent, body-similar pair #3001/#3002 clusters together, and the
// cluster carries per-pair evidence (similarity on both axes + matched
// excerpts), never a bare verdict.
func TestCensusClustersBodySimilarPair(t *testing.T) {
	rep := Census(censusBacklog, 0, 0)

	c := clusterOf(rep, 3001)
	if c == nil {
		t.Fatalf("issue #3001 was not clustered; want it grouped with its body-twin #3002\nreport: %+v", rep)
	}
	if got := clusterOf(rep, 3002); got == nil || got.ID != c.ID {
		t.Fatalf("#3002 not in the same cluster as #3001: %+v", rep.Clusters)
	}
	if len(c.Members) != 2 {
		t.Fatalf("cluster members = %v, want exactly {3001,3002}", c.Members)
	}
	if c.Keep != 3001 {
		t.Fatalf("cluster keep = %d, want the oldest (lowest) number 3001", c.Keep)
	}
	if len(c.Pairs) == 0 {
		t.Fatalf("cluster carries no pair evidence; a proposal must never be a bare verdict")
	}
	p := c.Pairs[0]
	if p.Similarity < CensusDefaultThreshold {
		t.Fatalf("top pair similarity %.3f below threshold %.2f", p.Similarity, CensusDefaultThreshold)
	}
	// Body-only twins: the title+body axis must be what matched, not the titles.
	if p.MatchedOn != MatchedOnTitleBody {
		t.Fatalf("matched_on = %q, want %q for a body-similar/title-divergent twin", p.MatchedOn, MatchedOnTitleBody)
	}
	if p.BodyScore <= p.TitleScore {
		t.Fatalf("expected body axis to dominate for divergent titles: title=%.3f body=%.3f", p.TitleScore, p.BodyScore)
	}
	if p.ExcerptA == "" || p.ExcerptB == "" {
		t.Fatalf("pair evidence lost a matched excerpt: %+v", p)
	}
	// Both issues carry the loader labels — shared-label evidence must surface.
	if len(p.SharedLabels) == 0 {
		t.Fatalf("pair evidence dropped the shared labels (enhancement, loader): %+v", p)
	}
}

// TestCensusLeavesUnrelatedApart — the done condition's second half: the two
// unrelated issues (#3100 docs, #3200 dispatch), carrying the same template
// skeleton, are never clustered — not with the twins and not with each other.
func TestCensusLeavesUnrelatedApart(t *testing.T) {
	rep := Census(censusBacklog, 0, 0)

	if len(rep.Clusters) != 1 {
		t.Fatalf("clusters = %d, want exactly 1 (the #3001/#3002 twins)\n%+v", len(rep.Clusters), rep.Clusters)
	}
	for _, n := range []int{3100, 3200} {
		if c := clusterOf(rep, n); c != nil {
			t.Fatalf("unrelated issue #%d was clustered into %+v", n, c)
		}
	}
}

// TestCensusSuppressesTemplateFamily — the anti-avalanche guard. A producer's
// epic sub-issues share one long templated body: distinct work ("Dimension A"
// vs "Dimension B") that is near-identical on the body axis, NOT duplicates of
// each other. Single-linkage union-find would otherwise glue the whole family
// into one giant component — proposing an absurd "close them all as a dup of
// one" and burying the genuine small twin pairs. The census must suppress any
// component larger than CensusMaxCluster as a template family (reported via
// SuppressedFamilies/SuppressedIssues, never a silent drop) while still
// surfacing a real title-divergent, body-similar twin pair in the same backlog.
func TestCensusSuppressesTemplateFamily(t *testing.T) {
	// A long shared body block identical across the family, so the family is a
	// dense high-similarity clique the naive union-find would collapse.
	shared := "## Current state\n" +
		"This work is one facet of the concept-popularization epic tracked in " +
		"docs/notes/CONCEPT-POPULARIZATION-EPIC.md, which coordinates a dozen " +
		"parallel adoption workstreams under a single shared template so each " +
		"sub-issue reads the same skeleton with only its facet name changed.\n" +
		"## In scope\n" +
		"Advance this facet against the shared scorecard behind `fak score seo` " +
		"and report progress on the epic's shared dashboard for the gardener to track.\n" +
		"## Done condition\n" +
		"The facet clears the shared scorecard bar and the epic dashboard reflects it.\n"

	family := make([]BacklogIssue, 0, CensusMaxCluster+2)
	for i := 0; i < CensusMaxCluster+2; i++ {
		family = append(family, BacklogIssue{
			Number: 4000 + i,
			Title:  fmt.Sprintf("Dimension %c — distinct adoption facet of the popularization epic", 'A'+i),
			Labels: []Label{{Name: "adoption"}, {Name: "popularization"}},
			Body:   shared + "## Facet\n" + fmt.Sprintf("facet-%d\n", i),
		})
	}

	// A genuine title-divergent, body-similar twin pair (the #2401/#2417 shape),
	// unrelated to the family, which must still surface after the family is cut.
	family = append(family,
		BacklogIssue{
			Number: 5001,
			Title:  "feat(cache): evict cold radix entries under memory pressure",
			Body: "## Current state\n" +
				"The radix cache never evicts cold entries, so it grows without bound and " +
				"peak resident memory climbs until the process is OOM-killed under sustained load.\n" +
				"## In scope\n" +
				"Evict least-recently-touched radix entries once memory pressure crosses the " +
				"high-water mark, so the cache tracks the working set instead of growing forever.\n",
		},
		BacklogIssue{
			Number: 5002,
			Title:  "reclaim radix cache memory by dropping stale entries when pressure is high",
			Body: "## Current state\n" +
				"Under sustained load the radix cache grows without bound because cold entries " +
				"are never reclaimed, so peak resident memory climbs until an OOM kill.\n" +
				"## In scope\n" +
				"Drop least-recently-used radix entries once memory pressure passes the high-water " +
				"threshold, so the cache tracks the working set rather than growing unbounded.\n",
		},
	)

	rep := Census(family, 0, 0)

	if rep.SuppressedFamilies == 0 {
		t.Fatalf("template family of %d was not suppressed: %+v", CensusMaxCluster+2, rep)
	}
	if rep.SuppressedIssues < CensusMaxCluster+1 {
		t.Fatalf("suppressed issue count = %d, want the whole family (>= %d)", rep.SuppressedIssues, CensusMaxCluster+1)
	}
	for _, c := range rep.Clusters {
		if len(c.Members) > CensusMaxCluster {
			t.Fatalf("emitted cluster %d has %d members, above the %d cap (avalanche leaked)", c.ID, len(c.Members), CensusMaxCluster)
		}
	}
	// The genuine twin pair still surfaces as its own small cluster.
	c := clusterOf(rep, 5001)
	if c == nil {
		t.Fatalf("genuine twin #5001 was lost when the family was suppressed: %+v", rep.Clusters)
	}
	if got := clusterOf(rep, 5002); got == nil || got.ID != c.ID {
		t.Fatalf("twin #5001/#5002 not clustered together: %+v", rep.Clusters)
	}
	if len(c.Members) != 2 {
		t.Fatalf("twin cluster = %v, want exactly {5001,5002}", c.Members)
	}
}

// exactSupersetBacklog reproduces the live duplicate shape without coupling
// production logic or the regression to live issue numbers: two normalized-
// title-equal filings share more than four fifths of the shorter body, the
// later filing lightly edits the final section, then appends a work breakdown.
// The adjacent control is a legitimate template family with equal titles but
// distinct bodies; title equality alone must not let it escape suppression.
func exactSupersetBacklog() []BacklogIssue {
	shared := strings.Repeat(
		"The architecture contract keeps model identity, resource lifecycle, and receipt evidence explicit across every execution plane. ",
		18,
	)
	tail := strings.Repeat("The existing contract remains the authority for compatibility and rollback. ", 5)
	issues := []BacklogIssue{
		{
			Number: 6001,
			Title:  "feat(model): compose model runtimes through one architecture contract",
			Labels: []Label{{Name: "architecture"}, {Name: "model"}},
			Body: "## Current state\n" + shared +
				"The follow-on issues should cover descriptors and observability.\n" + tail,
		},
		{
			Number: 6002,
			Title:  "  FEAT(model):   compose model runtimes through one architecture contract ",
			Labels: []Label{{Name: "architecture"}, {Name: "model"}},
			Body: "<!-- producer marker -->\n## Current state\n" + shared +
				"The follow-on issues cover descriptors and observability.\n" + tail +
				strings.Repeat("Work breakdown: implement and witness one bounded composition leaf. ", 8),
		},
	}

	controlShared := strings.Repeat(
		"All facets use the same scorecard, witness format, rollout boundary, and shared epic dashboard. ",
		20,
	)
	for i := 0; i < CensusMaxCluster+2; i++ {
		issues = append(issues, BacklogIssue{
			Number: 6100 + i,
			Title:  "docs(adoption): advance one distinct template-family facet",
			Labels: []Label{{Name: "adoption"}},
			Body:   fmt.Sprintf("Facet %d has a distinct audience, artifact, and acceptance witness.\n%s", i, controlShared),
		})
	}
	return issues
}

func TestCensusExactTitleBodySupersetPrecedesFamilySuppression(t *testing.T) {
	rep := Census(exactSupersetBacklog(), 0, 0)

	c := clusterOf(rep, 6001)
	if c == nil || len(c.Members) != 2 || c.Members[0] != 6001 || c.Members[1] != 6002 {
		t.Fatalf("exact-title/body-superset cluster = %+v, want exactly {6001,6002}\nreport: %+v", c, rep)
	}
	exactPairs := 0
	for _, cluster := range rep.Clusters {
		for _, p := range cluster.Pairs {
			if p.A == 6001 && p.B == 6002 {
				exactPairs++
				if p.MatchedOn != MatchedOnExactBodySuperset || p.Reason != CensusReasonExactBodySuperset {
					t.Fatalf("exact pair lacks typed reason: %+v", p)
				}
				if p.CommonPrefixChars < censusSupersetPrefixMinChars || p.ShorterBodyChars <= 0 || p.LongerBodyChars <= p.ShorterBodyChars {
					t.Fatalf("exact pair lacks falsifiable prefix/length evidence: %+v", p)
				}
			}
		}
	}
	if exactPairs != 1 {
		t.Fatalf("exact pair emitted %d times, want exactly once: %+v", exactPairs, rep.Clusters)
	}
	if rep.SuppressedFamilies != 1 || rep.SuppressedIssues != CensusMaxCluster+2 {
		t.Fatalf("equal-title template control escaped family suppression: families=%d issues=%d report=%+v",
			rep.SuppressedFamilies, rep.SuppressedIssues, rep)
	}
	for n := 6100; n < 6100+CensusMaxCluster+2; n++ {
		if got := clusterOf(rep, n); got != nil {
			t.Fatalf("legitimate template sibling #%d leaked as a duplicate proposal: %+v", n, got)
		}
	}
}
