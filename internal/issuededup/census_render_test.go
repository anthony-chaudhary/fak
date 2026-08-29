package issuededup

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestRenderCensusNamesEvidenceNotBareVerdict pins the acceptance gate's
// headline clause — "every proposal names its evidence (pair similarity +
// matched excerpts), never a bare verdict" — at the layer the issue gardener
// actually reads: the rendered markdown. Census() (the data) is covered by
// census_test.go; this covers RenderCensus() (the report). It asserts the
// rendered proposal for the #3001/#3002 body-twin carries the keep/close
// proposal, BOTH axis scores (the "show both scores rather than silently
// changing the metric" transition discipline), the matched excerpt prose, the
// shared-label evidence, and the confirm-before-closing advisory — so a bare
// verdict can never render green.
func TestRenderCensusNamesEvidenceNotBareVerdict(t *testing.T) {
	md := RenderCensus(Census(censusBacklog, 0, 0))

	wants := []string{
		"# Backlog duplicate census", // the gardener's report header
		"**#3001**",                  // the proposed canonical (keep) issue
		"#3001 ↔ #3002",              // the twin pair, named — not an anonymous count
		"(title ",                    // the title-only axis score is shown...
		"/ body ",                    // ...alongside the title+body axis score (both, never one)
		"on " + MatchedOnTitleBody,   // which axis the link matched on
		"shared labels: ",            // shared-label evidence surfaced
		"loader",                     // ...the actual shared label, not just the header
		"safetensors",                // matched excerpt prose (evidence, not a bare number)
		"  - #3001: ",                // the excerpt cell for the kept issue
		"Advisory — confirm before closing as dup", // the confirm-before-closing discipline stands
		"never writes to GitHub",                   // the census is read-only/advisory, stated in-report
	}
	for _, w := range wants {
		if !strings.Contains(md, w) {
			t.Errorf("rendered census missing evidence fragment %q\n--- report ---\n%s", w, md)
		}
	}
}

// TestRenderCensusEmptyIsAdvisoryOnly covers the no-cluster branch: an empty
// backlog renders the advisory-only footer, never a false duplicate claim.
func TestRenderCensusEmptyIsAdvisoryOnly(t *testing.T) {
	md := RenderCensus(Census(nil, 0, 0))
	if !strings.Contains(md, "No duplicate clusters found") {
		t.Errorf("empty census should render the no-clusters advisory, got:\n%s", md)
	}
	if strings.Contains(md, "close ") {
		t.Errorf("empty census must not propose any close:\n%s", md)
	}
}

// TestCensusJSONCarriesPerPairEvidence pins the "markdown + JSON output the
// gardener can consume" in-scope item at the JSON axis: the same CensusReport
// the `fak issue dedup --json` path serializes must carry the per-pair evidence
// fields (both axis scores, the matched axis, the excerpts, the shared labels),
// so the machine-readable report is never a bare verdict either.
func TestCensusJSONCarriesPerPairEvidence(t *testing.T) {
	rep := Census(censusBacklog, 0, 0)
	if len(rep.Clusters) == 0 || len(rep.Clusters[0].Pairs) == 0 {
		t.Fatalf("fixture census produced no evidenced cluster to serialize: %+v", rep)
	}
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		t.Fatalf("marshal census report: %v", err)
	}
	got := string(b)
	for _, key := range []string{
		`"title_score"`, `"body_score"`, `"matched_on"`,
		`"excerpt_a"`, `"excerpt_b"`, `"shared_labels"`,
	} {
		if !strings.Contains(got, key) {
			t.Errorf("census JSON dropped per-pair evidence key %s\n%s", key, got)
		}
	}
}

func TestRenderCensusNamesExactSupersetReasonAndMeasurements(t *testing.T) {
	rep := Census(exactSupersetBacklog(), 0, 0)
	md := RenderCensus(rep)
	for _, want := range []string{
		"#6001 ↔ #6002",
		"on " + MatchedOnExactBodySuperset,
		"reason " + CensusReasonExactBodySuperset,
		"exact normalized title",
		"common body prefix:",
		"shorter chars; longer body:",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("rendered exact/superset proposal missing %q\n--- report ---\n%s", want, md)
		}
	}

	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal exact/superset report: %v", err)
	}
	for _, want := range []string{
		`"reason":"` + CensusReasonExactBodySuperset + `"`,
		`"common_prefix_chars"`,
		`"shorter_body_chars"`,
		`"longer_body_chars"`,
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("exact/superset JSON evidence missing %s: %s", want, b)
		}
	}
}
