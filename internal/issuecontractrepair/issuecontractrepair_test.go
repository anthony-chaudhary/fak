package issuecontractrepair

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func issue(number int, title string) issuecontract.IssueDraft {
	return issuecontract.IssueDraft{Number: number, Title: title}
}

func review(ok bool, score int, reasons []string, missing []string) issuecontract.Review {
	return issuecontract.Review{OK: ok, Score: issuecontract.Score{Total: score}, Reasons: reasons, MissingFields: missing}
}

func TestRepairKinds(t *testing.T) {
	if got := RepairKinds([]string{issuecontract.ReasonNotDispatchLeaf}); strings.Join(got, ",") != "split" {
		t.Fatalf("split = %+v", got)
	}
	reasons := []string{issuecontract.ReasonAgentIncomplete, issuecontract.ReasonNoiseIncomplete, issuecontract.ReasonScopeIncomplete, issuecontract.ReasonUnrouted}
	if got := RepairKinds(reasons); strings.Join(got, ",") != "noise,scope,route" {
		t.Fatalf("dedup = %+v", got)
	}
	if got := RepairKinds([]string{"SOMETHING_NEW"}); strings.Join(got, ",") != "other" {
		t.Fatalf("other = %+v", got)
	}
	if got := RepairKinds(nil); strings.Join(got, ",") != "other" {
		t.Fatalf("empty = %+v", got)
	}
	if got := RepairKinds([]string{issuecontract.ReasonUnexpandedTemplate}); strings.Join(got, ",") != "template" {
		t.Fatalf("template = %+v", got)
	}
	if got := PrimaryKind([]string{"route", "split"}); got != "split" {
		t.Fatalf("primary = %q", got)
	}
}

func TestFieldScaffold(t *testing.T) {
	out := FieldScaffold([]string{"done_condition", "witness"})
	if len(out) != 2 || out[0].Field != "done_condition" || !strings.Contains(out[0].Question, "observable state") || !strings.Contains(out[1].Question, "evidence") {
		t.Fatalf("scaffold = %+v", out)
	}
	out = FieldScaffold([]string{"mystery_field"})
	if out[0].Field != "mystery_field" || !strings.Contains(out[0].Question, "mystery_field") {
		t.Fatalf("generic = %+v", out)
	}
}

func TestBuildRepairRow(t *testing.T) {
	if row := BuildRepairRow(issue(1, "done"), review(true, 100, nil, nil), RouteProposal{}, issuecontract.TemplateRepairPlan{}, false, MinScore); row != nil {
		t.Fatalf("passing row = %+v", row)
	}
	row := BuildRepairRow(issue(1207, "fix thing"), review(false, 8, []string{issuecontract.ReasonScopeIncomplete}, []string{"done_condition", "witness"}), RouteProposal{}, issuecontract.TemplateRepairPlan{}, false, MinScore)
	if row == nil || row.Kind != "scope" || row.Ready || len(row.MissingFields) != 2 || row.ProposedLane != nil || row.ProposedHeader != nil {
		t.Fatalf("scope row = %+v", row)
	}
	row = BuildRepairRow(issue(1496, "docs: fix typo"), review(false, 0, []string{issuecontract.ReasonUnrouted}, nil), RouteProposal{Lane: "docs", Confidence: "exact-scope"}, issuecontract.TemplateRepairPlan{}, false, MinScore)
	if row.Kind != "route" || row.ProposedLane == nil || *row.ProposedLane != "docs" || row.RouteConfidence == nil || *row.RouteConfidence != "exact-scope" {
		t.Fatalf("route row = %+v", row)
	}
	row = BuildRepairRow(issue(1612, "unrouted"), review(false, 0, []string{issuecontract.ReasonUnrouted}, nil), RouteProposal{Confidence: "none"}, issuecontract.TemplateRepairPlan{}, false, MinScore)
	if row.Kind != "route" || row.ProposedLane != nil {
		t.Fatalf("unset route = %+v", row)
	}
	plan := issuecontract.TemplateRepairPlan{IssueNumber: 1545, ProposedNormalizedHeader: "N=27 Lane=api/provider"}
	row = BuildRepairRow(issue(1545, "template"), review(false, 25, []string{issuecontract.ReasonUnexpandedTemplate}, nil), RouteProposal{}, plan, true, MinScore)
	if row.Kind != "template" || !row.Ready || row.ProposedHeader == nil || *row.ProposedHeader != "N=27 Lane=api/provider" {
		t.Fatalf("template row = %+v", row)
	}
	row = BuildRepairRow(issue(1545, "template"), review(false, 25, []string{issuecontract.ReasonUnexpandedTemplate}, nil), RouteProposal{}, issuecontract.TemplateRepairPlan{}, false, MinScore)
	if row.Kind != "template" || row.Ready || row.ProposedHeader != nil {
		t.Fatalf("template without plan = %+v", row)
	}
}

func TestManifestSchemaCountsAndLaneFilter(t *testing.T) {
	issues := []issuecontract.IssueDraft{issue(1207, "a"), issue(1852, "b"), issue(2000, "c")}
	reviews := map[int]issuecontract.Review{
		1207: review(false, 8, []string{issuecontract.ReasonScopeIncomplete}, []string{"done_condition"}),
		1852: review(false, 8, []string{issuecontract.ReasonScopeIncomplete}, []string{"done_condition"}),
		2000: review(true, 100, nil, nil),
	}
	manifest := BuildManifest("/repo", issues, Options{
		AsOf: "2026-07-01",
		Review: func(i issuecontract.IssueDraft) issuecontract.Review {
			return reviews[i.Number]
		},
	})
	if manifest.Schema != Schema || manifest.Counts.CandidatesExamined != 3 || manifest.Counts.NeedsRepair != 2 || len(manifest.Issues) != 2 || manifest.Counts.ByKind["scope"] != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}

	filtered := BuildManifest("/repo", []issuecontract.IssueDraft{issue(1, "a"), issue(2, "b"), issue(3, "c")}, Options{
		Lane: "docs", AsOf: "2026-07-01",
		Review: func(i issuecontract.IssueDraft) issuecontract.Review {
			return review(false, 0, []string{issuecontract.ReasonScopeIncomplete}, []string{"done_condition"})
		},
		Route: func(i issuecontract.IssueDraft, _ issuecontract.Review) RouteProposal {
			switch i.Number {
			case 1:
				return RouteProposal{Lane: "docs"}
			case 2:
				return RouteProposal{Lane: "tools"}
			default:
				return RouteProposal{BlockedLane: "docs"}
			}
		},
	})
	if len(filtered.Issues) != 2 || filtered.Issues[0].Number != 1 || filtered.Issues[1].Number != 3 {
		t.Fatalf("filtered = %+v", filtered.Issues)
	}
}

func TestActionsAndRender(t *testing.T) {
	manifest := Manifest{AsOf: "2026-07-01", Counts: Counts{CandidatesExamined: 2, NeedsRepair: 1, ByKind: map[string]int{"scope": 1}}, Issues: []RepairRow{
		{Number: 1207, Kind: "scope", Ready: false, Score: 8, Title: "fix thing", Reasons: []string{issuecontract.ReasonScopeIncomplete}, NextAction: "do x"},
	}}
	actions := BuildActions(manifest)
	if len(actions) != 1 || actions[0].Cmd != nil || actions[0].Reason != issuecontract.ReasonScopeIncomplete {
		t.Fatalf("actions = %+v", actions)
	}
	md := RenderMarkdown(manifest)
	if !strings.Contains(strings.ToLower(md), "issue-contract repairs") || !strings.Contains(md, "#1207") || !strings.Contains(md, "scope") {
		t.Fatalf("markdown = %s", md)
	}
}
