package issuepolicy

import (
	"strings"
	"testing"
)

func TestShiftLeftTaskBriefReadiness(t *testing.T) {
	complete := map[string]string{
		"Value":           "- For: operators choosing the next leaf\n- Problem: intent is reconstructed after dispatch\n- Today: read several records and infer it\n- Better because: one short frame rejects low-value work early",
		"Outcome":         "operators see typed readiness",
		"Scope / tree":    "internal/issuepolicy/** only",
		"Dependencies":    "none",
		"Acceptance":      "focused test passes",
		"Witness / proof": "go test ./internal/issuepolicy -count=1",
		"Placement":       "gen/now; P1; milestone G0; issuecontract lane",
	}
	body := func(fields map[string]string) string {
		order := []string{"Value", "Outcome", "Scope / tree", "Dependencies", "Acceptance", "Witness / proof", "Placement"}
		out := ""
		for _, name := range order {
			if value, ok := fields[name]; ok {
				out += "## " + name + "\n\n" + value + "\n\n"
			}
		}
		return out
	}
	t.Run("ready", func(t *testing.T) {
		got := ReviewIssueDraft(IssueDraft{Title: "add typed task briefs", Body: body(complete)}, Options{})
		if !got.BriefReadiness.Ready || !got.BriefReadiness.Enforced {
			t.Fatalf("readiness = %+v", got.BriefReadiness)
		}
		for name, field := range got.BriefReadiness.Fields {
			if field.Status != "present" {
				t.Fatalf("%s = %+v", name, field)
			}
		}
	})
	t.Run("typed unknown", func(t *testing.T) {
		fields := cloneBriefFields(complete)
		fields["Dependencies"] = "unknown(reason: upstream issue has not been assigned)"
		got := ReviewIssueDraft(IssueDraft{Title: "add typed task briefs", Body: body(fields)}, Options{})
		field := got.BriefReadiness.Fields["dependencies"]
		if !got.BriefReadiness.Ready || field.Status != "unknown" || field.Reason != "upstream issue has not been assigned" {
			t.Fatalf("readiness = %+v", got.BriefReadiness)
		}
	})
	t.Run("missing whole value frame", func(t *testing.T) {
		fields := cloneBriefFields(complete)
		delete(fields, "Value")
		got := ReviewIssueDraft(IssueDraft{Title: "add typed task briefs", Body: body(fields)}, Options{})
		if got.BriefReadiness.Ready || got.Dispatchability == Dispatchable {
			t.Fatalf("review accepted a shift-left brief with no value frame: %+v", got)
		}
		for _, name := range []string{"beneficiary", "problem", "alternative", "advantage"} {
			field := got.BriefReadiness.Fields[name]
			if field.Status != "missing" || field.RepairAction == "" {
				t.Fatalf("%s = %+v, want missing with repair", name, field)
			}
		}
	})
	cases := map[string]string{"outcome": "Outcome", "scope": "Scope / tree", "dependencies": "Dependencies", "acceptance": "Acceptance", "witness": "Witness / proof", "placement": "Placement"}
	for field, heading := range cases {
		t.Run("missing "+field, func(t *testing.T) {
			fields := cloneBriefFields(complete)
			delete(fields, heading)
			got := ReviewIssueDraft(IssueDraft{Title: "add typed task briefs", Body: body(fields)}, Options{})
			state := got.BriefReadiness.Fields[field]
			if got.BriefReadiness.Ready || got.Dispatchability == Dispatchable || state.Status != "missing" || state.RepairAction == "" {
				t.Fatalf("review = %+v", got)
			}
		})
	}

	valueCases := map[string]string{
		"beneficiary": "For:",
		"problem":     "Problem:",
		"alternative": "Today:",
		"advantage":   "Better because:",
	}
	for field, label := range valueCases {
		t.Run("missing value "+field, func(t *testing.T) {
			fields := cloneBriefFields(complete)
			lines := strings.Split(fields["Value"], "\n")
			kept := lines[:0]
			for _, line := range lines {
				if !strings.Contains(line, label) {
					kept = append(kept, line)
				}
			}
			fields["Value"] = strings.Join(kept, "\n")
			got := ReviewIssueDraft(IssueDraft{Title: "add typed task briefs", Body: body(fields)}, Options{})
			state := got.BriefReadiness.Fields[field]
			if got.BriefReadiness.Ready || got.Dispatchability == Dispatchable || state.Status != "missing" || state.RepairAction == "" {
				t.Fatalf("review = %+v", got)
			}
		})
	}
	t.Run("typed unknown value", func(t *testing.T) {
		fields := cloneBriefFields(complete)
		fields["Value"] = strings.Replace(fields["Value"], "Today: read several records and infer it", "Today: unknown(reason: discovery spike must observe current workflow)", 1)
		got := ReviewIssueDraft(IssueDraft{Title: "add typed task briefs", Body: body(fields)}, Options{})
		field := got.BriefReadiness.Fields["alternative"]
		if !got.BriefReadiness.Ready || field.Status != "unknown" || field.Reason != "discovery spike must observe current workflow" {
			t.Fatalf("readiness = %+v", got.BriefReadiness)
		}
	})
}

func TestLegacyIssueContractDoesNotEnforceShiftLeftBrief(t *testing.T) {
	got := ReviewIssueDraft(IssueDraft{Title: "legacy issue", Body: "## Scope\n\ninternal/foo/**\n\n## Definition of done\n\ntest passes\n\n## Witness\n\ngo test ./internal/foo"}, Options{})
	if got.BriefReadiness.Enforced {
		t.Fatalf("legacy contract unexpectedly migrated: %+v", got.BriefReadiness)
	}
}

// A legacy brief is not enforced, but if it still carries a Value section the
// four value fields stay reported so making the frame default-on does not
// silently drop a readout callers already had.
func TestLegacyValueFrameStillReportsValueFields(t *testing.T) {
	body := "## Value\n\n- For: a maintainer\n- Problem: no readout\n- Today: read the body\n- Better because: it is checkable\n\n## Scope\n\ninternal/foo/**\n\n## Definition of done\n\ntest passes\n\n## Witness\n\ngo test ./internal/foo"
	got := ReviewIssueDraft(IssueDraft{Title: "legacy issue with a value frame", Body: body}, Options{})
	if got.BriefReadiness.Enforced {
		t.Fatalf("legacy contract unexpectedly migrated: %+v", got.BriefReadiness)
	}
	for _, name := range []string{"beneficiary", "problem", "alternative", "advantage"} {
		if field := got.BriefReadiness.Fields[name]; field.Status != "present" {
			t.Fatalf("%s = %+v, want present", name, field)
		}
	}
}

func cloneBriefFields(in map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range in {
		out[k] = v
	}
	return out
}
