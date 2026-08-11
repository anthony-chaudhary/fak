package issuepolicy

import "testing"

func TestShiftLeftTaskBriefReadiness(t *testing.T) {
	complete := map[string]string{
		"Outcome":         "operators see typed readiness",
		"Scope / tree":    "internal/issuepolicy/** only",
		"Dependencies":    "none",
		"Acceptance":      "focused test passes",
		"Witness / proof": "go test ./internal/issuepolicy -count=1",
		"Placement":       "gen/now; P1; milestone G0; issuecontract lane",
	}
	body := func(fields map[string]string) string {
		order := []string{"Outcome", "Scope / tree", "Dependencies", "Acceptance", "Witness / proof", "Placement"}
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
}

func TestLegacyIssueContractDoesNotEnforceShiftLeftBrief(t *testing.T) {
	got := ReviewIssueDraft(IssueDraft{Title: "legacy issue", Body: "## Scope\n\ninternal/foo/**\n\n## Definition of done\n\ntest passes\n\n## Witness\n\ngo test ./internal/foo"}, Options{})
	if got.BriefReadiness.Enforced {
		t.Fatalf("legacy contract unexpectedly migrated: %+v", got.BriefReadiness)
	}
}

func cloneBriefFields(in map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range in {
		out[k] = v
	}
	return out
}
