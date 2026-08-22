package toolcallcontrol

import (
	"encoding/json"
	"testing"
)

func TestClampResultCountEnforcesContractWithoutModelRoundTrip(t *testing.T) {
	contract := ResultCountContract{
		Tool: "github.search_issues", ArgumentPointer: "/options/per_page",
		Minimum: 1, Maximum: 500, ReductionSafe: true,
	}
	got := ClampResultCount(ModeEnforce, contract.Tool, json.RawMessage(`{"query":"is:open","options":{"per_page":500}}`), contract, 10, false)
	if got.Decision != "clamp" || got.Reason != "requested_items_above_policy_maximum" {
		t.Fatalf("verdict = %#v", got)
	}
	if got.ModelRoundTrips != 0 {
		t.Fatalf("model round trips = %d, want 0", got.ModelRoundTrips)
	}
	if len(got.Changes) != 1 || got.Changes[0].From != 500 || got.Changes[0].To != 10 || got.Changes[0].Path != "/options/per_page" {
		t.Fatalf("changes = %#v", got.Changes)
	}
	var effective struct {
		Query   string `json:"query"`
		Options struct {
			PerPage int64 `json:"per_page"`
		} `json:"options"`
	}
	if err := json.Unmarshal(got.EffectiveArgs, &effective); err != nil {
		t.Fatal(err)
	}
	if effective.Query != "is:open" || effective.Options.PerPage != 10 {
		t.Fatalf("effective args = %s", got.EffectiveArgs)
	}
}

func TestClampResultCountNeverRaisesOrInjects(t *testing.T) {
	contract := ResultCountContract{Tool: "read", ArgumentPointer: "/limit", Minimum: 1, Maximum: 500, ReductionSafe: true}
	for _, test := range []struct {
		name     string
		args     string
		proposed int64
		reason   string
	}{
		{name: "lower request", args: `{"limit":5}`, proposed: 10, reason: "within_budget"},
		{name: "omitted", args: `{"path":"README.md"}`, proposed: 10, reason: "argument_missing"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := ClampResultCount(ModeEnforce, contract.Tool, json.RawMessage(test.args), contract, test.proposed, false)
			if got.Decision != "pass" || got.Reason != test.reason || string(got.EffectiveArgs) != test.args {
				t.Fatalf("verdict = %#v", got)
			}
		})
	}
}

func TestClampResultCountFailsOpenForUncontractedValues(t *testing.T) {
	base := ResultCountContract{Tool: "read", ArgumentPointer: "/limit", Minimum: 1, Maximum: 500, ReductionSafe: true}
	tests := []struct {
		name     string
		mode     Mode
		tool     string
		args     string
		contract ResultCountContract
		exempt   bool
	}{
		{name: "observe", mode: ModeShadow, tool: "read", args: `{"limit":500}`, contract: base},
		{name: "wrong tool", mode: ModeEnforce, tool: "write", args: `{"limit":500}`, contract: base},
		{name: "float", mode: ModeEnforce, tool: "read", args: `{"limit":500.0}`, contract: base},
		{name: "string", mode: ModeEnforce, tool: "read", args: `{"limit":"500"}`, contract: base},
		{name: "negative", mode: ModeEnforce, tool: "read", args: `{"limit":-1}`, contract: base},
		{name: "overflow", mode: ModeEnforce, tool: "read", args: `{"limit":9223372036854775808}`, contract: base},
		{name: "unsafe", mode: ModeEnforce, tool: "read", args: `{"limit":500}`, contract: ResultCountContract{Tool: "read", ArgumentPointer: "/limit", Minimum: 1, Maximum: 500}},
		{name: "exempt", mode: ModeEnforce, tool: "read", args: `{"limit":500}`, contract: base, exempt: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := ClampResultCount(test.mode, test.tool, json.RawMessage(test.args), test.contract, 10, test.exempt)
			if got.Decision != "pass" || string(got.EffectiveArgs) != test.args {
				t.Fatalf("verdict = %#v", got)
			}
		})
	}
}

func TestClampResultCountDecodesJSONPointer(t *testing.T) {
	contract := ResultCountContract{Tool: "read", ArgumentPointer: "/page~1size/max~0items", Minimum: 1, Maximum: 100, ReductionSafe: true}
	got := ClampResultCount(ModeEnforce, "read", json.RawMessage(`{"page/size":{"max~items":100}}`), contract, 8, false)
	if got.Decision != "clamp" || string(got.EffectiveArgs) != `{"page/size":{"max~items":8}}` {
		t.Fatalf("verdict = %#v", got)
	}
}
