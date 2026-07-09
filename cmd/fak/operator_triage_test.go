package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/operatorbrief"
)

// TestOperatorTriageSelfcheck exercises the deterministic, no-I/O proof through
// the real subcommand dispatch — no key, no network, no fixtures.
func TestOperatorTriageSelfcheck(t *testing.T) {
	var out, errb bytes.Buffer
	code := runOperatorHeavinessGroup(&out, &errb, []string{"triage", "selfcheck"})
	if code != 0 {
		t.Fatalf("operator triage selfcheck should exit 0, got %d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "SELFCHECK OK") {
		t.Fatalf("want SELFCHECK OK banner, got %q", out.String())
	}
}

// TestOperatorTriageGateRoutesAndPages drives the lens over a brief artifact: a
// genuine authority decision keeps the gate paging (exit 1) while a runnable
// "cadence incomplete" page is routed off to the fleet in the JSON view.
func TestOperatorTriageGateRoutesAndPages(t *testing.T) {
	brief := operatorbrief.Report{
		Schema: operatorbrief.Schema,
		Human: []operatorbrief.Item{
			{Bucket: "human", Source: "release", Severity: "decision", Title: "release decision needed", Detail: "approve the tagged build before publish", Action: "approve the release"},
			{Bucket: "human", Source: "cadence", Severity: "page", Title: "cadence incomplete", Detail: "cadence report incomplete", Action: "repair scores, then rerun `fak cadence`"},
		},
	}
	path := writeOperatorBriefJSON(t, "brief.json", brief)

	var out, errb bytes.Buffer
	code := runOperatorHeavinessGroup(&out, &errb, []string{"triage", "--brief", path, "--json", "--check"})
	if code != 1 {
		t.Fatalf("a residual authority decision should page, got exit %d stderr=%s", code, errb.String())
	}
	var view operatorTriageView
	if err := json.Unmarshal(out.Bytes(), &view); err != nil {
		t.Fatalf("triage --json did not parse: %v\n%s", err, out.String())
	}
	if len(view.Human) != 1 || view.Human[0].Source != "release" {
		t.Fatalf("want only the release decision residual, got %+v", view.Human)
	}
	if len(view.Reassignments) != 1 || view.Reassignments[0].Source != "cadence" {
		t.Fatalf("want the cadence page routed to the fleet, got %+v", view.Reassignments)
	}
	// The reassignment must carry the deciding choicetriage rung, not just its
	// destination — an operator reading the lens needs the "why" (item 5).
	if strings.TrimSpace(view.Reassignments[0].Reason) == "" {
		t.Fatalf("reassignment dropped the deciding rung/reason, got %+v", view.Reassignments[0])
	}
	if view.GateExit == nil || *view.GateExit != 1 {
		t.Fatalf("JSON view should carry the gate decision, got %+v", view.GateExit)
	}
}

// TestOperatorTriageGateClearsWhenNoResidual: a brief whose only human item is a
// runnable page gates clean after triage, and the human summary names the route.
func TestOperatorTriageGateClearsWhenNoResidual(t *testing.T) {
	brief := operatorbrief.Report{
		Schema: operatorbrief.Schema,
		Human: []operatorbrief.Item{
			{Bucket: "human", Source: "cadence", Severity: "page", Title: "cadence incomplete", Detail: "cadence report incomplete", Action: "regenerate `fak cadence --json` and pass it with --cadence"},
		},
	}
	path := writeOperatorBriefJSON(t, "brief.json", brief)

	var out, errb bytes.Buffer
	code := runOperatorHeavinessGroup(&out, &errb, []string{"triage", "--brief", path, "--check"})
	if code != 0 {
		t.Fatalf("a brief with only a runnable page should gate clean, got exit %d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "routed to fleet") {
		t.Fatalf("want a routed-to-fleet summary line, got %q", out.String())
	}
}
