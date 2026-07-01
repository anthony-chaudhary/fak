package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/productscorecard"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

// productscorecard_contexthealth_test.go — issue #1579 witness: `go test
// ./internal/scorecardpane ./cmd/fak -run ContextHealth`. Proves the product
// scorecard CLI renders the closed scorecardpane.ContextHealthSeverity enum (not
// free-text severity) in both --json and the default human render, derived from
// whatever managed-context SLO rows the payload carries.

// TestProductScorecardContextHealthDefaultsFreshWithNoManagedContextData proves an
// absent managed_context corpus entry (the fixture workspace declares no SLO
// fixtures) classifies as fresh rather than erroring or omitting the field.
func TestProductScorecardContextHealthDefaultsFreshWithNoManagedContextData(t *testing.T) {
	root := makeProductScorecardWorkspace(t)
	var out, errb bytes.Buffer
	code := runProductScorecard(&out, &errb, []string{"--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("parse json: %v\n%s", err, out.String())
	}
	got, ok := payload["context_health"].(string)
	if !ok {
		t.Fatalf("payload missing string context_health field: %#v", payload)
	}
	if got != string(scorecardpane.ContextHealthFresh) {
		t.Errorf("context_health = %q, want %q", got, scorecardpane.ContextHealthFresh)
	}
	if !scorecardpane.ValidContextHealthSeverity(scorecardpane.ContextHealthSeverity(got)) {
		t.Errorf("context_health %q must be a member of the closed vocabulary", got)
	}

	out.Reset()
	errb.Reset()
	code = runProductScorecard(&out, &errb, []string{"--workspace", root})
	if code != 0 {
		t.Fatalf("default render exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "context-health: fresh") {
		t.Errorf("default render missing context-health line:\n%s", out.String())
	}
}

// TestProductScorecardContextHealthReflectsFailingSLO proves failing
// managed-context SLO rows surface as the matching closed severity, not the raw
// debt integer alone.
func TestProductScorecardContextHealthReflectsFailingSLO(t *testing.T) {
	payload := productscorecard.Payload{
		Corpus: map[string]any{
			"managed_context": map[string]any{
				"rows": []map[string]any{
					{"area": "assumption", "status": "pass"},
					{"area": "visibility", "status": "pass"},
					{"area": "objective", "status": "pass"},
					{"area": "reset", "status": "fail"},
					{"area": "budget", "status": "pass"},
					{"area": "query", "status": "pass"},
					{"area": "cache", "status": "pass"},
					{"area": "memory", "status": "pass"},
				},
			},
		},
	}
	got := productScorecardContextHealth(payload)
	if got != scorecardpane.ContextHealthResetImminent {
		t.Fatalf("productScorecardContextHealth(failing reset SLO) = %q, want reset_imminent", got)
	}

	payload.Corpus["managed_context"] = map[string]any{
		"rows": []map[string]any{{"area": "assumption", "status": "query"}},
	}
	got = productScorecardContextHealth(payload)
	if got != scorecardpane.ContextHealthQueryNeeded {
		t.Fatalf("productScorecardContextHealth(failing assumption SLO) = %q, want query_needed", got)
	}

	payload.Corpus["managed_context"] = map[string]any{
		"rows": []map[string]any{{"area": "objective", "status": "drifted"}},
	}
	got = productScorecardContextHealth(payload)
	if got != scorecardpane.ContextHealthObjectiveDrift {
		t.Fatalf("productScorecardContextHealth(failing objective SLO) = %q, want objective_drift", got)
	}
}

// TestProductScorecardSLORowsAcceptsJSONRoundTrippedRows proves the row-normalizer
// handles a []any-of-map[string]any shape (what a --compare baseline's
// json.Unmarshal into map[string]any produces), not just the live
// []map[string]any ManagedContextSLOReport builds in-process.
func TestProductScorecardSLORowsAcceptsJSONRoundTrippedRows(t *testing.T) {
	raw := []byte(`{"rows":[{"area":"budget","status":"fail"}]}`)
	var mc map[string]any
	if err := json.Unmarshal(raw, &mc); err != nil {
		t.Fatal(err)
	}
	rows := productScorecardSLORows(mc["rows"])
	if len(rows) != 1 {
		t.Fatalf("productScorecardSLORows(json rows) = %#v, want 1 row", rows)
	}
	sig := scorecardpane.ContextHealthSignalsFromSLORows(rows)
	if !sig.BudgetFailing {
		t.Errorf("signals = %+v, want BudgetFailing", sig)
	}
}
