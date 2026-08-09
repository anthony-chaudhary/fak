package sessionaudit

// deferrefault_test.go — the acceptance witness for #3625. The issue's done condition is one
// sentence with two clauses, and each has a test here that fails if the clause regresses:
//
//	"the auditor reports a per-session defer_refault_rate" -> TestAuditDeferRefault (rate column)
//	                                                       -> TestAuditDeferRefaultTranscript (per session)
//	"sessions above a threshold get a DEFER_DEFEATED finding" -> TestAuditDeferRefault (verdict column)
//
// The rest pins the judgement calls the lens makes, because each is a place a later refactor
// could quietly change what the rate MEANS: the threshold is exclusive, a small sample HOLDS
// instead of accusing, an unattributed fault dilutes rather than inflates, and a split
// assistant record must not double-count one search.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var fixedDeferTime = time.Date(2026, 8, 7, 9, 30, 0, 0, time.UTC)

// fault is a fault-sequence literal shorthand: one search that materialized these tool names.
func fault(tools ...string) DeferFault { return DeferFault{Tools: tools} }

// TestAuditDeferRefault is the core done-condition witness: a synthetic tool-search fault
// SEQUENCE in, a per-session defer_refault_rate plus DEFER_OK/DEFER_DEFEATED verdict out.
func TestAuditDeferRefault(t *testing.T) {
	cases := []struct {
		name             string
		faults           []DeferFault
		wantVerdict      DeferRefaultVerdict
		wantRate         float64
		wantSearches     int
		wantMaterialized int
		wantDistinct     int
		wantRefaults     int
		wantChurned      []string
	}{{
		// The headline failure the issue exists to catch: the model keeps searching for the
		// same two schemas, so two thirds of every materialization is a redundant re-load.
		name:             "same schemas faulted back in defeats the defer",
		faults:           []DeferFault{fault("Alpha", "Beta"), fault("Alpha", "Beta"), fault("Alpha", "Beta")},
		wantVerdict:      DeferDefeated,
		wantRate:         4.0 / 6.0,
		wantSearches:     3,
		wantMaterialized: 6,
		wantDistinct:     2,
		wantRefaults:     4,
		wantChurned:      []string{"Alpha", "Beta"},
	}, {
		// Wide but flat: four DISTINCT cold schemas faulted in once each is the defer doing
		// precisely its job, and must never be flagged however many faults it took.
		name:             "distinct first-time faults are healthy",
		faults:           []DeferFault{fault("Alpha"), fault("Beta"), fault("Gamma"), fault("Delta")},
		wantVerdict:      DeferOK,
		wantRate:         0,
		wantSearches:     4,
		wantMaterialized: 4,
		wantDistinct:     4,
		wantRefaults:     0,
	}, {
		name:        "no fault at all is the ideal defer",
		faults:      nil,
		wantVerdict: DeferOK,
	}, {
		// The threshold is EXCLUSIVE: a rate landing exactly on it is not yet a defeat.
		name:             "rate exactly at the threshold is not defeated",
		faults:           []DeferFault{fault("Alpha", "Beta"), fault("Alpha", "Beta")},
		wantVerdict:      DeferOK,
		wantRate:         0.5,
		wantSearches:     2,
		wantMaterialized: 4,
		wantDistinct:     2,
		wantRefaults:     2,
		wantChurned:      []string{"Alpha", "Beta"},
	}, {
		// A high rate on a tiny sample HOLDS: three materializations cannot carry a verdict
		// about whether the defer holds across a session.
		name:             "high rate under the sample floor holds",
		faults:           []DeferFault{fault("Alpha"), fault("Alpha"), fault("Alpha")},
		wantVerdict:      DeferOK,
		wantRate:         2.0 / 3.0,
		wantSearches:     3,
		wantMaterialized: 3,
		wantDistinct:     1,
		wantRefaults:     2,
		wantChurned:      []string{"Alpha"},
	}, {
		// Unattributed faults count as EVENTS but contribute no materialization, so they can
		// only dilute the rate — never invent one.
		name:             "unattributed faults claim no rate",
		faults:           []DeferFault{{CallID: "s1"}, {CallID: "s2"}},
		wantVerdict:      DeferOK,
		wantRate:         0,
		wantSearches:     2,
		wantMaterialized: 0,
		wantDistinct:     0,
		wantRefaults:     0,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AuditDeferRefault("sess-1", tc.faults, DefaultDeferRefaultThreshold, fixedDeferTime)
			if got.Verdict != tc.wantVerdict {
				t.Errorf("Verdict = %q, want %q (finding: %s)", got.Verdict, tc.wantVerdict, got.Finding)
			}
			if !nearly(got.RefaultRate, tc.wantRate) {
				t.Errorf("defer_refault_rate = %v, want %v", got.RefaultRate, tc.wantRate)
			}
			if got.Searches != tc.wantSearches {
				t.Errorf("Searches = %d, want %d", got.Searches, tc.wantSearches)
			}
			if got.Materializations != tc.wantMaterialized {
				t.Errorf("Materializations = %d, want %d", got.Materializations, tc.wantMaterialized)
			}
			if got.DistinctTools != tc.wantDistinct {
				t.Errorf("DistinctTools = %d, want %d", got.DistinctTools, tc.wantDistinct)
			}
			if got.Refaults != tc.wantRefaults {
				t.Errorf("Refaults = %d, want %d", got.Refaults, tc.wantRefaults)
			}
			if strings.Join(got.RefaultedTools, ",") != strings.Join(tc.wantChurned, ",") {
				t.Errorf("RefaultedTools = %v, want %v", got.RefaultedTools, tc.wantChurned)
			}
			// Every row is a dated, schema-versioned artifact whose finding NAMES its verdict,
			// so a reader never has to infer the outcome from the numbers.
			if got.Schema != deferRefaultSchema {
				t.Errorf("Schema = %q, want %q", got.Schema, deferRefaultSchema)
			}
			if got.Generated != "2026-08-07T09:30:00Z" {
				t.Errorf("Generated = %q, want the supplied clock", got.Generated)
			}
			if got.Session != "sess-1" {
				t.Errorf("Session = %q, want sess-1", got.Session)
			}
			if !strings.HasPrefix(got.Finding, string(tc.wantVerdict)) {
				t.Errorf("Finding %q does not lead with its verdict %q", got.Finding, tc.wantVerdict)
			}
		})
	}
}

// TestAuditDeferRefaultUnattributedDilutesRate proves the conservative direction: adding
// faults whose names could not be recovered must never push a session INTO a defeat.
func TestAuditDeferRefaultUnattributedDilutesRate(t *testing.T) {
	churn := []DeferFault{fault("Alpha", "Beta"), fault("Alpha", "Beta"), fault("Alpha", "Beta")}
	defeated := AuditDeferRefault("s", churn, DefaultDeferRefaultThreshold, fixedDeferTime)
	if defeated.Verdict != DeferDefeated {
		t.Fatalf("precondition: want DEFER_DEFEATED, got %q", defeated.Verdict)
	}
	withBlanks := AuditDeferRefault("s", append(churn, DeferFault{}, DeferFault{}), DefaultDeferRefaultThreshold, fixedDeferTime)
	if withBlanks.Unattributed != 2 {
		t.Errorf("Unattributed = %d, want 2", withBlanks.Unattributed)
	}
	if !nearly(withBlanks.RefaultRate, defeated.RefaultRate) {
		t.Errorf("unattributed faults changed the rate: %v -> %v", defeated.RefaultRate, withBlanks.RefaultRate)
	}
	if withBlanks.Searches != 5 {
		t.Errorf("Searches = %d, want 5 (the model did ask, five times)", withBlanks.Searches)
	}
}

// TestAuditDeferRefaultZeroThresholdFallsBack pins that a zero-valued caller gets the
// documented default rather than a threshold of 0 that flags every session with one repeat.
func TestAuditDeferRefaultZeroThresholdFallsBack(t *testing.T) {
	got := AuditDeferRefault("s", []DeferFault{fault("Alpha"), fault("Beta"), fault("Gamma"), fault("Alpha")}, 0, fixedDeferTime)
	if got.Threshold != DefaultDeferRefaultThreshold {
		t.Errorf("Threshold = %v, want the %v default", got.Threshold, DefaultDeferRefaultThreshold)
	}
	if got.Verdict != DeferOK {
		t.Errorf("Verdict = %q, want DEFER_OK at a 0.25 rate", got.Verdict)
	}
}

// TestDeferFaultsFromTranscript is the transcript-pass witness: search calls pair with their
// results by tool_use id across intervening records, non-search tools are ignored, an errored
// search materializes nothing, and a split record cannot double-count one search.
func TestDeferFaultsFromTranscript(t *testing.T) {
	path := writeDeferTranscript(t, "extract.jsonl",
		searchCall("s1"),
		otherToolCall("r1", "Read"),
		searchResult("s1", "Alpha", "Beta"),
		searchCall("s2"),
		searchCall("s2"), // a split/duplicated assistant record replaying the same call
		searchResult("s2", "Alpha"),
		searchCall("s3"),
		erroredSearchResult("s3"),
	)
	faults, err := DeferFaultsFromTranscript(path)
	if err != nil {
		t.Fatalf("DeferFaultsFromTranscript: %v", err)
	}
	if len(faults) != 3 {
		t.Fatalf("faults = %d (%+v), want 3 searches", len(faults), faults)
	}
	if got := strings.Join(faults[0].Tools, ","); got != "Alpha,Beta" {
		t.Errorf("fault 0 tools = %q, want Alpha,Beta", got)
	}
	if got := strings.Join(faults[1].Tools, ","); got != "Alpha" {
		t.Errorf("fault 1 tools = %q, want Alpha", got)
	}
	if len(faults[2].Tools) != 0 {
		t.Errorf("errored search materialized %v, want nothing", faults[2].Tools)
	}
	if faults[0].CallID != "s1" {
		t.Errorf("CallID = %q, want s1 so a flagged session walks back to the turn", faults[0].CallID)
	}
}

// TestAuditDeferRefaultTranscript is the captured-transcript witness the issue asks for: a
// completed transcript in, a per-session rate + DEFER_DEFEATED finding out.
func TestAuditDeferRefaultTranscript(t *testing.T) {
	path := writeDeferTranscript(t, "defeated-session.jsonl",
		searchCall("s1"), searchResult("s1", "Alpha", "Beta"),
		searchCall("s2"), searchResult("s2", "Alpha", "Beta"),
		searchCall("s3"), searchResult("s3", "Alpha", "Beta"),
	)
	got, err := AuditDeferRefaultTranscript(path, DefaultDeferRefaultThreshold, fixedDeferTime)
	if err != nil {
		t.Fatalf("AuditDeferRefaultTranscript: %v", err)
	}
	if got.Verdict != DeferDefeated {
		t.Errorf("Verdict = %q, want DEFER_DEFEATED (finding: %s)", got.Verdict, got.Finding)
	}
	if !nearly(got.RefaultRate, 4.0/6.0) {
		t.Errorf("defer_refault_rate = %v, want 4/6", got.RefaultRate)
	}
	// The row is per SESSION, named from the transcript file the way Analyze names it.
	if got.Session != "defeated-session" {
		t.Errorf("Session = %q, want defeated-session", got.Session)
	}
	for _, want := range []string{"DEFER_DEFEATED", "Alpha", "Beta", "defer_refault_rate"} {
		if !strings.Contains(got.Finding, want) {
			t.Errorf("finding %q missing %q", got.Finding, want)
		}
	}
	t.Logf("captured transcript audit: %s", got.Finding)
}

// TestAuditDeferRefaultTranscriptMissingFileErrors proves a missing transcript is an error,
// never a clean DEFER_OK — an unreadable session must not read as one whose defer held.
func TestAuditDeferRefaultTranscriptMissingFileErrors(t *testing.T) {
	if _, err := AuditDeferRefaultTranscript(filepath.Join(t.TempDir(), "nope.jsonl"), 0, fixedDeferTime); err == nil {
		t.Fatal("missing transcript = nil error, want a loud read failure")
	}
}

// TestDeferMaterializedTools pins the tolerant decoder: the shapes it understands, and that
// everything else yields NO names so the caller holds instead of guessing.
func TestDeferMaterializedTools(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"function envelopes in a text block",
			`[{"type":"text","text":"<functions><function>{\"name\":\"Alpha\"}</function><function>{\"name\":\"Beta\"}</function></functions>"}]`,
			"Alpha,Beta"},
		{"descriptor array with direct names",
			`[{"name":"Alpha"},{"name":"Beta"}]`, "Alpha,Beta"},
		{"bare json string", `"<function>{\"name\":\"Alpha\"}</function>"`, "Alpha"},
		// Prose that merely MENTIONS a name is not a materialization.
		{"prose without envelopes", `[{"type":"text","text":"the name field is absent here"}]`, ""},
		{"unparseable payload inside an envelope", `[{"type":"text","text":"<function>not json</function>"}]`, ""},
		{"empty", ``, ""},
		{"object shape is unrecognized", `{"tools":["Alpha"]}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := strings.Join(deferMaterializedTools(json.RawMessage(tc.raw)), ",")
			if got != tc.want {
				t.Errorf("deferMaterializedTools(%s) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

// TestDeferRefaultAuditJSON pins the wire shape a downstream reader binds to — above all the
// `defer_refault_rate` key the done condition names by that exact spelling.
func TestDeferRefaultAuditJSON(t *testing.T) {
	got := AuditDeferRefault("sess-1", []DeferFault{fault("Alpha", "Beta"), fault("Alpha", "Beta"), fault("Alpha", "Beta")},
		DefaultDeferRefaultThreshold, fixedDeferTime)
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var round map[string]any
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	for _, key := range []string{
		"schema", "generated", "session", "verdict", "searches", "materializations",
		"distinct_tools", "refaults", "defer_refault_rate", "threshold", "refaulted_tools", "finding",
	} {
		if _, ok := round[key]; !ok {
			t.Errorf("audit JSON missing %q key; got %v", key, round)
		}
	}
	if round["verdict"] != "DEFER_DEFEATED" {
		t.Errorf("verdict = %v, want DEFER_DEFEATED", round["verdict"])
	}
	// Unattributed is refusal-only: it must stay absent on a fully attributed session.
	if _, ok := round["unattributed"]; ok {
		t.Errorf("unattributed present on a fully attributed session: %v", round)
	}
}

// --- helpers -------------------------------------------------------------------------

func nearly(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

func writeDeferTranscript(t *testing.T, name string, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	return path
}

func searchCall(id string) string {
	return mustLineNoT(map[string]any{
		"type": "assistant",
		"message": map[string]any{
			"id": "m-" + id,
			"content": []any{map[string]any{
				"type": "tool_use", "id": id, "name": "ToolSearch",
				"input": map[string]any{"query": "select:Alpha"},
			}},
		},
	})
}

func otherToolCall(id, name string) string {
	return mustLineNoT(map[string]any{
		"type": "assistant",
		"message": map[string]any{
			"id":      "m-" + id,
			"content": []any{map[string]any{"type": "tool_use", "id": id, "name": name}},
		},
	})
}

func searchResult(id string, names ...string) string {
	var sb strings.Builder
	sb.WriteString("<functions>")
	for _, n := range names {
		sb.WriteString(`<function>{"description":"d","name":"` + n + `","parameters":{}}</function>`)
	}
	sb.WriteString("</functions>")
	return mustLineNoT(map[string]any{
		"type": "user",
		"message": map[string]any{
			"content": []any{map[string]any{
				"type": "tool_result", "tool_use_id": id,
				"content": []any{map[string]any{"type": "text", "text": sb.String()}},
			}},
		},
	})
}

func erroredSearchResult(id string) string {
	return mustLineNoT(map[string]any{
		"type": "user",
		"message": map[string]any{
			"content": []any{map[string]any{
				"type": "tool_result", "tool_use_id": id, "is_error": true,
				"content": []any{map[string]any{"type": "text", "text": "no matching tool"}},
			}},
		},
	})
}

// mustLineNoT marshals a record for the fixture builders above, which are called in argument
// position where a *testing.T is not to hand. The inputs are literals in this file, so a
// marshal failure is a programming error in the test itself and panicking is the clearest
// possible signal.
func mustLineNoT(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}
