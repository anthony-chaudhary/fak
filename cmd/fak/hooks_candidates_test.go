package main

// hooks_candidates_test.go — the REPORTING half of the candidate denominator (#5602).
//
// internal/hooks/candidates*_test.go pin the ledger (unreported ≠ zero) and its coverage (every
// staged gate records one). Those tests stop at the package boundary. This file pins what a
// consumer actually reads: the `gates` array in `fak hooks pre-commit --json`, and the one human
// line the clean path prints.
//
// The distinction under test is the whole point of the issue. A gate that judged 40 files and
// found nothing, and a gate whose filter admitted nothing at all, produced a BYTE-IDENTICAL
// payload before this landed. Rendering "unreported" as 0 — the obvious Go default, since
// gateReport.Candidates could have been an int — would rebuild that ambiguity in the one place
// every downstream dashboard reads. Hence the pointer, and hence these tests.

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// decodeFindingsJSON runs emitFindingsJSON and decodes the payload a consumer would parse. The
// scope argument (#5603) is the staged population with no operator narrowing — the posture these
// denominator tests are about; hooks_scope_test.go varies it.
func decodeFindingsJSON(t *testing.T, findings []hooks.Finding, skipped []string, gates []gateReport) map[string]any {
	t.Helper()
	var stdout, stderr bytes.Buffer
	emitFindingsJSON(&stdout, &stderr, findings, skipped, gates, runScope{Population: scopePopulationStaged})
	if stderr.Len() > 0 {
		t.Fatalf("emitFindingsJSON wrote to stderr: %s", stderr.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("payload is not valid JSON: %v\n%s", err, stdout.String())
	}
	return payload
}

// TestGateReportUnreportedIsNullNotZero is the central claim of the reporting layer, stated
// against the wire bytes rather than the Go value: a gate that recorded no denominator must
// serialize as JSON null, and a gate that recorded the number 0 must serialize as 0. If these two
// ever collapse, "this gate judged nothing" becomes unreadable from "this gate did not say", and
// the payload is back to covering two worlds with one shape.

func TestGateReportCarriesElapsedNanoseconds(t *testing.T) {
	d := &hooks.StagedDiff{}
	r := buildGateReport(d, "CONCEPT_ADMISSION", 0, false, 125*time.Millisecond)
	if r.ElapsedNS != int64(125*time.Millisecond) {
		t.Fatalf("elapsed_ns = %d, want %d", r.ElapsedNS, int64(125*time.Millisecond))
	}
	payload := decodeFindingsJSON(t, nil, nil, []gateReport{r})
	gates := payload["gates"].([]any)
	got := int64(gates[0].(map[string]any)["elapsed_ns"].(float64))
	if got != r.ElapsedNS {
		t.Fatalf("JSON elapsed_ns = %d, want %d", got, r.ElapsedNS)
	}
}

func TestGateReportUnreportedIsNullNotZero(t *testing.T) {
	d := &hooks.StagedDiff{}
	d.NoteCandidates("GOFMT", 0, "staged .go file(s)") // ran, judged nothing — a REAL answer
	// SECRET_SHAPE deliberately records nothing: UNREPORTED.

	reports := []gateReport{
		buildGateReport(d, "GOFMT", 0, false),
		buildGateReport(d, "SECRET_SHAPE", 0, false),
	}

	// The Go values, first: zero is addressable, absent is nil.
	if reports[0].Candidates == nil {
		t.Fatalf("GOFMT recorded 0 candidates; buildGateReport dropped it to UNREPORTED")
	}
	if got := *reports[0].Candidates; got != 0 {
		t.Fatalf("GOFMT candidates = %d, want 0", got)
	}
	if reports[1].Candidates != nil {
		t.Fatalf("SECRET_SHAPE recorded nothing; buildGateReport invented a denominator of %d", *reports[1].Candidates)
	}

	// Now the wire, which is what a consumer sees.
	payload := decodeFindingsJSON(t, nil, nil, reports)
	gates, ok := payload["gates"].([]any)
	if !ok || len(gates) != 2 {
		t.Fatalf("gates array = %#v, want 2 entries", payload["gates"])
	}
	gofmt := gates[0].(map[string]any)
	secret := gates[1].(map[string]any)

	if v, present := gofmt["candidates"]; !present || v == nil {
		t.Errorf("GOFMT judged zero candidates but serialized as %v; a consumer cannot tell it ran", v)
	} else if n, isNum := v.(float64); !isNum || n != 0 {
		t.Errorf("GOFMT candidates serialized as %#v, want the number 0", v)
	}
	if v, present := secret["candidates"]; !present {
		t.Errorf("SECRET_SHAPE has no candidates key at all; the key must always be present so a\n" +
			"reader never has to treat absent as a third state")
	} else if v != nil {
		t.Errorf("SECRET_SHAPE reported no denominator but serialized as %#v, want null", v)
	}
	// The unit travels with a real count and is omitted for an absent one.
	if gofmt["unit"] != "staged .go file(s)" {
		t.Errorf("GOFMT unit = %#v, want %q", gofmt["unit"], "staged .go file(s)")
	}
	if _, present := secret["unit"]; present {
		t.Errorf("SECRET_SHAPE reported no denominator; a unit for a number that does not exist is noise")
	}
}

// TestEmitFindingsJSONStaysAdditive pins the compatibility promise the issue makes in as many
// words: "leaving findings, count, skipped_gates and skipped_count exactly as they are". A
// consumer written against #5299's payload must still parse this one unchanged, so the four
// original keys keep their names, types and meanings and `gates` is purely additive.
func TestEmitFindingsJSONStaysAdditive(t *testing.T) {
	payload := decodeFindingsJSON(t, nil, nil, nil)

	want := []string{
		"findings", "count", "skipped_gates", "skipped_count", // #5299
		"gates",                    // #5602
		"scope", "scope_narrowing", // #5603
	}
	known := make(map[string]bool, len(want))
	for _, key := range want {
		known[key] = true
		if _, present := payload[key]; !present {
			t.Errorf("key %q missing; every key is always present so a reader never treats absent as none", key)
		}
	}
	// The relation this stands for, not today's total: the epic only ADDS, so the
	// payload carries exactly the documented keys and nothing is renamed or removed.
	for key := range payload {
		if !known[key] {
			t.Errorf("undocumented key %q in payload (%v); this epic only ADDS — the four #5299 keys, plus `gates`\n"+
				"(#5602), plus scope/scope_narrowing (#5603), and nothing renamed or removed",
				key, payload)
		}
	}
	// Empty must be [] and 0, never null: the nil-to-empty normalization is the reason a consumer
	// can iterate without a nil check.
	for _, key := range []string{"findings", "skipped_gates", "gates"} {
		if arr, ok := payload[key].([]any); !ok || arr == nil {
			t.Errorf("%s = %#v on an empty run, want an empty JSON array", key, payload[key])
		}
	}
	if payload["count"] != float64(0) || payload["skipped_count"] != float64(0) {
		t.Errorf("count/skipped_count = %v/%v, want 0/0", payload["count"], payload["skipped_count"])
	}
}

// TestCleanRunSummarySeparatesJudgedFromIdle is the human-path claim. The clean path used to
// print NOTHING, and silence reads as "checked, nothing owed" when it equally means "nothing was
// checked". One line must therefore name all three states separately — judged, judged-nothing,
// and no-denominator — because folding the last into the second is the same lie in smaller type.
func TestCleanRunSummarySeparatesJudgedFromIdle(t *testing.T) {
	n := func(i int) *int { return &i }
	reports := []gateReport{
		{Gate: "GOFMT", Candidates: n(12), Unit: "staged .go file(s)", Findings: 0},
		{Gate: "PUBLIC_LEAK", Candidates: n(40), Unit: "staged file(s) scanned", Findings: 0},
		{Gate: "INDEX_SYNC", Candidates: n(0), Unit: "staged index file(s)", Findings: 0},
		{Gate: "DOC_PLACEMENT", Candidates: nil, Findings: 0}, // UNREPORTED
		{Gate: "DUPLICATION", Skipped: true},                  // named by the degraded-run line instead
	}

	got := cleanRunSummary(reports, 40, runScope{Population: scopePopulationStaged})

	for _, want := range []string{
		"2 judged candidates",     // GOFMT + PUBLIC_LEAK
		"1 judged nothing",        // INDEX_SYNC — a real zero
		"1 report no denominator", // DOC_PLACEMENT — said nothing at all
		"40 staged file(s)",       // the domain the whole run quantified over
		"PUBLIC_LEAK 40",          // the biggest denominator leads
		"GOFMT 12",                //
	} {
		if !strings.Contains(got, want) {
			t.Errorf("clean summary missing %q\ngot: %s", want, got)
		}
	}
	// A skipped gate is NOT counted as having judged nothing — #5299's degraded-run line owns it,
	// and double-reporting it here would inflate the idle bucket with gates that never ran.
	if strings.Contains(got, "DUPLICATION") {
		t.Errorf("clean summary names the skipped gate DUPLICATION; the degraded-run line owns it\ngot: %s", got)
	}
}

// TestCleanRunSummaryOmitsUnreportedClauseWhenEveryGateAnswered keeps the line honest in the
// healthy case: once every gate reports a denominator there is no ambiguity left to disclose, and
// a permanent ", 0 report no denominator" would train a reader to skip the clause that matters.
func TestCleanRunSummaryOmitsUnreportedClauseWhenEveryGateAnswered(t *testing.T) {
	n := func(i int) *int { return &i }
	got := cleanRunSummary([]gateReport{
		{Gate: "GOFMT", Candidates: n(3), Unit: "staged .go file(s)"},
		{Gate: "INDEX_SYNC", Candidates: n(0), Unit: "staged index file(s)"},
	}, 3, runScope{Population: scopePopulationStaged})

	if strings.Contains(got, "no denominator") {
		t.Errorf("every gate answered, but the summary still discloses an unreported bucket\ngot: %s", got)
	}
	if !strings.Contains(got, "1 judged nothing") {
		t.Errorf("summary lost the real zero (INDEX_SYNC judged nothing)\ngot: %s", got)
	}
}
