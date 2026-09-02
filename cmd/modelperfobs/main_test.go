// Package main tests for the behavior the modelperfobs CLI drives end to end:
// ParseBackend (proxy --backend validation), ReadObservations + Summarize +
// WriteMarkdown (the report --input/--format pipeline), and the
// cache-state-bench --verify witness validation errors. All resource-free:
// no network, no model weights, no GPU.
package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

func TestParseBackend(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantErr   string
		wantHost  string
		wantHTTPS bool
	}{
		{"http backend accepted", "http://127.0.0.1:8080", "", "127.0.0.1:8080", false},
		{"https backend accepted", "https://api.example.com/v1", "", "api.example.com", true},
		{"empty backend refused", "", "absolute http(s) URL", "", false},
		{"bare host refused", "api.example.com", "absolute http(s) URL", "", false},
		{"scheme without host refused", "http://", "absolute http(s) URL", "", false},
		{"non-http scheme refused", "ftp://api.example.com", "http or https", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u, err := modelperfobs.ParseBackend(tt.raw)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("ParseBackend(%q) error = %v, want containing %q", tt.raw, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseBackend(%q) unexpected error: %v", tt.raw, err)
			}
			if u.Host != tt.wantHost {
				t.Fatalf("ParseBackend(%q) host = %q, want %q", tt.raw, u.Host, tt.wantHost)
			}
			if (u.Scheme == "https") != tt.wantHTTPS {
				t.Fatalf("ParseBackend(%q) scheme = %q", tt.raw, u.Scheme)
			}
		})
	}
}

func observationLine(id string, status int, prompt, completion int64, duration, ttft, tpot float64, errMsg string) string {
	b, _ := json.Marshal(modelperfobs.Observation{
		Schema: modelperfobs.Schema, RequestID: id, Backend: "bench", Status: status,
		PromptTokens: prompt, CompletionTokens: completion,
		DurationMS: duration, TTFTMS: ttft, TPOTMS: tpot, Error: errMsg,
	})
	return string(b)
}

func TestReadObservations(t *testing.T) {
	valid := observationLine("r1", 200, 100, 20, 50, 10, 5, "") + "\n" +
		observationLine("r2", 200, 200, 40, 70, 12, 6, "")
	tests := []struct {
		name    string
		input   string
		wantN   int
		wantErr string
	}{
		{"two valid rows parse", valid + "\n", 2, ""},
		{"empty input parses to zero rows", "", 0, ""},
		{"malformed line names its position", valid + "\n{not json}\n", 0, "decode observation 3"},
		{"wrong schema names both values", `{"schema":"other/1"}` + "\n", 0, `schema "other/1", want "fak-model-perf/1"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rows, err := modelperfobs.ReadObservations(strings.NewReader(tt.input))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("ReadObservations error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ReadObservations unexpected error: %v", err)
			}
			if len(rows) != tt.wantN {
				t.Fatalf("ReadObservations rows = %d, want %d", len(rows), tt.wantN)
			}
		})
	}
}

func TestSummarizeAggregation(t *testing.T) {
	lines := []string{
		observationLine("ok1", 200, 10, 5, 10, 20, 5, ""),
		observationLine("ok2", 200, 20, 5, 20, 40, 10, ""),
		observationLine("ok3", 200, 40, 5, 100, 80, 20, ""),
		observationLine("http-fail", 503, 15, 0, 30, 0, 0, ""),
		observationLine("err-fail", 0, 15, 0, 30, 0, 0, "connection reset"),
	}
	var rows []modelperfobs.Observation
	for _, l := range lines {
		var row modelperfobs.Observation
		if err := json.Unmarshal([]byte(l), &row); err != nil {
			t.Fatalf("fixture decode: %v", err)
		}
		rows = append(rows, row)
	}
	s := modelperfobs.Summarize(rows)
	if s.Requests != 5 {
		t.Fatalf("Requests = %d, want 5", s.Requests)
	}
	if s.Errors != 2 {
		t.Fatalf("Errors = %d, want 2 (status>=400 and explicit error both count)", s.Errors)
	}
	if s.PromptTokens != 100 || s.CompletionTokens != 15 {
		t.Fatalf("tokens = %d/%d, want 100/15", s.PromptTokens, s.CompletionTokens)
	}
	// quantile picks the sorted nearest-rank element: of {10,20,30,30,100},
	// p50 is 30 and p95 is 100.
	if s.DurationP50MS != 30 || s.DurationP95MS != 100 {
		t.Fatalf("duration p50/p95 = %v/%v, want 30/100", s.DurationP50MS, s.DurationP95MS)
	}
	if s.Schema != "fak-model-perf-summary/1" {
		t.Fatalf("Schema = %q", s.Schema)
	}
}

func TestSummarizeDiagnosis(t *testing.T) {
	row := func(ttft, tpot float64, status int, errMsg string) modelperfobs.Observation {
		return modelperfobs.Observation{DurationMS: 100, TTFTMS: ttft, TPOTMS: tpot, Status: status, Error: errMsg}
	}
	tests := []struct {
		name       string
		rows       []modelperfobs.Observation
		wantBottle string
		wantCheck  string
	}{
		{"no data at all", nil, "no-data", "route an OpenAI-compatible request"},
		{"any failed request dominates", []modelperfobs.Observation{row(10, 5, 200, ""), row(10, 5, 500, "")}, "reliability", "group failing observations"},
		{"success without stream timing", []modelperfobs.Observation{row(0, 0, 200, "")}, "missing-stream-timing", "enable streaming"},
		{"slow decode dominates", []modelperfobs.Observation{row(90, 150, 200, "")}, "decode", "memory bandwidth"},
		{"prefill or queue", []modelperfobs.Observation{row(800, 100, 200, "")}, "prefill-or-queue", "sweep prompt length"},
		{"healthy workload orchestration", []modelperfobs.Observation{row(50, 40, 200, "")}, "workload-orchestration", "join request IDs"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := modelperfobs.Summarize(tt.rows)
			if s.LikelyBottleneck != tt.wantBottle {
				t.Fatalf("LikelyBottleneck = %q, want %q", s.LikelyBottleneck, tt.wantBottle)
			}
			if !strings.Contains(s.NextCheck, tt.wantCheck) {
				t.Fatalf("NextCheck = %q, want containing %q", s.NextCheck, tt.wantCheck)
			}
		})
	}
}

func TestReportRenderingBothFormats(t *testing.T) {
	s := modelperfobs.Summarize([]modelperfobs.Observation{
		{Status: 200, PromptTokens: 10, CompletionTokens: 5, DurationMS: 100, TTFTMS: 90, TPOTMS: 150},
	})
	var md bytes.Buffer
	if err := modelperfobs.WriteMarkdown(&md, s); err != nil {
		t.Fatalf("WriteMarkdown: %v", err)
	}
	for _, want := range []string{
		"# Model performance observation report",
		"- Requests: **1** (0 errors)",
		"- Tokens: **10 prompt / 5 completion**",
		"Likely bottleneck: **decode**",
		"- Next check: profile device residency",
	} {
		if !strings.Contains(md.String(), want) {
			t.Fatalf("markdown missing %q:\n%s", want, md.String())
		}
	}
	// The report --format json branch encodes the same summary for machine readers.
	var jsonOut bytes.Buffer
	if err := json.NewEncoder(&jsonOut).Encode(s); err != nil {
		t.Fatalf("json encode: %v", err)
	}
	var back modelperfobs.Summary
	if err := json.Unmarshal(jsonOut.Bytes(), &back); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if back != s {
		t.Fatalf("json round trip changed the summary: %+v vs %+v", back, s)
	}
}

func TestReadStateReportValidationErrors(t *testing.T) {
	fullProvenance := `"evidence_kind":"observed","scope":"probe_scope","command":"fak bench","code_state":"clean","captured_at":"2026-08-30T00:00:00Z","note":"n"`
	schema := `"schema":"` + modelperfobs.StateBenchmarkSchema + `"`
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{"trailing JSON refused", `{` + schema + `,"verdict":"admitted","provenance":{` + fullProvenance + `},"arms":[]}` + "\n{}\n", "trailing JSON"},
		{"unknown field refused", `{"schema":"x","bogus":1}`, "unknown field"},
		{"unsupported provenance refused", `{` + schema + `,"provenance":{"evidence_kind":"hallucinated"}}`, "unsupported provenance"},
		{"incomplete provenance refused", `{` + schema + `,"provenance":{"evidence_kind":"observed"}}`, "provenance is incomplete"},
		{"in-process report cannot claim external backend", `{` + schema + `,"provenance":{"evidence_kind":"observed","scope":"in_process_fak_workflow_cache","external_backend_claims":true,"command":"c","code_state":"s","captured_at":"2026-08-30T00:00:00Z","note":"n"}}`, "cannot claim an external backend"},
		{"report without arms refused", `{` + schema + `,"provenance":{` + fullProvenance + `},"arms":[]}`, "no arms"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := modelperfobs.ReadStateReport(strings.NewReader(tt.body))
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ReadStateReport error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestWriteStateReportRejectsInvalidWitness(t *testing.T) {
	var report modelperfobs.StateReport // zero value: empty provenance, no arms
	var out bytes.Buffer
	if err := modelperfobs.WriteStateReport(&out, report, true); err == nil {
		t.Fatal("WriteStateReport accepted a report with no provenance and no arms")
	}
	if out.Len() != 0 {
		t.Fatalf("WriteStateReport wrote %d bytes before refusing the report", out.Len())
	}
}
