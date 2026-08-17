package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workaccount"
)

func TestRunInfoWorkCoverageTextAndJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runInfoWorkCoverage(&out, &errOut, false); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{"WORK ACCOUNTING COVERAGE · 10 declared", "provider_prompt_cache", "context_elision", "not_yet_measurable", "safety_intervention", "intentionally_excluded"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("report missing %q:\n%s", want, out.String())
		}
	}
	out.Reset()
	if code := runInfoWorkCoverage(&out, &errOut, true); code != 0 {
		t.Fatalf("json code=%d", code)
	}
	var report workaccount.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != "fak.info.work-accounting-coverage/1" || len(report.Mechanisms) != 10 {
		t.Fatalf("json report=%#v", report)
	}
}

func TestWorkCoverageHasNoUnavailableMechanisms(t *testing.T) {
	if text := workCoverageUnavailableText(); text != "" {
		t.Fatalf("measurable registry still reports unavailable coverage: %s", text)
	}
}

func TestWorkCoverageJSONUsesStableFieldsNotDisplayParsing(t *testing.T) {
	raw, err := marshalWorkCoverageForTest()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range [][]byte{
		[]byte(`"source_id":"provider_cache"`),
		[]byte(`"id":"context_elision"`),
		[]byte(`"id":"schema_tool_filtering"`),
		[]byte(`"status":"overlapping"`),
		[]byte(`"reason":`),
	} {
		if !bytes.Contains(raw, want) {
			t.Fatalf("coverage JSON missing %s: %s", want, raw)
		}
	}
	if bytes.Contains(raw, []byte(`"status":"not_yet_measurable"`)) {
		t.Fatalf("coverage JSON retained obsolete unavailable status: %s", raw)
	}
}
