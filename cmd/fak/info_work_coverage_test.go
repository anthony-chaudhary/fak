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

func TestWorkCoverageUnavailableIsNamedNotZero(t *testing.T) {
	text := workCoverageUnavailableText()
	for _, want := range []string{"context_elision", "model_routing", "schema_tool_filtering"} {
		if !strings.Contains(text, want) {
			t.Fatalf("unavailable text missing %q: %s", want, text)
		}
	}
	if strings.Contains(text, "0") {
		t.Fatalf("unavailable coverage rendered as zero: %s", text)
	}
}

func TestWorkCoverageJSONUsesStableFieldsNotDisplayParsing(t *testing.T) {
	raw, err := marshalWorkCoverageForTest()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"source_id":"provider_cache"`)) || !bytes.Contains(raw, []byte(`"status":"not_yet_measurable"`)) || !bytes.Contains(raw, []byte(`"reason":`)) {
		t.Fatalf("coverage JSON=%s", raw)
	}
}
