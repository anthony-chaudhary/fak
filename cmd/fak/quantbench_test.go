package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantbench"
)

func TestQuantbenchSelfTestJSON(t *testing.T) {
	var out, err bytes.Buffer
	if code := runQuantbench(&out, &err, []string{"--self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, err.String())
	}
	var report quantbench.SelfTestReport
	if e := json.Unmarshal(out.Bytes(), &report); e != nil {
		t.Fatal(e)
	}
	if !report.Pass || report.Schema != quantbench.Schema {
		t.Fatalf("report=%+v", report)
	}
}

func TestQuantbenchUnknownFormatIsTyped(t *testing.T) {
	in := `{"artifact":{"format":"futureq","version":"1","recipe":"x"},"runtime":{"name":"vllm","version":"1"}}`
	var parsed quantbench.Input
	if e := json.Unmarshal([]byte(in), &parsed); e != nil {
		t.Fatal(e)
	}
	got := quantbench.Evaluate(parsed)
	if got.Outcome != quantbench.OutcomeAbstain || got.ReasonCode != quantbench.ReasonUnknownFormat {
		t.Fatalf("got %+v", got)
	}
}
