package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestScoreNegationOperatorTextAndJSON(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runNegationOperatorScore(&out, &errb, []string{"--benchmark-delta", "1.5", "--unknown-fallback-rate", "0.2"}); code != 0 {
		t.Fatalf("text code=%d err=%s", code, errb.String())
	}
	for _, want := range []string{"benchmark_delta=1.500", "enumerable_domain_coverage=1.000", "unknown_fallback_rate=0.200"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("stdout %q missing %q", out.String(), want)
		}
	}
	out.Reset()
	errb.Reset()
	if code := runNegationOperatorScore(&out, &errb, []string{"--json"}); code != 0 {
		t.Fatalf("json code=%d err=%s", code, errb.String())
	}
	var payload struct {
		Corpus map[string]any `json:"corpus"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"benchmark_delta", "enumerable_domain_coverage", "unknown_fallback_rate"} {
		if _, ok := payload.Corpus[key]; !ok {
			t.Fatalf("JSON missing %s: %s", key, out.String())
		}
	}
}
