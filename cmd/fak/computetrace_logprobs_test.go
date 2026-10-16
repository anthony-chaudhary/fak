package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestComputeTraceLogprobsScoresOracleBattery(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": " Paris. In 1923, the International"}}})
	}))
	defer srv.Close()
	var out, errb bytes.Buffer
	code := runComputeTraceLogprobs(&out, &errb, []string{"--endpoint", srv.URL, "--ref", "../../pkg/referenceoracle/testdata/v41-reference-oracle.json", "--json"})
	var rep struct {
		Verdict           string `json:"verdict"`
		GreedyMatchTokens int    `json:"greedy_match_tokens"`
		Prompts           []struct {
			Prompt            string `json:"prompt"`
			GreedyMatchTokens int    `json:"greedy_match_tokens"`
		} `json:"prompts"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("exit %d: %v\n%s%s", code, err, out.String(), errb.String())
	}
	if len(rep.Prompts) != 4 || rep.Prompts[1].Prompt != "The capital of France is" || rep.Prompts[1].GreedyMatchTokens != 8 {
		t.Fatalf("report %+v", rep)
	}
}

func TestComputeTraceLogprobsUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runComputeTraceLogprobs(&out, &errb, []string{"--endpoint", "http://x"}); code != 2 {
		t.Fatalf("exit %d, want 2", code)
	}
}
