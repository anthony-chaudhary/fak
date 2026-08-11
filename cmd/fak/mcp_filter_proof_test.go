package main

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestMCPFilterProofCommand(t *testing.T) {
	var out, errw bytes.Buffer
	if code := runMCPFilterProof(&out, &errw, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errw.String())
	}
	var got struct {
		Verdict               string  `json:"verdict"`
		TaskSuccessRate       float64 `json:"task_success_rate"`
		SearchRecall          float64 `json:"search_recall"`
		FirstCallRouteSuccess float64 `json:"first_call_route_success"`
		Active                struct {
			SavedBytes int `json:"saved_bytes"`
		} `json:"active"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "PASS" || got.TaskSuccessRate != 1 || got.SearchRecall != 1 || got.FirstCallRouteSuccess != 1 || got.Active.SavedBytes <= 0 {
		t.Fatalf("proof=%+v", got)
	}
}
