package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestExpertTraceRanksAndOverlapCLI(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.jsonl"), filepath.Join(dir, "b.jsonl")
	header := `{"schema":"fak.expert-route.trace/v1","layers":1,"experts":4,"top_k":2}` + "\n"
	if err := os.WriteFile(a, []byte(header+`{"layer":0,"token":0,"phase":"decode","experts":[0,1]}
{"layer":0,"token":1,"phase":"decode","experts":[2,3]}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte(header+`{"layer":0,"token":0,"phase":"decode","experts":[0,2]}
{"layer":0,"token":1,"phase":"decode","experts":[2,3]}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runExpertTrace(&out, &errb, []string{"analyze", a, "--json", "--ranks", "2"}); code != 0 {
		t.Fatalf("analyze exit %d: %s", code, errb.String())
	}
	var analysis struct {
		RankBalance []struct {
			Ranks     int `json:"ranks"`
			CountEven struct {
				Steps int     `json:"steps"`
				MoeLB float64 `json:"moe_lb"`
			} `json:"count_even"`
		} `json:"rank_balance"`
	}
	if err := json.Unmarshal(out.Bytes(), &analysis); err != nil {
		t.Fatal(err)
	}
	if len(analysis.RankBalance) != 1 || analysis.RankBalance[0].Ranks != 2 || analysis.RankBalance[0].CountEven.Steps != 1 || analysis.RankBalance[0].CountEven.MoeLB != 2 {
		t.Fatalf("rank_balance = %+v", analysis.RankBalance)
	}
	out.Reset()
	if code := runExpertTrace(&out, &errb, []string{"overlap", a, b, "--json", "--phase", "decode", "--ranks", "2"}); code != 0 {
		t.Fatalf("overlap exit %d: %s", code, errb.String())
	}
	var overlap struct {
		Schema         string  `json:"schema"`
		Streams        int     `json:"streams"`
		Steps          int     `json:"steps"`
		UniqueReads    int     `json:"unique_reads"`
		SharedFraction float64 `json:"shared_fraction"`
		RankBalance    []any   `json:"rank_balance"`
	}
	if err := json.Unmarshal(out.Bytes(), &overlap); err != nil {
		t.Fatal(err)
	}
	if overlap.Schema != "fak.expert-route.overlap/v1" || overlap.Streams != 2 || overlap.Steps != 2 || overlap.UniqueReads != 5 || overlap.SharedFraction != 0.375 || len(overlap.RankBalance) != 1 {
		t.Fatalf("overlap = %+v", overlap)
	}
	if code := runExpertTrace(&out, &errb, []string{"overlap", a}); code != 2 {
		t.Fatalf("single-stream overlap exit = %d, want 2", code)
	}
}
