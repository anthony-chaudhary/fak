package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestExpertTraceAnalyzeCLI(t *testing.T) {
	dir := t.TempDir()
	trace := filepath.Join(dir, "route.jsonl")
	body := `{"schema":"fak.expert-route.trace/v1","layers":1,"experts":4,"top_k":2}
{"layer":0,"token":0,"phase":"prefill","experts":[0,1]}
{"layer":0,"token":1,"phase":"decode","experts":[0,2]}
`
	if err := os.WriteFile(trace, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	usage, strix := filepath.Join(dir, "hist.json"), filepath.Join(dir, "strix.json")
	var out, errb bytes.Buffer
	code := runExpertTrace(&out, &errb, []string{"analyze", trace, "--json", "--capacities", "1,2", "--usage-out", usage, "--strix-out", strix, "--expert-bytes", "64"})
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	var rep model.ExpertRouteAnalysis
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Schema != model.ExpertRouteAnalysisSchema || rep.Routes != 2 || rep.Picks != 4 || rep.UniqueExperts != 3 || len(rep.Cache) != 2 {
		t.Fatalf("report = %+v", rep)
	}
	h, err := model.LoadExpertUsageHistogram(usage)
	if err != nil || h.Count(0, 0) != 2 {
		t.Fatalf("usage histogram err=%v count(0,0)=%v, want 2", err, h.Count(0, 0))
	}
	raw, err := os.ReadFile(strix)
	if err != nil {
		t.Fatal(err)
	}
	var st model.StrixMoECacheTrace
	if err := json.Unmarshal(raw, &st); err != nil || st.Schema != model.StrixMoECacheTraceSchema || len(st.Events) != 4 || st.ExpertBytes != 64 {
		t.Fatalf("strix trace err=%v %+v", err, st)
	}
	if code := runExpertTrace(&out, &errb, []string{"analyze"}); code != 2 {
		t.Fatalf("missing path exit = %d, want 2", code)
	}
}

// fak-test:runtime fast est=10ms
func TestExpertTracePolicyWeightedCLI(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weighted.jsonl")
	body := `{"layer":0,"token":0,"experts":[0],"weights":[1]}
{"layer":0,"token":1,"experts":[1],"weights":[0.01]}
{"layer":0,"token":2,"experts":[2],"weights":[0.01]}
{"layer":0,"token":3,"experts":[0],"weights":[1]}
`
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runExpertTrace(&out, &errb, []string{"policy", path, "--capacity", "2", "--json"}); code != 0 {
		t.Fatalf("exit=%d error=%s", code, errb.String())
	}
	var report struct {
		ScoreInput  string `json:"score_input"`
		Measurement string `json:"measurement"`
		Rows        []struct {
			Hits   int `json:"hits"`
			Misses int `json:"misses"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.ScoreInput != "selected-route-weights" || report.Measurement != "simulated" || len(report.Rows) != 4 {
		t.Fatalf("invalid policy report: %s", out.String())
	}
	if report.Rows[0].Hits != 0 || report.Rows[2].Hits != 1 || report.Rows[3].Hits != 1 {
		t.Fatalf("unexpected policy hits: %s", out.String())
	}
	for _, body := range []string{`{"layer":0,"experts":[0]}`, `{"layer":0,"experts":[0],"weights":[-1]}`, `{"layer":0,"experts":[0],"weights":[1,2]}`, `{"layer":0,"experts":[0,0],"weights":[0.5,0.5]}`} {
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		out.Reset()
		errb.Reset()
		if code := runExpertTrace(&out, &errb, []string{"policy", path, "--capacity", "2", "--json"}); code != 1 || out.Len() != 0 {
			t.Fatalf("invalid trace exit=%d output=%s", code, out.String())
		}
	}
}

// fak-test:runtime fast est=10ms
func TestExpertTracePolicyRejectsExtraArguments(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runExpertTrace(&out, &errb, []string{"policy", "trace.jsonl", "--capacity", "2", "extra"}); code != 2 {
		t.Fatalf("surplus arg exit=%d", code)
	}
}

// fak-test:runtime fast est=10ms
func TestExpertTracePolicyRejectsUnknownArm(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weighted.jsonl")
	if err := os.WriteFile(path, []byte(`{"layer":0,"experts":[0],"weights":[1]}`), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runExpertTrace(&out, &errb, []string{"policy", path, "--capacity", "2", "--policies", "nonexistent"}); code != 2 {
		t.Fatalf("unknown policy exit=%d", code)
	}
}
