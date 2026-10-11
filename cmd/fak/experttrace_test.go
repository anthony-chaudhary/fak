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
