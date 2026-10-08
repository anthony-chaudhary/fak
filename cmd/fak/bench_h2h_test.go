package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/h2hbench"
)

// TestBenchH2HRunThenReport drives the real verb end to end: run against two
// streaming arms writes the ledger, report reads it back as agent JSON.
func TestBenchH2HRunThenReport(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":10}}}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	ledger := filepath.Join(t.TempDir(), "h2h.jsonl")
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"run", "--ledger", ledger, "--model", "m", "--run-id", "r1",
		"--arm", "fak=" + srv.URL + "/v1", "--arm", "llama=" + srv.URL + "/v1",
		"--cold", "64", "--reps", "1", "--decode", "4", "--turns", "2", "--system", "32", "--turn-tokens", "8"})
	if code != 0 {
		t.Fatalf("run exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "where fak loses") {
		t.Fatalf("run output: %s", out.String())
	}
	out.Reset()
	if code := runBenchH2H(&out, &errb, []string{"report", "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("report exit %d: %s", code, errb.String())
	}
	var rep h2hbench.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("report json: %v\n%s", err, out.String())
	}
	if rep.Schema != h2hbench.ReportSchema || rep.RunID != "r1" || rep.Rows != 2+2+4 {
		t.Fatalf("report = %+v", rep)
	}
	if code := runBenchH2H(&out, &errb, []string{"run", "--model", "m"}); code != 2 {
		t.Fatalf("missing --arm must exit 2, got %d", code)
	}
}
