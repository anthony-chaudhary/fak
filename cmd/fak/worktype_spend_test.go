package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/worktype"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestWorktypeSpendJoinsAndPreservesCoverage(t *testing.T) {
	d := t.TempDir()
	u := filepath.Join(d, "u")
	c := filepath.Join(d, "c")
	o := filepath.Join(d, "o")
	rows := []gatewayusageledger.Row{gatewayusageledger.NewRow("exit", "guard", "", "a", time.Second, nil, gatewayusageledger.Counters{InputTokens: 100}, time.Unix(1, 0)), gatewayusageledger.NewRow("exit", "guard", "", "b", time.Second, nil, gatewayusageledger.Counters{InputTokens: 300}, time.Unix(2, 0))}
	var buf bytes.Buffer
	for _, r := range rows {
		x, _ := json.Marshal(r)
		buf.Write(x)
		buf.WriteByte('\n')
	}
	_ = os.WriteFile(u, buf.Bytes(), 0600)
	_ = os.WriteFile(c, []byte(`{"schema":"fak-worktype/1","trace_id":"a","pattern_id":"wp.issue-to-patch"}`+"\n"), 0600)
	_ = os.WriteFile(o, []byte(`{"schema":"fak-session-outcomes/1","session_id":"a","outcome":"shipped_commit","provenance":"independent_witness"}`+"\n"), 0600)
	var out, er bytes.Buffer
	if runWorktypeSpend(&out, &er, []string{"--usage-ledger", u, "--classifications", c, "--outcomes", o, "--json"}) != 0 {
		t.Fatal(er.String())
	}
	var got worktype.SpendReport
	_ = json.Unmarshal(out.Bytes(), &got)
	if got.Coverage.RowCount != 2 || got.Coverage.ClassifiedRows != 1 || got.Coverage.TotalTokens != 400 || got.Coverage.CoveredTokens != 100 {
		t.Fatalf("%+v", got.Coverage)
	}
	for _, g := range got.Groups {
		if g.PatternID == "wp.issue-to-patch" && (g.AcceptedRate != 1 || g.Drilldown[0].TraceID != "a") {
			t.Fatalf("%+v", g)
		}
	}
}
func TestNormalizeSpendOutcomeUnknown(t *testing.T) {
	if normalizeSpendOutcome("plausible") != "unknown" {
		t.Fatal("must abstain")
	}
}
