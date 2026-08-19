package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/providercost"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestProviderCostCLIImportReportAndReconcile(t *testing.T) {
	d := t.TempDir()
	ledger := filepath.Join(d, "cost.jsonl")
	input := filepath.Join(d, "input.jsonl")
	registry := filepath.Join(d, "registry.jsonl")
	line := `{"schema":"fak-provider-cost-ledger/1","provider":"openai","provider_row_id":"row-1","session_id":"sess-1","interval_start":"2026-08-01","interval_end":"2026-08-02","billed_micro_usd":42,"currency":"USD","export_id":"export-1","exported_at":"2026-08-03T00:00:00Z","source":"provider-billing-export"}` + "\n"
	if err := os.WriteFile(input, []byte(line), 0600); err != nil {
		t.Fatal(err)
	}
	r, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "root", LaunchKind: "guard", Runtime: "codex", SessionID: "sess-1", Now: time.Unix(1, 0)})
	if err := (sessionregistry.Store{Path: registry}).Register(r); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	runProviderCost(&out, &errOut, []string{"import", "--ledger", ledger, "--input", input})
	var imp providercost.ImportReport
	if err := json.NewDecoder(&out).Decode(&imp); err != nil {
		t.Fatal(err)
	}
	if imp.Imported != 1 {
		t.Fatal(imp)
	}
	out.Reset()
	runProviderCost(&out, &errOut, []string{"report", "--ledger", ledger, "--registry", registry})
	var rep providercost.Report
	if err := json.NewDecoder(&out).Decode(&rep); err != nil {
		t.Fatal(err)
	}
	if len(rep.Roots) != 1 || rep.Roots[0].BilledMicroUSD != 42 {
		t.Fatalf("report=%+v", rep)
	}
}
