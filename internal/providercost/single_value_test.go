package providercost

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Five tiny rejected imports share one seed file; the runtime estimate is unmeasured.
// fak-test:runtime fast est=10ms lane=default
func TestParseAndImportRejectTrailingRowData(t *testing.T) {
	t.Parallel()
	const row = `{"schema":"fak-provider-cost-ledger/1","provider":"openai","provider_row_id":"r1","session_id":"s","export_id":"e","exported_at":"now","source":"billing-export","billed_micro_usd":7,"currency":"USD"}`
	next := strings.Replace(row, `"r1"`, `"r2"`, 1)
	rows, err := Parse([]byte(row + " \t\r\n" + next + "\n"))
	if err != nil || len(rows) != 2 || rows[0].ProviderRowID != "r1" || rows[1].ProviderRowID != "r2" || rows[0].BilledMicroUSD == nil || *rows[0].BilledMicroUSD != 7 {
		t.Fatalf("valid rows = %+v, err = %v", rows, err)
	}
	path := filepath.Join(t.TempDir(), "cost.jsonl")
	seed := []byte(row + "\n")
	if err := os.WriteFile(path, seed, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tail := range []string{" " + next, " null", " 7", " ]", " garbage"} {
		input := next + "\n" + row + tail + "\n"
		if got, err := Parse([]byte(input)); err == nil || got != nil || !strings.Contains(err.Error(), "line 2:") {
			t.Errorf("Parse tail %q = %+v, %v; want line 2 error and no rows", tail, got, err)
		}
		if _, err := Import(path, strings.NewReader(input)); err == nil {
			t.Errorf("Import accepted tail %q", tail)
		}
		got, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(got, seed) {
			t.Fatalf("Import tail %q changed ledger: %q, %v", tail, got, err)
		}
	}
}
