package providercost

import (
	"bytes"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"testing"
	"time"
)

func amount(v MicroUSD) *MicroUSD { return &v }
func reg(id, root, session string) sessionregistry.Record {
	r, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: id, RootRegistrationID: root, LaunchKind: "guard", Runtime: "codex", SessionID: session, Now: time.Unix(1, 0)})
	return r
}
func TestFoldKeepsProvidersRootsRetriesMissingAndAmbiguousSeparate(t *testing.T) {
	rows := []Row{{Schema: Schema, Provider: "anthropic", ProviderRowID: "a1", SessionID: "s1", ExportID: "e1", ExportedAt: "now", Source: "billing-export", BilledMicroUSD: amount(10), Currency: "USD"}, {Schema: Schema, Provider: "openai", ProviderRowID: "o1", SessionID: "s2", ExportID: "e2", ExportedAt: "now", Source: "billing-export", BilledMicroUSD: amount(20), Currency: "USD"}, {Schema: Schema, Provider: "openai", ProviderRowID: "o2", SessionID: "missing", ExportID: "e2", ExportedAt: "now", Source: "billing-export"}, {Schema: Schema, Provider: "anthropic", ProviderRowID: "a2", SessionID: "amb", ExportID: "e1", ExportedAt: "now", Source: "billing-export", BilledMicroUSD: amount(30), Currency: "USD"}}
	regs := []sessionregistry.Record{reg("r1", "r1", "s1"), reg("r1-retry", "r1", "s1"), reg("r2", "r2", "s2"), reg("x", "x", "amb"), reg("y", "y", "amb")}
	got := Fold(rows, regs)
	if len(got.Roots) != 2 || got.Roots[0].RootRegistrationID != "r1" || got.Roots[0].BilledMicroUSD != 10 || got.Roots[1].BilledMicroUSD != 20 {
		t.Fatalf("roots=%+v", got.Roots)
	}
	if got.Coverage.AttributedRows != 2 || got.Coverage.MissingRows != 1 || got.Coverage.AmbiguousRows != 1 || got.Coverage.TotalBilledMicroUSD != 60 || got.Coverage.AttributedBilledMicroUSD != 30 {
		t.Fatalf("coverage=%+v", got.Coverage)
	}
}
func TestImportDeduplicatesProviderRowsAndPreservesUnknown(t *testing.T) {
	path := t.TempDir() + "/cost.jsonl"
	line := `{"schema":"fak-provider-cost-ledger/1","provider":"openai","provider_row_id":"r1","session_id":"s","export_id":"e","exported_at":"now","source":"billing-export"}` + "\n"
	r, err := Import(path, bytes.NewBufferString(line+line))
	if err != nil {
		t.Fatal(err)
	}
	if r.Imported != 1 || r.Duplicates != 1 || r.UnknownAmount != 1 {
		t.Fatalf("report=%+v", r)
	}
	rows, err := Read(path)
	if err != nil || len(rows) != 1 || rows[0].BilledMicroUSD != nil {
		t.Fatalf("rows=%+v err=%v", rows, err)
	}
}
func TestReconcileNeverTreatsUnknownAsKnownAmount(t *testing.T) {
	rows := []Row{{Schema: Schema, Provider: "p", ProviderRowID: "1", SessionID: "s", ExportID: "e", ExportedAt: "now", Source: "x"}}
	expected := MicroUSD(0)
	r := Reconcile(rows, "p", 1, &expected)
	if !r.RowsMatch || r.AmountMatch == nil || !*r.AmountMatch {
		t.Fatalf("reconcile=%+v", r)
	}
	if Fold(rows, nil).Coverage.AmountRows != 0 {
		t.Fatal("unknown amount counted as zero-valued fact")
	}
}
