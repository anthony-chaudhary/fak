package cachevaluereport

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFabricTrack1RowMapsControlledLedger(t *testing.T) {
	path := filepath.Join("..", "..", "experiments", "microcontext", "s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json")
	row, err := FabricTrack1Row(path)
	if err != nil {
		t.Fatal(err)
	}
	if row.SessionType != "fabric-shared-base" || row.Provider != "fak-inkernel" || row.Mechanism != "fabric_radix_prefix" || row.Turns != 8 || row.PromptTokens != 5608 || row.ReusedTokens != 4859 {
		t.Fatalf("row=%+v", row)
	}
}

func TestFabricTrack1RowRejectsSyntheticSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "synthetic.json")
	if err := os.WriteFile(path, []byte(`{"schema":"fak-microcontext-run/1"}`), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := FabricTrack1Row(path); err == nil {
		t.Fatal("expected provenance refusal")
	}
}
