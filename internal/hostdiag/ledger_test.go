package hostdiag

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCensusMixedLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hostdiag.jsonl")
	data := `{"schema":"fak.hostdiag-correlation.v1","correlation_id":"x"}` + "\n" + `{"schema":"fak.hostdiag-census.v1","sample_id":"s","sampled_at_ms":2,"pid":1,"process_start_ms":1}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := LoadCensus(path)
	if err != nil || len(got) != 1 || got[0].PID != 1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestLoadCensusRejectsMalformed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hostdiag.jsonl")
	if err := os.WriteFile(path, []byte("bad\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCensus(path); err == nil {
		t.Fatal("accepted malformed ledger")
	}
}
