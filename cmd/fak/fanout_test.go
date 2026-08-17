package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFanoutTrendRendersLedger(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "fanout.jsonl")
	rows := `{"schema":"fak-fanout-reuse-ledger/1","date":"2026-08-01","profile":"research","agents":4,"sub_turns":4,"prefix_tokens":1000,"cross_uplift_p50":3000,"prefix_tokens_saved":3000,"tax_clawed_back":0.75}
{"schema":"fak-fanout-reuse-ledger/1","date":"2026-08-08","profile":"research","agents":4,"sub_turns":4,"prefix_tokens":1000,"cross_uplift_p50":3100,"prefix_tokens_saved":3100,"tax_clawed_back":0.76}
`
	if err := os.WriteFile(ledger, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	if code := runFanoutTrend(&out, &errOut, []string{"--ledger", ledger}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"fan-out reuse trend", "2026-08-01", "2026-08-08", "first"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
}
