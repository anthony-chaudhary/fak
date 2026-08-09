package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDormancyJSONWitness(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ledger.jsonl")
	data := "{\"loop_id\":\"planned\",\"last_active_at\":\"2026-08-08T10:00:00Z\",\"expected_interval_seconds\":300,\"sleep_until\":\"2026-08-08T13:00:00Z\"}\n{\"loop_id\":\"hung\",\"last_active_at\":\"2026-08-08T11:00:00Z\",\"expected_interval_seconds\":300}\n"
	if err := os.WriteFile(p, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errs bytes.Buffer
	if rc := runDormancy(&out, &errs, []string{"--ledger", p, "--json", "--now", "2026-08-08T12:00:00Z"}); rc != 0 {
		t.Fatalf("rc=%d err=%s", rc, errs.String())
	}
	for _, want := range []string{`"intentionally_dormant": 1`, `"stuck": 1`, `"status": "intentionally_dormant"`, `"status": "stuck"`} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %s:\n%s", want, out.String())
		}
	}
}
