package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDispatchWorktreeABProducesScoreboardRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ab.json")
	raw := `{"baseline":{"wave_id":"fixed-wave","resolved":2,"duration_seconds":1200,"poison_incidents":1,"peak_concurrency":5},"isolated":{"wave_id":"fixed-wave","resolved":2,"duration_seconds":600,"poison_incidents":0,"peak_concurrency":8}}`
	if err := os.WriteFile(path, []byte(raw), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runDispatchWorktreeAB(&out, &errb, []string{"--in", path}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	for _, want := range []string{"baseline: 6.00 issues/h", "isolated: 12.00 issues/h", "ISOLATION_POISON_FREE"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

func TestDispatchWorktreeABRefusesDifferentWaves(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ab.json")
	if err := os.WriteFile(path, []byte(`{"baseline":{"wave_id":"fixed-wave","resolved":2,"duration_seconds":1},"isolated":{"wave_id":"fixed-wave","resolved":3,"duration_seconds":1}}`), 0644); err != nil {
		t.Fatal(err)
	}
	if code := runDispatchWorktreeAB(&bytes.Buffer{}, &bytes.Buffer{}, []string{"--in", path}); code != 3 {
		t.Fatalf("code=%d want 3", code)
	}
}
