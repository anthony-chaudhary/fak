package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestDispatchLatCommandGolden(t *testing.T) {
	p := filepath.Join(t.TempDir(), "loops.jsonl")
	f, _ := os.Create(p)
	for i := 1; i <= 100; i++ {
		f.WriteString(`{"metrics":{"preflight_ms":` + strconv.Itoa(i) + `,"tick_total_ms":` + itoa(i*2) + `}}` + "\n")
	}
	f.Close()
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	cmdDispatchLat([]string{p})
	w.Close()
	os.Stdout = old
	var b bytes.Buffer
	b.ReadFrom(r)
	got := b.String()
	if !strings.Contains(got, "preflight                100       50       90       99") || !strings.Contains(got, "total                    100      100      180      198") {
		t.Fatalf("out=%s", got)
	}
}

func TestDispatchLatSinceFiltersOldEvents(t *testing.T) {
	p := filepath.Join(t.TempDir(), "loops.jsonl")
	now := time.Now().UnixNano()
	old := now - int64(2*time.Hour)
	rows := `{"ts_unix_nano":` + strconv.FormatInt(old, 10) + `,"metrics":{"preflight_ms":999}}` + "\n" + `{"ts_unix_nano":` + strconv.FormatInt(now, 10) + `,"metrics":{"preflight_ms":7}}` + "\n"
	if err := os.WriteFile(p, []byte(rows), 0600); err != nil {
		t.Fatal(err)
	}
	oldOut := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	cmdDispatchLat([]string{"--since", "1h", p})
	w.Close()
	os.Stdout = oldOut
	var b bytes.Buffer
	b.ReadFrom(r)
	if !strings.Contains(b.String(), "preflight                  1        7        7        7") {
		t.Fatalf("out=%s", b.String())
	}
}
