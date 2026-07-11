package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
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
