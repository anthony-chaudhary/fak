package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func TestSessionIndexHealthMissingDefaultIndexUsesStatusContract(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	var out, errb bytes.Buffer
	if code := runSessionIndexHealth(&out, &errb, nil); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "RED (index_missing)") || strings.Contains(errb.String(), "--index is required") {
		t.Fatalf("stdout=%s stderr=%s", out.String(), errb.String())
	}
}

func TestSessionIndexHealthMissingIndexHumanAndJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.json")
	var out, errb bytes.Buffer
	if code := runSessionIndexHealth(&out, &errb, []string{"--index", path, "--now", "2000000"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "RED (index_missing)") {
		t.Fatalf("%s", out.String())
	}
	out.Reset()
	if code := runSessionIndexHealth(&out, &errb, []string{"--index", path, "--json", "--now", "2000000"}); code != 0 {
		t.Fatal(code)
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["schema"] != "fak-session-history-status/1" || got["verdict"] != "RED" {
		t.Fatalf("%v", got)
	}
}
