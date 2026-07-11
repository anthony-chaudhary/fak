package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResumeCapReadsLiveSeatHeadroom(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sessions.json")
	if err := os.WriteFile(p, []byte(`{"accounts":[{"available":true,"active_sessions":1},{"available":true,"active_sessions":2},{"available":false,"active_sessions":0}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runResumeCap(&out, &errOut, []string{"--sessions", p, "--floor", "4", "--ceiling", "20", "--seat-cap", "5"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"cap": 7`) || !strings.Contains(out.String(), `"headroom": 7`) {
		t.Fatalf("out=%s", out.String())
	}
}
