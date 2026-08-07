package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestToolWidthReportAndRegressionExit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "width.jsonl")
	rows := `{"lane":"cmd","engine":"claude","model":"m","tool_calls":2,"success":true}
{"lane":"cmd","engine":"claude","model":"m","tool_calls":1,"success":true}
{"lane":"cmd","engine":"claude","model":"m","tool_calls":4,"client_suppressed":true}
`
	if err := os.WriteFile(path, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runToolWidth(&out, &errOut, []string{"--input", path, "--baseline", "0.9", "--min-drop", "0.2"})
	if code != 3 {
		t.Fatalf("code=%d stderr=%s output=%s", code, errOut.String(), out.String())
	}
	for _, want := range []string{`"batched_turn_rate": 0.5`, `"client_suppressed_rate": 0.3333333333333333`, `"regressed": true`, `"outcome_rate": 0.6666666666666666`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %s: %s", want, out.String())
		}
	}
}
