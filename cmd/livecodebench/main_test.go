package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunFixtureSmokeJSONClaimDisallowed(t *testing.T) {
	fixture := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "fixture.json")
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	code := run([]string{"--fixture", fixture, "--check", "--json"})
	_ = w.Close()
	os.Stdout = oldStdout
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stdout=%s", code, out.String())
	}
	if !strings.Contains(out.String(), `"result_claim_allowed": false`) {
		t.Fatalf("json did not pin result_claim_allowed=false:\n%s", out.String())
	}
}
