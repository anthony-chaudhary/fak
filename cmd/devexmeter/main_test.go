package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunGatePassesOnStrictDrop(t *testing.T) {
	dir := t.TempDir()
	issue := filepath.Join(dir, "issue.json")
	ledger := filepath.Join(dir, "meter.jsonl")
	write(t, issue, `{"number":2166,"labels":[{"name":"dev-ex"},{"name":"friction/retry-after-refusal"}]}`)
	write(t, ledger, `{"issue":2166,"class":"retry-after-refusal","window":"before","value":10}
{"issue":2166,"class":"retry-after-refusal","window":"after","value":6}
`)
	var out, errb bytes.Buffer
	code := run(&out, &errb, []string{"--issue", issue, "--ledger", ledger})
	if code != 0 {
		t.Fatalf("run exit=%d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	for _, want := range []string{"PASS", "class=retry-after-refusal", "delta=-4"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("stdout missing %q in %q", want, out.String())
		}
	}
}

func TestRunGateHoldsFlatMeter(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "meter.jsonl")
	write(t, ledger, `{"issue":77,"class":"livelock","window":"before","value":3}
{"issue":77,"class":"livelock","window":"after","value":3}
`)
	var out, errb bytes.Buffer
	code := run(&out, &errb, []string{"--issue-number", "77", "--class", "livelock", "--ledger", ledger, "--json"})
	if code != 3 {
		t.Fatalf("run exit=%d stderr=%q stdout=%q, want NOT_YET exit 3", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"verdict":"NOT_YET"`) || !strings.Contains(out.String(), "strict drop") {
		t.Fatalf("stdout = %q, want NOT_YET strict-drop witness", out.String())
	}
}

func write(t *testing.T, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
