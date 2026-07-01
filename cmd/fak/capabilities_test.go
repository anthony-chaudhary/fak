package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestCapabilitiesJSONListsToolbelt(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runCapabilities(&out, &errb, []string{"--root", root, "--json"}); rc != 0 {
		t.Fatalf("runCapabilities rc=%d stderr=%s", rc, errb.String())
	}
	var resp struct {
		Cards []struct {
			Name string `json:"name"`
		} `json:"cards"`
	}
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("capabilities --json invalid: %v\n%s", err, out.String())
	}
	seen := map[string]bool{}
	for _, c := range resp.Cards {
		seen[c.Name] = true
	}
	for _, want := range []string{"memory-driver:recall", "memory-driver:compact", "fak index lane", "fak_changes", "dos_arbitrate"} {
		if !seen[want] {
			t.Fatalf("capabilities --json missing %s; got %v", want, seen)
		}
	}
}

func TestCapabilitiesIntentRanksMemoryRunFirst(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runCapabilities(&out, &errb, []string{"--root", root, "compact", "my", "context"}); rc != 0 {
		t.Fatalf("runCapabilities rc=%d stderr=%s", rc, errb.String())
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) < 2 || !strings.HasPrefix(lines[1], "memory-driver:compact") {
		t.Fatalf("capabilities table top row = %v, want memory-driver:compact first", lines)
	}
	if !strings.Contains(out.String(), "fak_memory_run") {
		t.Fatalf("capabilities table missing fak_memory_run call:\n%s", out.String())
	}
}

func TestCapabilitiesRejectsNegativeLimit(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runCapabilities(&out, &errb, []string{"--root", root, "--limit", "-1"}); rc == 0 {
		t.Fatalf("negative limit should fail, got rc=0 out=%s", out.String())
	}
}
