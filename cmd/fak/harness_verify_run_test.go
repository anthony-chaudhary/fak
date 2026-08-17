package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestHarnessVerifyRunCapturedDeviation(t *testing.T) {
	lockPath := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema, Assets: []harnesscompose.EffectiveAsset{
		{Kind: "instruction", ID: "tone", Value: "concise", Source: "team:support"},
		{Kind: "policy", ID: "tools", Grants: []string{"search"}, Denies: []string{"shell"}, Source: "company:security"},
		{Kind: "tool", ID: "search", Source: "repo:defaults"},
	}})
	var lock harnessresolve.Lock
	raw, err := os.ReadFile(lockPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &lock); err != nil {
		t.Fatal(err)
	}
	observation := filepath.Join(t.TempDir(), "run.json")
	writeOverrideFile(t, observation, `{"schema":"fak-harness-runtime-observation/1","lock_id":"`+lock.ID+`","run_id":"run-42","capabilities":[{"capability":"instruction:tone","source":"task:hotfix","value":"verbose"},{"capability":"policy:tools","source":"company:security","grants":["search"],"denies":["shell"]},{"capability":"tool:deploy","source":"runtime:plugin"}],"events":[{"kind":"route","capability":"route:model","source":"gateway","outcome":"selected fast"},{"kind":"approval","capability":"tool:deploy","source":"policy","outcome":"allowed"}]}`)
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"verify-run", "--lock", lockPath, "--observation", observation})
	for _, want := range []string{
		"HARNESS VERIFY RUN | DEVIATION", "capabilities: matched=1 changed=1 added=1 omitted=1",
		"instruction:tone | changed | expected team:support | runtime task:hotfix | changed source,value",
		"policy:tools | matched", "tool:deploy | added | runtime runtime:plugin", "tool:search | omitted | expected repo:defaults",
		"route | route:model | from gateway | selected fast", "approval | tool:deploy | from policy | allowed",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q:\n%s", want, out.String())
		}
	}
	if code != 3 || errb.Len() != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
}

func TestHarnessVerifyRunRefusesWrongLock(t *testing.T) {
	lockPath := writePreviewLock(t, harnessresolve.Lock{Schema: harnessresolve.LockSchema})
	observation := filepath.Join(t.TempDir(), "run.json")
	writeOverrideFile(t, observation, `{"schema":"fak-harness-runtime-observation/1","lock_id":"sha256:wrong","run_id":"run-1","capabilities":[]}`)
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"verify-run", "--lock", lockPath, "--observation", observation})
	if code != 1 || out.Len() != 0 || !strings.Contains(errb.String(), "does not match verified lock") {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}
