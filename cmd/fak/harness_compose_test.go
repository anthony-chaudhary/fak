package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessComposeCLIMixedLayersAndProvenance(t *testing.T) {
	path := filepath.Join(t.TempDir(), "assets.json")
	raw := `{"schema":"fak.harness-assets/v1alpha1","layers":[
		{"id":"company","scope":"company","assets":[{"kind":"policy","id":"tools","grants":["search","shell"],"denies":["refund"]},{"kind":"workflow","id":"audit","value":"record","mandatory":true}]},
		{"id":"person","scope":"person","assets":[{"kind":"ui","id":"density","value":"compact"}]},
		{"id":"matter-7","scope":"project","assets":[{"kind":"memory","id":"matter","boundary":"matter-7","value":"private"},{"kind":"policy","id":"tools","denies":["shell"]}]},
		{"id":"legal","scope":"domain","assets":[{"kind":"instruction","id":"citations","value":"primary-only"},{"kind":"tool","id":"research","value":"legal-search"}]}
	]}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	selection := filepath.Join(filepath.Dir(path), "selection.json")
	if err := os.WriteFile(selection, []byte(`{"layers":["company","person","matter-7","legal"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"compose", "--assets", path, "--selection", selection})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{`"kind": "instruction"`, `"kind": "memory"`, `"kind": "policy"`, `"kind": "tool"`, `"kind": "ui"`, `"kind": "workflow"`, `"source": "legal"`, `"source": "matter-7"`, `"action": "narrow"`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %s:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), `"shell"`) && !strings.Contains(out.String(), `"denies": [\n        "refund",\n        "shell"`) {
		t.Fatalf("shell was not narrowed:\n%s", out.String())
	}
}

func TestHarnessComposeCLIRefusesPolicyWidening(t *testing.T) {
	path := filepath.Join(t.TempDir(), "assets.json")
	raw := `{"schema":"fak.harness-assets/v1alpha1","layers":[{"id":"company","scope":"company","assets":[{"kind":"policy","id":"tools","grants":["search"],"denies":["shell"]}]},{"id":"task","scope":"task","assets":[{"kind":"policy","id":"tools","grants":["shell"]}]}]}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runHarness(&out, &errb, []string{"compose", "--assets", path, "--layer", "company", "--layer", "task"}); code != 1 {
		t.Fatalf("code=%d output=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), `layer "task" policy/tools: policy privilege widening for shell`) {
		t.Fatalf("stderr=%s", errb.String())
	}
}
