package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessResolveCLIEmitsImmutableExplainableLock(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "product.json")
	selection := filepath.Join(dir, "selection.json")
	raw := `{"schema":"fak.harness-product/v1alpha1","roots":["legal-pack"],"compatibility":{"os":["linux"],"arch":["amd64"],"contract":"v1"},"budget":{"context_tokens":2000,"memory_mib":512,"workers":2},"components":[{"id":"kernel","version":"1.0.0","digest":"sha256:kernel","source":"registry/kernel","provides":["runtime"],"cost":{"context_tokens":500,"memory_mib":256},"evidence":{"authority":"fixture","source":"kernel"}},{"id":"legal-pack","version":"1.2.0","digest":"sha256:legal","source":"registry/legal","requires":[{"capability":"runtime","range":">=1.0.0"},{"capability":"search","range":">=1.0.0"}],"cost":{"context_tokens":1000,"memory_mib":64,"workers":1},"evidence":{"authority":"fixture","source":"legal"}},{"id":"search","version":"1.1.0","digest":"sha256:search","source":"registry/search","provides":["search"],"evidence":{"authority":"fixture","source":"search"}}],"assets":{"schema":"fak.harness-assets/v1alpha1","layers":[{"id":"company","scope":"company","assets":[{"kind":"policy","id":"tools","grants":["search"],"denies":["shell"],"lock":true},{"kind":"workflow","id":"audit","value":"record","mandatory":true}]},{"id":"legal","scope":"domain","assets":[{"kind":"instruction","id":"citations","value":"primary-only"}]}]}}`
	if err := os.WriteFile(manifest, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(selection, []byte(`{"layers":["company","legal"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"resolve", "--manifest", manifest, "--selection", selection, "--os", "linux", "--arch", "amd64", "--contract", "v1"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{`"schema": "fak.harness-product-lock/v1alpha2"`, `"id": "sha256:`, `"version": "1.2.0"`, `"digest": "sha256:legal"`, `"source": "registry/legal"`, `"kind": "instruction"`, `"kind": "policy"`, `"kind": "workflow"`, `"explain": [`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %s:\n%s", want, out.String())
		}
	}
}

func TestHarnessResolveCLIRefusesCompatibilityBeforeLock(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "product.json")
	selection := filepath.Join(dir, "selection.json")
	raw := `{"schema":"fak.harness-product/v1alpha1","roots":["kernel"],"compatibility":{"os":["linux"]},"components":[{"id":"kernel","version":"1.0.0","digest":"sha256:k","source":"r","evidence":{"authority":"f","source":"f"}}],"assets":{"schema":"fak.harness-assets/v1alpha1","layers":[{"id":"company","scope":"company","assets":[{"kind":"workflow","id":"audit","value":"record","mandatory":true}]}]}}`
	os.WriteFile(manifest, []byte(raw), 0o600)
	os.WriteFile(selection, []byte(`{"layers":["company"]}`), 0o600)
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"resolve", "--manifest", manifest, "--selection", selection, "--os", "windows", "--arch", "amd64", "--contract", "v1"})
	if code != 1 || !strings.Contains(errb.String(), `incompatible OS "windows"`) {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if out.Len() != 0 {
		t.Fatalf("unexpected lock=%s", out.String())
	}
}

func TestHarnessStageTemplatesCompletePublicResolveToVerifyPath(t *testing.T) {
	var manifestOut, selectionOut, errb bytes.Buffer
	if code := runHarness(&manifestOut, &errb, []string{"resolve", "--example", "manifest"}); code != 0 {
		t.Fatalf("manifest example code=%d stderr=%s", code, errb.String())
	}
	if code := runHarness(&selectionOut, &errb, []string{"resolve", "--example", "selection"}); code != 0 {
		t.Fatalf("selection example code=%d stderr=%s", code, errb.String())
	}
	if strings.Contains(manifestOut.String(), "search_kb") || strings.Contains(manifestOut.String(), "customer-support") {
		t.Fatalf("generic template leaked study answer: %s", manifestOut.String())
	}

	dir := t.TempDir()
	manifest := filepath.Join(dir, "product.json")
	selection := filepath.Join(dir, "selection.json")
	lockPath := filepath.Join(dir, "product.lock.json")
	if err := os.WriteFile(manifest, manifestOut.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(selection, selectionOut.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	var resolveOut bytes.Buffer
	if code := runHarness(&resolveOut, &errb, []string{"resolve", "--manifest", manifest, "--selection", selection, "--os", "linux", "--arch", "amd64", "--contract", "v1", "--output", lockPath}); code != 0 {
		t.Fatalf("resolve code=%d stdout=%s stderr=%s", code, resolveOut.String(), errb.String())
	}
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("resolved lock: %v", err)
	}

	var observationOut bytes.Buffer
	if code := runHarness(&observationOut, &errb, []string{"verify-run", "--lock", lockPath, "--print-observation-template"}); code != 0 {
		t.Fatalf("observation template code=%d stderr=%s", code, errb.String())
	}
	observation := filepath.Join(dir, "observation.json")
	if err := os.WriteFile(observation, bytes.ReplaceAll(observationOut.Bytes(), []byte("replace-with-runtime-run-id"), []byte("run-template-witness")), 0o600); err != nil {
		t.Fatal(err)
	}
	var verifyOut bytes.Buffer
	if code := runHarness(&verifyOut, &errb, []string{"verify-run", "--lock", lockPath, "--observation", observation}); code != 0 || !strings.Contains(verifyOut.String(), "HARNESS VERIFY RUN | VERIFIED") {
		t.Fatalf("verify code=%d stdout=%s stderr=%s", code, verifyOut.String(), errb.String())
	}
}
