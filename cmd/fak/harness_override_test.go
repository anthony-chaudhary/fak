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

func TestHarnessOverrideLiveResolvePreviewSpine(t *testing.T) {
	dir := t.TempDir()
	product := filepath.Join(dir, "product.json")
	selection := filepath.Join(dir, "selection.json")
	current := filepath.Join(dir, "current.json")
	override := filepath.Join(dir, "operator-override.json")
	candidateProduct := filepath.Join(dir, "candidate-product.json")
	candidateSelection := filepath.Join(dir, "candidate-selection.json")
	candidate := filepath.Join(dir, "candidate.json")
	manifest := `{"schema":"fak.harness-product/v1alpha1","roots":["kernel"],"compatibility":{"os":["linux"],"arch":["amd64"],"contract":"v1"},"components":[{"id":"kernel","version":"1.0.0","digest":"sha256:k","source":"registry:kernel","provides":["runtime"],"evidence":{"authority":"fixture","source":"kernel"}}],"assets":{"schema":"fak.harness-assets/v1alpha1","layers":[{"id":"defaults","scope":"company","assets":[{"kind":"instruction","id":"response-style","value":"concise"}]}]}}`
	writeOverrideFile(t, product, manifest)
	writeOverrideFile(t, selection, `{"layers":["defaults"]}`)

	currentRaw := runHarnessJSON(t, []string{"resolve", "--manifest", product, "--selection", selection, "--os", "linux", "--arch", "amd64", "--contract", "v1"})
	if err := os.WriteFile(current, currentRaw, 0o600); err != nil {
		t.Fatal(err)
	}

	var overrideOut, overrideErr bytes.Buffer
	code := runHarness(&overrideOut, &overrideErr, []string{"override", "--lock", current, "--capability", "instruction:response-style", "--value", "detailed", "--output", override})
	if code != 0 || overrideErr.Len() != 0 {
		t.Fatalf("override code=%d stderr=%q", code, overrideErr.String())
	}
	for _, want := range []string{"HARNESS OVERRIDE | PROPOSAL", "instruction:response-style | from defaults | changeable by re-resolve", "change: replace | value detailed", "written: " + override} {
		if !strings.Contains(overrideOut.String(), want) {
			t.Fatalf("missing %q in captured override:\n%s", want, overrideOut.String())
		}
	}

	var productDoc map[string]any
	if err := json.Unmarshal([]byte(manifest), &productDoc); err != nil {
		t.Fatal(err)
	}
	var generated struct {
		Layers []any `json:"layers"`
	}
	raw, err := os.ReadFile(override)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &generated); err != nil {
		t.Fatal(err)
	}
	assets := productDoc["assets"].(map[string]any)
	assets["layers"] = append(assets["layers"].([]any), generated.Layers...)
	writeOverrideJSON(t, candidateProduct, productDoc)
	writeOverrideFile(t, candidateSelection, `{"layers":["defaults","operator-override"]}`)
	candidateRaw := runHarnessJSON(t, []string{"resolve", "--manifest", candidateProduct, "--selection", candidateSelection, "--os", "linux", "--arch", "amd64", "--contract", "v1"})
	if err := os.WriteFile(candidate, candidateRaw, 0o600); err != nil {
		t.Fatal(err)
	}

	var previewOut, previewErr bytes.Buffer
	code = runHarness(&previewOut, &previewErr, []string{"preview", "--current", current, "--candidate", candidate, "--view", "json"})
	if code != 3 || previewErr.Len() != 0 {
		t.Fatalf("preview code=%d stderr=%q output=%s", code, previewErr.String(), previewOut.String())
	}
	for _, want := range []string{`"reason": "behavior-change"`, `"layer": "operator-override"`, `"capability": "instruction:response-style"`} {
		if !strings.Contains(previewOut.String(), want) {
			t.Fatalf("missing %s in preview:\n%s", want, previewOut.String())
		}
	}
}

func TestHarnessOverrideRefusesLockedCapability(t *testing.T) {
	path := writePreviewLock(t, fixtureInspectLock(true))
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"override", "--lock", path, "--capability", "instruction:response-style", "--value", "detailed"})
	if code != 1 || !strings.Contains(errb.String(), `locked by company:defaults and cannot be overridden`) || out.Len() != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}

func runHarnessJSON(t *testing.T, argv []string) []byte {
	t.Helper()
	var out, errb bytes.Buffer
	if code := runHarness(&out, &errb, argv); code != 0 {
		t.Fatalf("%v code=%d stderr=%q", argv, code, errb.String())
	}
	return append([]byte(nil), out.Bytes()...)
}

func writeOverrideFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeOverrideJSON(t *testing.T, path string, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

func fixtureInspectLock(locked bool) harnessresolve.Lock {
	return harnessresolve.Lock{Schema: harnessresolve.LockSchema, Assets: []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "response-style", Value: "concise", Source: "company:defaults", Locked: locked}}}
}
