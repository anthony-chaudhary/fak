package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessinit"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestHarnessMixLiveCLI(t *testing.T) {
	dir := t.TempDir()
	a := mixCLILock(t, "support", []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "support", Value: "resolve tickets", Source: "support"}})
	b := mixCLILock(t, "research", []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "citations", Value: "cite sources", Source: "research"}})
	ap := filepath.Join(dir, "a.lock.json")
	bp := filepath.Join(dir, "b.lock.json")
	outp := filepath.Join(dir, "mixed.lock.json")
	writeOverrideJSON(t, ap, a)
	writeOverrideJSON(t, bp, b)
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"mix", "--import", ap, "--import", bp, "--context-budget", "300", "--output", outp})
	if code != 0 || errb.Len() != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	for _, want := range []string{"HARNESS MIX | VERIFIED", "imports: 2", "components: 2", "next: fak harness inspect"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q:\n%s", want, out.String())
		}
	}
	var inspect, inspectErr bytes.Buffer
	if code = runHarness(&inspect, &inspectErr, []string{"inspect", "--lock", outp}); code != 0 {
		t.Fatalf("inspect code=%d err=%s", code, inspectErr.String())
	}
	if !strings.Contains(inspect.String(), "instruction:support") || !strings.Contains(inspect.String(), "instruction:citations") {
		t.Fatalf("inspect=%s", inspect.String())
	}
	if _, err := os.Stat(outp + ".mix.json"); err != nil {
		t.Fatal(err)
	}
}

func mixCLILock(t *testing.T, id string, assets []harnesscompose.EffectiveAsset) harnessresolve.Lock {
	t.Helper()
	l := harnessresolve.Lock{Schema: harnessresolve.LockSchema, Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "v1"}, Components: []harnessresolve.LockedComponent{{ID: id, Version: "1.0.0", Digest: "sha256:" + id, Source: "registry/" + id, Reason: "selected import", Provider: id, Provides: []string{id}, Compatibility: harnessresolve.Compatibility{OS: []string{"linux"}, Arch: []string{"amd64"}, Contract: "v1"}, Cost: harnessresolve.Budget{ContextTokens: 100}, Adapters: []string{"instruction"}}}, Assets: assets}
	if err := harnessresolve.ReidentifyLock(&l); err != nil {
		t.Fatal(err)
	}
	return l
}

func TestHarnessMixCleanRoomGeneratedProductLaunch(t *testing.T) {
	if testing.Short() {
		t.Skip("clean-room generated module runs go")
	}
	dir := t.TempDir()
	productDir := filepath.Join(dir, "product")
	result, err := harnessinit.Init(harnessinit.Options{Dir: productDir, Module: "example.com/mixed-support"})
	if err != nil {
		t.Fatal(err)
	}
	a := mixCLILock(t, "support", []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "support", Value: "resolve support tickets", Source: "support"}})
	b := mixCLILock(t, "research", []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "citations", Value: "cite support evidence", Source: "research"}})
	ap := filepath.Join(dir, "support.lock.json")
	bp := filepath.Join(dir, "research.lock.json")
	mixed := filepath.Join(dir, "mixed.lock.json")
	writeOverrideJSON(t, ap, a)
	writeOverrideJSON(t, bp, b)
	var out, errb bytes.Buffer
	if code := runHarness(&out, &errb, []string{"mix", "--import", ap, "--import", bp, "--output", mixed}); code != 0 {
		t.Fatalf("mix code=%d err=%s", code, errb.String())
	}
	cmd := exec.Command("go", "run", "./cmd/product", "--selfcheck", "--product-lock", mixed)
	cmd.Dir = result.Directory
	cmd.Env = append(os.Environ(), "GOWORK=off")
	raw, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("launch: %v\n%s", err, raw)
	}
	text := string(raw)
	for _, want := range []string{`"type":"harness.locked"`, `support@1.0.0#sha256:support`, `research@1.0.0#sha256:research`, `cite support evidence`, `resolve support tickets`, `"type":"turn.completed"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q:\n%s", want, text)
		}
	}
}
