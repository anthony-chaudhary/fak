package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessinit"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestHarnessDeriveImportPlusDeltaSpine(t *testing.T) {
	dir := t.TempDir()
	basePath := filepath.Join(dir, "support.lock.json")
	derivedPath := filepath.Join(dir, "my-support.lock.json")
	receiptPath := filepath.Join(dir, "my-support.derive.json")
	base := deriveFixture(t, false, false)
	writeOverrideJSON(t, basePath, base)
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"derive", "--from", basePath, "--set", "instruction:response-style=detailed", "--layer", "my-support", "--output", derivedPath, "--receipt", receiptPath})
	if code != 0 || errb.Len() != 0 {
		t.Fatalf("code=%d stderr=%q", code, errb.String())
	}
	for _, want := range []string{"HARNESS DERIVE | VERIFIED", "deltas: 1 | layer my-support", "next: fak harness preview", "inspect: fak harness inspect"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q:\n%s", want, out.String())
		}
	}
	var derived harnessresolve.Lock
	raw, err := os.ReadFile(derivedPath)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &derived); err != nil {
		t.Fatal(err)
	}
	if err = harnessresolve.VerifyLock(derived); err != nil {
		t.Fatal(err)
	}
	if derived.ID == base.ID || derived.Assets[0].Value != "detailed" || !strings.Contains(derived.Assets[0].Source, "from company:support") {
		t.Fatalf("derived=%+v", derived)
	}
	var previewOut, previewErr bytes.Buffer
	if code = runHarness(&previewOut, &previewErr, []string{"preview", "--current", basePath, "--candidate", derivedPath, "--view", "json"}); code != 3 || previewErr.Len() != 0 {
		t.Fatalf("preview code=%d err=%q out=%s", code, previewErr.String(), previewOut.String())
	}
	if !strings.Contains(previewOut.String(), `"capability": "instruction:response-style"`) || !strings.Contains(previewOut.String(), `"layer": "derive:my-support (from company:support)"`) {
		t.Fatalf("preview=%s", previewOut.String())
	}
	var inspectOut, inspectErr bytes.Buffer
	if code = runHarness(&inspectOut, &inspectErr, []string{"inspect", "--lock", derivedPath}); code != 0 || inspectErr.Len() != 0 {
		t.Fatalf("inspect code=%d err=%q", code, inspectErr.String())
	}
	if !strings.Contains(inspectOut.String(), "HARNESS INSPECT | VERIFIED") || !strings.Contains(inspectOut.String(), "derive:my-support (from company:support)") {
		t.Fatalf("inspect=%s", inspectOut.String())
	}
	if raw, err = os.ReadFile(receiptPath); err != nil || !bytes.Contains(raw, []byte(`"base_id": "`+base.ID+`"`)) || !bytes.Contains(raw, []byte(`"result_id": "`+derived.ID+`"`)) {
		t.Fatalf("receipt err=%v raw=%s", err, raw)
	}
}

func TestHarnessDeriveRefusesTamperLockedMandatoryAndUnsupported(t *testing.T) {
	cases := []struct {
		name string
		lock harnessresolve.Lock
		args []string
		want string
	}{
		{"tamper", deriveFixture(t, false, false), []string{"--set", "instruction:response-style=x"}, "digest mismatch"},
		{"locked", deriveFixture(t, true, false), []string{"--set", "instruction:response-style=x"}, "locked by"},
		{"mandatory", deriveFixture(t, false, true), []string{"--set", "instruction:response-style=x"}, "mandatory"},
		{"unsupported-adapter", deriveFixture(t, false, false), []string{"--set", "tool:search=x"}, "not active"},
	}
	cases[0].lock.ID = "sha256:tampered"
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			base := filepath.Join(dir, "base.json")
			out := filepath.Join(dir, "out.json")
			writeOverrideJSON(t, base, tc.lock)
			argv := append([]string{"derive", "--from", base, "--output", out}, tc.args...)
			var stdout, stderr bytes.Buffer
			code := runHarness(&stdout, &stderr, argv)
			if code != 1 || !strings.Contains(stderr.String(), tc.want) || stdout.Len() != 0 {
				t.Fatalf("code=%d out=%q err=%q want=%q", code, stdout.String(), stderr.String(), tc.want)
			}
			if _, err := os.Stat(out); !os.IsNotExist(err) {
				t.Fatalf("output exists err=%v", err)
			}
		})
	}
}

func deriveFixture(t *testing.T, locked, mandatory bool) harnessresolve.Lock {
	t.Helper()
	lock := harnessresolve.Lock{Schema: harnessresolve.LockSchema, Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "v1"}, Components: []harnessresolve.LockedComponent{{ID: "support", Version: "1.0.0", Digest: "sha256:support", Source: "registry/support", Reason: "root", Provider: "support"}}, Assets: []harnesscompose.EffectiveAsset{{Kind: "instruction", ID: "response-style", Value: "concise", Source: "company:support", Locked: locked, Mandatory: mandatory}}}
	if err := harnessresolve.ReidentifyLock(&lock); err != nil {
		t.Fatal(err)
	}
	return lock
}

func TestHarnessDeriveCleanRoomGeneratedProductLaunch(t *testing.T) {
	if testing.Short() {
		t.Skip("clean-room generated module runs go")
	}
	dir := t.TempDir()
	productDir := filepath.Join(dir, "product")
	result, err := harnessinit.Init(harnessinit.Options{Dir: productDir, Module: "example.com/derived-support"})
	if err != nil {
		t.Fatal(err)
	}
	basePath := filepath.Join(dir, "support.lock.json")
	derivedPath := filepath.Join(dir, "my-support.lock.json")
	base := deriveFixture(t, false, false)
	writeOverrideJSON(t, basePath, base)
	var deriveOut, deriveErr bytes.Buffer
	if code := runHarness(&deriveOut, &deriveErr, []string{"derive", "--from", basePath, "--set", "instruction:response-style=answer with cited support steps", "--output", derivedPath}); code != 0 {
		t.Fatalf("derive code=%d err=%s", code, deriveErr.String())
	}
	cmd := exec.Command("go", "run", "./cmd/product", "--selfcheck", "--product-lock", derivedPath)
	cmd.Dir = result.Directory
	cmd.Env = append(os.Environ(), "GOWORK=off")
	raw, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("launch: %v\n%s", err, raw)
	}
	text := string(raw)
	for _, want := range []string{`"type":"harness.locked"`, `\"lock_id\":\"`, `instruction/response-style@derive:local (from company:support)`, `locked | answer with cited support steps | offline reply`, `"type":"turn.completed"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q:\n%s", want, text)
		}
	}
}
