package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessserver"
)

func TestHarnessInitCLI(t *testing.T) {
	var out, errb bytes.Buffer
	root := filepath.Join(t.TempDir(), "product")
	code := runHarness(&out, &errb, []string{"init", "--dir", root, "--module", "example.test/product", "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got struct {
		ContractVersion string   `json:"contract_version"`
		Created         []string `json:"created"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ContractVersion != "v1alpha1" || len(got.Created) == 0 {
		t.Fatalf("result=%s", out.String())
	}
}

func TestHarnessInitCLISeedsVersionedHost(t *testing.T) {
	var out, errb bytes.Buffer
	root := filepath.Join(t.TempDir(), "product")
	code := runHarness(&out, &errb, []string{"init", "--dir", root, "--module", "example.test/product", "--host", "codex", "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got struct {
		Host    string   `json:"host"`
		Created []string `json:"created"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Host != "codex" {
		t.Fatalf("host=%q", got.Host)
	}
	found := map[string]bool{}
	for _, path := range got.Created {
		found[path] = true
	}
	for _, path := range []string{"product.json", "product.lock.json"} {
		if !found[path] {
			t.Fatalf("created=%v missing %s", got.Created, path)
		}
	}
}

func TestHarnessServerReceiptCrossesInitResolveVerifyImmutably(t *testing.T) {
	root := t.TempDir()
	serverDir := filepath.Join(root, "server-product")
	harnessDir := filepath.Join(root, "harness-product")
	if err := os.MkdirAll(serverDir, 0o755); err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(serverDir, "server-receipt.json")
	receiptRaw, err := os.ReadFile(filepath.Join("..", "..", "internal", "serverproduct", "testdata", "valid", "fixed-port.receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(receiptPath, receiptRaw, 0o444); err != nil {
		t.Fatal(err)
	}
	lifecycleLog := filepath.Join(serverDir, "lifecycle.jsonl")
	if err := os.WriteFile(lifecycleLog, []byte("{\"operation\":\"up\",\"owner\":\"server-product\"}\n"), 0o444); err != nil {
		t.Fatal(err)
	}
	receiptBefore := harnessServerFileDigest(t, receiptPath)
	lifecycleBefore := harnessServerFileDigest(t, lifecycleLog)

	var initOut, errb bytes.Buffer
	code := runHarness(&initOut, &errb, []string{
		"init", "--dir", harnessDir, "--module", "example.test/server-consumer",
		"--server-receipt", receiptPath,
		"--server-model", "local-code",
		"--server-protocol", "openai-http",
		"--server-protocol-revision", "2026-01",
		"--server-capabilities", "chat.completions,models.list",
		"--server-min-generation", "1",
		"--json",
	})
	if code != 0 {
		t.Fatalf("init code=%d stderr=%s", code, errb.String())
	}
	bindingPath := filepath.Join(harnessDir, harnessserver.BindingFileName)
	if !strings.Contains(initOut.String(), bindingPath) {
		t.Fatalf("init result does not name server binding: %s", initOut.String())
	}

	var manifestOut, selectionOut bytes.Buffer
	if code := runHarness(&manifestOut, &errb, []string{"resolve", "--example", "manifest"}); code != 0 {
		t.Fatalf("manifest example code=%d stderr=%s", code, errb.String())
	}
	if code := runHarness(&selectionOut, &errb, []string{"resolve", "--example", "selection"}); code != 0 {
		t.Fatalf("selection example code=%d stderr=%s", code, errb.String())
	}
	manifestPath := filepath.Join(harnessDir, "product.json")
	selectionPath := filepath.Join(harnessDir, "selection.json")
	lockPath := filepath.Join(harnessDir, "product.lock.json")
	if err := os.WriteFile(manifestPath, manifestOut.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(selectionPath, selectionOut.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	var resolveOut bytes.Buffer
	code = runHarness(&resolveOut, &errb, []string{
		"resolve", "--manifest", manifestPath, "--selection", selectionPath,
		"--os", "linux", "--arch", "amd64", "--contract", "v1",
		"--server-binding", bindingPath, "--output", lockPath,
	})
	if code != 0 || !strings.Contains(resolveOut.String(), `"schema": "fak.harness-server-resolution/v1"`) {
		t.Fatalf("resolve code=%d stdout=%s stderr=%s", code, resolveOut.String(), errb.String())
	}

	var observationOut bytes.Buffer
	if code := runHarness(&observationOut, &errb, []string{"verify-run", "--lock", lockPath, "--print-observation-template"}); code != 0 {
		t.Fatalf("observation template code=%d stderr=%s", code, errb.String())
	}
	observationPath := filepath.Join(harnessDir, "observation.json")
	observation := bytes.ReplaceAll(observationOut.Bytes(), []byte("replace-with-runtime-run-id"), []byte("run-server-receipt-witness"))
	if err := os.WriteFile(observationPath, observation, 0o600); err != nil {
		t.Fatal(err)
	}
	var verifyOut bytes.Buffer
	code = runHarness(&verifyOut, &errb, []string{
		"verify-run", "--lock", lockPath, "--observation", observationPath, "--server-binding", bindingPath,
	})
	if code != 0 || !strings.Contains(verifyOut.String(), "SERVER RECEIPT | VERIFIED") || !strings.Contains(verifyOut.String(), "HARNESS VERIFY RUN | VERIFIED") {
		t.Fatalf("verify code=%d stdout=%s stderr=%s", code, verifyOut.String(), errb.String())
	}
	if got := harnessServerFileDigest(t, receiptPath); got != receiptBefore {
		t.Fatalf("receipt mutated: before=%s after=%s", receiptBefore, got)
	}
	if got := harnessServerFileDigest(t, lifecycleLog); got != lifecycleBefore {
		t.Fatalf("lifecycle log mutated: before=%s after=%s", lifecycleBefore, got)
	}
}

func harnessServerFileDigest(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
