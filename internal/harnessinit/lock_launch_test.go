package harnessinit

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestGeneratedProductLaunchesDistinctVerifiedLocks(t *testing.T) {
	root := t.TempDir()
	if _, err := Init(Options{Dir: root, Module: "example.test/contextual"}); err != nil {
		t.Fatal(err)
	}
	legal := testLock(t, "legal", "citations", "cite-primary")
	coding := testLock(t, "coding", "shell", "repo-shell")
	legalPath := filepath.Join(root, "legal.lock.json")
	codingPath := filepath.Join(root, "coding.lock.json")
	writeJSON(t, legalPath, legal)
	writeJSON(t, codingPath, coding)
	run := func(lock string) string {
		t.Helper()
		cmd := exec.Command("go", "run", "./cmd/product", "--selfcheck", "--product-lock", lock)
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "GOTOOLCHAIN=auto")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("run %s: %v\n%s", lock, err, out)
		}
		return string(out)
	}
	legalOut := run(legalPath)
	codingOut := run(codingPath)
	for _, want := range []string{`"type":"harness.locked"`, legal.ID, `\"profile\":\"legal\"`, `legal | cite-primary | offline reply:`} {
		if !strings.Contains(legalOut, want) {
			t.Fatalf("legal missing %s:\n%s", want, legalOut)
		}
	}
	for _, want := range []string{coding.ID, `\"profile\":\"coding\"`, `coding | repo-shell | offline reply:`} {
		if !strings.Contains(codingOut, want) {
			t.Fatalf("coding missing %s:\n%s", want, codingOut)
		}
	}
	if legalOut == codingOut {
		t.Fatal("distinct locks produced identical launch")
	}
	if matches, _ := filepath.Glob(filepath.Join(root, ".*")); len(matches) != 0 {
		t.Fatalf("launch stranded host files: %v", matches)
	}
	if err := os.Remove(legalPath); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(codingPath); err != nil {
		t.Fatal(err)
	}
	stock := exec.Command("go", "run", "./cmd/product", "--selfcheck")
	stock.Dir = root
	out, err := stock.CombinedOutput()
	if err != nil {
		t.Fatalf("stock: %v\n%s", err, out)
	}
	if strings.Contains(string(out), "harness.locked") {
		t.Fatalf("stock retained lock state:\n%s", out)
	}
}

func testLock(t *testing.T, domain, id, value string) harnesskit.ProductLock {
	t.Helper()
	components := []harnesskit.LockedComponent{{ID: domain + "-pack", Version: "1.0.0", Digest: "sha256:" + domain, Source: "fixture/" + domain, Reason: "selected root", Provider: "manifest"}}
	assets := []harnesskit.LockedAsset{{Kind: "policy", ID: "floor", Source: "company", Locked: true}, {Kind: "instruction", ID: id, Value: value, Source: domain}}
	canonical := struct {
		Schema      string                       `json:"schema"`
		ID          string                       `json:"id"`
		Environment harnesskit.LockEnvironment   `json:"environment"`
		Budget      harnesskit.LockBudget        `json:"budget"`
		Components  []harnesskit.LockedComponent `json:"components"`
		Assets      []harnesskit.LockedAsset     `json:"assets"`
		AssetTrace  json.RawMessage              `json:"asset_trace"`
		Decisions   json.RawMessage              `json:"decisions"`
	}{Schema: harnesskit.ProductLockSchema, Components: components, Assets: assets}
	raw, err := json.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return harnesskit.ProductLock{Schema: harnesskit.ProductLockSchema, ID: "sha256:" + hex.EncodeToString(sum[:]), Components: components, Assets: assets}
}
func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}
