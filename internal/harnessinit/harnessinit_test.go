package harnessinit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInitCreatesIdempotentPublicProductAndPreservesUserFiles(t *testing.T) {
	root := t.TempDir()
	result, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Created) != 7 {
		t.Fatalf("created=%v", result.Created)
	}
	mainBody, err := os.ReadFile(filepath.Join(root, "generated", "runtime.go"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(mainBody), "github.com/anthony-chaudhary/fak/internal/") {
		t.Fatal("generated product imports internal package")
	}
	userPath := filepath.Join(root, "product", "config.go")
	if err := os.WriteFile(userPath, []byte("package product\n// user edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	second, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(userPath)
	if !strings.Contains(string(got), "user edit") {
		t.Fatalf("user file overwritten: %s", got)
	}
	if len(second.Created) != 0 || len(second.Updated) != 0 {
		t.Fatalf("rerun changed generated files: %+v", second)
	}
}

func TestInitRefusesUnownedGeneratedPath(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "generated", "runtime.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package generated\n// mine\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err == nil || !strings.Contains(err.Error(), "refusing to overwrite user-owned") {
		t.Fatalf("err=%v", err)
	}
	body, _ := os.ReadFile(path)
	if !strings.Contains(string(body), "mine") {
		t.Fatal("foreign file changed")
	}
}

func TestInitPublishesProvenanceAndOwnershipMetadata(t *testing.T) {
	root := t.TempDir()
	if _, err := Init(Options{Dir: root, Module: "example.test/acme"}); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"go.mod", "go.sum", "cmd/product/main.go", "generated/runtime.go", "harness.lock.json"} {
		body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		if path != "go.sum" && !strings.Contains(string(body), generatorID) {
			t.Fatalf("%s lacks generator provenance", path)
		}
	}
	lock, err := os.ReadFile(filepath.Join(root, "harness.lock.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{ContractVersion, DefaultFAKVersion, `"README.md": "user"`, `"go.sum": "generated"`, `"build":`, `"run":`, `"upgrade":`} {
		if !strings.Contains(string(lock), want) {
			t.Fatalf("manifest lacks %q: %s", want, lock)
		}
	}
}

func TestInitRefusesUnrecognizedGoSum(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "go.sum")
	if err := os.WriteFile(path, []byte("example.test/user v1.0.0 h1:user-owned\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err == nil || !strings.Contains(err.Error(), "refusing to overwrite") {
		t.Fatalf("err=%v", err)
	}
	body, _ := os.ReadFile(path)
	if string(body) != "example.test/user v1.0.0 h1:user-owned\n" {
		t.Fatal("foreign go.sum changed")
	}
}

func TestInitUpgradesOwnedGeneratedGoSum(t *testing.T) {
	root := t.TempDir()
	if _, err := Init(Options{Dir: root, Module: "example.test/acme"}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "go.sum")
	if err := os.WriteFile(path, []byte("old generated sums\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Updated) != 1 || result.Updated[0] != "go.sum" {
		t.Fatalf("updated=%v", result.Updated)
	}
	body, _ := os.ReadFile(path)
	if !strings.Contains(string(body), DefaultFAKVersion) {
		t.Fatalf("go.sum not restored: %s", body)
	}
}
