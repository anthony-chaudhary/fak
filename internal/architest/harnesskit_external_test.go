package architest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const harnesskitModule = "github.com/anthony-chaudhary/fak"

func TestHarnessKitLockV2Export(t *testing.T) {
	root := repositoryRoot(t)
	consumer := `package main
import (
    "encoding/json"
    lockv2 "github.com/anthony-chaudhary/fak/pkg/harnesskit/lockv2"
)
func main() {
    lock := &lockv2.Lock{
        Schema: lockv2.ProductLockSchemaV2,
        Platforms: []lockv2.PlatformRequirement{{OS: "linux", Arch: "amd64"}},
        Budget: lockv2.LockBudget{ContextTokens: 512, Workers: 1},
        Components: []lockv2.LockedComponent{{ID: "runtime", Version: "1.0.0", Digest: "sha256:fixture", Source: "registry/runtime"}},
        Assets: []lockv2.LockedAsset{{Kind: "secret", ID: "token", Ref: "env:FIXTURE_TOKEN", Source: "env"}},
    }
    id, err := lockv2.CanonicalID(lock)
    if err != nil { panic(err) }
    lock.ID = id
    raw, err := json.Marshal(lock)
    if err != nil { panic(err) }
    parsed, err := lockv2.Parse(raw)
    if err != nil { panic(err) }
    if err := lockv2.ValidateSecretContracts(parsed); err != nil { panic(err) }
}`
	dir := writeExternalModule(t, root, consumer)
	runGo(t, dir, true, "run", ".")

	output := runGo(t, dir, true, "list", "-json", harnesskitModule+"/pkg/harnesskit/lockv2")
	var info struct {
		Imports []string
		Deps    []string
	}
	if err := json.Unmarshal([]byte(output), &info); err != nil {
		t.Fatalf("decode go list output: %v", err)
	}
	for _, imported := range append(info.Imports, info.Deps...) {
		if strings.HasPrefix(imported, harnesskitModule+"/internal/") {
			t.Fatalf("pkg/harnesskit/lockv2 imports internal package %q", imported)
		}
	}
}

func TestHarnesskitExternalImportBoundary(t *testing.T) {
	if testing.Short() {
		t.Skip("external module compile witness")
	}
	root := repositoryRoot(t)

	positive := `package main
import (
    "fmt"
    kit "github.com/anthony-chaudhary/fak/pkg/harnesskit"
)
func main() {
    product, err := kit.New("example/support", "v1").WithProfile(kit.Profile{
        ID: "default",
        Capabilities: []kit.Capability{"kb.read"},
        Extensions: []kit.Extension{{
            ID: "kb", Plane: kit.PlaneTools, Compatibility: kit.ContractVersion,
            Provenance: kit.Provenance{Source: "example.org/kb", Version: "v1.0.0", Digest: "sha256:fixture"},
        }},
    }).WithTransport(kit.Transport{ID: "stdio", Provenance: kit.Provenance{Source: "example.org/stdio", Version: "v1.0.0"}}).Build()
    if err != nil { panic(err) }
    fmt.Println(product.Spec().ID)
}`
	dir := writeExternalModule(t, root, positive)
	runGo(t, dir, true, "run", ".")

	negative := `package main
import _ "github.com/anthony-chaudhary/fak/internal/abi"
func main() {}`
	os.WriteFile(filepath.Join(dir, "main.go"), []byte(negative), 0o600)
	output := runGo(t, dir, false, "build", ".")
	if !strings.Contains(output, "use of internal package") && !strings.Contains(output, "not allowed") {
		t.Fatalf("negative fixture failed for wrong reason:\n%s", output)
	}
}

func repositoryRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root, err := filepath.Abs(filepath.Join(filepath.Dir(file), "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func writeExternalModule(t *testing.T, root, main string) string {
	t.Helper()
	dir := t.TempDir()
	gomod := "module example.com/clean-harness-product\n\ngo 1.26\n\nrequire " + harnesskitModule + " v0.0.0\n\nreplace " + harnesskitModule + " => " + filepath.ToSlash(root) + "\n"
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(gomod), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(main), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func runGo(t *testing.T, dir string, wantSuccess bool, args ...string) string {
	t.Helper()
	cmd := exec.Command("go", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	out, err := cmd.CombinedOutput()
	if wantSuccess && err != nil {
		t.Fatalf("go %s failed: %v\n%s", strings.Join(args, " "), err, out)
	}
	if !wantSuccess && err == nil {
		t.Fatalf("go %s unexpectedly succeeded\n%s", strings.Join(args, " "), out)
	}
	return string(out)
}
