package architest

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const harnesskitModule = "github.com/anthony-chaudhary/fak"

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
