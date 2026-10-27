package validate

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// fak-test:runtime fast est=10ms lane=default
func TestValidateBuildImportersBoundsLinkedOutputs(t *testing.T) {
	const metadata = "main fixture.test/tools/first ./tools/first\nleaf fixture.test/leaf ./leaf\nmain fixture.test/internal/second ./internal/second\nother fixture.test/other ./other\n"
	targets := []string{"./tools/first", "./leaf", "./internal/second", "./other"}
	build := func(pkgs ...string) []string {
		return append([]string{"build", "-trimpath", "-buildvcs=false", "-o", "/dev/null"}, pkgs...)
	}
	want := [][]string{
		{"env", "GOFLAGS"},
		append([]string{"list", "-trimpath", "-buildvcs=false", "-f", "{{.Name}} {{.ImportPath}}{{range .Match}} {{.}}{{end}}"}, targets...),
		build("fixture.test/leaf", "fixture.test/other"),
		build("fixture.test/tools/first"),
		build("fixture.test/internal/second"),
	}
	for _, scenario := range []string{"success", "link failure", "canceled", "malformed metadata", "empty metadata", "duplicate metadata", "list failure", "partial metadata", "library failure", "flags fallback", "env failure"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			deadline, _ := ctx.Deadline()
			var calls [][]string
			out, ok := validateBuildImporters(ctx, targets, "/dev/null", func(got context.Context, args ...string) (string, bool) {
				if d, _ := got.Deadline(); got != ctx || d != deadline {
					t.Fatal("build invocation replaced the overall context or deadline")
				}
				calls = append(calls, append([]string(nil), args...))
				if args[0] == "env" {
					if scenario == "env failure" {
						return "environment read failed", false
					}
					if scenario == "flags fallback" {
						return "-gcflags=-N", true
					}
					return "-p=1", true
				}
				if args[0] == "list" {
					switch scenario {
					case "partial metadata":
						return "main fixture.test/tools/first ./tools/first", true
					case "malformed metadata":
						return "unexpected metadata fields", true
					case "empty metadata":
						return "", true
					case "duplicate metadata":
						return metadata + "leaf fixture.test/leaf ./leaf\n", true
					case "list failure":
						return "package load failed", false
					}
					return metadata, true
				}
				if scenario == "library failure" && len(calls) == 3 {
					return "library compile failed", false
				}
				if len(calls) == 4 {
					if scenario == "link failure" {
						return "undefined: linkSymbol", false
					}
					if scenario == "canceled" {
						cancel()
					}
				}
				return "", true
			})
			count := len(want)
			switch scenario {
			case "success":
				if !ok || out != "" {
					t.Fatalf("build = %q, %v", out, ok)
				}
			case "link failure":
				count = 4
				if ok || !strings.Contains(out, "fixture.test/tools/first: undefined: linkSymbol") {
					t.Fatalf("missing failing link identity/detail: %q, %v", out, ok)
				}
			case "canceled":
				count = 4
				if ok || !strings.Contains(out, context.Canceled.Error()) {
					t.Fatalf("canceled build = %q, %v", out, ok)
				}
			case "library failure":
				count = 3
				if ok || !strings.Contains(out, "library compile failed") {
					t.Fatalf("library failure = %q, %v", out, ok)
				}
			case "flags fallback":
				legacy := [][]string{{"env", "GOFLAGS"}, validateGoCheckArgs("build", targets)}
				if !ok || !reflect.DeepEqual(calls, legacy) {
					t.Fatalf("legacy build = %q, %v, calls %v", out, ok, calls)
				}
				return
			case "env failure":
				count = 1
				if ok || !strings.Contains(out, "read importer build flags: environment read failed") {
					t.Fatalf("environment failure = %q, %v", out, ok)
				}
			default:
				count = 2
				if ok || !strings.Contains(out, "select importer build packages:") {
					t.Fatalf("metadata failure = %q, %v", out, ok)
				}
			}
			// Exact argv equality proves no selected importer is omitted or repeated,
			// libraries stay batched, each main links alone, and failure is fail-fast.
			if !reflect.DeepEqual(calls, want[:count]) {
				t.Fatalf("calls = %v, want %v", calls, want[:count])
			}
		})
	}
	t.Run("no unnecessary invocation", func(t *testing.T) {
		neverRun := func(context.Context, ...string) (string, bool) {
			t.Fatal("unexpected subprocess invocation")
			return "", false
		}
		if out, ok := validateBuildImporters(context.Background(), nil, "/dev/null", neverRun); !ok || out != "" {
			t.Fatalf("empty selection = %q, %v", out, ok)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if out, ok := validateBuildImporters(ctx, targets, "/dev/null", neverRun); ok || !strings.Contains(out, context.Canceled.Error()) {
			t.Fatalf("pre-canceled selection = %q, %v", out, ok)
		}
	})
	t.Run("runner paths", func(t *testing.T) {
		if validateBuildNullOutput("NUL", false) != "NUL" || validateBuildNullOutput("NUL", true) != "/dev/null" || validateBuildNullOutput("/dev/null", false) != "/dev/null" {
			t.Fatal("null output does not follow the runner OS")
		}
		ctx := context.Background()
		args := []string{"list", "-trimpath", "-buildvcs=false", "-f", "{{.Name}} {{.ImportPath}}{{range .Match}} {{.}}{{end}}", "./leaf"}
		native := validateBuildImporterListCommand(ctx, "source root", false, args)
		if native.Dir != "source root" || !reflect.DeepEqual(native.Args[1:], args) {
			t.Fatalf("native command = %v in %q", native.Args, native.Dir)
		}
		wsl := validateBuildImporterListCommand(ctx, "/tmp/source ' root", true, args)
		wantWSL := "set -euo pipefail; cd '/tmp/source '\"'\"' root'; exec go 'list' '-trimpath' '-buildvcs=false' '-f' '{{.Name}} {{.ImportPath}}{{range .Match}} {{.}}{{end}}' './leaf'"
		if !reflect.DeepEqual(wsl.Args[1:], []string{"-e", "bash", "-lc", wantWSL}) {
			t.Fatalf("WSL command = %v", wsl.Args)
		}
	})
}

// fak-test:runtime medium est=5s lane=default
func TestValidateBuildImportersRealGoParity(t *testing.T) {
	t.Setenv("GOWORK", "off")
	t.Setenv("GOFLAGS", "-p=1")
	t.Setenv("GOOS", runtime.GOOS)
	t.Setenv("GOARCH", runtime.GOARCH)
	root := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module fixture.test/importerbuild\n\ngo 1.21\n")
	write("leaf/leaf.go", "package leaf\nfunc Value() int { return 1 }\n")
	write("cmd/first/main.go", "package main\nimport \"fixture.test/importerbuild/leaf\"\nfunc main() { _ = leaf.Value() }\n")
	write("tools/second/main.go", "package main\nfunc main() {}\n")
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	targets := []string{"./leaf", "./cmd/first", "./tools/second"}
	if out, ok := validateRunGoCheckWithin(ctx, root, false, validateGoCheckArgs("build", targets)...); !ok {
		t.Fatalf("legacy mixed build: %s", out)
	}
	if out, ok := validateBuildImportersWithin(ctx, root, false, targets); !ok {
		t.Fatalf("split mixed build: %s", out)
	}
	if out, ok := validateBuildImportersWithin(ctx, root, false, []string{"./cmd/first"}); !ok {
		t.Fatalf("singleton command build: %s", out)
	}
	for _, name := range []string{"first", "first.exe", "second", "second.exe"} {
		if _, err := os.Stat(filepath.Join(root, name)); !os.IsNotExist(err) {
			t.Fatalf("unexpected installed executable %s: %v", name, err)
		}
	}
	// This compiles successfully but cannot link: a compile-only replacement
	// would incorrectly pass. Keep the bad main outside cmd/ as a naming guard.
	write("tools/second/main.go", "package main\nimport _ \"unsafe\"\n//go:linkname absent fixture.test/importerbuild/missingLinkSymbol\nfunc absent()\nfunc main() { absent() }\n")
	legacyOut, legacyOK := validateRunGoCheckWithin(ctx, root, false, validateGoCheckArgs("build", targets)...)
	splitOut, splitOK := validateBuildImportersWithin(ctx, root, false, targets)
	for name, result := range map[string]struct {
		out string
		ok  bool
	}{"legacy": {legacyOut, legacyOK}, "split": {splitOut, splitOK}} {
		if result.ok || !strings.Contains(result.out, "missingLinkSymbol") || !strings.Contains(result.out, "relocation target") {
			t.Fatalf("%s did not detect the missing link symbol: success=%v output=%s", name, result.ok, result.out)
		}
	}
}

// fak-test:runtime medium est=3s lane=default
func TestValidateBuildImportersMetadataStderr(t *testing.T) {
	// Build a tiny native fake Go executable, rather than a shell script, so the
	// same test checks the real stdout/stderr command runner on every host OS.
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture.test/fakego\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	const source = `package main
import("fmt"; "os")
func main() {
 fmt.Fprintln(os.Stderr, "go: successful diagnostic chatter")
 if len(os.Args) < 2 { os.Exit(2) }
 switch os.Args[1] {
 case "env": fmt.Println("-p=1")
 case "list": fmt.Println("main fixture.test/fakego/tool ./tool")
 case "build":
 default: os.Exit(2)
 }
}
`
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Fatal(err)
	}
	name := "go"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binDir := filepath.Join(root, "bin")
	if err := os.Mkdir(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	t.Setenv("GOWORK", "off")
	t.Setenv("GOFLAGS", "-p=1")
	t.Setenv("GOOS", runtime.GOOS)
	t.Setenv("GOARCH", runtime.GOARCH)
	cmd := windowgate.CommandContext(ctx, goTool, "build", "-trimpath", "-buildvcs=false", "-o", filepath.Join(binDir, name), ".")
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build metadata fixture: %v: %s", err, out)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	if out, ok := validateBuildImportersWithin(ctx, root, false, []string{"./tool"}); !ok || out != "" {
		t.Fatalf("successful stderr polluted metadata: %q, %v", out, ok)
	}
}
