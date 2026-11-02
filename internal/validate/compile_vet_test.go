package validate

import (
	"bytes"
	"encoding/json"
	"go/format"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestValidateCompileVetCoverage(t *testing.T) {
	coverage := &validateVetCoverage{dir: "/snapshot", targets: []string{"./app", "./consumer"}}
	for _, tc := range []struct {
		name, selector, dir, goos string
		wslTests, workspace       bool
		coverage                  *validateVetCoverage
		targets                   []string
		want                      bool
	}{
		{"native", "^$", "/snapshot", "linux", true, false, coverage, []string{"./app"}, true},
		{"native-windows", "^$", "/snapshot", "windows", false, false, coverage, []string{"./app"}, true},
		{"missing-proof", "^$", "/snapshot", "linux", false, false, nil, []string{"./app"}, false},
		{"behavior", "TestApp", "/snapshot", "linux", false, false, coverage, []string{"./app"}, false},
		{"equivalent-regex", "^(?:)$", "/snapshot", "linux", false, false, coverage, []string{"./app"}, false},
		{"different-workspace", "^$", "/other", "linux", false, false, coverage, []string{"./app"}, false},
		{"unknown-package", "^$", "/snapshot", "linux", false, false, coverage, []string{"./other"}, false},
		{"partial-coverage", "^$", "/snapshot", "linux", false, false, coverage, []string{"./app", "./other"}, false},
		{"rematerialized-wsl", "^$", "/snapshot", "windows", true, false, coverage, []string{"./app"}, false},
		{"runner-mismatch", "^$", "/snapshot", "windows", true, true, coverage, []string{"./app"}, false},
		{"same-guest", "^$", "/snapshot", "windows", true, true, &validateVetCoverage{dir: "/snapshot", wsl: true, targets: []string{"./app"}}, []string{"./app"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.coverage.coversCompileTests(tc.selector, tc.dir, tc.goos, tc.wslTests, tc.workspace, tc.targets); got != tc.want {
				t.Fatalf("coverage=%v, want %v", got, tc.want)
			}
		})
	}
}

// fak-test:runtime medium est=30s lane=default
func TestValidateCompileVetRetainsTestDiagnostics(t *testing.T) {
	if testing.Short() {
		t.Skip("real Go vet integration")
	}
	for _, pkg := range []string{"app", "app_test"} {
		for _, testOnly := range []bool{false, true} {
			t.Run(pkg+map[bool]string{false: "/full", true: "/test-only"}[testOnly], func(t *testing.T) {
				root := compileVetFixture(t, "package "+pkg+"\nimport(\"fmt\";\"testing\")\nfunc TestBad(t *testing.T){fmt.Printf(\"%d\", \"bad\")}\n")
				args := []string{"--root", root, "--mine", "app/app_test.go", "--test-run", "^$", "--wsl-tests=false", "--progress=false", "--json"}
				if testOnly {
					args = append(args, "--test-only")
				}
				var out, stderr bytes.Buffer
				code := Run(&out, &stderr, args)
				var result validateResult
				if err := json.Unmarshal(out.Bytes(), &result); err != nil {
					t.Fatalf("receipt: %v; %s; %s", err, out.String(), stderr.String())
				}
				if code == 0 || result.OK {
					t.Fatalf("bad test vet passed: code=%d receipt=%+v", code, result)
				}
				phase := "vet"
				if testOnly {
					phase = "test"
				}
				found := false
				for _, p := range result.Phases {
					found = found || p.Name == phase && p.Status == "failed"
				}
				if !found {
					t.Fatalf("missing failed %s phase: %+v", phase, result.Phases)
				}
			})
		}
	}
}

// fak-test:runtime medium est=15s lane=default
func TestValidateCompileVetPreservesInitializationPaths(t *testing.T) {
	if testing.Short() {
		t.Skip("real Go initialization integration")
	}
	root := compileVetFixture(t, `package app
import("os";"runtime";"testing")
var initialized bool
func init(){_,p,_,ok:=runtime.Caller(0);if !ok{os.Exit(21)};if _,e:=os.Stat(p);e!=nil{os.Exit(22)};initialized=true}
func TestMain(m *testing.M){if !initialized{os.Exit(23)};_,p,_,_:=runtime.Caller(0);if _,e:=os.Stat(p);e!=nil{os.Exit(24)};os.Exit(m.Run())}
func TestUnselected(t *testing.T){t.Fatal("must not run")}
`)
	var out, stderr bytes.Buffer
	code := Run(&out, &stderr, []string{"--root", root, "--mine", "app/app_test.go", "--test-run", "^$", "--wsl-tests=false", "--progress=false", "--json"})
	var result validateResult
	if err := json.Unmarshal(out.Bytes(), &result); err != nil || code != 0 || !result.OK {
		t.Fatalf("initialization paths: code=%d err=%v receipt=%s stderr=%s", code, err, out.String(), stderr.String())
	}
	for _, phase := range result.Phases {
		if phase.Name == "test" && phase.Status == "ok" && phase.Detail == "default test vet reused from successful explicit vet in the same workspace and runner" {
			return
		}
	}
	t.Fatalf("successful compile did not witness vet reuse: %+v", result.Phases)
}

func compileVetFixture(t *testing.T, tests string) string {
	t.Helper()
	root := t.TempDir()
	for name, body := range map[string]string{"go.mod": "module fixture.test/compilevet\n\ngo 1.26\n", "app/app.go": "package app\nfunc Value() int { return 1 }\n", "app/app_test.go": tests} {
		if filepath.Ext(name) == ".go" {
			formatted, err := format.Source([]byte(body))
			if err != nil {
				t.Fatal(err)
			}
			body = string(formatted)
		}
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, args := range [][]string{{"init", "-q"}, {"add", "go.mod", "app/app.go", "app/app_test.go"}, {"-c", "user.name=Validator Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"}} {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("fixture git: %v: %s", err, out)
		}
	}
	return root
}
