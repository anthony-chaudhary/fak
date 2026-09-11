// Behavior contract harness for the `make metal-check` cgo/device gate
// (fak issue #12784; independent test-author lane W2).
//
// Embeds oracleMain: a standalone reference implementation of the gate that
// dispatches purely on METALPROBE_* env vars, standing in for the
// not-yet-integrated real cmd/metalprobe. The harness proves the contract
// is executable, deterministic, and fully specified; the real implementation
// must satisfy the same observable behavior and is conformance-checked
// against this harness at integration time. Stdlib only; test code only.
package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// oracleMain is the contract oracle: a complete standalone metal-check
// implementation whose observable behavior (stdout, stderr, exit code) is
// the contract under test. It reads only METALPROBE_* env vars.
const oracleMain = `package main

import (
	"fmt"
	"os"
)

func main() {
	arch := os.Getenv("METALPROBE_ARCH")
	goarch := os.Getenv("METALPROBE_GOARCH")
	if arch != "darwin" || goarch != "arm64" {
		fmt.Printf("metal-check: not applicable on %s/%s (Apple Silicon only)\n", arch, goarch)
		os.Exit(0)
	}
	if os.Getenv("METALPROBE_CGO") == "0" {
		fmt.Fprintf(os.Stderr, "metal-check: FAIL cgo disabled (CGO_ENABLED=0); prerequisite: unset CGO_ENABLED\n")
		os.Exit(1)
	}
	if os.Getenv("METALPROBE_XCODE_CLT") != "1" {
		fmt.Fprintf(os.Stderr, "metal-check: FAIL Xcode Command Line Tools missing; prerequisite: xcode-select --install\n")
		os.Exit(1)
	}
	if os.Getenv("METALPROBE_CLANG") != "1" {
		fmt.Fprintf(os.Stderr, "metal-check: FAIL clang compiler missing; prerequisite: install Xcode Command Line Tools\n")
		os.Exit(1)
	}
	if os.Getenv("METALPROBE_GO_TOOL") != "1" {
		fmt.Fprintf(os.Stderr, "metal-check: FAIL Go toolchain missing; prerequisite: install go\n")
		os.Exit(1)
	}
	if os.Getenv("METALPROBE_DEVICE") == "1" {
		fmt.Printf("metal: compiled=true available=true device=Apple M3 Pro\n")
		fmt.Printf("metal-check: PASS (compiled=yes available=yes device=Apple M3 Pro)\n")
		os.Exit(0)
	}
	fmt.Printf("metal: compiled=true available=false device=\n")
	fmt.Printf("metal-check: PASS (compiled=yes available=no device= -> CPU-fallback build; serving will not use Metal device)\n")
	os.Exit(0)
}
`

// metalCheckSpec is one row of the behavioral matrix: full env context plus
// the exact observable outcomes the contract requires.
type metalCheckSpec struct {
	name            string
	env             map[string]string
	wantExit        int
	stdoutContains  []string
	stderrContains  []string
	stdoutForbidden []string
}

// metalEnv returns a complete, explicit env map for one oracle run. Every
// spec sets all seven keys so cases are independent of each other and of the
// host environment.
func metalEnv(arch, goarch, cgo, clt, clang, gotool, device string) map[string]string {
	return map[string]string{
		"METALPROBE_ARCH":      arch,
		"METALPROBE_GOARCH":    goarch,
		"METALPROBE_CGO":       cgo,
		"METALPROBE_XCODE_CLT": clt,
		"METALPROBE_CLANG":     clang,
		"METALPROBE_GO_TOOL":   gotool,
		"METALPROBE_DEVICE":    device,
	}
}

// metalCheckSpecs is the required 9-case matrix (plus nothing else).
func metalCheckSpecs() []metalCheckSpec {
	appleSilicon := metalEnv("darwin", "arm64", "1", "1", "1", "1", "1")
	notApplicable := metalEnv("windows", "amd64", "1", "1", "1", "1", "1")
	return []metalCheckSpec{
		{
			name:           "windows-amd64-not-applicable",
			env:            notApplicable,
			wantExit:       0,
			stdoutContains: []string{"not applicable on windows/amd64"},
		},
		{
			name:           "linux-arm64-not-applicable",
			env:            metalEnv("linux", "arm64", "1", "1", "1", "1", "1"),
			wantExit:       0,
			stdoutContains: []string{"not applicable on linux/arm64"},
		},
		{
			name:            "darwin-arm64-cgo-disabled",
			env:             withEnv(appleSilicon, map[string]string{"METALPROBE_CGO": "0"}),
			wantExit:        1,
			stderrContains:  []string{"cgo"},
			stdoutForbidden: []string{"PASS"},
		},
		{
			name:            "darwin-arm64-no-xcode-clt",
			env:             withEnv(appleSilicon, map[string]string{"METALPROBE_XCODE_CLT": "0"}),
			wantExit:        1,
			stderrContains:  []string{"Xcode Command Line Tools"},
			stdoutForbidden: []string{"PASS"},
		},
		{
			name:            "darwin-arm64-no-clang",
			env:             withEnv(appleSilicon, map[string]string{"METALPROBE_CLANG": "0"}),
			wantExit:        1,
			stderrContains:  []string{"clang"},
			stdoutForbidden: []string{"PASS"},
		},
		{
			name:            "darwin-arm64-no-go-toolchain",
			env:             withEnv(appleSilicon, map[string]string{"METALPROBE_GO_TOOL": "0"}),
			wantExit:        1,
			stderrContains:  []string{"Go toolchain"},
			stdoutForbidden: []string{"PASS"},
		},
		{
			name:     "darwin-arm64-device-available",
			env:      appleSilicon,
			wantExit: 0,
			stdoutContains: []string{
				"metal: compiled=",
				"compiled=true available=true device=Apple M3 Pro",
				"metal-check: PASS (compiled=yes available=yes device=Apple M3 Pro)",
			},
		},
		{
			name:     "darwin-arm64-cpu-fallback",
			env:      withEnv(appleSilicon, map[string]string{"METALPROBE_DEVICE": "0"}),
			wantExit: 0,
			stdoutContains: []string{
				"metal: compiled=",
				"compiled=true available=false device=",
				"CPU-fallback",
				"metal-check: PASS (compiled=yes available=no",
			},
		},
		{
			name:           "darwin-x86_64-not-applicable",
			env:            metalEnv("darwin", "x86_64", "1", "1", "1", "1", "1"),
			wantExit:       0,
			stdoutContains: []string{"not applicable on darwin/x86_64"},
		},
	}
}

// withEnv returns a copy of base with the given overrides applied, so specs
// never share mutable maps.
func withEnv(base, overrides map[string]string) map[string]string {
	env := make(map[string]string, len(base)+len(overrides))
	for k, v := range base {
		env[k] = v
	}
	for k, v := range overrides {
		env[k] = v
	}
	return env
}

// deriveExpectedFailure re-derives the contract's prerequisite precedence
// from an env map, mirroring the oracle's check order.
func deriveExpectedFailure(env map[string]string) (int, string) {
	if env["METALPROBE_CGO"] == "0" {
		return 1, "cgo"
	}
	if env["METALPROBE_XCODE_CLT"] != "1" {
		return 1, "Xcode Command Line Tools"
	}
	if env["METALPROBE_CLANG"] != "1" {
		return 1, "clang"
	}
	if env["METALPROBE_GO_TOOL"] != "1" {
		return 1, "Go toolchain"
	}
	return 0, ""
}

func isNotApplicable(env map[string]string) bool {
	return env["METALPROBE_ARCH"] != "darwin" || env["METALPROBE_GOARCH"] != "arm64"
}

// buildOracle writes oracleMain to a temp dir and compiles it as a single
// self-contained file (no module needed). Skips cleanly if go is unavailable.
func buildOracle(t *testing.T) string {
	t.Helper()
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH; contract oracle cannot be built")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "main.go")
	if err := os.WriteFile(src, []byte(oracleMain), 0o644); err != nil {
		t.Fatalf("write oracle main.go: %v", err)
	}
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	bin := filepath.Join(dir, "oracle"+suffix)
	build := exec.Command(goTool, "build", "-o", bin, src)
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	var diag bytes.Buffer
	build.Stdout = &diag
	build.Stderr = &diag
	if err := build.Run(); err != nil {
		t.Fatalf("go build oracle: %v\n%s", err, diag.String())
	}
	return bin
}

// runOracle executes the oracle binary with a scrubbed env (host
// METALPROBE_* leakage removed) plus the spec's explicit values.
func runOracle(t *testing.T, bin string, env map[string]string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(bin)
	full := make([]string, 0, len(os.Environ())+len(env))
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "METALPROBE_") {
			continue
		}
		full = append(full, kv)
	}
	for k, v := range env {
		full = append(full, k+"="+v)
	}
	cmd.Env = full
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()
	code := 0
	if runErr != nil {
		exitErr, ok := runErr.(*exec.ExitError)
		if !ok {
			t.Fatalf("run oracle %s: %v", bin, runErr)
		}
		code = exitErr.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

// assertSpec checks one oracle run against its spec row, including the
// mandated per-shape assertions (exit-1 names the prerequisite and leaks no
// PASS; N/A prints "not applicable"; darwin pass rows carry verdict grammar).
func assertSpec(t *testing.T, spec metalCheckSpec, code int, stdout, stderr string) {
	t.Helper()
	if code != spec.wantExit {
		t.Fatalf("exit code = %d, want %d (stdout=%q stderr=%q)", code, spec.wantExit, stdout, stderr)
	}
	for _, want := range spec.stderrContains {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr = %q, want substring %q", stderr, want)
		}
	}
	for _, want := range spec.stdoutContains {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout = %q, want substring %q", stdout, want)
		}
	}
	for _, forbid := range spec.stdoutForbidden {
		if strings.Contains(stdout, forbid) {
			t.Errorf("stdout = %q, forbidden substring %q", stdout, forbid)
		}
	}
	if spec.wantExit != 0 && strings.Contains(stdout, "PASS") {
		t.Errorf("exit-1 case leaked PASS on stdout: %q", stdout)
	}
	if isNotApplicable(spec.env) {
		if !strings.Contains(strings.ToLower(stdout), "not applicable") {
			t.Errorf("not-applicable case stdout = %q, want \"not applicable\" (case-insensitive)", stdout)
		}
		if strings.Contains(stdout, "metal: compiled=") {
			t.Errorf("not-applicable case must not emit device verdict grammar: %q", stdout)
		}
	}
}

func findSpec(t *testing.T, name string) metalCheckSpec {
	t.Helper()
	for _, spec := range metalCheckSpecs() {
		if spec.name == name {
			return spec
		}
	}
	t.Fatalf("spec %q not found", name)
	return metalCheckSpec{}
}

// TestMetalCheckBehavioralMatrix drives the full 9-case contract matrix
// against the compiled oracle binary.
func TestMetalCheckBehavioralMatrix(t *testing.T) {
	bin := buildOracle(t)
	for _, spec := range metalCheckSpecs() {
		t.Run(spec.name, func(t *testing.T) {
			code, stdout, stderr := runOracle(t, bin, spec.env)
			assertSpec(t, spec, code, stdout, stderr)
		})
	}
}

// TestContractOracleDeterminism proves the device-available and CPU-fallback
// verdicts are byte-identical across repeated runs under identical env.
func TestContractOracleDeterminism(t *testing.T) {
	bin := buildOracle(t)
	for _, name := range []string{"darwin-arm64-device-available", "darwin-arm64-cpu-fallback"} {
		spec := findSpec(t, name)
		code1, out1, err1 := runOracle(t, bin, spec.env)
		code2, out2, err2 := runOracle(t, bin, spec.env)
		if code1 != code2 {
			t.Errorf("%s: exit code differs across runs: %d vs %d", name, code1, code2)
		}
		if bytes.Compare([]byte(out1), []byte(out2)) != 0 {
			t.Errorf("%s: stdout differs across runs:\n%q\nvs\n%q", name, out1, out2)
		}
		if bytes.Compare([]byte(err1), []byte(err2)) != 0 {
			t.Errorf("%s: stderr differs across runs:\n%q\nvs\n%q", name, err1, err2)
		}
		assertSpec(t, spec, code1, out1, err1)
	}
}

// TestSpecSelfConsistency validates every spec row against the contract's
// own invariants without running the oracle: N/A rows are a portable
// Makefile no-op, darwin pass rows never expect "not applicable", and fail
// rows must name the first failing prerequisite per the contract's
// precedence order.
func TestSpecSelfConsistency(t *testing.T) {
	envKeys := [...]string{
		"METALPROBE_ARCH", "METALPROBE_GOARCH", "METALPROBE_CGO",
		"METALPROBE_XCODE_CLT", "METALPROBE_CLANG", "METALPROBE_GO_TOOL",
		"METALPROBE_DEVICE",
	}
	for _, spec := range metalCheckSpecs() {
		t.Run(spec.name, func(t *testing.T) {
			env := spec.env
			for _, key := range envKeys {
				if _, ok := env[key]; !ok {
					t.Errorf("env incomplete: missing %s", key)
				}
			}
			joinedOut := strings.Join(spec.stdoutContains, "; ")
			joinedErr := strings.Join(spec.stderrContains, "; ")
			if isNotApplicable(env) {
				if spec.wantExit != 0 {
					t.Errorf("N/A spec must expect exit 0, got %d", spec.wantExit)
				}
				if joinedErr != "" {
					t.Errorf("N/A spec must not expect stderr output: %q", joinedErr)
				}
				if !strings.Contains(strings.ToLower(joinedOut), "not applicable") {
					t.Errorf("N/A spec must expect \"not applicable\" on stdout, got %q", joinedOut)
				}
				for _, flag := range []string{"cgo", "Xcode", "clang", "Go toolchain", "PASS", "metal: compiled="} {
					if strings.Contains(joinedOut, flag) {
						t.Errorf("N/A spec expectation %q must not depend on tool flags or verdict grammar (tool flags are not meaningful off darwin/arm64)", flag)
					}
				}
				return
			}
			if strings.Contains(joinedOut, "not applicable") {
				t.Errorf("darwin/arm64 spec must never expect \"not applicable\": %q", joinedOut)
			}
			wantExit, marker := deriveExpectedFailure(env)
			if spec.wantExit != wantExit {
				t.Errorf("wantExit %d disagrees with env-derived failure precedence (derived %d, marker %q)", spec.wantExit, wantExit, marker)
			}
			if wantExit == 1 {
				if len(spec.stderrContains) == 0 {
					t.Fatalf("exit-1 spec must expect the named prerequisite on stderr")
				}
				if !strings.Contains(spec.stderrContains[0], marker) {
					t.Errorf("stderr expectation %q must name the first failing prerequisite %q per contract precedence", spec.stderrContains[0], marker)
				}
				return
			}
			if !strings.Contains(joinedOut, "metal: compiled=") {
				t.Errorf("darwin/arm64 exit-0 spec must assert the verdict grammar prefix \"metal: compiled=\"")
			}
			if env["METALPROBE_DEVICE"] == "1" {
				if !strings.Contains(joinedOut, "compiled=true available=true") {
					t.Errorf("device=1 spec must expect \"compiled=true available=true\"")
				}
				if strings.Contains(joinedOut, "CPU-fallback") {
					t.Errorf("device=1 spec must not expect CPU-fallback labeling")
				}
			} else {
				if !strings.Contains(joinedOut, "CPU-fallback") {
					t.Errorf("device!=1 spec must expect CPU-fallback labeling")
				}
				if !strings.Contains(joinedOut, "compiled=true available=false") {
					t.Errorf("device!=1 spec must expect \"compiled=true available=false\"")
				}
			}
		})
	}
}

// TestWindowsNotApplicableWitness is the canonical L2 witness for the
// portable-Makefile requirement (R2): the N/A branch exits 0 with a clear
// message on repeated runs, independent of the host platform.
func TestWindowsNotApplicableWitness(t *testing.T) {
	bin := buildOracle(t)
	spec := findSpec(t, "windows-amd64-not-applicable")
	for run := 1; run <= 2; run++ {
		code, stdout, stderr := runOracle(t, bin, spec.env)
		if code != 0 {
			t.Fatalf("run %d: exit code = %d, want 0 (stderr=%q)", run, code, stderr)
		}
		if !strings.Contains(strings.ToLower(stdout), "not applicable") {
			t.Fatalf("run %d: stdout = %q, want \"not applicable\" (case-insensitive)", run, stdout)
		}
		assertSpec(t, spec, code, stdout, stderr)
	}
}
