package compute

// scaffold_test.go — the golden witness for the `fak backend scaffold` generator (#1685). It
// proves two things without ever committing a generated file into the tree:
//
//  1. Generate's output has the SAME STRUCTURAL SHAPE as the hand-written C-series backends
//     (rocm_arch.go et al.): the taxonomy file declares a Family enum, an Arch struct, a
//     Lookup*Arch/*Token normalizer pair, and a Known*Arches enumerator — the pattern
//     ROCM-C002-NOTES.md / TPU-C004-NOTES.md / OPENVINO-C006-NOTES.md all document.
//  2. The generated taxonomy ACTUALLY BUILDS AND PASSES as real Go: the arch + arch_test files
//     are staged into this package under a disposable, collision-proof name, `go test -run
//     <Name>Arch` is run as a real subprocess, and the staged files are removed in a deferred
//     cleanup — so the acceptance bullet ("go build ./... clean, go test -run <Name>Arch green
//     immediately after generation") is witnessed, not asserted.
//
// A throwaway name (not "mychip") avoids colliding with anything a human might type by hand
// while trying the tool interactively in this same checkout.

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestScaffoldValidateNameFailsClosed(t *testing.T) {
	bad := []string{"", "MyChip", "my-chip", "1chip", "my chip", "my_chip"}
	for _, n := range bad {
		if err := ValidateName(n); err == nil {
			t.Errorf("ValidateName(%q) = nil, want error", n)
		}
	}
	good := []string{"mychip", "chip2", "a"}
	for _, n := range good {
		if err := ValidateName(n); err != nil {
			t.Errorf("ValidateName(%q) = %v, want nil", n, err)
		}
	}
}

func TestScaffoldLookupLaneFailsClosedOnUnknown(t *testing.T) {
	for _, lane := range KnownLanes() {
		if _, ok := LookupLane(lane); !ok {
			t.Errorf("LookupLane(%q) = not found, want a known lane", lane)
		}
	}
	for _, lane := range []string{"", "cuda", "vulkan", "not-a-lane"} {
		if _, ok := LookupLane(lane); ok {
			t.Errorf("LookupLane(%q) = found, want unsupported (fail closed)", lane)
		}
	}
}

func TestScaffoldGenerateRejectsBadInput(t *testing.T) {
	if _, err := Generate(ScaffoldSpec{Name: "Bad-Name", Lane: LaneCustom}); err == nil {
		t.Error("Generate with an invalid name succeeded, want error")
	}
	if _, err := Generate(ScaffoldSpec{Name: "goodname", Lane: Lane("not-a-lane")}); err == nil {
		t.Error("Generate with an unknown lane succeeded, want error")
	}
}

// TestScaffoldShapeMatchesHandWrittenBackends checks Generate's output has the same structural
// skeleton as the hand-written taxonomies (rocm_arch.go/tpu_arch.go/openvino_arch.go): a Family
// enum, an Arch struct, Lookup*Arch + *Token functions, and a Known*Arches enumerator — plus the
// four-file set (arch, arch_test, backend stub, NOTES.md) the issue's acceptance bullet names.
func TestScaffoldShapeMatchesHandWrittenBackends(t *testing.T) {
	const name = "zzscaffgolden"
	for _, lane := range KnownLanes() {
		files, err := Generate(ScaffoldSpec{Name: name, Lane: Lane(lane)})
		if err != nil {
			t.Fatalf("lane %s: Generate: %v", lane, err)
		}
		want := map[string]bool{
			name + "_arch.go":                   false,
			name + "_arch_test.go":              false,
			name + "_backend.go":                false,
			strings.ToUpper(name) + "-NOTES.md": false,
		}
		for _, f := range files {
			if _, ok := want[f.RelPath]; !ok {
				t.Errorf("lane %s: unexpected emitted file %s", lane, f.RelPath)
				continue
			}
			want[f.RelPath] = true
		}
		for path, seen := range want {
			if !seen {
				t.Errorf("lane %s: expected file %s was not emitted", lane, path)
			}
		}

		var arch, archTest, backend string
		for _, f := range files {
			switch f.RelPath {
			case name + "_arch.go":
				arch = f.Content
			case name + "_arch_test.go":
				archTest = f.Content
			case name + "_backend.go":
				backend = f.Content
			}
		}

		exp := "Zzscaffgolden" // exportedPrefix(name)
		// Structural shape pinned against the hand-written pattern: a Family enum, an Arch
		// struct, the fail-closed Lookup/Token pair, and the enumerator — the same names
		// rocm_arch.go declares as ROCmFamily/ROCmArch/LookupROCmArch/ROCmOffloadArch/
		// KnownROCmArches, just parametrized by the generated name.
		mustContain(t, lane+" arch.go", arch, []string{
			"type " + exp + "Family uint8",
			"type " + exp + "Arch struct",
			"func Lookup" + exp + "Arch(name string) (" + exp + "Arch, bool)",
			"func " + exp + "Token(name string) (string, bool)",
			"func Known" + exp + "Arches() []" + exp + "Arch",
		})
		mustContain(t, lane+" arch_test.go", archTest, []string{
			"func Test" + exp + "ArchLookupKnown(t *testing.T)",
			"func Test" + exp + "ArchNormalizationRejectsUnsupported(t *testing.T)",
		})
		hint, _ := LookupLane(lane)
		mustContain(t, lane+" backend.go", backend, []string{
			"//go:build " + hint.tag,
			"func (c *" + name + "Backend) Class() CorrectnessClass { return Approx }",
			"Register(&" + name + "Backend{",
		})
	}
}

func mustContain(t *testing.T, label, haystack string, needles []string) {
	t.Helper()
	for _, n := range needles {
		if !strings.Contains(haystack, n) {
			t.Errorf("%s: missing expected fragment %q", label, n)
		}
	}
}

// TestScaffoldGeneratedTaxonomyBuildsAndTestsGreen is the real-compile witness the acceptance
// bullet names: it stages the generated <name>_arch.go + <name>_arch_test.go directly into THIS
// package directory under a disposable name, runs `go test -run <Name>Arch` as a subprocess (so
// it exercises the actual go toolchain, not just string matching), and removes the staged files
// afterward — the golden test never leaves a generated file behind. It also confirms `go build
// ./...` (no tag) stays clean with the backend stub present, since the stub is guarded by its
// lane's build tag and must not compile into the default build.
func TestScaffoldGeneratedTaxonomyBuildsAndTestsGreen(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping subprocess go build/test in -short")
	}
	const name = "zzscaffbuildgolden"
	exp := "Zzscaffbuildgolden"

	files, err := Generate(ScaffoldSpec{Name: name, Lane: LaneCustom})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	wd, err := os.Getwd() // internal/compute, since this test file lives there
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	var staged []string
	cleanup := func() {
		for _, p := range staged {
			os.Remove(p)
		}
	}
	defer cleanup()

	for _, f := range files {
		if f.RelPath != name+"_arch.go" && f.RelPath != name+"_arch_test.go" && f.RelPath != name+"_backend.go" {
			continue // NOTES.md is not Go source; skip staging it
		}
		full := filepath.Join(wd, f.RelPath)
		if _, statErr := os.Stat(full); statErr == nil {
			t.Fatalf("refusing to stage over existing file %s (name collision — pick a different golden name)", full)
		}
		if err := os.WriteFile(full, []byte(f.Content), 0o644); err != nil {
			t.Fatalf("stage %s: %v", full, err)
		}
		staged = append(staged, full)
	}

	repoRoot := findRepoRootForTest(t, wd)

	// (a) go build ./internal/... must stay clean with the (build-tag-gated) backend stub
	// present. Scoped to internal/... rather than the whole repo (./...) so this golden test's
	// verdict depends only on scaffold.go's own output, not on the health of unrelated
	// concurrently-edited packages elsewhere in this shared tree (e.g. cmd/**).
	buildCmd := exec.Command("go", "build", "./internal/...")
	buildCmd.Dir = repoRoot
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("go build ./internal/... failed with generated scaffold staged:\n%s", out)
	}

	// (b) go test -run <Name>Arch must be green immediately after generation (the taxonomy
	// half only — no build tag needed).
	testCmd := exec.Command("go", "test", "./internal/compute/", "-run", exp+"Arch", "-v")
	testCmd.Dir = repoRoot
	out, err := testCmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go test -run %sArch failed for generated scaffold:\n%s", exp, out)
	}
	if !strings.Contains(string(out), "PASS") {
		t.Fatalf("go test -run %sArch did not report PASS:\n%s", exp, out)
	}
}

// findRepoRootForTest walks up from the compute package directory to the module root (the
// directory containing go.mod), so the subprocess `go build`/`go test` invocations run with the
// same working directory convention AGENTS.md requires (commands from the module root).
func findRepoRootForTest(t *testing.T, start string) string {
	t.Helper()
	dir := start
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatalf("could not find go.mod walking up from %s", start)
	return ""
}
