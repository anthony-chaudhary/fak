package compute

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func captureConfigSyntax(n ast.Node) string {
	var b bytes.Buffer
	if err := format.Node(&b, token.NewFileSet(), n); err != nil {
		panic(err)
	}
	return strings.Join(strings.Fields(b.String()), "")
}

// This checks actual CUDA source wiring, not runtime locking or device execution.
func captureConfigRuntimeContract(src []byte) error {
	f, err := parser.ParseFile(token.NewFileSet(), "cuda_graph_once_qualification.go", src, 0)
	if err != nil {
		return err
	}
	functions := map[string]*ast.FuncDecl{}
	defaultOff := false
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok {
			functions[fn.Name.Name] = fn
		}
		if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.VAR {
			for _, item := range g.Specs {
				v := item.(*ast.ValueSpec)
				for _, name := range v.Names {
					if name.Name == "cudaCaptureOnceMode" {
						defaultOff = len(v.Values) == 0 && captureConfigSyntax(v.Type) == "cudaCaptureOnceConfiguration"
					}
				}
			}
		}
	}
	if !defaultOff {
		return fmt.Errorf("qualification mode must be a zero-value, not build-tag consent")
	}
	for _, imp := range f.Imports {
		if imp.Path.Value == `"os"` {
			return fmt.Errorf("capture-once must not read runtime environment")
		}
	}
	for _, name := range []string{"configureCUDACaptureOnce", "GraphOnceBegin", "GraphOnceReplay"} {
		fn := functions[name]
		if fn == nil || len(fn.Body.List) < 3 || captureConfigSyntax(fn.Body.List[0]) != "cudaMu.Lock()" || captureConfigSyntax(fn.Body.List[1]) != "defercudaMu.Unlock()" {
			return fmt.Errorf("%s must own cudaMu for mode access", name)
		}
		body := captureConfigSyntax(fn.Body)
		if name == "configureCUDACaptureOnce" {
			for _, required := range []string{
				"ifc==nil||c!=cudaDev{returnErrCUDACaptureOnceUnsupported}",
				"ownerLive:=cudaGraphOnceOwner!=nil||c.graphOnce!=nil||c.capturing",
				"returncudaCaptureOnceMode.set(enabled,cudaAsyncGraphOnceAllowed(),graphEnabled,cudaKernelProfileOn,ownerLive)",
			} {
				if !strings.Contains(body, required) {
					return fmt.Errorf("configuration lost guard: %s", required)
				}
			}
			if strings.Contains(body, "C.") {
				return fmt.Errorf("configuration must not submit or tear down native work")
			}
		} else {
			gate := strings.Index(body, "!cudaCaptureOnceMode.enabled")
			native := strings.Index(body, "C.")
			if gate < 0 || native < 0 || gate >= native {
				return fmt.Errorf("%s must gate native work on explicit mode", name)
			}
		}
	}
	return nil
}

func captureConfigRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// fak-test:runtime unit est=2ms
func TestCUDACaptureOnceConfigurationSourceContract(t *testing.T) {
	src := captureConfigRead(t, "cuda_graph_once_qualification.go")
	if err := captureConfigRuntimeContract(src); err != nil {
		t.Fatal(err)
	}
	for _, m := range []struct{ old, new string }{
		{"var cudaCaptureOnceMode cudaCaptureOnceConfiguration", "var cudaCaptureOnceMode = cudaCaptureOnceConfiguration{enabled: true}"},
		{"cudaMu.Lock()", "unrelatedMu.Lock()"},
		{"!cudaCaptureOnceMode.enabled", "false"},
		{"c == nil || c != cudaDev", "c == nil"},
	} {
		mutant := bytes.Replace(src, []byte(m.old), []byte(m.new), 1)
		if bytes.Equal(src, mutant) || captureConfigRuntimeContract(mutant) == nil {
			t.Fatalf("source contract did not reject %q mutation", m.old)
		}
	}
	if !bytes.HasPrefix(src, []byte("//go:build cuda && cgo && fak_cuda_capture_once_qualification\n")) {
		t.Fatal("implementation escaped qualification-only build")
	}
	off := captureConfigRead(t, "cuda_graph_once_off.go")
	if bytes.Contains(off, []byte("configureCUDACaptureOnce")) {
		t.Fatal("ordinary CUDA build must not accept an inert configuration")
	}
	// Existing ordinary graph lifetime and shared native execution remain outside
	// this setting. These files must not acquire any read of the new mode/API.
	for _, path := range []string{"cuda.go", filepath.Join("cuda", "spec_graph.cuh")} {
		b := captureConfigRead(t, path)
		if bytes.Contains(b, []byte("cudaCaptureOnceMode")) || bytes.Contains(b, []byte("ConfigureCUDACaptureOnce")) {
			t.Fatalf("capture-once configuration changed ordinary graph wiring in %s", path)
		}
	}
}

// fak-test:runtime unit est=1ms
func TestCUDACaptureOnceConfigurationWitnessContract(t *testing.T) {
	modelPath := filepath.Join("..", "model", "cuda_capture_once_required_device_test.go")
	for _, path := range []string{"cuda_graph_once_required_device_test.go", "cuda_graph_once_required_capacity_test.go", modelPath} {
		src := string(captureConfigRead(t, path))
		if strings.Contains(src, "FAK_CUDA_CAPTURE_ONCE") {
			t.Fatalf("retired runtime setting remains in witness %s", path)
		}
		for _, required := range []string{"ConfigureCUDACaptureOnce(be, true)", "ConfigureCUDACaptureOnce(be, false)", "t.Cleanup(func()"} {
			if !strings.Contains(src, required) {
				t.Fatalf("missing checked configuration/cleanup %q in %s", required, path)
			}
		}
	}
	model := string(captureConfigRead(t, modelPath))
	last := -1
	for _, marker := range []string{
		"compute.ConfigureCUDACaptureOnce(be, false)",
		"ref, err := m.NewBackendSessionChecked(be)",
		"\n\tref.Close()",
		"compute.ConfigureCUDACaptureOnce(be, true)",
		"selected, err := m.NewBackendSessionChecked(be)",
	} {
		i := strings.Index(model, marker)
		if i <= last {
			t.Fatalf("eager OFF -> close -> selected ON ordering lost at %q", marker)
		}
		last = i
	}
	capacity := string(captureConfigRead(t, "cuda_graph_once_required_capacity_test.go"))
	if !strings.Contains(capacity, "!cudaCaptureOnceMode.enabled") {
		t.Fatal("capacity attribution must prove the new non-capacity admission prerequisite")
	}
	doc := string(captureConfigRead(t, filepath.Join("..", "..", "docs", "cuda-capture-once-required-device.md")))
	if strings.Contains(doc, "FAK_CUDA_CAPTURE_ONCE=1") || !strings.Contains(doc, "fak_cuda_capture_once_capacity_witness") {
		t.Fatal("recipe must retire environment enablement and include the separate capacity witness")
	}
	for _, name := range []string{"TestCUDAGraphOnceRequiredBorrowedLifetime", "TestCUDAGraphOnceRequiredDenseModel", "TestCUDAGraphOnceRequiredLastCapacity"} {
		if strings.Count(doc, name) < 2 {
			t.Fatalf("recipe must run and JSON-verify %s", name)
		}
	}
}
