package stepobs

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// vulkan_kernel_source_invariant_test.go — the fail-to-pass SYMPTOM WITNESS for the
// "Vulkan emits no kernel step" defect (#TICKET-15).
//
// The defect was that no file in internal/compute/vulkan*.go contained a single
// computetrace.Record call, so fak_engine_kernel_observed read 0 forever on the one
// backend the appliance actually serves through.
//
// Why this witness is a SOURCE invariant and not a behavioural one. A behavioural
// test of the emitter has to CALL the emitter, so it references symbols the fix
// introduces (computetrace.Emitting, vulkanKernelEvent, ...). Those symbols do not
// exist at the parent commit, so the parent test binary does not COMPILE — and a
// witness that cannot build witnesses nothing. That is exactly the
// SYMPTOM_PARENT_BUILD abstention this file replaces.
//
// Parsing the producer's own source sidesteps it: go/parser reads the Vulkan files
// as text and this test references ZERO symbols the fix introduces, so it builds
// identically at the parent and at the fix. At the parent the first assertion fails
// because no emission site exists; at the fix they all pass because one does.
//
// It also pins the two properties that make the emission HONEST rather than merely
// present — the part a bare presence check would miss:
//
//   - the seam is gated on the narrow computetrace.Emitting() predicate and the
//     Vulkan family never gates on computetrace.Enabled(). Enabled() also gates the
//     bounded activation-sample capture on the decode hot path
//     (internal/model/v41_activation_trace.go), so a Vulkan site gated on it — or a
//     later edit that widens it — silently switches on activation sampling for every
//     Vulkan forward.
//   - the Vulkan family never emits the host_monotonic timer domain. A host clock
//     wearing a device timer domain is fabricated telemetry, the exact failure this
//     metrics effort exists to eliminate.
//
// fak-test:runtime fast est=1s — parses only internal/compute/vulkan*.go; no device,
// no dispatch, no model load.

// siComputeDir is internal/compute relative to this package (go test runs with the
// package directory as cwd).
const siComputeDir = "../compute"

// siParsedFile is one parsed producer source file plus its raw text.
type siParsedFile struct {
	name string // base name, for failure messages
	src  string
	file *ast.File
}

// siParseVulkanFamily parses every internal/compute/vulkan*.go source file (test
// files excluded: they are not production emission points) and FAILS CLOSED — an
// unreadable file, a parse error, or an empty file set is a test failure, never a
// silent pass. An empty parse would make every assertion below vacuous, which is the
// failure mode a source-invariant witness is most exposed to.
func siParseVulkanFamily(t *testing.T) []siParsedFile {
	t.Helper()
	entries, err := os.ReadDir(siComputeDir)
	if err != nil {
		t.Fatalf("read %s: %v (the producer dir must exist for this witness)", siComputeDir, err)
	}
	var names []string
	for _, e := range entries {
		n := e.Name()
		if e.IsDir() || !strings.HasPrefix(n, "vulkan") || !strings.HasSuffix(n, ".go") ||
			strings.HasSuffix(n, "_test.go") {
			continue
		}
		names = append(names, n)
	}
	if len(names) == 0 {
		t.Fatalf("no internal/compute/vulkan*.go source file was found; refusing to treat an " +
			"empty parse as a satisfied invariant")
	}
	sort.Strings(names)

	fset := token.NewFileSet()
	out := make([]siParsedFile, 0, len(names))
	for _, name := range names {
		path := filepath.Join(siComputeDir, name)
		src, readErr := os.ReadFile(path)
		if readErr != nil {
			t.Fatalf("read %s: %v", path, readErr)
		}
		if len(src) == 0 {
			t.Fatalf("%s is empty; refusing to treat an empty source as a satisfied invariant", path)
		}
		parsed, parseErr := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", path, parseErr)
		}
		out = append(out, siParsedFile{name: name, src: string(src), file: parsed})
	}
	return out
}

// siSelectorCallsIn returns every pkg.Sel(...) call spelled under n.
func siSelectorCallsIn(n ast.Node, pkg, sel string) []*ast.CallExpr {
	var out []*ast.CallExpr
	ast.Inspect(n, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		se, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		id, ok := se.X.(*ast.Ident)
		if ok && id.Name == pkg && se.Sel.Name == sel {
			out = append(out, call)
		}
		return true
	})
	return out
}

// siSelectionsIn returns every pkg.Sel selector expression in the file, as
// "pkg.Sel" strings. A PARSE-level scan: it reports what the source SPELLS, which
// is the invariant under test, and it needs no type information so it behaves
// identically at the parent and at the fix.
func siSelectionsIn(f ast.Node, pkg string) []string {
	var out []string
	ast.Inspect(f, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		id, ok := sel.X.(*ast.Ident)
		if ok && id.Name == pkg {
			out = append(out, pkg+"."+sel.Sel.Name)
		}
		return true
	})
	return out
}

// siRecordEmitterFuncs returns the names of top-level funcs that call
// computetrace.Record, in first-seen order across the family.
func siRecordEmitterFuncs(f ast.Node) []string {
	var out []string
	for _, decl := range f.(*ast.File).Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		if len(siSelectorCallsIn(fn.Body, "computetrace", "Record")) > 0 {
			out = append(out, fn.Name.Name)
		}
	}
	return out
}

// siTopLevelFunc returns the named top-level func declaration, or nil.
func siTopLevelFunc(f ast.Node, name string) *ast.FuncDecl {
	for _, decl := range f.(*ast.File).Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == name {
			return fn
		}
	}
	return nil
}

// siPlainCallsIn returns every unqualified name(...) call under n.
func siPlainCallsIn(n ast.Node, name string) []*ast.CallExpr {
	var out []*ast.CallExpr
	ast.Inspect(n, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, ok := call.Fun.(*ast.Ident); ok && id.Name == name {
			out = append(out, call)
		}
		return true
	})
	return out
}

// siStringConsts returns the string values of every top-level const whose name
// mentions TimerDomain, keyed by const name.
func siStringConsts(f ast.Node) map[string]string {
	out := map[string]string{}
	for _, decl := range f.(*ast.File).Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.CONST {
			continue
		}
		for _, spec := range gd.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for i, name := range vs.Names {
				if !strings.Contains(name.Name, "TimerDomain") || i >= len(vs.Values) {
					continue
				}
				if lit, ok := vs.Values[i].(*ast.BasicLit); ok && lit.Kind == token.STRING {
					if v, err := strconv.Unquote(lit.Value); err == nil {
						out[name.Name] = v
					}
				}
			}
		}
	}
	return out
}

// siFileList names the parsed family for failure messages.
func siFileList(files []siParsedFile) string {
	names := make([]string, 0, len(files))
	for _, f := range files {
		names = append(names, f.name)
	}
	return strings.Join(names, ", ")
}

// TestVulkanKernelEmissionSourceInvariant is the designated red-then-green symptom
// witness for #TICKET-15. Deliberately a source invariant (see the file header): it
// compiles at the parent commit, where it FAILS, and at the fix commit, where it
// PASSES.
//
// Land-gate selector:  --symptom-test '^TestVulkanKernelEmissionSourceInvariant$'
func TestVulkanKernelEmissionSourceInvariant(t *testing.T) {
	files := siParseVulkanFamily(t)

	// ---- (1) the defect itself: the Vulkan family emitted NO kernel event ------
	var emitters []string
	for _, f := range files {
		emitters = append(emitters, siRecordEmitterFuncs(f.file)...)
	}
	if len(emitters) == 0 {
		t.Fatalf("no computetrace.Record emission point exists anywhere in the Vulkan backend "+
			"(%s): fak_engine_kernel_observed would read 0 forever on a Vulkan-served model, which "+
			"is the defect this witness reproduces", siFileList(files))
	}
	if len(emitters) != 1 {
		t.Fatalf("the Vulkan backend declares %d kernel-event emission points (%v); it must have "+
			"exactly one so a dispatch can never be double-counted", len(emitters), emitters)
	}

	// The site must be on the matmul path, not an incidental helper: a non-matmul
	// emitter leaves the decode GEMV — the kernel the appliance runs per layer — dark.
	if !siEmissionReachesMatMul(files, emitters[0]) {
		t.Fatalf("the Vulkan kernel-event emission point %q is not reachable from "+
			"(*vulkanBackend).MatMul; the decode GEMV would still emit nothing", emitters[0])
	}

	// ---- (2) the gate must be the NARROW predicate, never Enabled() -----------
	// This is the assertion that carries the real invariant.
	var sawEmitting bool
	for _, f := range files {
		for _, sel := range siSelectionsIn(f.file, "computetrace") {
			switch sel {
			case "computetrace.Emitting":
				sawEmitting = true
			case "computetrace.Enabled":
				t.Fatalf("%s references computetrace.Enabled(). The Vulkan kernel seam must gate "+
					"on computetrace.Emitting() so an attached metrics observer alone is enough to be "+
					"fed; Enabled() must stay the recorder-only gate because it also switches on the "+
					"decode-hot-path activation-sample capture in internal/model", f.name)
			}
		}
	}
	if !sawEmitting {
		t.Fatalf("no file in the Vulkan family (%s) references computetrace.Emitting(); the emission "+
			"point must be reachable WITHOUT the `fak computetrace` artifact recorder installed",
			siFileList(files))
	}

	// ---- (3) the domains are declared, and no host clock can be emitted --------
	device, unavailable := "", ""
	for _, f := range files {
		for name, value := range siStringConsts(f.file) {
			switch name {
			case "VulkanTimerDomainDevice":
				device = value
			case "VulkanTimerDomainUnavailable":
				unavailable = value
			}
		}
	}
	if device == "" {
		t.Fatal("the Vulkan family does not declare a non-empty VulkanTimerDomainDevice constant, " +
			"so a device-measured kernel duration has no provenance label")
	}
	if unavailable == "" {
		t.Fatal("the Vulkan family does not declare a non-empty VulkanTimerDomainUnavailable " +
			"constant, so a kernel whose device timer could not be read has no honest label")
	}
	if device == unavailable {
		t.Fatalf("the device and unavailable Vulkan timer domains are the same string %q; a consumer "+
			"could not tell a measurement from an unavailable marker", device)
	}

	// A host-monotonic spelling anywhere in the Vulkan family would let a host clock
	// masquerade as a device measurement.
	for _, f := range files {
		if strings.Contains(f.src, `"host_monotonic"`) {
			t.Fatalf("%s spells the host_monotonic timer domain; the Vulkan emitter must report a "+
				"device measurement or an explicit unavailable marker, never a host clock", f.name)
		}
	}
}

// siEmissionReachesMatMul reports whether the named emitter is reachable from
// (*vulkanBackend).MatMul: either the emitter IS MatMul's body, or MatMul calls it.
func siEmissionReachesMatMul(files []siParsedFile, emitter string) bool {
	for _, f := range files {
		matmul := siTopLevelFunc(f.file, "MatMul")
		if matmul == nil || matmul.Body == nil || matmul.Recv == nil || len(matmul.Recv.List) == 0 {
			continue
		}
		if len(siSelectorCallsIn(matmul.Body, "", emitter)) > 0 {
			return true
		}
		if len(siPlainCallsIn(matmul.Body, emitter)) > 0 {
			return true
		}
	}
	return false
}
