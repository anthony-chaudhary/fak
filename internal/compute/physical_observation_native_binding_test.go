package compute

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=10s lane=default
func TestVulkanUploadAttributionNativeBinding(t *testing.T) {
	t.Parallel()
	// Compile the real Go wrapper prefix and accounting helpers with explicit
	// native-call stubs. This proves Go gating, lock/defer order and unwind behavior;
	// it is neither a cgo/link witness nor evidence of physical GPU transfers.
	upload := nativeAttributionDeclaration(t, "vulkan.go", "uploadClass")
	start, end := strings.Index(upload, "vulkanMu.Lock()"), strings.Index(upload, "hb, ok :=")
	if start < 0 || end < start {
		t.Fatal("uploadClass no longer exposes its accounting prefix")
	}
	prefix := upload[start:end]
	if !strings.Contains(prefix, "defer vulkanMu.Unlock()") ||
		!strings.Contains(prefix, "if vulkanUploadAttribution.enabled {") ||
		!strings.Contains(prefix, "defer vulkanBeginUploadAttributionLocked(class)()") {
		t.Fatal("upload attribution must be gated and deferred inside the existing lock")
	}
	state := nativeAttributionDeclaration(t, "vulkan.go", "vulkanUploadAttribution")
	if !strings.Contains(state, "C.fvk_dispatch_profile_enabled() != 0") {
		t.Fatal("upload attribution must capture the native startup dispatch-profile gate")
	}
	shim, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	getter := sourceSection(t, string(shim), `extern "C" int fvk_dispatch_profile_enabled`, "static inline void dp_inc")
	if !strings.Contains(string(shim), "static const bool g_dp_on =") ||
		!strings.Contains(getter, "return g_dp_on ? 1 : 0;") || strings.Contains(getter, "getenv") {
		t.Fatal("native profile getter must only expose the captured immutable gate")
	}
	snapshot := nativeAttributionDeclaration(t, "vulkan_device_debug.go", "backendExecutionSnapshotLocked")
	for _, required := range []string{
		"vulkanUploadAttributionSnapshotLocked(uint64(h2dBytes))",
		"H2DAttribution: attribution", "H2DAttributionObserved: attributionObserved",
	} {
		if !strings.Contains(snapshot, required) {
			t.Fatalf("existing execution snapshot is missing %q", required)
		}
	}
	if strings.Count(snapshot, "C.fvk_transfer_counters(") != 1 {
		t.Fatal("snapshot attribution must reuse the existing native transfer sample")
	}
	var helpers strings.Builder
	for _, name := range []string{
		"vulkanBeginUploadAttributionLocked", "vulkanUploadCounterLocked",
		"vulkanObserveUploadTotalLocked", "vulkanUploadAttributionSnapshotLocked",
	} {
		helpers.WriteString(nativeAttributionDeclaration(t, "vulkan_device_debug.go", name))
		helpers.WriteByte('\n')
	}
	for _, forbidden := range []string{"recover(", "fvk_sync", "fvk_batch_flush", "fvk_device_"} {
		if strings.Contains(prefix+helpers.String(), forbidden) {
			t.Fatalf("upload attribution must not recover, wait, flush or query devices: %s", forbidden)
		}
	}
	goHelpers := strings.NewReplacer("C.uint64_t", "uint64", "C.fvk_transfer_counters", "stubTransfers", "C.fvk_direct_h2d_bytes", "stubDirect").Replace(helpers.String())
	goState := strings.ReplaceAll(state, "C.fvk_dispatch_profile_enabled", "stubProfileEnabled")
	goStateOff := strings.NewReplacer("vulkanUploadAttribution", "disabledUploadAttribution", "stubProfileEnabled", "stubProfileDisabled").Replace(goState)
	fixture := "package compute\nimport \"fmt\"\n" + goState + "\n" + goStateOff + "\n" +
		nativeAttributionDeclaration(t, "physical_observation.go", "BackendCounterSnapshot") + "\n" +
		nativeAttributionDeclaration(t, "physical_observation.go", "validateBackendTransferCounterPairs") + "\n" +
		goHelpers + "\nfunc upload(class MemoryClass, operation func()) {\n" + prefix + "operation()\n}\n" + `
type MemoryClass string
type checkedLock struct { held bool }
func (m *checkedLock) Lock() { if m.held { panic("double lock") }; m.held = true }
func (m *checkedLock) Unlock() { if !m.held { panic("unlock without lock") }; m.held = false }
var vulkanMu checkedLock
var reads, directReads int
var total, count, direct uint64
var nativeAvailable = true
var nativeProfile = 1
var profileReads int
func stubProfileEnabled() int { profileReads++; return nativeProfile }
func stubProfileDisabled() int { return 0 }
func stubTransfers(hc, hb, dc, db, cc, cb *uint64) int {
 if !vulkanMu.held { panic("transfer sample outside lock") }; reads++
 if !nativeAvailable { return 0 }; *hc, *hb = count, total; return 1
}
func stubDirect() uint64 {
 if !vulkanMu.held { panic("direct sample outside lock") }; directReads++; return direct
}
func snapshot() (BackendH2DAttribution, bool, error) {
 vulkanMu.Lock(); defer vulkanMu.Unlock()
 return vulkanUploadAttributionSnapshotLocked(total)
}
`
	reducer, err := os.ReadFile("physical_h2d_attribution.go")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for name, body := range map[string]string{
		"go.mod": "module nativebinding\n\ngo 1.26\n", "binding.go": fixture,
		"reducer.go": string(reducer), "binding_test.go": nativeAttributionFixtureTest,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "test", "-trimpath", "-p=1", "-vet=off", "-count=1", "-timeout=20s", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=0", "GOPROXY=off", "GOSUMDB=off", "GOFLAGS=")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("extracted Go control-flow witness: %v\n%s", err, output)
	}
}

func nativeAttributionDeclaration(t *testing.T, path, name string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range file.Decls {
		match := false
		switch d := decl.(type) {
		case *ast.FuncDecl:
			match = d.Name.Name == name
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					match = match || s.Name.Name == name
				case *ast.ValueSpec:
					for _, n := range s.Names {
						match = match || n.Name == name
					}
				}
			}
		}
		if match {
			return string(raw[fset.Position(decl.Pos()).Offset:fset.Position(decl.End()).Offset])
		}
	}
	t.Fatalf("missing %s in %s", name, path)
	return ""
}

const nativeAttributionFixtureTest = `package compute
import "testing"
func TestExtractedUploadControlFlow(t *testing.T) {
 if !vulkanUploadAttribution.enabled || disabledUploadAttribution.enabled || profileReads != 1 {
  t.Fatal("upload attribution did not capture enabled/disabled native gates exactly once")
 }
 nativeProfile = 0
 if !vulkanUploadAttribution.enabled || profileReads != 1 {
  t.Fatal("upload attribution reread the startup gate")
 }
 vulkanUploadAttribution.enabled = false
 upload("weights", func() { total += 5; count++ })
 if a, observed, err := snapshot(); err != nil || observed || a != (BackendH2DAttribution{}) || reads != 0 || directReads != 0 {
  t.Fatalf("unprofiled path added reads or evidence: %+v %v %v %d %d", a, observed, err, reads, directReads)
 }
 vulkanUploadAttribution.enabled = true
 upload("weights", func() { total += 8; count++ })
 sentinel := new(int)
 func() {
  defer func() { if got := recover(); got != sentinel { t.Fatalf("panic changed or swallowed: %v", got) } }()
  upload("activation", func() { total += 7; count++; panic(sentinel) })
 }()
 if vulkanMu.held || reads != 4 || vulkanUploadAttribution.bytes.Weights != 8 || vulkanUploadAttribution.bytes.Activation != 7 {
  t.Fatalf("partial upload lost attribution or deferred sample: %+v reads=%d locked=%v", vulkanUploadAttribution.bytes, reads, vulkanMu.held)
 }
 total += 11; count++; direct = 9
 a, observed, err := snapshot()
 if err != nil || !observed || a.Weights != 8 || a.Activation != 7 || a.UnattributedBytes != 16 || a.DirectMappedBytes != 9 || directReads != 1 || reads != 4 {
  t.Fatalf("residual/direct snapshot is inconsistent: %+v %v %v", a, observed, err)
 }
 // A failed native tuple poisons future attribution instead of guessing bytes.
 nativeAvailable = false
 upload("weights", func() {})
 nativeAvailable = true
 if a, observed, err := snapshot(); err == nil || observed || a != (BackendH2DAttribution{}) {
  t.Fatalf("failed native sample did not remain invalid: %+v %v %v", a, observed, err)
 }
 // Reset detection spans uploads, not just one before/after interval.
 vulkanUploadAttribution.err = nil
 total = 1; count = 1
 upload("weights", func() { total++; count++ })
 total = 100; count = 100
 if _, observed, err := snapshot(); err == nil || observed { t.Fatal("counter reset recovered silently") }
 // A nonzero byte/zero count tuple must not produce attribution.
 vulkanUploadAttribution.err = nil; vulkanUploadAttribution.lastTotal = 0
 count = 0
 upload("weights", func() {})
 if _, observed, err := snapshot(); err == nil || observed { t.Fatal("invalid native tuple was accepted") }
 if profileReads != 1 { t.Fatal("upload or snapshot reread the startup gate") }
}
`
