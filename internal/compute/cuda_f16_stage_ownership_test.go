//go:build cuda && cgo

package compute

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
	"unsafe"
)

// Per-call synthetic allocation and release; the real pure Go placement policy
// supplies the charge. Estimates only, never measured.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAF16StageFailureRetiresDestination(t *testing.T) {
	for _, budget := range []bool{false, true} {
		for _, managed := range []bool{false, true} {
			for _, unknown := range []bool{false, true} {
				for _, cleanupPanic := range []bool{false, true} {
					t.Run(fmt.Sprintf("budget%v/managed%v/unknown%v/cleanup%v", budget, managed, unknown, cleanupPanic), func(t *testing.T) {
						c := &cudaBackend{dlUsed: 77, managedN: 3}
						if budget {
							c.budgetBytes = 1000
						}
						memory := make([]byte, 128)
						ptr := unsafe.Pointer(&memory[0])
						dst := &cudaBuf{ptr: ptr, n: len(memory), managed: managed, class: MemoryWeights}
						c.accountWeightPlacement(dst, len(memory))
						var primary any = &DeviceAllocError{Bytes: 256, Site: "f16-stage", Class: MemoryScratchpad}
						if unknown {
							primary = &struct{ marker string }{"unknown allocation failure"}
						}
						allocations, releases := 0, 0
						var caught any
						func() {
							defer func() { caught = recover() }()
							c.allocateF16UploadStage(dst, 256, func(bytes int) *cudaBuf {
								allocations++
								if bytes != 256 {
									t.Errorf("stage bytes=%d", bytes)
								}
								panic(primary)
							}, func(p unsafe.Pointer) {
								releases++
								if p != ptr || dst.ptr != nil || dst.budgetedWeightBytes != 0 || dst.managedWeight {
									t.Error("release owner or detached charges incorrect")
								}
								if c.dlUsed != 77 || c.managedN != 3 {
									t.Error("charges not restored before release")
								}
								if cleanupPanic {
									panic("secondary cleanup failure")
								}
							})
							t.Error("failed allocation returned")
						}()
						if caught != primary {
							t.Fatalf("primary changed: %v", caught)
						}
						if allocations != 1 || releases != 1 || dst.ptr != nil || c.dlUsed != 77 || c.managedN != 3 {
							t.Fatalf("ownership counts=%d/%d accounting=%d/%d", allocations, releases, c.dlUsed, c.managedN)
						}
						if dst.n != 128 || dst.managed != managed || dst.class != MemoryWeights {
							t.Fatal("unrelated destination metadata changed")
						}
						if c.immutableWeightUploadCalls != 0 || c.immutableWeightUploadTransferBytes != 0 || c.immutableWeightUploadResidentBytes != 0 {
							t.Fatal("failed stage reported an immutable upload")
						}
					})
				}
			}
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAF16StageSuccessPreservesOwners(t *testing.T) {
	for _, managed := range []bool{false, true} {
		t.Run(fmt.Sprint(managed), func(t *testing.T) {
			c := &cudaBackend{budgetBytes: 1000, dlUsed: 77, managedN: 3}
			destination := make([]byte, 128)
			scratch := make([]byte, 256)
			dst := &cudaBuf{ptr: unsafe.Pointer(&destination[0]), n: 128, managed: managed, class: MemoryWeights}
			stage := &cudaBuf{ptr: unsafe.Pointer(&scratch[0]), n: 256, class: MemoryScratchpad}
			c.accountWeightPlacement(dst, 128)
			before := *dst
			dl, mn := c.dlUsed, c.managedN
			allocations, releases := 0, 0
			got := c.allocateF16UploadStage(dst, 256, func(bytes int) *cudaBuf {
				allocations++
				if bytes != 256 {
					t.Error("wrong stage size")
				}
				return stage
			}, func(unsafe.Pointer) { releases++ })
			if got != stage || allocations != 1 || releases != 0 || dst.ptr != before.ptr || dst.budgetedWeightBytes != before.budgetedWeightBytes || dst.managedWeight != before.managedWeight || c.dlUsed != dl || c.managedN != mn {
				t.Fatal("successful staging changed destination ownership")
			}
			if got.budgetedWeightBytes != 0 || got.managedWeight {
				t.Fatal("scratch gained weight placement charges")
			}
		})
	}
}

// Actual uploadF16 remains the consumer; no CUDA calls during this contract check.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAF16StageNativeBindingAndOrder(t *testing.T) {
	raw, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "cuda.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	bodies := map[string]string{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Body != nil {
			bodies[fn.Name.Name] = string(raw[fs.Position(fn.Body.Pos()).Offset:fs.Position(fn.Body.End()).Offset])
		}
	}
	body := bodies["uploadF16"]
	for _, want := range []string{"c.devF16(t.Shape, t.Layout)", "c.allocateF16UploadStage(buf, len(f)*4,", "c.dallocClass(bytes, MemoryScratchpad, \"f16-stage\")", "C.fcuda_free(ptr)", "C.fcuda_f32_to_f16_T(", "C.fcuda_f32_to_f16(", "C.fcuda_free(stage.ptr)", "c.accountImmutableWeightUpload(", "uploadCache["} {
		if !strings.Contains(body, want) {
			t.Errorf("missing upload owner operation %q", want)
		}
	}
	prev := -1
	for _, step := range []string{"c.devF16(", "if len(f) == 0", "c.allocateF16UploadStage(", "C.fcuda_h2d(", "if t.Layout == ColMajor", "C.fcuda_free(stage.ptr)", "c.accountImmutableWeightUpload(", "uploadCache["} {
		i := strings.Index(body, step)
		if i <= prev {
			t.Fatalf("wrong owner order for %q", step)
		}
		prev = i
	}
	for _, want := range []string{"cudaMu.Lock()", "defer cudaMu.Unlock()", "return c.uploadF16(t, hb, f, hp)"} {
		if !strings.Contains(bodies["uploadClass"], want) {
			t.Errorf("missing serialized consumer %q", want)
		}
	}
	for _, bad := range []string{"cudaMu.Lock()", ".Free(", "fcuda_h2d", "uploadCache", "accountImmutableWeightUpload"} {
		if strings.Contains(bodies["allocateF16UploadStage"], bad) {
			t.Errorf("stage-allocation boundary expanded: %q", bad)
		}
	}
}
