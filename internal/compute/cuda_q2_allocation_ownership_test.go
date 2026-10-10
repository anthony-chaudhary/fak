//go:build cuda && cgo

package compute

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

// Synthetic operations only; no CUDA calls. Placement uses the real pure Go
// accounting leaf, including budget-disabled and managed-code behavior.
// Estimate only; not timed.
// fak-test:runtime fast est=1ms lane=default
func TestCUDADevQ2AllocationFailureOwnership(t *testing.T) {
	for _, budgeted := range []bool{false, true} {
		for _, managed := range []bool{false, true} {
			for _, failAt := range []int{1, 2} {
				for _, unknown := range []bool{false, true} {
					t.Run(fmt.Sprintf("budget%v/managed%v/fail%d/unknown%v", budgeted, managed, failAt, unknown), func(t *testing.T) {
						c := &cudaBackend{dlUsed: 77, managedN: 3}
						if budgeted {
							c.budgetBytes = 1000
						}
						var primary any = &DeviceAllocError{Bytes: 8, Site: "q2-scale", Class: MemoryWeights}
						if unknown {
							primary = &struct{ marker string }{"original"}
						}
						var storage []byte
						var code *cudaBuf
						attempts, releases := 0, 0
						var got any
						func() {
							defer func() { got = recover() }()
							c.devQ2WithOperations([]int{2, 32}, 32, 2, func(bytes int) *cudaBuf {
								attempts++
								if failAt == 1 {
									panic(primary)
								}
								if bytes != 16 {
									t.Errorf("code bytes=%d", bytes)
								}
								storage = make([]byte, bytes)
								code = &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: bytes, managed: managed, class: MemoryWeights}
								c.accountWeightPlacement(code, bytes)
								return code
							}, func(bytes int) *cudaBuf {
								attempts++
								if bytes != 8 {
									t.Errorf("scale bytes=%d", bytes)
								}
								panic(primary)
							}, func(ptr unsafe.Pointer) {
								releases++
								if code == nil || ptr != unsafe.Pointer(&storage[0]) || code.ptr != nil || code.budgetedWeightBytes != 0 || code.managedWeight {
									t.Error("unowned or undetached cleanup")
								}
								panic("secondary cleanup")
							})
							t.Error("allocation failure returned")
						}()
						if got != primary {
							t.Fatalf("lost primary identity: %v", got)
						}
						if attempts != failAt || releases != failAt-1 {
							t.Fatalf("attempts/releases=%d/%d", attempts, releases)
						}
						if c.dlUsed != 77 || c.managedN != 3 {
							t.Fatalf("placement leak: %d/%d", c.dlUsed, c.managedN)
						}
						if c.immutableWeightUploadCalls != 0 || c.immutableWeightUploadTransferBytes != 0 || c.immutableWeightUploadResidentBytes != 0 {
							t.Fatal("allocation failure changed upload telemetry")
						}
					})
				}
			}
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDADevQ2SuccessfulAndEmptyLayout(t *testing.T) {
	for _, empty := range []bool{false, true} {
		for _, managed := range []bool{false, true} {
			t.Run(fmt.Sprintf("empty%v/managed%v", empty, managed), func(t *testing.T) {
				c := &cudaBackend{budgetBytes: 1000, dlUsed: 77, managedN: 3}
				shape := []int{2, 32}
				nScales := 2
				if empty {
					shape[0] = 0
					nScales = 0
				}
				var owners []*cudaBuf
				var storage [][]byte
				var labels []string
				releases := 0
				alloc := func(bytes int, label string) *cudaBuf {
					n := bytes
					if n == 0 {
						n = 1
					}
					b := make([]byte, n)
					storage = append(storage, b)
					owner := &cudaBuf{ptr: unsafe.Pointer(&b[0]), n: bytes, managed: managed, class: MemoryWeights}
					owners = append(owners, owner)
					labels = append(labels, label)
					return owner
				}
				tensor, buf := c.devQ2WithOperations(shape, 32, nScales, func(bytes int) *cudaBuf {
					owner := alloc(bytes, "codes")
					c.accountWeightPlacement(owner, bytes)
					return owner
				}, func(bytes int) *cudaBuf { return alloc(bytes, "scales") }, func(unsafe.Pointer) { releases++ })
				if releases != 0 || !reflect.DeepEqual(labels, []string{"codes", "scales"}) || len(storage) != 2 {
					t.Fatal("allocation/publish order mismatch")
				}
				if buf != owners[0] || tensor.buf != buf || buf.ptr != owners[0].ptr || buf.scales != owners[1].ptr || buf.scalesN != nScales*4 || buf.n != shape[0]*shape[1]/4 {
					t.Fatal("code/scale ownership mismatch")
				}
				if tensor.Dtype != Q2_0 || tensor.Layout != RowMajor || !reflect.DeepEqual(tensor.Shape, shape) || tensor.Quant.Block != 32 || tensor.Quant.Axis != 2 || tensor.Quant.Bits != 2 || !tensor.Quant.Symmetric {
					t.Fatal("Q2 layout changed")
				}
				wantDL, wantManaged := int64(77+shape[0]*shape[1]/4), 3
				if managed {
					wantDL = 77
					wantManaged++
				}
				if c.dlUsed != wantDL || c.managedN != wantManaged {
					t.Fatalf("placement changed: %d/%d", c.dlUsed, c.managedN)
				}
				if owners[1].budgetedWeightBytes != 0 || owners[1].managedWeight {
					t.Fatal("scale allocation gained a placement charge")
				}
				shape[0] = 99
				if tensor.Shape[0] == 99 {
					t.Fatal("shape alias")
				}
			})
		}
	}
}

// Pin the allocation transaction to the real packed upload owner and native leaves.
// fak-test:runtime fast est=10ms lane=default
func TestCUDADevQ2NativeAllocationBinding(t *testing.T) {
	raw, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "cuda.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	bodies := map[string]string{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Body != nil {
			bodies[fn.Name.Name] = string(raw[set.Position(fn.Body.Pos()).Offset:set.Position(fn.Body.End()).Offset])
		}
	}
	for _, want := range []string{"c.devQ2WithOperations(", "c.dallocWeight,", "c.dallocClass(bytes, MemoryWeights, \"q2-scale\")", "C.fcuda_free(ptr)"} {
		if !strings.Contains(bodies["devQ2"], want) {
			t.Errorf("native allocation binding missing %q", want)
		}
	}
	for _, name := range []string{"uploadQ2Resident"} {
		if !strings.Contains(bodies[name], "c.devQ2(") {
			t.Errorf("actual caller %s disconnected", name)
		}
	}
	if !strings.Contains(bodies["uploadClass"], "return c.uploadQ2Resident(t, hb)") || !strings.Contains(bodies["uploadClass"], "cudaMu.Lock()") || !strings.Contains(bodies["uploadClass"], "defer cudaMu.Unlock()") {
		t.Fatal("upload owner no longer serialized")
	}
	for _, forbidden := range []string{".Free(", "cudaMu.Lock()", "uploadCache", "fcuda_h2d"} {
		if strings.Contains(bodies["devQ2WithOperations"], forbidden) {
			t.Errorf("allocation transaction exceeds scope: %s", forbidden)
		}
	}
}
