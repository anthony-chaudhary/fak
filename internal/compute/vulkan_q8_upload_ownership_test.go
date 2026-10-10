//go:build vulkan && (windows || linux) && cgo

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

// Synthetic per-call operations; no Vulkan initialization or native operations.
// Estimate only; not timed.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanQ8UploadUnwindsAllocationsAndPlacement(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		last := 2
		if chunked {
			last = 6
		}
		for failAt := 1; failAt <= last; failAt++ {
			for _, hostvis := range []bool{false, true} {
				t.Run(fmt.Sprintf("chunked%v/fail%d/mixedHost%v", chunked, failAt, hostvis), func(t *testing.T) {
					v := &vulkanBackend{haveQ8: true, dlUsed: 77, hostvisN: 3}
					if chunked {
						v.maxBufferBytes = 32
					}
					codes := make([]int8, 96)
					scales := []float32{1, 2, 3}
					var primary any = &DeviceAllocError{Bytes: 4, Site: "synthetic Q8", Class: MemoryWeights}
					if failAt%2 == 1 {
						primary = &struct{ cause string }{"unknown primary"}
					}
					attempts, uploads := 0, 0
					owners := map[unsafe.Pointer]*vulkanBuf{}
					released := map[unsafe.Pointer]int{}
					var backing [][]byte
					var got any
					func() {
						defer func() { got = recover() }()
						v.uploadQ8WithOperations([]int{3, 32}, codes, scales, 32, func(bytes int, site string) *vulkanBuf {
							attempts++
							if attempts == failAt {
								panic(primary)
							}
							b := make([]byte, bytes)
							backing = append(backing, b)
							ptr := unsafe.Pointer(&b[0])
							owner := &vulkanBuf{ptr: ptr, n: bytes, class: MemoryWeights}
							if hostvis && attempts%2 == 0 {
								owner.hostVisibleWeight = true
								v.hostvisN++
							} else {
								owner.budgetedWeightBytes = int64(bytes)
								v.dlUsed += int64(bytes)
							}
							owners[ptr] = owner
							return owner
						}, func(dst, src unsafe.Pointer, bytes int) {
							if owners[dst] == nil || src == nil || bytes <= 0 {
								t.Error("invalid upload")
							}
							uploads++
						}, func(ptr unsafe.Pointer) {
							owner := owners[ptr]
							if owner == nil {
								t.Error("unknown release")
							} else if owner.ptr != nil || owner.budgetedWeightBytes != 0 || owner.hostVisibleWeight {
								t.Error("owner not detached/decharged before release")
							}
							released[ptr]++
							panic("secondary release")
						})
						t.Error("failed upload returned")
					}()
					if got != primary {
						t.Fatalf("primary identity lost: %v", got)
					}
					if attempts != failAt || len(released) != failAt-1 || len(backing) != failAt-1 {
						t.Fatalf("attempts/releases=%d/%d", attempts, len(released))
					}
					for ptr := range owners {
						if released[ptr] != 1 {
							t.Error("allocation not released once")
						}
					}
					if v.dlUsed != 77 || v.hostvisN != 3 {
						t.Fatalf("placement not restored: %d/%d", v.dlUsed, v.hostvisN)
					}
					expectedUploads := 0
					if chunked {
						expectedUploads = ((failAt - 1) / 2) * 2
					}
					if uploads != expectedUploads {
						t.Fatalf("upload order changed: got %d want %d", uploads, expectedUploads)
					}
					if !reflect.DeepEqual(scales, []float32{1, 2, 3}) || !reflect.DeepEqual(codes, make([]int8, 96)) {
						t.Error("host payload changed")
					}
				})
			}
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestVulkanQ8UploadSuccessKeepsPlacementAndLayout(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		t.Run(fmt.Sprint(chunked), func(t *testing.T) {
			v := &vulkanBackend{haveQ8: true, dlUsed: 77, hostvisN: 3}
			if chunked {
				v.maxBufferBytes = 32
			}
			codes := make([]int8, 96)
			for i := range codes {
				codes[i] = int8(i)
			}
			scales := []float32{1, 2, 3}
			var owners []*vulkanBuf
			var backing [][]byte
			var events []string
			var uploadSizes []int
			var copied [][]byte
			released := 0
			out := v.uploadQ8WithOperations([]int{3, 32}, codes, scales, 32, func(bytes int, site string) *vulkanBuf {
				b := make([]byte, bytes)
				backing = append(backing, b)
				owner := &vulkanBuf{ptr: unsafe.Pointer(&b[0]), n: bytes, class: MemoryWeights, budgetedWeightBytes: int64(bytes)}
				v.dlUsed += int64(bytes)
				owners = append(owners, owner)
				events = append(events, "alloc")
				return owner
			}, func(dst, src unsafe.Pointer, bytes int) {
				events = append(events, "upload")
				uploadSizes = append(uploadSizes, bytes)
				payload := append([]byte(nil), unsafe.Slice((*byte)(src), bytes)...)
				copied = append(copied, payload)
				copy(unsafe.Slice((*byte)(dst), bytes), payload)
			}, func(unsafe.Pointer) { released++ })
			buf := out.buf.(*vulkanBuf)
			if released != 0 || v.dlUsed != 185 || v.hostvisN != 3 || out.Dtype != Q8_0 || !reflect.DeepEqual(out.Shape, []int{3, 32}) || out.Quant.Block != 32 {
				t.Fatal("successful ownership/placement changed")
			}
			for _, owner := range owners {
				if owner.ptr == nil || owner.budgetedWeightBytes != int64(owner.n) {
					t.Fatal("successful original allocation was retired")
				}
			}
			if chunked {
				if len(buf.q8Chunks) != 3 || !reflect.DeepEqual(uploadSizes, []int{32, 4, 32, 4, 32, 4}) {
					t.Fatal("chunk layout changed")
				}
				for i, c := range buf.q8Chunks {
					if c.rowStart != i || c.rows != 1 || c.ptr != owners[2*i].ptr || c.scalePtr != owners[2*i+1].ptr {
						t.Fatal("chunk handles changed")
					}
				}
			} else if buf.ptr != owners[0].ptr || buf.scalePtr != owners[1].ptr || buf.scaleBudgetedBytes != 12 || !reflect.DeepEqual(uploadSizes, []int{96, 12}) {
				t.Fatal("single layout changed")
			}
			pairs := 1
			if chunked {
				pairs = 3
			}
			wantEvents := []string{}
			for i := 0; i < pairs; i++ {
				wantEvents = append(wantEvents, "alloc", "alloc", "upload", "upload")
			}
			if !reflect.DeepEqual(events, wantEvents) {
				t.Fatalf("upload sequencing: %v", events)
			}
			if len(copied) != len(backing) {
				t.Fatal("copy ownership count mismatch")
			}
			for i := range copied {
				if !reflect.DeepEqual(copied[i], backing[i]) {
					t.Fatal("uploaded bytes changed")
				}
			}
		})
	}
}

// Exact public helper-to-native binding, with no Vulkan call at test time.
// fak-test:runtime fast est=10ms lane=default
func TestVulkanQ8UploadNativeOperationBinding(t *testing.T) {
	raw, err := os.ReadFile("vulkan.go")
	if err != nil {
		t.Fatal(err)
	}
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "vulkan.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Name.Name == "uploadQ8Locked" && fn.Body != nil {
			body = string(raw[set.Position(fn.Body.Pos()).Offset:set.Position(fn.Body.End()).Offset])
		}
	}
	for _, want := range []string{"v.uploadQ8WithOperations(", "v.dallocWeightFor,", "C.fvk_h2d(dst, src, C.size_t(bytes))", "C.fvk_free(ptr)"} {
		if !strings.Contains(body, want) {
			t.Errorf("missing native binding %q", want)
		}
	}
	for _, forbidden := range []string{".Free(", "batchFlush", "batchBegin", "submission"} {
		if strings.Contains(body, forbidden) {
			t.Errorf("wrapper changes native retirement/status policy: %q", forbidden)
		}
	}
}

// Successful chunk publication transfers both code and scale placement charges
// into the chunk owner. Native retirement is replaced only by a per-call seam.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanQ8ChunkFreeRestoresScalePlacement(t *testing.T) {
	for _, placement := range []string{"device", "host", "host-scales", "host-codes"} {
		t.Run(placement, func(t *testing.T) {
			v := &vulkanBackend{haveQ8: true, maxBufferBytes: 32, dlUsed: 77, hostvisN: 3}
			var owners []*vulkanBuf
			var storage [][]byte
			out := v.uploadQ8WithOperations([]int{3, 32}, make([]int8, 96), []float32{1, 2, 3}, 32, func(bytes int, site string) *vulkanBuf {
				data := make([]byte, bytes)
				storage = append(storage, data)
				owner := &vulkanBuf{ptr: unsafe.Pointer(&data[0]), n: bytes, class: MemoryWeights}
				scale := len(owners)%2 == 1
				host := placement == "host" || placement == "host-scales" && scale || placement == "host-codes" && !scale
				if host {
					owner.hostVisibleWeight = true
					v.hostvisN++
				} else {
					owner.budgetedWeightBytes = int64(bytes)
					v.dlUsed += int64(bytes)
				}
				owners = append(owners, owner)
				return owner
			}, func(unsafe.Pointer, unsafe.Pointer, int) {}, func(unsafe.Pointer) { t.Fatal("successful upload unwound") })
			db := out.buf.(*vulkanBuf)
			rows := db.q8Chunks
			if len(rows) != 3 || len(owners) != 6 {
				t.Fatal("fixture did not chunk")
			}
			for i, row := range rows {
				code, scale := owners[2*i], owners[2*i+1]
				if row.budgetedWeightBytes != code.budgetedWeightBytes || row.hostVisibleWeight != code.hostVisibleWeight || row.scaleBudgetedBytes != scale.budgetedWeightBytes || row.scaleHostVisible != scale.hostVisibleWeight {
					t.Fatal("published chunk lost placement metadata")
				}
			}
			var released []unsafe.Pointer
			v.freeQ8ChunksWithOperation(db, func(ptr unsafe.Pointer) { released = append(released, ptr) })
			expected := []unsafe.Pointer{}
			for i := 0; i < 3; i++ {
				expected = append(expected, owners[2*i+1].ptr, owners[2*i].ptr)
			}
			if !reflect.DeepEqual(released, expected) {
				t.Fatal("scale/code retirement order or identity changed")
			}
			if v.dlUsed != 77 || v.hostvisN != 3 {
				t.Fatalf("successful cleanup leaked charges: %d/%d", v.dlUsed, v.hostvisN)
			}
			if db.q8Chunks != nil {
				t.Fatal("chunk list retained")
			}
			for _, row := range rows {
				if row.ptr != nil || row.scalePtr != nil || row.n != 0 || row.scaleN != 0 || row.budgetedWeightBytes != 0 || row.hostVisibleWeight || row.scaleBudgetedBytes != 0 || row.scaleHostVisible {
					t.Fatal("retired chunk retained ownership or charges")
				}
			}
			v.freeQ8ChunksWithOperation(db, func(unsafe.Pointer) { t.Fatal("repeated successful cleanup released again") })
			if v.dlUsed != 77 || v.hostvisN != 3 || len(storage) != 6 {
				t.Fatal("repeat cleanup changed accounting")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVulkanQ8ChunkFreePublicNativeBinding(t *testing.T) {
	raw, err := os.ReadFile("vulkan.go")
	if err != nil {
		t.Fatal(err)
	}
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "vulkan.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Name.Name == "Free" && fn.Recv != nil && fn.Body != nil {
			body = string(raw[set.Position(fn.Body.Pos()).Offset:set.Position(fn.Body.End()).Offset])
		}
	}
	for _, want := range []string{"vulkanMu.Lock()", "defer vulkanMu.Unlock()", "v.freeQ8ChunksWithOperation(db, func(ptr unsafe.Pointer) { C.fvk_free(ptr) })", "if db.ptr == nil"} {
		if !strings.Contains(body, want) {
			t.Errorf("public Free lost binding %q", want)
		}
	}
	lock, cleanup := strings.Index(body, "vulkanMu.Lock()"), strings.Index(body, "v.freeQ8ChunksWithOperation(")
	if lock < 0 || cleanup <= lock || cleanup >= strings.Index(body, "if db.ptr == nil") {
		t.Fatal("chunk cleanup escaped serialized public owner")
	}
}
