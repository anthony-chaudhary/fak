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

func cudaCloneOwnershipSource() (*cudaKV, [][]float32) {
	backing := make([][]float32, 6)
	k := &cudaKV{cfg: KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2}, K: make([]dslice, 2), Kraw: make([]dslice, 2), V: make([]dslice, 2), pos: []int{4, 7}}
	for i := range backing {
		backing[i] = []float32{float32(i), 1, 2, 3, 4, 5}
		row := dslice{ptr: unsafe.Pointer(&backing[i][0]), len: 4, cap: 6}
		switch i % 3 {
		case 0:
			k.K[i/3] = row
		case 1:
			k.Kraw[i/3] = row
		case 2:
			k.V[i/3] = row
		}
	}
	return k, backing
}

// Synthetic per-call callbacks only; no native allocation, copy, or release.
// Estimate only; not timed.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAKVCloneUnwindsAllocationPanic(t *testing.T) {
	for _, failAt := range []int{1, 2, 6} {
		for _, unknown := range []bool{false, true} {
			t.Run(fmt.Sprintf("allocation%d/unknown%v", failAt, unknown), func(t *testing.T) {
				source, sourceBacking := cudaCloneOwnershipSource()
				beforeK, beforeRaw, beforeV := append([]dslice(nil), source.K...), append([]dslice(nil), source.Kraw...), append([]dslice(nil), source.V...)
				beforePos := append([]int(nil), source.pos...)
				var primary any = &DeviceAllocError{Bytes: 16, Site: "synthetic clone", Class: MemoryKVCache}
				if unknown {
					primary = &struct{ marker string }{"original"}
				}
				var storage [][]float32
				owned := map[unsafe.Pointer]bool{}
				released := map[unsafe.Pointer]int{}
				attempts, copies := 0, 0
				var got any
				func() {
					defer func() { got = recover() }()
					source.cloneWithOperations(func(bytes int, site string) unsafe.Pointer {
						attempts++
						if bytes != 16 || !strings.Contains(site, "clone layer ") {
							t.Errorf("allocation contract: %d %q", bytes, site)
						}
						if attempts == failAt {
							panic(primary)
						}
						b := make([]float32, 4)
						storage = append(storage, b)
						ptr := unsafe.Pointer(&b[0])
						owned[ptr] = true
						return ptr
					}, func(dst, src unsafe.Pointer, bytes int) {
						if !owned[dst] || src == nil || bytes != 16 {
							t.Error("invalid copy callback")
						}
						copies++
					}, func(ptr unsafe.Pointer) {
						if !owned[ptr] {
							t.Error("released source or unknown pointer")
						}
						released[ptr]++
						// Every cleanup fails, so later owned rows must still be attempted.
						panic("secondary cleanup")
					})
					t.Error("allocation failure returned a clone")
				}()
				if got != primary {
					t.Fatalf("primary identity lost: got %v want %v", got, primary)
				}
				if attempts != failAt || copies != failAt-1 || len(released) != failAt-1 {
					t.Fatalf("attempts/copies/releases = %d/%d/%d", attempts, copies, len(released))
				}
				for ptr := range owned {
					if released[ptr] != 1 {
						t.Errorf("owned pointer release count = %d", released[ptr])
					}
				}
				if !reflect.DeepEqual(source.K, beforeK) || !reflect.DeepEqual(source.Kraw, beforeRaw) || !reflect.DeepEqual(source.V, beforeV) || !reflect.DeepEqual(source.pos, beforePos) {
					t.Fatal("source metadata changed")
				}
				if len(sourceBacking) != 6 || len(storage) != failAt-1 {
					t.Fatal("synthetic backing ownership changed")
				}
			})
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAKVCloneSuccessfulOwnershipAndEmptyRows(t *testing.T) {
	source, backing := cudaCloneOwnershipSource()
	source.Kraw[0].len = 0
	var storage [][]float32
	var sites []string
	var sources []unsafe.Pointer
	releases := 0
	clone := source.cloneWithOperations(func(bytes int, site string) unsafe.Pointer {
		sites = append(sites, site)
		b := make([]float32, bytes/4)
		storage = append(storage, b)
		return unsafe.Pointer(&b[0])
	}, func(dst, src unsafe.Pointer, bytes int) {
		sources = append(sources, src)
		copy(unsafe.Slice((*float32)(dst), bytes/4), unsafe.Slice((*float32)(src), bytes/4))
	}, func(unsafe.Pointer) { releases++ })
	expectedSites := []string{"kv-key-clone layer 0", "kv-value-clone layer 0", "kv-key-clone layer 1", "kv-pre-rope-key-clone layer 1", "kv-value-clone layer 1"}
	expectedSources := []unsafe.Pointer{source.K[0].ptr, source.V[0].ptr, source.K[1].ptr, source.Kraw[1].ptr, source.V[1].ptr}
	if !reflect.DeepEqual(sites, expectedSites) || !reflect.DeepEqual(sources, expectedSources) || releases != 0 {
		t.Fatalf("successful copy order/cleanup: %v %v %d", sites, sources, releases)
	}
	if clone.Kraw[0] != (dslice{}) || !reflect.DeepEqual(clone.cfg, source.cfg) || clone.be != source.be || !reflect.DeepEqual(clone.pos, source.pos) {
		t.Fatal("clone metadata mismatch")
	}
	for _, pair := range [][2][]dslice{{source.K, clone.K}, {source.Kraw, clone.Kraw}, {source.V, clone.V}} {
		for i, src := range pair[0] {
			dst := pair[1][i]
			if src.len == 0 {
				continue
			}
			if dst.ptr == src.ptr || dst.len != src.len || dst.cap != src.len || !reflect.DeepEqual(unsafe.Slice((*float32)(dst.ptr), dst.len), unsafe.Slice((*float32)(src.ptr), src.len)) {
				t.Fatal("clone payload/ownership mismatch")
			}
		}
	}
	clone.pos[0] = 99
	if source.pos[0] != 4 || len(storage) != 5 || len(backing) != 6 {
		t.Fatal("source position alias or backing mismatch")
	}
	empty := (&cudaKV{}).cloneWithOperations(func(int, string) unsafe.Pointer { t.Fatal("empty allocation"); return nil }, func(unsafe.Pointer, unsafe.Pointer, int) { t.Fatal("empty copy") }, func(unsafe.Pointer) { t.Fatal("empty release") })
	if empty == nil || len(empty.pos) != 0 {
		t.Fatal("empty clone mismatch")
	}
}

// Pin the synthetic seam to the public, serialized native owner. No CUDA calls.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVCloneProductionOperationBinding(t *testing.T) {
	raw, err := os.ReadFile("cuda_kv.go")
	if err != nil {
		t.Fatal(err)
	}
	set := token.NewFileSet()
	parsed, err := parser.ParseFile(set, "cuda_kv.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, decl := range parsed.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "Clone" || fn.Recv == nil {
			continue
		}
		ptr, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		owner, ok := ptr.X.(*ast.Ident)
		if !ok || owner.Name != "cudaKV" {
			continue
		}
		body = string(raw[set.Position(fn.Body.Pos()).Offset:set.Position(fn.Body.End()).Offset])
	}
	for _, want := range []string{"cudaMu.Lock()", "defer cudaMu.Unlock()", "return k.cloneWithOperations(", "k.be.dallocKV(bytes, site).ptr", "C.fcuda_d2d(dst, src, C.size_t(bytes))", "C.fcuda_free(ptr)"} {
		if !strings.Contains(body, want) {
			t.Errorf("public Clone missing binding %q", want)
		}
	}
	lock, call := strings.Index(body, "cudaMu.Lock()"), strings.Index(body, "return k.cloneWithOperations(")
	if lock < 0 || call <= lock || strings.Contains(body, ".Free()") {
		t.Fatal("Clone lost serialization or recursively frees")
	}
}
