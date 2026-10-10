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

// The scoped cache entry is synthetic and has no device allocation. Cleanup
// restores only its own key; the production Upload still acquires cudaMu.
func cudaQ8WidthCache(t *testing.T, c *cudaBackend, host []float32, shape []int) Tensor {
	t.Helper()
	key := ucKey{uintptr(unsafe.Pointer(&host[0])), Q8_0, RowMajor}
	cached := makeTensor(c, Q8_0, RowMajor, append([]int(nil), shape...), &QuantSpec{Block: 32, Axis: 2, Bits: 8, Symmetric: true}, &cudaBuf{})
	cudaMu.Lock()
	old, exists := uploadCache[key]
	uploadCache[key] = cached
	cudaMu.Unlock()
	t.Cleanup(func() {
		cudaMu.Lock()
		defer cudaMu.Unlock()
		if exists {
			uploadCache[key] = old
		} else {
			delete(uploadCache, key)
		}
	})
	return cached
}

// fak-test:runtime fast est=1ms lane=default
// Unmeasured estimate. Every input is rejected before a native operation.
func TestCUDAF32Q8RejectsIncompleteRowBlocks(t *testing.T) {
	for _, shape := range [][]int{{1, 1}, {2, 31}, {1, 33}, {2, 63}, {8, 12}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			c := &cudaBackend{dlUsed: 17, managedN: 2}
			host := make([]float32, shape[0]*shape[1])
			host[len(host)-1] = 1
			if shape[0] == 8 {
				cudaQ8WidthCache(t, c, host, []int{1, 96})
			}
			var got any
			func() { defer func() { got = recover() }(); c.Upload(NewF32(nil, shape, host), Q8_0) }()
			err, ok := got.(*CUDAOpError)
			if !ok || err.Op != "Upload" || err.Site != "upload-q8" || err.Class != MemoryWeights || !strings.Contains(err.Msg, "divisible by 32") {
				t.Fatalf("wrong refusal=%#v", got)
			}
			if c.dlUsed != 17 || c.managedN != 2 || c.immutableWeightUploadCalls != 0 {
				t.Fatal("invalid width changed placement/upload accounting")
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
// Unmeasured estimate. Valid-width cache returns avoid native CUDA execution.
func TestCUDAF32Q8AlignedCachedWidthsRemainAdmitted(t *testing.T) {
	for _, shape := range [][]int{{1, 32}, {2, 64}, {1, 96}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			c := &cudaBackend{}
			host := make([]float32, shape[0]*shape[1])
			cached := cudaQ8WidthCache(t, c, host, shape)
			got := c.Upload(NewF32(nil, shape, host), Q8_0)
			if got.buf != cached.buf || got.Dtype != Q8_0 || got.Numel() != len(host) {
				t.Fatal("aligned cached upload changed")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
// Unmeasured estimate. AST checks constrain the guard's actual cache/native order
// and its explicit empty-input exception; no zero-sized native allocation is run.
func TestCUDAQ8WidthGuardPrecedesCacheAndPreservesEmptyPolicy(t *testing.T) {
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
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
			bodies[fn.Name.Name] = string(raw[fs.Position(fn.Body.Pos()).Offset:fs.Position(fn.Body.End()).Offset])
		}
	}
	body := bodies["uploadClass"]
	guard := "store == Q8_0 && len(f) > 0 && len(t.Shape) == 2 && t.Shape[1]%q8DeviceBlock != 0"
	at := strings.Index(body, guard)
	cache := strings.Index(body, "uploadCache[")
	call := strings.Index(body, "return c.uploadQ8(t, hb, f, hp)")
	if at < 0 || cache <= at || call <= cache {
		t.Fatal("width guard no longer precedes cache and native narrowing")
	}
	if !strings.Contains(bodies["uploadQ8"], "if len(f) == 0") || !strings.Contains(bodies["uploadQ8"], "c.devQ8(t.Shape, blk, out*nblk)") {
		t.Fatal("empty-host path changed")
	}
	// The documented defect is not approximation noise: a nonzero tail beyond the
	// complete-block prefix is never visited, so its exact dot contribution is lost.
	width := 33
	weight := make([]float32, width)
	input := make([]float32, width)
	weight[32], input[32] = 1, 1
	var full, prefix float32
	for i := range weight {
		full += weight[i] * input[i]
	}
	for i := 0; i < (width/32)*32; i++ {
		prefix += weight[i] * input[i]
	}
	if full != 1 || prefix != 0 {
		t.Fatal("nonaligned-tail discriminator became vacuous")
	}
}
