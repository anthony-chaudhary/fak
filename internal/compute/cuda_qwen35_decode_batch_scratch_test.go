package compute

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"testing"
)

const cudaQwen35ScratchFixtureLayers = 48

var cudaQwen35ScratchFixtureChecksum int

// Preserve the original non-inlined CUDA helper's dynamic container sizes.
// Device allocation is replaced with supplied values; *byte substitutes only
// the opaque *cudaBuf pointer. This measures host containers, not CUDA execution.
//
//go:noinline
func legacyCUDAQwen35DecodeBatchContainers(allocations []qwen35GDNAllocation, values []Tensor, owners []*byte) (Tensor, *byte) {
	tensors := make([]Tensor, 0, len(allocations))
	buffers := make([]*byte, 0, len(allocations))
	for i := range allocations {
		tensors = append(tensors, values[i])
		buffers = append(buffers, owners[i])
	}
	return tensors[8], buffers[8]
}

func cudaQwen35DecodeBatchScratchFixture(lanes int) ([]qwen35GDNAllocation, []Tensor, []*byte) {
	allocations := qwen35GDNAllocations("gdn-batch-", "-", lanes, lanes, 8, 4, 8, 2, 16)
	values := make([]Tensor, len(allocations))
	owners := make([]*byte, len(allocations))
	for i, allocation := range allocations {
		values[i] = Tensor{Dtype: F32, Layout: RowMajor, Shape: allocation.shape}
		owners[i] = new(byte)
		*owners[i] = byte(i)
	}
	return allocations, values, owners
}

func legacyCUDAQwen35DecodeBatchScratchRequest(allocations []qwen35GDNAllocation, values []Tensor, owners []*byte) int {
	checksum := 0
	for layer := 0; layer < cudaQwen35ScratchFixtureLayers; layer++ {
		output, owner := legacyCUDAQwen35DecodeBatchContainers(allocations, values, owners)
		checksum += output.Shape[0] + int(*owner)
	}
	return checksum
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDAQwen35DecodeBatchScratchLegacyMetric(t *testing.T) {
	for _, lanes := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("lanes-%d", lanes), func(t *testing.T) {
			allocations, values, owners := cudaQwen35DecodeBatchScratchFixture(lanes)
			if len(allocations) != 9 {
				t.Fatalf("GDN scratch recipe has %d entries, want nine", len(allocations))
			}
			got := testing.AllocsPerRun(20, func() {
				cudaQwen35ScratchFixtureChecksum = legacyCUDAQwen35DecodeBatchScratchRequest(allocations, values, owners)
			})
			if got <= 0 {
				t.Fatalf("legacy container allocation baseline must be positive, got %g", got)
			}
			if want := cudaQwen35ScratchFixtureLayers * (lanes + 8); cudaQwen35ScratchFixtureChecksum != want {
				t.Fatalf("legacy consumer checksum=%d, want %d", cudaQwen35ScratchFixtureChecksum, want)
			}
			t.Logf("cohort=%d layers=%d entries=9 legacy_container_allocs_per_request=%g fresh_request=true pointer_type=*byte software_only=true", lanes, cudaQwen35ScratchFixtureLayers, got)
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDAQwen35DecodeBatchScratchStorage(t *testing.T) {
	allocations, values, owners := cudaQwen35DecodeBatchScratchFixture(2)
	var scratch qwen35DecodeBatchScratch[*byte]
	for _, prefix := range []int{9, 1, 4, 0} {
		tensors, buffers := scratch.prepare(len(allocations))
		if len(tensors) != 0 || len(buffers) != 0 || cap(tensors) != 9 || cap(buffers) != 9 {
			t.Fatal("scratch must provide empty nine-entry containers")
		}
		for i := 0; i < prefix; i++ {
			tensors = append(tensors, values[i])
			buffers = append(buffers, owners[i])
		}
		var output Tensor
		var owner *byte
		if prefix > 0 {
			output, owner = tensors[prefix-1], buffers[prefix-1]
		}
		scratch.reset()
		for i := range scratch.tensors {
			if !reflect.DeepEqual(scratch.tensors[i], Tensor{}) || scratch.buffers[i] != nil {
				t.Fatalf("prefix %d retained owner at index %d", prefix, i)
			}
		}
		if prefix > 0 && (output.Shape[0] != 2 || owner != owners[prefix-1]) {
			t.Fatal("reset changed a copied output value or its buffer owner")
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDAQwen35DecodeBatchScratchRequestIsolation(t *testing.T) {
	_, aValues, aOwners := cudaQwen35DecodeBatchScratchFixture(2)
	_, bValues, bOwners := cudaQwen35DecodeBatchScratchFixture(4)
	var a, b qwen35DecodeBatchScratch[*byte]
	aTensors, aBuffers := a.prepare(9)
	aTensors, aBuffers = append(aTensors, aValues...), append(aBuffers, aOwners...)
	bTensors, bBuffers := b.prepare(9)
	bTensors, bBuffers = append(bTensors, bValues...), append(bBuffers, bOwners...)
	if &aTensors[0] == &bTensors[0] || &aBuffers[0] == &bBuffers[0] {
		t.Fatal("independent requests share scratch storage")
	}
	a.reset()
	if bTensors[8].Shape[0] != 4 || bBuffers[8] != bOwners[8] {
		t.Fatal("resetting one request changed another request's containers")
	}
	b.reset()
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDAQwen35DecodeBatchScratchFallback(t *testing.T) {
	for _, count := range []int{8, 10} {
		var scratch qwen35DecodeBatchScratch[*byte]
		first, firstOwners := scratch.prepare(count)
		second, secondOwners := scratch.prepare(count)
		if len(first) != 0 || len(firstOwners) != 0 || cap(first) != count || cap(firstOwners) != count {
			t.Fatalf("count %d lost the ordinary allocation fallback", count)
		}
		first, second = append(first, Tensor{Shape: []int{2}}), append(second, Tensor{Shape: []int{4}})
		firstOwners, secondOwners = append(firstOwners, new(byte)), append(secondOwners, new(byte))
		if &first[0] == &second[0] || &firstOwners[0] == &secondOwners[0] || first[0].Shape[0] != 2 {
			t.Fatalf("count %d fallback containers alias", count)
		}
		scratch.reset()
		if first[0].Shape[0] != 2 || second[0].Shape[0] != 4 {
			t.Fatalf("count %d reset changed independently allocated fallback data", count)
		}
	}
}

// Match the real layer helper's call/return boundary and deferred cleanup.
//
//go:noinline
func candidateCUDAQwen35DecodeBatchContainers(allocations []qwen35GDNAllocation, values []Tensor, owners []*byte, scratch *qwen35DecodeBatchScratch[*byte]) (Tensor, *byte) {
	tensors, buffers := scratch.prepare(len(allocations))
	defer scratch.reset()
	for i := range allocations {
		tensors = append(tensors, values[i])
		buffers = append(buffers, owners[i])
	}
	return tensors[8], buffers[8]
}

func candidateCUDAQwen35DecodeBatchScratchRequest(allocations []qwen35GDNAllocation, values []Tensor, owners []*byte) int {
	var scratch qwen35DecodeBatchScratch[*byte]
	checksum := 0
	for layer := 0; layer < cudaQwen35ScratchFixtureLayers; layer++ {
		output, owner := candidateCUDAQwen35DecodeBatchContainers(allocations, values, owners, &scratch)
		checksum += output.Shape[0] + int(*owner)
	}
	return checksum
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDAQwen35DecodeBatchScratchMetric(t *testing.T) {
	for _, lanes := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("lanes-%d", lanes), func(t *testing.T) {
			allocations, values, owners := cudaQwen35DecodeBatchScratchFixture(lanes)
			before := testing.AllocsPerRun(20, func() {
				cudaQwen35ScratchFixtureChecksum = legacyCUDAQwen35DecodeBatchScratchRequest(allocations, values, owners)
			})
			after := testing.AllocsPerRun(20, func() {
				cudaQwen35ScratchFixtureChecksum = candidateCUDAQwen35DecodeBatchScratchRequest(allocations, values, owners)
			})
			if before <= 0 || after > before/2 {
				t.Fatalf("container allocations/request before=%g after=%g; need a positive baseline and at least 50%% reduction", before, after)
			}
			if want := cudaQwen35ScratchFixtureLayers * (lanes + 8); cudaQwen35ScratchFixtureChecksum != want {
				t.Fatalf("candidate consumer checksum=%d, want %d", cudaQwen35ScratchFixtureChecksum, want)
			}
			t.Logf("cohort=%d layers=%d entries=9 legacy_container_allocs_per_request=%g candidate_container_allocs_per_request=%g reduction_pct=%g fresh_request=true pointer_type=*byte software_only=true", lanes, cudaQwen35ScratchFixtureLayers, before, after, 100*(before-after)/before)
		})
	}
}

// fak-test:justify why=contract when=changed:internal/compute/cuda_qwen35_decode_batch.go twin=TestCUDAQwen35DecodeBatchScratchStorage
// fak-test:runtime fast est=100ms lane=default
func TestCUDAQwen35DecodeBatchScratchCallerWiring(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "cuda_qwen35_decode_batch.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var caller, layer *ast.FuncDecl
	for _, declaration := range file.Decls {
		fn, ok := declaration.(*ast.FuncDecl)
		if !ok {
			continue
		}
		switch fn.Name.Name {
		case "Qwen35IndependentDecodeBatch":
			caller = fn
		case "qwen35BatchGDNLocked":
			layer = fn
		}
	}
	if caller == nil || layer == nil {
		t.Fatal("CUDA batch entrypoint or layer helper missing")
	}
	localOwner, sharedAcrossLayers := false, false
	ast.Inspect(caller.Body, func(node ast.Node) bool {
		if value, ok := node.(*ast.ValueSpec); ok && len(value.Names) == 1 && value.Names[0].Name == "batchScratch" {
			if generic, ok := value.Type.(*ast.IndexExpr); ok {
				name, ok := generic.X.(*ast.Ident)
				localOwner = ok && name.Name == "qwen35DecodeBatchScratch"
			}
		}
		call, ok := node.(*ast.CallExpr)
		if !ok || len(call.Args) != 5 {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "qwen35BatchGDNLocked" {
			return true
		}
		address, ok := call.Args[4].(*ast.UnaryExpr)
		if ok && address.Op == token.AND {
			name, ok := address.X.(*ast.Ident)
			sharedAcrossLayers = ok && name.Name == "batchScratch"
		}
		return true
	})
	prepared, resetOnReturn := false, false
	ast.Inspect(layer.Body, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok {
			if selector, ok := call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == "prepare" {
				name, ok := selector.X.(*ast.Ident)
				prepared = ok && name.Name == "scratch"
			}
		}
		if deferred, ok := node.(*ast.DeferStmt); ok {
			if selector, ok := deferred.Call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == "reset" {
				name, ok := selector.X.(*ast.Ident)
				resetOnReturn = ok && name.Name == "scratch"
			}
		}
		return true
	})
	if !localOwner || !sharedAcrossLayers || !prepared || !resetOnReturn {
		t.Fatalf("CUDA caller storage wiring: local=%t passed=%t prepared=%t deferred_reset=%t", localOwner, sharedAcrossLayers, prepared, resetOnReturn)
	}
}
