package compute

import (
	"context"
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime slow est=60s lane=optin
// The estimate is the subprocess deadline, not a measured duration. This host
// witness executes the exact production cache-hit branches and Tensor.Numel.
// It does not execute upload admission, allocation, copies, frees, or CUDA. The
// fixtures satisfy the outer nonempty, rank-two, and complete-row preconditions.
func TestCUDAUploadCacheGeometryHost(t *testing.T) {
	assertCUDAUploadLockedBoundary(t)
	if os.Getenv("FAK_TEST_CUDA_UPLOAD_CACHE_GEOMETRY_HOST") != "1" {
		t.Skip("set FAK_TEST_CUDA_UPLOAD_CACHE_GEOMETRY_HOST=1 for the host cache witness")
	}
	var program strings.Builder
	program.WriteString(cudaUploadGeometryHostFixture)
	program.Write(cudaUploadGeometryType(t, "cuda_upload.go", "cudaQ8UploadSource"))
	program.WriteByte('\n')
	src, fn, fs := cudaUploadGeometryMethod(t, "compute.go", "Tensor", "Numel")
	program.Write(src[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
	program.WriteByte('\n')
	for _, route := range []struct{ file, method string }{
		{"cuda.go", "uploadClassLocked"},
		{"cuda_upload.go", "uploadQ2Resident"},
		{"cuda_upload.go", "uploadRawKQuant"},
	} {
		src, fn, fs := cudaUploadGeometryMethod(t, route.file, "*cudaBackend", route.method)
		var branch *ast.IfStmt
		ast.Inspect(fn.Body, func(node ast.Node) bool {
			stmt, ok := node.(*ast.IfStmt)
			if !ok {
				return true
			}
			init, ok := stmt.Init.(*ast.AssignStmt)
			if !ok || len(init.Lhs) != 2 || len(init.Rhs) != 1 {
				return true
			}
			lookup, ok := init.Rhs[0].(*ast.IndexExpr)
			if !ok {
				return true
			}
			cache, ok := lookup.X.(*ast.Ident)
			if !ok || cache.Name != "uploadCache" {
				return true
			}
			if branch != nil {
				t.Fatalf("%s has multiple cache-hit branches", route.method)
			}
			branch = stmt
			return true
		})
		if branch == nil || branch.Else != nil {
			t.Fatalf("%s cache-hit boundary changed", route.method)
		}
		returns := 0
		ast.Inspect(branch.Body, func(node ast.Node) bool {
			ret, ok := node.(*ast.ReturnStmt)
			if !ok {
				return true
			}
			returns++
			if len(ret.Results) != 1 {
				t.Fatalf("%s cache return changed", route.method)
			}
			cached, ok := ret.Results[0].(*ast.Ident)
			if !ok || cached.Name != "cached" {
				t.Fatalf("%s cache no longer returns cached", route.method)
			}
			return true
		})
		if returns != 1 || (route.method != "uploadQ2Resident" && len(branch.Body.List) != 1) {
			t.Fatalf("%s cache-return boundary changed", route.method)
		}
		body := src[fs.Position(branch.Pos()).Offset:fs.Position(branch.End()).Offset]
		t.Logf("%s/%s cache branch SHA256=%x", route.file, route.method, sha256.Sum256(body))
		prelude := ""
		if route.method == "uploadQ2Resident" {
			// Outer admission already validated these immutable descriptor fields.
			prelude = "out, in := t.Shape[0], t.Shape[1]\nblk, scales := t.Quant.Block, t.Quant.Scale\n"
		}
		fmt.Fprintf(&program, "func probe%s(t Tensor, store Dtype, hp uintptr) Tensor {\n%s%s\nreturn Tensor{}\n}\n", route.method, prelude, body)
	}
	dir := t.TempDir()
	probe := filepath.Join(dir, "cache_geometry_test.go")
	if err := os.WriteFile(probe, []byte(program.String()), 0600); err != nil {
		t.Fatal(err)
	}
	goExe := ""
	if root := runtime.GOROOT(); root != "" {
		goExe = filepath.Join(root, "bin", "go")
		if runtime.GOOS == "windows" {
			goExe += ".exe"
		}
	} else {
		var err error
		goExe, err = exec.LookPath("go")
		if err != nil {
			t.Fatalf("host cache witness needs Go on PATH: %v", err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", "^TestCacheGeometry$", probe)
	cmd.Dir = dir
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		switch key {
		case "GOWORK", "GOENV", "GOTOOLCHAIN", "CGO_ENABLED", "GOMAXPROCS", "GOFLAGS":
			continue
		}
		cmd.Env = append(cmd.Env, entry)
	}
	cmd.Env = append(cmd.Env, "GOWORK=off", "GOENV=off", "GOTOOLCHAIN=local", "CGO_ENABLED=0", "GOMAXPROCS=1", "GOFLAGS=")
	out, err := cmd.CombinedOutput()
	t.Logf("host cache witness (%s):\n%s", goExe, out)
	if ctx.Err() != nil {
		t.Fatalf("host cache witness deadline: %v", ctx.Err())
	}
	if err != nil {
		t.Fatalf("host cache witness failed: %v", err)
	}
}

func cudaUploadGeometryMethod(t *testing.T, path, receiver, name string) ([]byte, *ast.FuncDecl, *token.FileSet) {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, path, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var found *ast.FuncDecl
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != name || fn.Recv == nil || len(fn.Recv.List) != 1 || fn.Body == nil {
			continue
		}
		r := fn.Recv.List[0].Type
		actual := ""
		if ptr, ok := r.(*ast.StarExpr); ok {
			actual = "*"
			r = ptr.X
		}
		id, ok := r.(*ast.Ident)
		if !ok || actual+id.Name != receiver {
			continue
		}
		if found != nil {
			t.Fatalf("duplicate %s.%s", receiver, name)
		}
		found = fn
	}
	if found == nil {
		t.Fatalf("missing %s.%s in %s", receiver, name, path)
	}
	return src, found, fs
}

func cudaUploadGeometryType(t *testing.T, path, name string) []byte {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, path, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var found []byte
	for _, decl := range file.Decls {
		group, ok := decl.(*ast.GenDecl)
		if !ok || group.Tok != token.TYPE {
			continue
		}
		for _, spec := range group.Specs {
			typ, ok := spec.(*ast.TypeSpec)
			if !ok || typ.Name.Name != name {
				continue
			}
			if found != nil {
				t.Fatalf("duplicate %s in %s", name, path)
			}
			found = append([]byte("type "), src[fs.Position(typ.Pos()).Offset:fs.Position(typ.End()).Offset]...)
		}
	}
	if found == nil {
		t.Fatalf("missing %s in %s", name, path)
	}
	return found
}

const cudaUploadGeometryHostFixture = `package compute

import (
	"runtime"
	"slices"
	"testing"
	"unsafe"
)

// Only the surrounding types/storage are host fixtures. The lookup, predicate,
// cached return, and Numel method are extracted unchanged from production.
type Dtype uint8
const (F32 Dtype = iota; F16; Q8_0; Q2_0; Q4_K)
type Layout uint8
const (RowMajor Layout = iota; ColMajor)
type QuantSpec struct { Block int; Scale []float32 }
type Buffer interface { Ready() bool }
type HostBuffer interface { Buffer; F32() []float32; I8() []int8 }
type geometryHostBuffer struct { codes []int8 }
func (*geometryHostBuffer) Ready() bool { return true }
func (*geometryHostBuffer) F32() []float32 { return nil }
func (h *geometryHostBuffer) I8() []int8 { return h.codes }
type cudaBuf struct { hostKeep HostBuffer }
func (*cudaBuf) Ready() bool { return true }
type Tensor struct { Dtype Dtype; Layout Layout; Shape []int; Quant *QuantSpec; buf Buffer; owner int }
type ucKey struct { hp uintptr; dt Dtype; lo Layout }
var uploadCache map[ucKey]Tensor

func TestCacheGeometry(t *testing.T) {
	for _, tc := range []struct {
		name string
		probe func(Tensor, Dtype, uintptr) Tensor
		source, stored Dtype
		layout Layout
		first, second []int
	}{
		{"f32", probeuploadClassLocked, F32, F32, RowMajor, []int{2,64}, []int{4,32}},
		{"f16-row", probeuploadClassLocked, F32, F16, RowMajor, []int{2,64}, []int{4,32}},
		{"f16-col", probeuploadClassLocked, F32, F16, ColMajor, []int{2,64}, []int{4,32}},
		{"q8-from-f32", probeuploadClassLocked, F32, Q8_0, RowMajor, []int{2,64}, []int{4,32}},
		{"q2-resident", probeuploadQ2Resident, Q2_0, Q2_0, RowMajor, []int{2,64}, []int{4,32}},
		{"raw-k", probeuploadRawKQuant, Q4_K, Q4_K, RowMajor, []int{2,512}, []int{4,256}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Retain actual host storage; no host pointer is submitted to CUDA.
			storage := make([]int8, 4096)
			hp := uintptr(unsafe.Pointer(&storage[0]))
			defer runtime.KeepAlive(storage)
			input := Tensor{Dtype:tc.source, Layout:tc.layout, Shape:slices.Clone(tc.first)}
			resident := Tensor{Dtype:tc.stored, Layout:tc.layout, Shape:slices.Clone(tc.first), owner:7}
			if tc.stored == Q2_0 {
				scales := []float32{1,1,1,1}
				input.Quant = &QuantSpec{Block:32,Scale:scales}
				resident.Quant = &QuantSpec{Block:32}
				resident.buf = &cudaBuf{hostKeep:&cudaQ8UploadSource{
					HostBuffer:&geometryHostBuffer{codes:storage[:32]}, scales:scales,
					rows:tc.first[0], cols:tc.first[1], block:32,
				}}
			}
			key := ucKey{hp,tc.stored,tc.layout}
			uploadCache = map[ucKey]Tensor{key:resident}
			if got := tc.probe(input,tc.stored,hp); got.owner != resident.owner || !slices.Equal(got.Shape,input.Shape) {
				t.Fatal("identical geometry lost its cache hit")
			}
			input.Shape = slices.Clone(tc.second)
			if input.Numel() != resident.Numel() || slices.Equal(input.Shape,resident.Shape) {
				t.Fatal("fixture does not distinguish geometry from element count")
			}
			if got := tc.probe(input,tc.stored,hp); got.owner != 0 {
				t.Fatalf("stale geometry cache hit: got %v for requested %v",got.Shape,input.Shape)
			}
			if len(uploadCache) != 1 || uploadCache[key].owner != resident.owner || !slices.Equal(uploadCache[key].Shape,tc.first) {
				t.Fatal("geometry miss mutated the prior owner or cache")
			}
			input.Shape = slices.Clone(tc.first)
			if got := tc.probe(input,tc.stored,hp); got.owner != resident.owner {
				t.Fatal("geometry miss invalidated an existing exact hit")
			}
			otherLayout := RowMajor
			if tc.layout == RowMajor { otherLayout = ColMajor }
			uploadCache = map[ucKey]Tensor{{hp,tc.stored,otherLayout}:resident}
			if got := tc.probe(input,tc.stored,hp); got.owner != 0 {
				t.Fatal("layout-key isolation changed")
			}
			otherDtype := F32
			if tc.stored == F32 { otherDtype = F16 }
			uploadCache = map[ucKey]Tensor{{hp,otherDtype,tc.layout}:resident}
			if got := tc.probe(input,tc.stored,hp); got.owner != 0 {
				t.Fatal("dtype-key isolation changed")
			}
		})
	}
}
`
