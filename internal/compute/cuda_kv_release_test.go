package compute

import (
	"context"
	"crypto/sha256"
	"flag"
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

var cudaKVReleaseHost = flag.Bool("fak-cuda-kv-release-host", false, "run the exact-source contiguous KV shared-stop ownership witness")

// The default lane binds the early guard to the actual Free method without
// compiling CUDA or spawning a child compiler. Behavior is checked separately.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVReleaseEntryGuard(t *testing.T) {
	source, err := os.ReadFile("cuda_kv.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "cuda_kv.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range file.Decls {
		f, ok := decl.(*ast.FuncDecl)
		if !ok || f.Name.Name != "Free" || f.Recv == nil || f.Body == nil {
			continue
		}
		ptr, ok := f.Recv.List[0].Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		id, ok := ptr.X.(*ast.Ident)
		if !ok || id.Name != "cudaKV" {
			continue
		}
		want := []string{
			"cudaMu.Lock()", "defer cudaMu.Unlock()",
			"if k.releaseRetained || cudaQ56KMMVQReleaseBlocked() { k.retainReleaseOwnerLocked() return }",
			"k.be.graphOnceInvalidateKVLocked(k)",
		}
		if len(f.Body.List) != len(want)+1 {
			t.Fatal("Free changed effects around the pre-mutation shared-stop guard")
		}
		for i, expected := range want {
			stmt := f.Body.List[i]
			got := strings.Join(strings.Fields(string(source[fs.Position(stmt.Pos()).Offset:fs.Position(stmt.End()).Offset])), " ")
			if got != expected {
				t.Fatalf("Free statement %d = %q, want %q", i, got, expected)
			}
		}
		return
	}
	t.Fatal("missing actual contiguous KV Free")
}

// Executes intact production ownership/release methods and their generic row
// helpers. Shared status, graph invalidation and native free are mocked host
// boundaries. This is not cgo, device, constructor rollback or mid-free failure
// evidence. Child compilation is opt-in and deadline-bound.
// fak-test:runtime slow est=60s lane=optin
func TestCUDAKVReleaseHostContract(t *testing.T) {
	if !*cudaKVReleaseHost {
		t.Skip("set -fak-cuda-kv-release-host for the CPU ownership witness")
	}
	var program strings.Builder
	program.WriteString(cudaKVReleaseHostFixture)
	wanted := map[string]bool{
		"dslice": false, "cudaKV": false, "cudaKVReleaseOwners": false,
		"*cudaKV.retainReleaseOwnerLocked": false, "*cudaKV.Free": false,
		"*cudaKV.ResidentBytes":    false,
		"clearDeviceSliceMetadata": false, "releaseDeviceSlice": false, "releaseKVDeviceSlices": false,
	}
	for _, path := range []string{"cuda_kv.go", "backend_helpers.go"} {
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, path, source, 0)
		if err != nil {
			t.Fatal(err)
		}
		emit := func(key string, node ast.Node, prefix string) {
			seen, ok := wanted[key]
			if !ok {
				return
			}
			if seen {
				t.Fatalf("duplicate real declaration %s", key)
			}
			wanted[key] = true
			body := source[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset]
			t.Logf("%s/%s SHA256=%x", path, key, sha256.Sum256(body))
			program.WriteString(prefix)
			program.Write(body)
			program.WriteByte('\n')
		}
		for _, decl := range file.Decls {
			switch node := decl.(type) {
			case *ast.GenDecl:
				for _, spec := range node.Specs {
					switch spec := spec.(type) {
					case *ast.TypeSpec:
						emit(spec.Name.Name, spec, "type ")
					case *ast.ValueSpec:
						if node.Tok == token.VAR && len(spec.Names) == 1 {
							emit(spec.Names[0].Name, spec, "var ")
						}
					}
				}
			case *ast.FuncDecl:
				key := node.Name.Name
				if node.Recv != nil {
					r := node.Recv.List[0].Type
					prefix := ""
					if ptr, ok := r.(*ast.StarExpr); ok {
						prefix, r = "*", ptr.X
					}
					id, ok := r.(*ast.Ident)
					if !ok {
						continue
					}
					key = prefix + id.Name + "." + key
				}
				emit(key, node, "")
			}
		}
	}
	for key, seen := range wanted {
		if !seen {
			t.Fatalf("missing real declaration %s", key)
		}
	}
	dir := t.TempDir()
	probe := filepath.Join(dir, "kv_release_test.go")
	if err := os.WriteFile(probe, []byte(program.String()), 0600); err != nil {
		t.Fatal(err)
	}
	goExe := filepath.Join(runtime.GOROOT(), "bin", "go")
	if runtime.GOOS == "windows" {
		goExe += ".exe"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", "^TestKVReleaseHost$", probe)
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
	output, err := cmd.CombinedOutput()
	t.Logf("exact-source host witness:\n%s", output)
	if err != nil {
		t.Fatal(err)
	}
}

const cudaKVReleaseHostFixture = `package compute

import (
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"unsafe"
	"weak"
)

type Dtype int
const (F32 Dtype = iota; F16; BF16)
func (d Dtype) Bytes() int { if d == F32 { return 4 }; return 2 }
type KVConfig struct { NumLayers, NumKVHeads, HeadDim int }
type cudaBackend struct{}

var cudaMu sync.Mutex
var sharedStop bool
var queries, invalidations int
var released []unsafe.Pointer

func requireLock() {
	if cudaMu.TryLock() {
		cudaMu.Unlock()
		panic("CUDA owner boundary ran without cudaMu")
	}
}
func cudaQ56KMMVQReleaseBlocked() bool { requireLock(); queries++; return sharedStop }
func (*cudaBackend) graphOnceInvalidateKVLocked(*cudaKV) { requireLock(); invalidations++ }
var C = struct { fcuda_free func(unsafe.Pointer) }{
	fcuda_free: func(ptr unsafe.Pointer) {
		requireLock()
		if sharedStop { panic("native free reached under shared terminal state") }
		released = append(released, ptr)
	},
}

func reset() {
	cudaKVReleaseOwners, sharedStop = nil, false
	queries, invalidations, released = 0, 0, nil
}
func fresh(dtype Dtype) *cudaKV {
	k := &cudaKV{be: &cudaBackend{}, cfg: KVConfig{2, 1, 4}, maxPos: 17,
		K: make([]dslice, 2), Kraw: make([]dslice, 2), V: make([]dslice, 2), pos: []int{4, 7}}
	for l := range k.K {
		for i, rows := range [3][]dslice{k.K, k.Kraw, k.V} {
			d := dtype
			if i == 1 { d = F32 }
			rows[l] = dslice{ptr: unsafe.Pointer(new([64]byte)), len: 8, cap: 16, dtype: d}
		}
	}
	return k
}
func snapshot(k *cudaKV) cudaKV {
	out := *k
	out.K = append([]dslice(nil), k.K...)
	out.Kraw = append([]dslice(nil), k.Kraw...)
	out.V = append([]dslice(nil), k.V...)
	out.pos = append([]int(nil), k.pos...)
	out.releaseRetained, out.releaseNext = false, nil
	return out
}

func TestKVReleaseHost(t *testing.T) {
	for _, dtype := range []Dtype{F32, F16, BF16} {
		t.Run(fmt.Sprintf("shared-stop-dtype-%d", dtype), func(t *testing.T) {
			reset()
			a, b := fresh(dtype), fresh(dtype)
			beforeA, beforeB, bytes := snapshot(a), snapshot(b), a.ResidentBytes()
			wa, wb := weak.Make(a), weak.Make(b)
			sharedStop = true
			a.Free()
			b.Free()
			for i := 0; i < 2; i++ { a.Free(); b.Free() }
			if !reflect.DeepEqual(snapshot(a), beforeA) || !reflect.DeepEqual(snapshot(b), beforeB) || a.ResidentBytes() != bytes || bytes == 0 {
				t.Fatal("terminal refusal changed row metadata, geometry or resident accounting")
			}
			if queries != 2 || invalidations != 0 || len(released) != 0 ||
				cudaKVReleaseOwners != b || b.releaseNext != a || a.releaseNext != nil || !a.releaseRetained || !b.releaseRetained {
				t.Fatal("terminal refusal lost, duplicated or mutated an owner")
			}
			// Like the model caller, detach both handles. The intrusive root is
			// now the only strong reference to each wrapper, even after GC.
			a, b = nil, nil
			runtime.GC()
			if wa.Value() == nil || wb.Value() == nil || cudaKVReleaseOwners != wb.Value() || cudaKVReleaseOwners.releaseNext != wa.Value() {
				t.Fatal("detached terminal owners were not strongly retained")
			}
			// A retained owner is never retried, even if a mocked status clears.
			sharedStop = false
			cudaKVReleaseOwners.Free()
			if queries != 2 || invalidations != 0 || len(released) != 0 {
				t.Fatal("retained owner retried cleanup")
			}
			sharedStop = true
			empty := &cudaKV{be: &cudaBackend{}}
			empty.Free()
			if empty.releaseRetained || empty.releaseNext != nil || cudaKVReleaseOwners != wb.Value() {
				t.Fatal("empty refusal grew the retained owner list")
			}
		})
	}
	t.Run("healthy-release-order-and-metadata", func(t *testing.T) {
		reset()
		k := fresh(BF16)
		var want []unsafe.Pointer
		for l := range k.K { want = append(want, k.K[l].ptr, k.Kraw[l].ptr, k.V[l].ptr) }
		k.Free()
		if queries != 1 || invalidations != 1 || !reflect.DeepEqual(released, want) ||
			k.releaseRetained || k.releaseNext != nil || cudaKVReleaseOwners != nil || k.pos != nil || k.maxPos != 17 {
			t.Fatal("healthy cleanup or release order changed")
		}
		for l := range k.K {
			for i, rows := range [3][]dslice{k.K, k.Kraw, k.V} {
				dtype := BF16
				if i == 1 { dtype = F32 }
				if rows[l] != (dslice{dtype: dtype}) { t.Fatal("healthy row metadata/dtype changed") }
			}
		}
		k.Free()
		if len(released) != len(want) || cudaKVReleaseOwners != nil { t.Fatal("healthy repeat double-freed or retained") }
	})
}
`
