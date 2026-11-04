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

var cudaKVConsumerHost = flag.Bool("fak-cuda-kv-consumer-host", false, "run exact-source contiguous KV snapshot/evict terminal-entry witnesses")

// Require admission before graph invalidation, host allocation or publication.
// The nil snapshot error keeps its original precedence.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVConsumerEntryGuards(t *testing.T) {
	source, err := os.ReadFile("cuda_kv.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "cuda_kv.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	wanted := map[string][]string{
		"Evict": {
			"cudaMu.Lock()", "defer cudaMu.Unlock()",
			"if err := k.sharedStopErrorLocked(\"kv-evict\"); err != nil { panic(err) }",
			"k.be.graphOnceInvalidateKVLocked(k)",
		},
		"SnapshotToHost": {
			"if k == nil { return KVHostSnapshot{}, fmt.Errorf(\"cuda: cannot snapshot nil KV store\") }",
			"cudaMu.Lock()", "defer cudaMu.Unlock()",
			"if err := k.sharedStopErrorLocked(\"kv-snapshot\"); err != nil { return KVHostSnapshot{}, err }",
		},
		"sharedStopErrorLocked": {
			"if k.releaseRetained || cudaQ56KMMVQReleaseBlocked() { return &BackendError{Backend: \"cuda\", Site: site, Class: \"cuda-terminal-stream\", Message: \"shared private stream is uncertain; process teardown required\"} }",
			"return nil",
		},
	}
	for _, decl := range file.Decls {
		f, ok := decl.(*ast.FuncDecl)
		if !ok || f.Recv == nil || f.Body == nil {
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
		want, ok := wanted[f.Name.Name]
		if !ok {
			continue
		}
		if len(f.Body.List) < len(want) {
			t.Fatalf("missing %s admission prefix", f.Name.Name)
		}
		for i, expected := range want {
			stmt := f.Body.List[i]
			got := strings.Join(strings.Fields(string(source[fs.Position(stmt.Pos()).Offset:fs.Position(stmt.End()).Offset])), " ")
			if got != expected {
				t.Fatalf("%s statement %d = %q, want %q", f.Name.Name, i, got, expected)
			}
		}
		delete(wanted, f.Name.Name)
	}
	if len(wanted) != 0 {
		t.Fatalf("missing actual consumer methods: %v", wanted)
	}
}

// Extract actual methods, shared predicate, typed error, snapshot validation and
// cleanup helpers. Native boundaries and surrounding geometry are host fakes.
// Only the cgo scalar type C.float is renamed to float32; no control flow changes.
// Removed-admission controls must fail behaviorally, never by compilation.
// No native completion, numerical CUDA, mid-operation or rollback claim follows.
// fak-test:runtime slow est=180s lane=optin
func TestCUDAKVConsumerHostContract(t *testing.T) {
	if !*cudaKVConsumerHost {
		t.Skip("set -fak-cuda-kv-consumer-host for the CPU entry witness")
	}
	var program strings.Builder
	program.WriteString(cudaKVConsumerHostFixture)
	wanted := map[string]bool{
		"dslice": false, "cudaKV": false, "cudaKVReleaseOwners": false,
		"*cudaKV.retainReleaseOwnerLocked": false, "*cudaKV.Free": false,
		"*cudaKV.ResidentBytes": false, "*cudaKV.stride": false,
		"*cudaKV.sharedStopErrorLocked": false, "*cudaKV.Evict": false,
		"*cudaKV.SnapshotToHost": false, "*cudaBackend.compactDS": false, "offsetF": false,
		"cudaQ56KMMVQReleaseBlocked": false,
		"clearDeviceSliceMetadata":   false, "releaseDeviceSlice": false, "releaseKVDeviceSlices": false,
		"BackendError": false, "VulkanErrorClass": false, "*BackendError.Error": false,
		"KVHostSnapshot": false, "KVHostSnapshot.Validate": false,
	}
	for _, path := range []string{"cuda_kv.go", "cuda.go", "backend_helpers.go", "vulkan_errors.go", "kvhost.go"} {
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
	body := program.String()
	if strings.Count(body, "C.float") != 4 {
		t.Fatal("cgo float boundary changed; review host scalar aliases")
	}
	body = strings.ReplaceAll(body, "C.float", "float32")
	goExe := filepath.Join(runtime.GOROOT(), "bin", "go")
	if runtime.GOOS == "windows" {
		goExe += ".exe"
	}
	for _, tc := range []struct{ name, remove, run, failure string }{
		{"candidate", "", "^TestKVConsumerHost$", ""},
		{"snapshot-guard-removed", "\tif err := k.sharedStopErrorLocked(\"kv-snapshot\"); err != nil {\n\t\treturn KVHostSnapshot{}, err\n\t}\n", "^TestKVConsumerHost$/shared-f32$/snapshot$", "terminal snapshot was admitted"},
		{"evict-guard-removed", "\tif err := k.sharedStopErrorLocked(\"kv-evict\"); err != nil {\n\t\tpanic(err)\n\t}\n", "^TestKVConsumerHost$/shared-f32$/evict$", "terminal eviction was admitted"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			text := body
			if tc.remove != "" {
				if strings.Count(text, tc.remove) != 1 {
					t.Fatal("actual admission removal boundary changed")
				}
				text = strings.Replace(text, tc.remove, "", 1)
			}
			dir := t.TempDir()
			probe := filepath.Join(dir, "kv_consumer_test.go")
			if err := os.WriteFile(probe, []byte(text), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", tc.run, probe)
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
			t.Logf("source-derived host witness:\n%s", output)
			if tc.failure == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(string(output), tc.failure) || strings.Contains(string(output), "[build failed]") {
				t.Fatalf("removed guard must fail the behavioral assertion: %v\n%s", err, output)
			}
		})
	}
}

const cudaKVConsumerHostFixture = `package compute

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

type Dtype int
const (F32 Dtype = iota; F16; BF16)
func (d Dtype) Bytes() int { if d == F32 { return 4 }; return 2 }
type KVPrecision int
const (KVPrecisionF32 KVPrecision = iota; KVPrecisionF16; KVPrecisionBF16)
type KVConfig struct { NumLayers, NumKVHeads, HeadDim int; Precision KVPrecision; RopeTheta float64 }
func cloneKVConfig(c KVConfig) KVConfig { return c }
type MemoryClass string
const MemoryScratchpad MemoryClass = "scratch"
type DeviceAllocError struct { Bytes int; Site string; Class MemoryClass }
type cudaBuf struct { ptr unsafe.Pointer }
type cudaBackend struct { fenceGen uint64 }
var cudaMu sync.Mutex
var nativeStop, queries, invalidations, copies, allocs, frees, otherNative int

func requireLock() {
	if cudaMu.TryLock() { cudaMu.Unlock(); panic("CUDA consumer boundary outside cudaMu") }
}
func (*cudaBackend) graphOnceInvalidateKVLocked(*cudaKV) { requireLock(); invalidations++ }
func (*cudaBackend) dallocClass(int, MemoryClass, string) *cudaBuf { allocs++; panic("unexpected conversion allocation") }
func cudaAllocTag(MemoryClass, string, bool) (int32, int32) { return 1, 2 }
func cudaKVCheck(status int32) { if status != 0 { panic(status) } }
var C = struct {
	int func(any) int32
	double func(float64) float64
	size_t func(int) uintptr
	fak_v41_checked_stream_terminal func() int
	fcuda_d2h func(unsafe.Pointer, unsafe.Pointer, uintptr)
	fcuda_d2d func(unsafe.Pointer, unsafe.Pointer, uintptr)
	fcuda_rope_f32 func(*float32, int32, int32, int32, float64)
	fcuda_kv_store_typed func(unsafe.Pointer, *float32, int32, int32, int32) int32
	fcuda_kv_load_typed func(*float32, unsafe.Pointer, int32, int32) int32
	fcuda_malloc_tagged func(uintptr, int32, int32) unsafe.Pointer
	fcuda_free func(unsafe.Pointer)
}{
	int: func(v any) int32 { switch n := v.(type) { case int: return int32(n); case Dtype: return int32(n); default: panic("unexpected C integer") } },
	double: func(v float64) float64 { return v },
	size_t: func(v int) uintptr { return uintptr(v) },
	fak_v41_checked_stream_terminal: func() int { requireLock(); queries++; return nativeStop },
	fcuda_d2h: func(dst, src unsafe.Pointer, n uintptr) {
		requireLock(); copies++
		copy(unsafe.Slice((*float32)(dst), int(n/4)), unsafe.Slice((*float32)(src), int(n/4)))
	},
	fcuda_d2d: func(unsafe.Pointer, unsafe.Pointer, uintptr) { otherNative++; panic("unexpected D2D") },
	fcuda_rope_f32: func(*float32, int32, int32, int32, float64) { otherNative++; panic("unexpected RoPE") },
	fcuda_kv_store_typed: func(unsafe.Pointer, *float32, int32, int32, int32) int32 { otherNative++; panic("unexpected typed store") },
	fcuda_kv_load_typed: func(*float32, unsafe.Pointer, int32, int32) int32 { otherNative++; panic("unexpected typed load") },
	fcuda_malloc_tagged: func(uintptr, int32, int32) unsafe.Pointer { allocs++; panic("unexpected native allocation") },
	fcuda_free: func(unsafe.Pointer) { requireLock(); frees++ },
}

func clearCounters() { queries, invalidations, copies, allocs, frees, otherNative = 0, 0, 0, 0, 0, 0 }
func fresh(dtype Dtype) *cudaKV {
	cudaKVReleaseOwners, nativeStop = nil, 0
	clearCounters()
	k := &cudaKV{be: &cudaBackend{fenceGen: 41}, cfg: KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, Precision: KVPrecision(dtype), RopeTheta: 10000}, maxPos: 17,
		K: make([]dslice, 2), Kraw: make([]dslice, 2), V: make([]dslice, 2), pos: []int{0, 1, 2, 3}}
	for l := range k.K {
		for i, rows := range [3][]dslice{k.K, k.Kraw, k.V} {
			data := new([16]float32)
			for j := range data { data[j] = float32(100*l+10*i+j) }
			d := dtype; if i == 1 { d = F32 }
			rows[l] = dslice{ptr: unsafe.Pointer(&data[0]), len: 8, cap: 16, dtype: d}
		}
	}
	return k
}
func snapshot(k *cudaKV) cudaKV {
	out := *k
	out.K = append([]dslice(nil), k.K...); out.Kraw = append([]dslice(nil), k.Kraw...); out.V = append([]dslice(nil), k.V...)
	out.pos = append([]int(nil), k.pos...)
	return out
}
func checkError(t *testing.T, err error, site string) {
	t.Helper()
	e, ok := err.(*BackendError)
	if !ok || e.Backend != "cuda" || e.Class != "cuda-terminal-stream" || e.Site != site {
		t.Fatalf("wrong terminal error: %#v", err)
	}
}
func TestKVConsumerHost(t *testing.T) {
	for _, tc := range []struct { name string; dtype Dtype; retained bool }{
		{"shared-f32", F32, false}, {"shared-f16", F16, false}, {"shared-bf16", BF16, false},
		{"retained-f32", F32, true}, {"retained-f16", F16, true}, {"retained-bf16", BF16, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, operation := range []string{"snapshot", "evict"} {
				t.Run(operation, func(t *testing.T) {
					k := fresh(tc.dtype)
					nativeStop = 711
					if tc.retained {
						k.Free()
						if !k.releaseRetained || cudaKVReleaseOwners != k { t.Fatal("real Free did not retain owner") }
						nativeStop = 0 // retained ownership is independently non-reusable
						clearCounters()
					}
					before, root, bytes := snapshot(k), cudaKVReleaseOwners, k.ResidentBytes()
					if operation == "snapshot" {
						out, err := k.SnapshotToHost()
						if err == nil || !reflect.DeepEqual(out, KVHostSnapshot{}) { t.Fatal("terminal snapshot was admitted") }
						checkError(t, err, "kv-snapshot")
					} else {
						var failure any
						func() { defer func() { failure = recover() }(); k.Evict(2, 2) }()
						err, ok := failure.(error)
						if !ok { t.Fatal("terminal eviction was admitted") }
						checkError(t, err, "kv-evict")
					}
					wantQueries := 1; if tc.retained { wantQueries = 0 }
					if queries != wantQueries || invalidations != 0 || copies != 0 || allocs != 0 || frees != 0 || otherNative != 0 || k.be.fenceGen != 41 ||
						!reflect.DeepEqual(snapshot(k), before) || cudaKVReleaseOwners != root || k.ResidentBytes() != bytes {
						t.Fatal("terminal consumer changed ownership, accounting or completion state")
					}
				})
			}
		})
	}
	t.Run("healthy-f32-snapshot", func(t *testing.T) {
		k := fresh(F32); before := snapshot(k)
		out, err := k.SnapshotToHost()
		if err != nil { t.Fatal(err) }
		if queries != 1 || copies != 6 || invalidations != 0 || allocs != 0 || frees != 0 || otherNative != 0 || k.be.fenceGen != 47 || !reflect.DeepEqual(snapshot(k), before) || cudaKVReleaseOwners != nil {
			t.Fatal("healthy snapshot flow changed")
		}
		if out.Config != k.cfg || !reflect.DeepEqual(out.Pos, k.pos) { t.Fatal("healthy snapshot geometry changed") }
		for l := range k.K {
			for i, rows := range [3][][]float32{out.K, out.KRaw, out.V} {
				for j, value := range rows[l] { if value != float32(100*l+10*i+j) { t.Fatal("host boundary copy changed") } }
			}
		}
		out.K[0][0] = 999
		if *(*float32)(k.K[0].ptr) != 0 { t.Fatal("snapshot aliases source storage") }
	})
	t.Run("healthy-f32-tail-evict", func(t *testing.T) {
		k := fresh(F32); before := snapshot(k)
		if removed := k.Evict(2, 2); removed != 2 { t.Fatal("tail eviction count changed") }
		if queries != 1 || invalidations != 1 || copies != 0 || allocs != 0 || frees != 0 || otherNative != 0 || k.be.fenceGen != 41 || !reflect.DeepEqual(k.pos, []int{0, 1}) || cudaKVReleaseOwners != nil {
			t.Fatal("healthy tail eviction flow changed")
		}
		for l := range k.K {
			for i, rows := range [3][]dslice{k.K, k.Kraw, k.V} {
				want := [3][]dslice{before.K, before.Kraw, before.V}[i][l]; want.len = 4
				if rows[l] != want { t.Fatal("tail eviction changed storage ownership") }
			}
		}
	})
	t.Run("nil-snapshot-precedence", func(t *testing.T) {
		nativeStop = 711; clearCounters()
		out, err := (*cudaKV)(nil).SnapshotToHost()
		if err == nil || err.Error() != "cuda: cannot snapshot nil KV store" || !reflect.DeepEqual(out, KVHostSnapshot{}) || queries != 0 { t.Fatal("nil snapshot precedence changed") }
	})
}
`
