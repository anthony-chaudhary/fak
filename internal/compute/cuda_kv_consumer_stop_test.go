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

// Bind the native first-status latch and Go host-pin/failure ordering without
// pretending a source assertion executes CUDA or certifies a failed drain.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVSnapshotCheckedStatusContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	native := read("cuda/kv_precision.cuh")
	body := func(signature string) string {
		t.Helper()
		if strings.Count(native, signature) != 1 {
			t.Fatalf("missing/duplicate native adapter %s", signature)
		}
		start := strings.Index(native, signature)
		open := start + strings.Index(native[start:], "{")
		depth := 1
		for end := open + 1; end < len(native); end++ {
			switch native[end] {
			case '{':
				depth++
			case '}':
				depth--
			}
			if depth == 0 {
				return strings.Join(strings.Fields(native[open+1:end]), " ")
			}
		}
		t.Fatal("unterminated native adapter")
		return ""
	}
	for signature, expected := range map[string]string{
		"static int kv_snapshot_status(int status)":                                                      "if (status != 0 && g_cuda_cleanup_status == 0) g_cuda_cleanup_status = status; return status;",
		"extern \"C\" int fcuda_kv_snapshot_load_checked(float *dst, const void *src, int n, int dtype)": "if (v41_candidates_stream_uncertain()) return kv_snapshot_status(g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady); return kv_snapshot_status(fcuda_kv_load_typed(dst, src, n, dtype));",
		"extern \"C\" int fcuda_kv_snapshot_d2h_checked(void *dst, const void *src, size_t bytes)":       "if (v41_candidates_stream_uncertain()) return kv_snapshot_status(g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady); return kv_snapshot_status(fcuda_d2h_checked(dst, src, bytes));",
	} {
		if got := body(signature); got != expected {
			t.Fatalf("native status/ownership contract changed: %s: %s", signature, got)
		}
	}
	header := read("cuda_kv_precision.h")
	for _, declaration := range []string{
		"#include <stddef.h>",
		"int fcuda_kv_snapshot_load_checked(float *dst, const void *src, int n, int dtype);",
		"int fcuda_kv_snapshot_d2h_checked(void *dst, const void *src, size_t bytes);",
	} {
		if !strings.Contains(header, declaration) {
			t.Fatalf("missing native ABI declaration %q", declaration)
		}
	}
	start := strings.Index(native, "static int kv_snapshot_status")
	end := strings.Index(native[start:], "__global__ void k_kv_attention_typed")
	if end < 0 || strings.Contains(native[start:start+end], "#if") {
		t.Fatal("snapshot status adapters must be unconditional")
	}
	unit := read("cuda_kernels.cu")
	memory, precision := strings.Index(unit, "#include \"cuda/runtime_memory.cuh\""), strings.Index(unit, "#include \"cuda/kv_precision.cuh\"")
	if memory < 0 || precision < memory {
		t.Fatal("checked D2H definition must precede snapshot adapters")
	}
	goSource := read("cuda_kv.go")
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "cuda_kv.go", goSource, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "SnapshotToHost" || fn.Body == nil {
			continue
		}
		found = true
		text := goSource[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset]
		last := -1
		for _, token := range []string{
			"k.snapshotCopy = holder", "defer func()", "holder.pin.Pin(&out[0])",
			"C.fcuda_kv_snapshot_load_checked(", "C.fcuda_kv_snapshot_d2h_checked(",
			"copied = true", "atomic.AddUint64(&k.be.fenceGen, 1)", "C.fcuda_kv_snapshot_release_checked(", "holder.conversion = nil", "retired = true",
		} {
			at := strings.Index(text, token)
			if at <= last {
				t.Fatalf("missing/reordered snapshot ownership step %q", token)
			}
			last = at
		}
		for _, token := range []string{
			"if copied {\n\t\t\t\t\tholder.pin.Unpin()",
			"return nil, k.snapshotFailureLocked(\"kv-snapshot-convert\", status)",
			"return nil, k.snapshotFailureLocked(\"kv-snapshot-d2h\", status)",
			"if status != 0 || disposition != C.FCUDA_KV_SNAPSHOT_RETIRED {",
			"C.fcuda_kv_snapshot_release_failed(C.int(status))",
			"return nil, k.snapshotFailureLocked(\"kv-snapshot-release\", status)",
		} {
			if !strings.Contains(text, token) {
				t.Fatalf("missing checked snapshot outcome %q", token)
			}
		}
		if strings.Contains(text, "C.fcuda_d2h(") || strings.Contains(text, "defer C.fcuda_free(") || strings.Contains(text, "C.fcuda_free(holder.conversion.ptr)") {
			t.Fatal("unchecked copy or unconditional scratch cleanup returned")
		}
	}
	if !found {
		t.Fatal("missing actual snapshot method")
	}
}

// Extract actual methods, shared predicate, typed error, snapshot validation and
// cleanup helpers. Native boundaries and surrounding geometry are host fakes.
// Only the cgo scalar type C.float is renamed to float32; no control flow changes.
// Removed-admission controls must fail behaviorally, never by compilation.
// Checked-copy failure propagation is exercised through explicit native mocks.
// No actual native completion, numerical CUDA or rollback claim follows.
// fak-test:runtime slow est=180s lane=optin
func TestCUDAKVConsumerHostContract(t *testing.T) {
	if !*cudaKVConsumerHost {
		t.Skip("set -fak-cuda-kv-consumer-host for the CPU entry witness")
	}
	var program strings.Builder
	program.WriteString(cudaKVConsumerHostFixture)
	wanted := map[string]bool{
		"dslice": false, "cudaKV": false, "cudaKVReleaseOwners": false,
		"cudaKVSnapshotCopy":               false,
		"*cudaKV.retainReleaseOwnerLocked": false, "*cudaKV.Free": false,
		"*cudaKV.ResidentBytes": false, "*cudaKV.stride": false,
		"*cudaKV.sharedStopErrorLocked": false, "*cudaKV.Evict": false,
		"*cudaKV.snapshotFailureLocked": false,
		"*cudaKV.SnapshotToHost":        false, "*cudaBackend.compactDS": false, "offsetF": false,
		"cudaQ56KMMVQReleaseBlocked": false,
		"clearDeviceSliceMetadata":   false, "releaseDeviceSlice": false, "releaseKVDeviceSlices": false,
		"BackendError": false, "VulkanErrorClass": false, "*BackendError.Error": false,
		"KVHostSnapshot": false, "KVHostSnapshot.Validate": false,
		"CUDAOpError": false, "*CUDAOpError.Error": false,
	}
	for _, path := range []string{"cuda_kv.go", "cuda.go", "backend_helpers.go", "vulkan_errors.go", "kvhost.go", "cuda_errors.go"} {
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
	latch, err := os.ReadFile("devicefault.go")
	if err != nil {
		t.Fatal(err)
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
		{"snapshot-guard-removed", "\tif err := k.sharedStopErrorLocked(\"kv-snapshot\"); err != nil {\n\t\treturn KVHostSnapshot{}, err\n\t}\n", "^TestKVConsumerHost$/shared-f32$/snapshot$", "terminal snapshot entry was bypassed"},
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
			latchPath := filepath.Join(dir, "devicefault.go")
			if err := os.WriteFile(latchPath, latch, 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", tc.run, probe, latchPath)
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
	"runtime"
	"weak"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

type Backend interface{}
type Dtype int
const (F32 Dtype = iota; F16; BF16)
func (d Dtype) Bytes() int { if d == F32 { return 4 }; return 2 }
type KVPrecision int
const (KVPrecisionF32 KVPrecision = iota; KVPrecisionF16; KVPrecisionBF16)
type KVConfig struct { NumLayers, NumKVHeads, HeadDim int; Precision KVPrecision; RopeTheta float64 }
func cloneKVConfig(c KVConfig) KVConfig { return c }
type MemoryClass string
const MemoryScratchpad MemoryClass = "scratch"
const MemoryUnknown MemoryClass = "unknown"
type DeviceAllocError struct { Bytes int; Site string; Class MemoryClass }
func (*DeviceAllocError) Error() string { return "fixture allocation error" }
type Qwen35GDNKernelError struct { Stage string; Code int }
func (*Qwen35GDNKernelError) Error() string { return "fixture kernel error" }
type Qwen35GDNInvalidStateError struct { Operand string }
func (*Qwen35GDNInvalidStateError) Error() string { return "fixture invalid buffer" }
type cudaBuf struct { ptr unsafe.Pointer; n int }
type cudaBackend struct { fenceGen uint64; faultLatch *DeviceFaultLatch }
var cudaMu sync.Mutex
var nativeStop, queries, invalidations, copies, allocs, frees, otherNative int
var loadCalls, copyCalls, conversions, failCopyAt, failConversionAt int
var allocated []*cudaBuf
var released []unsafe.Pointer
var retainedHistory []*cudaKV
var lastDestination unsafe.Pointer
var copyPanic any
var releaseCalls, failReleaseAt int
var failReleaseStatus, failReleaseDisposition int32
const copyFailureStatus = 1057
const conversionFailureStatus = 17

func requireLock() {
	if cudaMu.TryLock() { cudaMu.Unlock(); panic("CUDA consumer boundary outside cudaMu") }
}
func (*cudaBackend) graphOnceInvalidateKVLocked(*cudaKV) { requireLock(); invalidations++ }
func (*cudaBackend) dallocClass(bytes int, class MemoryClass, site string) *cudaBuf {
	requireLock()
	if class != MemoryScratchpad || site != "kv-snapshot-conversion" || bytes <= 0 || bytes%4 != 0 { panic("unexpected conversion geometry") }
	allocs++
	storage := make([]float32, bytes/4)
	owner := &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: bytes}
	allocated = append(allocated, owner)
	return owner
}
func cudaAllocTag(MemoryClass, string, bool) (int32, int32) { return 1, 2 }
func cudaKVCheck(status int32) { if status != 0 { panic(status) } }
var C = struct {
	int func(any) int32
	double func(float64) float64
	size_t func(int) uintptr
	fak_v41_checked_stream_terminal func() int
	fcuda_kv_snapshot_d2h_checked func(unsafe.Pointer, unsafe.Pointer, uintptr) int32
	fcuda_kv_snapshot_load_checked func(*float32, unsafe.Pointer, int32, int32) int32
	fcuda_d2d func(unsafe.Pointer, unsafe.Pointer, uintptr)
	fcuda_rope_f32 func(*float32, int32, int32, int32, float64)
	fcuda_kv_store_typed func(unsafe.Pointer, *float32, int32, int32, int32) int32
	fcuda_kv_load_typed func(*float32, unsafe.Pointer, int32, int32) int32
	fcuda_malloc_tagged func(uintptr, int32, int32) unsafe.Pointer
	fcuda_free func(unsafe.Pointer)
	fcuda_kv_snapshot_release_checked func(unsafe.Pointer, uintptr, *int32) int32
	fcuda_kv_snapshot_release_failed func(int32) int32
	FCUDA_KV_SNAPSHOT_RETIRED int32
}{
	int: func(v any) int32 { switch n := v.(type) { case int: return int32(n); case Dtype: return int32(n); default: panic("unexpected C integer") } },
	double: func(v float64) float64 { return v },
	size_t: func(v int) uintptr { return uintptr(v) },
	fak_v41_checked_stream_terminal: func() int { requireLock(); queries++; return nativeStop },
	fcuda_kv_snapshot_d2h_checked: func(dst, src unsafe.Pointer, n uintptr) int32 {
		requireLock(); copyCalls++; lastDestination = dst
		if copyPanic != nil { panic(copyPanic) }
		if nativeStop != 0 { return int32(nativeStop) }
		copies++
		copy(unsafe.Slice((*float32)(dst), int(n/4)), unsafe.Slice((*float32)(src), int(n/4)))
		if copies == failCopyAt { nativeStop = copyFailureStatus; return copyFailureStatus }
		return 0
	},
	fcuda_kv_snapshot_load_checked: func(dst *float32, src unsafe.Pointer, n, dtype int32) int32 {
		requireLock(); loadCalls++
		if nativeStop != 0 { return int32(nativeStop) }
		conversions++
		if conversions == failConversionAt { nativeStop = conversionFailureStatus; return conversionFailureStatus }
		if dtype != int32(F16) && dtype != int32(BF16) { panic("unexpected conversion dtype") }
		// Float arrays are host boundary tokens, not a BF16/F16 numerical oracle.
		copy(unsafe.Slice(dst, int(n)), unsafe.Slice((*float32)(src), int(n)))
		return 0
	},
	fcuda_d2d: func(unsafe.Pointer, unsafe.Pointer, uintptr) { otherNative++; panic("unexpected D2D") },
	fcuda_rope_f32: func(*float32, int32, int32, int32, float64) { otherNative++; panic("unexpected RoPE") },
	fcuda_kv_store_typed: func(unsafe.Pointer, *float32, int32, int32, int32) int32 { otherNative++; panic("unexpected typed store") },
	fcuda_kv_load_typed: func(*float32, unsafe.Pointer, int32, int32) int32 { otherNative++; panic("unexpected typed load") },
	fcuda_malloc_tagged: func(uintptr, int32, int32) unsafe.Pointer { allocs++; panic("unexpected native allocation") },
	fcuda_free: func(ptr unsafe.Pointer) { requireLock(); if nativeStop != 0 { panic("cleanup attempted after snapshot failure") }; frees++; released = append(released, ptr) },
	FCUDA_KV_SNAPSHOT_RETIRED: 1,
	fcuda_kv_snapshot_release_checked: func(ptr unsafe.Pointer, bytes uintptr, disposition *int32) int32 {
		requireLock(); releaseCalls++; *disposition = 0
		if nativeStop != 0 { return int32(nativeStop) }
		if len(allocated) == 0 || allocated[len(allocated)-1].ptr != ptr || allocated[len(allocated)-1].n != int(bytes) { panic("wrong exact conversion release owner") }
		status := int32(0); *disposition = 1
		if releaseCalls == failReleaseAt { status = failReleaseStatus; *disposition = failReleaseDisposition }
		if *disposition == 1 { frees++; released = append(released, ptr) }
		if status != 0 { nativeStop = int(status) }
		return status
	},
	fcuda_kv_snapshot_release_failed: func(status int32) int32 {
		requireLock()
		if nativeStop != 0 { return int32(nativeStop) }
		if status == 0 { status = 1 }
		nativeStop = int(status)
		return status
	},
}

func clearCounters() {
	releaseCalls, failReleaseAt, failReleaseStatus, failReleaseDisposition = 0, 0, 0, 0
	queries, invalidations, copies, allocs, frees, otherNative = 0, 0, 0, 0, 0, 0
	loadCalls, copyCalls, conversions, failCopyAt, failConversionAt = 0, 0, 0, 0, 0
	allocated, released, lastDestination, copyPanic = nil, nil, nil, nil
}
func fresh(dtype Dtype) *cudaKV {
	// Failed pins are process-lifetime roots even when the test resets fake state.
	if cudaKVReleaseOwners != nil { retainedHistory = append(retainedHistory, cudaKVReleaseOwners) }
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
						if e, ok := err.(*BackendError); !ok || e.Backend != "cuda" || e.Class != "cuda-terminal-stream" || e.Site != "kv-snapshot" {
							t.Fatalf("terminal snapshot entry was bypassed: %#v", err)
						}
					} else {
						var failure any
						func() { defer func() { failure = recover() }(); k.Evict(2, 2) }()
						err, ok := failure.(error)
						if !ok { t.Fatal("terminal eviction was admitted") }
						checkError(t, err, "kv-evict")
					}
					wantQueries := 1; if tc.retained { wantQueries = 0 }
					if queries != wantQueries || invalidations != 0 || copyCalls != 0 || loadCalls != 0 || copies != 0 || allocs != 0 || frees != 0 || otherNative != 0 || k.be.fenceGen != 41 ||
						!reflect.DeepEqual(snapshot(k), before) || cudaKVReleaseOwners != root || k.ResidentBytes() != bytes {
						t.Fatal("terminal consumer changed ownership, accounting or completion state")
					}
				})
			}
		})
	}
	for _, tc := range []struct {
		name string; dtype Dtype; failCopy, failConvert, wantCopies, wantConverted, wantFrees, wantSuccessful int; latch bool
	}{
		{"f32-first-copy", F32, 1, 0, 1, 0, 0, 0, false},
		{"f32-later-copy", F32, 3, 0, 3, 0, 0, 2, true},
		{"f32-last-copy", F32, 6, 0, 6, 0, 0, 5, true},
		{"f16-converted-copy", F16, 1, 0, 1, 1, 0, 0, true},
		{"f16-raw-copy", F16, 2, 0, 2, 1, 1, 1, false},
		{"bf16-last-copy", BF16, 6, 0, 6, 4, 3, 5, false},
		{"f16-conversion-launch", F16, 0, 1, 0, 1, 0, 0, false},
		{"bf16-later-conversion", BF16, 0, 2, 2, 2, 1, 2, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			k := fresh(tc.dtype)
			if tc.latch { k.be.faultLatch = NewDeviceFaultLatch("cuda", 1) }
			before := snapshot(k)
			failCopyAt, failConversionAt = tc.failCopy, tc.failConvert
			out, err := k.SnapshotToHost()
			status, site := copyFailureStatus, "kv-snapshot-d2h"
			if tc.failConvert != 0 { status, site = conversionFailureStatus, "kv-snapshot-convert" }
			op, ok := err.(*CUDAOpError)
			if !ok || op.Op != "SnapshotToHost" || op.Site != site || op.Code != status || !reflect.DeepEqual(out, KVHostSnapshot{}) {
				t.Fatalf("failed snapshot published an image or lost primary status: out=%+v err=%#v", out, err)
			}
			if copies != tc.wantCopies || conversions != tc.wantConverted || allocs != tc.wantConverted || frees != tc.wantFrees ||
				k.be.fenceGen != uint64(41+tc.wantSuccessful) || invalidations != 0 || otherNative != 0 || nativeStop != status {
				t.Fatal("failure retried, freed an uncertain owner, or certified an unsuccessful copy")
			}
			holder := k.snapshotCopy
			if !k.releaseRetained || cudaKVReleaseOwners != k || k.releaseNext != nil || holder == nil || len(holder.host) != 8 {
				t.Fatal("failure lost the pinned host/input owner")
			}
			wantConversion := tc.wantConverted > tc.wantFrees
			if wantConversion {
				if holder.conversion == nil || holder.conversion != allocated[len(allocated)-1] { t.Fatal("lost exact conversion owner") }
				for _, ptr := range released { if ptr == holder.conversion.ptr { t.Fatal("failed conversion owner was freed") } }
			} else if holder.conversion != nil { t.Fatal("raw failure retained a previous retired conversion") }
			after := snapshot(k)
			after.releaseRetained, after.releaseNext, after.snapshotCopy = before.releaseRetained, before.releaseNext, before.snapshotCopy
			if !reflect.DeepEqual(after, before) { t.Fatal("failure changed source geometry or storage metadata") }
			if tc.latch {
				fault := k.be.faultLatch.Snapshot()
				if fault.Health != DevicePoisoned || fault.Faults != 1 || fault.LastCode != status || fault.LastSite != site { t.Fatal("real fault latch lost copy failure") }
			}
			queriesBefore, callsBefore, loadsBefore := queries, copyCalls, loadCalls
			k.Free()
			if retry, retryErr := k.SnapshotToHost(); retryErr == nil || !reflect.DeepEqual(retry, KVHostSnapshot{}) { t.Fatal("failed snapshot was retried") }
			if queries != queriesBefore || copyCalls != callsBefore || loadCalls != loadsBefore || nativeStop != status || k.snapshotCopy != holder {
				t.Fatal("retained failure touched native state or replaced its holder")
			}
			// A later native refusal must preserve the first status without copy.
			cudaMu.Lock()
			sticky := C.fcuda_kv_snapshot_d2h_checked(nil, nil, 0)
			cudaMu.Unlock()
			if int(sticky) != status || copies != tc.wantCopies || nativeStop != status { t.Fatal("native boundary lost first failure") }
			wk, wh := weak.Make(k), weak.Make(holder)
			address := uintptr(unsafe.Pointer(&holder.host[0]))
			k, holder, lastDestination = nil, nil, nil
			runtime.GC()
			if wk.Value() == nil || wh.Value() == nil || cudaKVReleaseOwners != wk.Value() || wk.Value().snapshotCopy != wh.Value() || uintptr(unsafe.Pointer(&wh.Value().host[0])) != address {
				t.Fatal("detached failed destination/pinner was not strongly retained")
			}
		})
	}
	t.Run("copy-panic-retains-before-unwind", func(t *testing.T) {
		k := fresh(F16)
		primary := &struct{ message string }{"native boundary panic"}
		copyPanic = primary
		var failure any
		func() { defer func() { failure = recover() }(); _, _ = k.SnapshotToHost() }()
		if failure != primary || !k.releaseRetained || cudaKVReleaseOwners != k || k.snapshotCopy == nil ||
			len(k.snapshotCopy.host) != 8 || k.snapshotCopy.conversion != allocated[0] || frees != 0 || k.be.fenceGen != 41 {
			t.Fatal("unexpected boundary panic lost pinned owners or primary identity")
		}
		runtime.GC()
	})
	for _, dtype := range []Dtype{F16, BF16} {
		for _, tc := range []struct{name string; at int; status, disposition int32}{
			{"retained-last", 4, 719, 0}, {"transferred-last", 4, 719, 1},
			{"transferred-first", 1, 719, 1}, {"missing-receipt", 4, 0, 0},
			{"malformed-receipt", 4, 0, 9}, {"error-malformed-receipt", 4, 719, 9},
		} {
			t.Run(fmt.Sprintf("release-%s-%d", tc.name, dtype), func(t *testing.T) {
				k := fresh(dtype); before := snapshot(k)
				if dtype == BF16 { k.be.faultLatch = NewDeviceFaultLatch("cuda", 1); before = snapshot(k) }
				failReleaseAt, failReleaseStatus, failReleaseDisposition = tc.at, tc.status, tc.disposition
				out, err := k.SnapshotToHost()
				wantStatus := int(tc.status); if wantStatus == 0 { wantStatus = 1 }
				op, ok := err.(*CUDAOpError)
				if !ok || op.Op != "SnapshotToHost" || op.Site != "kv-snapshot-release" || op.Code != wantStatus || !reflect.DeepEqual(out, KVHostSnapshot{}) { t.Fatalf("cleanup lost status or published image: out=%+v err=%#v", out, err) }
				wantCopies := 6; if tc.at == 1 { wantCopies = 1 }
				wantFrees := tc.at-1; if tc.disposition == 1 { wantFrees++ }
				if releaseCalls != tc.at || allocs != tc.at || conversions != tc.at || copies != wantCopies || frees != wantFrees || k.be.fenceGen != uint64(41+wantCopies) || nativeStop != wantStatus || invalidations != 0 || otherNative != 0 { t.Fatal("cleanup retried, ran later work, or lost successful-copy certificate") }
				if !k.releaseRetained || cudaKVReleaseOwners != k || k.releaseNext != nil { t.Fatal("cleanup failure lost source root") }
				holder := k.snapshotCopy
				if tc.disposition == 1 {
					if holder != nil { t.Fatal("transferred conversion remained a Go-owned retry target") }
				} else {
					if holder == nil || holder.conversion != allocated[len(allocated)-1] || holder.host != nil { t.Fatal("retained conversion lost identity or kept a completed host borrower") }
					for _, ptr := range released { if ptr == holder.conversion.ptr { t.Fatal("retained conversion already transferred") } }
				}
				after := snapshot(k); after.releaseRetained, after.releaseNext, after.snapshotCopy = before.releaseRetained, before.releaseNext, before.snapshotCopy
				if !reflect.DeepEqual(before, after) { t.Fatal("cleanup changed source metadata") }
				if dtype == BF16 { fault := k.be.faultLatch.Snapshot(); if fault.LastCode != wantStatus || fault.LastSite != "kv-snapshot-release" || fault.Health != DevicePoisoned { t.Fatal("real latch lost cleanup fault") } }
				k.Free(); _, retry := k.SnapshotToHost()
				if retry == nil || releaseCalls != tc.at || k.snapshotCopy != holder { t.Fatal("cleanup failure was retried or owner changed") }
				cudaMu.Lock(); sticky := C.fcuda_kv_snapshot_release_failed(888); cudaMu.Unlock()
				if int(sticky) != wantStatus || nativeStop != wantStatus { t.Fatal("later cleanup replaced first status") }
				runtime.GC() // completed-host holders must have unpinned despite cleanup failure
			})
		}
	}
	for _, dtype := range []Dtype{F16, BF16} {
		t.Run(fmt.Sprintf("healthy-narrowed-%d", dtype), func(t *testing.T) {
			k := fresh(dtype); before := snapshot(k)
			out, err := k.SnapshotToHost()
			if err != nil || len(out.K) != 2 || copies != 6 || conversions != 4 || allocs != 4 || frees != 4 || k.be.fenceGen != 47 ||
				nativeStop != 0 || k.snapshotCopy != nil || k.releaseRetained || cudaKVReleaseOwners != nil || !reflect.DeepEqual(snapshot(k), before) {
				t.Fatal("healthy narrowed snapshot ownership changed")
			}
			seen := map[unsafe.Pointer]bool{}
			for i, owner := range allocated { if released[i] != owner.ptr || seen[owner.ptr] { t.Fatal("healthy conversion cleanup order/identity changed") }; seen[owner.ptr] = true }
			runtime.GC() // healthy holders must have unpinned before becoming unreachable
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
