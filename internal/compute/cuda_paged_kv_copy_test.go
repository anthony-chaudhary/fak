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

// Untagged wrapper: a required invocation must not disappear when CUDA tags,
// a reachable backend, or the typed qualification archive are absent.
var cudaPagedCopyRequired = flag.Bool("fak-cuda-paged-kv-copy-required", false, "require real qualified partial-tail COW, checked completion and all-plane parity")
var cudaPagedCopyHost = flag.Bool("fak-cuda-paged-kv-copy-host", false, "run the exact-source host ownership/orchestration witness")
var cudaPagedCopyDeviceWitness func(*testing.T)

// fak-test:runtime slow est=30s lane=optin
func TestCUDAPagedKVCopyQualification(t *testing.T) {
	if !*cudaPagedCopyRequired {
		t.Skip("requires separately admitted CUDA hardware and typed copy archive")
	}
	if cudaPagedCopyDeviceWitness == nil {
		t.Fatal("required tagged CUDA partial-tail witness absent; compile with CGO_ENABLED=1 and -tags cuda")
	}
	t.Run("partial-tail-all-planes", cudaPagedCopyDeviceWitness)
}

// Executes intact production pool/owner methods and the real page refcounter.
// CUDA status, backend admission, allocation and transfers are explicit host
// boundaries. It is not native compilation, fault injection, completion or
// numerical/performance evidence. Child compilation is opt-in and deadline-bound.
// fak-test:runtime slow est=60s lane=optin
func TestCUDAPagedKVCopyHostContract(t *testing.T) {
	if !*cudaPagedCopyHost {
		t.Skip("set -fak-cuda-paged-kv-copy-host for the CPU contract witness")
	}
	var program strings.Builder
	program.WriteString(cudaPagedCopyHostFixture)
	wanted := map[string]bool{
		"cudaKVPagePool": false, "cudaPagedPoolKey": false, "cudaPagedKV": false,
		"cudaPagedKVCopyFailure": false, "pagedKVBlocks": false,
		"*cudaBackend.pagedPool": false, "*cudaBackend.newPagedKV": false,
		"*cudaBackend.restorePagedKVFromHost":   false,
		"*cudaPagedKVCopyFailure.Error":         false,
		"*cudaKVPagePool.retainCopyOwnerLocked": false,
		"*cudaKVPagePool.stopCopyLocked":        false, "*cudaKVPagePool.copyErrorLocked": false,
		"*cudaKVPagePool.allocLocked": false, "*cudaKVPagePool.growLocked": false,
		"*cudaKVPagePool.copyBlockLocked":       false,
		"*cudaPagedKV.requireCopyHealthyLocked": false, "*cudaPagedKV.writablePageLocked": false,
		"*cudaPagedKV.Clone": false, "*cudaPagedKV.Free": false, "*cudaPagedKV.releaseLocked": false,
		"*pagedKVBlocks.grow": false, "*pagedKVBlocks.alloc": false, "*pagedKVBlocks.check": false,
		"*pagedKVBlocks.retain": false, "*pagedKVBlocks.release": false, "*pagedKVBlocks.shared": false,
		"pagedKVGrowBlocks": false,
	}
	for _, path := range []string{"cuda_paged_kv.go", "cuda_paged_kv_blocks.go"} {
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
				if node.Tok == token.TYPE {
					for _, spec := range node.Specs {
						if typ, ok := spec.(*ast.TypeSpec); ok {
							emit(typ.Name.Name, typ, "type ")
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
	probe := filepath.Join(dir, "paged_copy_test.go")
	if err := os.WriteFile(probe, []byte(program.String()), 0600); err != nil {
		t.Fatal(err)
	}
	goExe := filepath.Join(runtime.GOROOT(), "bin", "go")
	if runtime.GOOS == "windows" {
		goExe += ".exe"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", "^TestPagedCopyHost$", probe)
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

// Source bindings complement the executed owner chain: every other actual pool
// consumer must refuse before its first mutation/transfer. They are not behavior
// proof for native calls or a replacement for the required device witness.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAPagedKVCopyConsumerGuards(t *testing.T) {
	source, err := os.ReadFile("cuda_paged_kv.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "cuda_paged_kv.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	direct := map[string]int{
		"writablePageLocked": 0, "appendRowsLocked": 0, "syncDeviceTableLocked": 0,
		"pagedAttentionLocked": 0, "gatherLocked": 0, "viewLocked": 0,
		"releaseLocked": 0, "loadFromHostLocked": 0,
		"pagedPanelAppendLocked": 0, "pagedPanelAttentionLocked": 0,
		"AppendKV": 2, "KeysView": 2, "ValuesView": 2, "Clone": 2,
	}
	sharedStopSeen := false
	for _, decl := range file.Decls {
		f, ok := decl.(*ast.FuncDecl)
		if !ok || f.Body == nil {
			continue
		}
		if f.Name.Name == "copyErrorLocked" {
			sharedStopSeen = true
			// Cached failure, qualified admission, then the shared terminal
			// query must remain ordered siblings. The last check must also run
			// when qualification is off and no Go fault latch is installed.
			if len(f.Body.List) != 4 {
				t.Fatal("shared stop query is not an unconditional pool boundary")
			}
			for i, want := range []string{
				"if status := p.copyFailure.Load(); status != 0",
				"if cudaPagedKVCopyQualification",
				"if C.fcuda_paged_kv_copy_terminal() != 0",
			} {
				stmt := f.Body.List[i]
				text := strings.Join(strings.Fields(string(source[fs.Position(stmt.Pos()).Offset:fs.Position(stmt.End()).Offset])), " ")
				if !strings.HasPrefix(text, want+" {") {
					t.Fatalf("pool check %d = %q, want %q", i, text, want)
				}
			}
		}
		at, ok := direct[f.Name.Name]
		if !ok {
			continue
		}
		if len(f.Body.List) <= at {
			t.Fatalf("missing first guard in %s", f.Name.Name)
		}
		stmt := f.Body.List[at]
		body := string(source[fs.Position(stmt.Pos()).Offset:fs.Position(stmt.End()).Offset])
		if body != "k.requireCopyHealthyLocked()" {
			t.Fatalf("%s first effect is %q", f.Name.Name, body)
		}
		delete(direct, f.Name.Name)
	}
	if !sharedStopSeen {
		t.Fatal("missing shared pool stop boundary")
	}
	if len(direct) != 0 {
		t.Fatalf("missing guarded consumers: %v", direct)
	}
	text := string(source)
	for _, binding := range []string{
		"if err := k.pool.copyErrorLocked(k); err != nil",
		"if status := p.copyFailure.Load(); status != 0",
		"k = c.newPagedKV(cloneKVConfig(state.Config))",
		"k != nil && k.pool.copyErrorLocked(k) == nil",
	} {
		if !strings.Contains(text, binding) {
			t.Fatalf("missing snapshot/restore/new-owner binding %q", binding)
		}
	}
	if KernelOpClass("fak_paged_kv_copy_blocks") != KernelOpCopy ||
		KernelOpClass("fak_paged_attention") != KernelOpAttention {
		t.Fatal("copy classification changed attention vocabulary")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAPagedKVCopyLaunchStatusContract(t *testing.T) {
	data, err := os.ReadFile("cuda/paged_attention_vllm.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	start := strings.Index(source, "extern \"C\" int fcuda_paged_kv_copy_blocks_f32(")
	if start < 0 {
		t.Fatal("missing actual copy adapter")
	}
	body := strings.Join(strings.Fields(source[start:]), " ")
	for _, token := range []string{
		"g_capture_open || g_q4k_qkv_async_pending != 0",
		"capture != cudaStreamCaptureStatusNone",
		"cudaPeekAtLastError()", "v41c_pending_contains(base)",
		"g_live.find", "g_managed_live.find",
		"cudaPointerGetAttributes", "capacity_blocks > SIZE_MAX / page_bytes",
		"FCUDA_TS(\"fak_paged_kv_copy_blocks\", planes,",
		"args, block_floats, dst, src)); status = cudaGetLastError()); return (int)status;",
	} {
		if !strings.Contains(body, token) {
			t.Fatalf("lost native preflight/launch binding %q", token)
		}
	}
	if strings.Count(body, "cudaGetLastError()") != 1 ||
		strings.Contains(body, "cudaMemcpy") || strings.Contains(body, "cudaStreamSynchronize") {
		t.Fatal("copy adapter consumed a second error, added a fence, or fell back after launch")
	}
}

const cudaPagedCopyHostFixture = `package compute

import (
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

const CUDAPagedKVBlockTokens = 16

type KVConfig struct{ NumLayers, NumKVHeads, HeadDim int }
type KVHostSnapshot struct{ Config KVConfig }

func (s KVHostSnapshot) Validate() error { return nil }
func (k *cudaPagedKV) loadFromHostLocked(KVHostSnapshot) {
	panic("unexpected healthy host restoration in failure-only fixture")
}

type KVStore interface {
	Clone() KVStore
	Free()
}
type cudaBackend struct{ fault bool }
type cudaBuf struct{ ptr unsafe.Pointer }
// Entry/COW fixture only; the dedicated snapshot fixture extracts the real holder.
type cudaPagedKVSnapshotCopy struct{}

func cloneKVConfig(c KVConfig) KVConfig { return c }
func itoaC(v int) string                { return strconv.Itoa(v) }
func (c *cudaBackend) dallocKV(int, string) *cudaBuf {
	panic("unexpected device allocation in pre-grown host fixture")
}
func AdmitDevice(c *cudaBackend, _ string) error {
	if c.fault {
		return fmt.Errorf("host device-fault boundary")
	}
	return nil
}

var cudaMu sync.Mutex
var cudaPagedPoolsMu sync.Mutex
var cudaPagedPools = map[cudaPagedPoolKey]*cudaKVPagePool{}
var cudaPagedKVCopyQualification bool
var copyStatus int32
var terminalStatus int32
var copyCalls, d2dCalls, freeCalls, terminalQueries int
var expectedPool *cudaKVPagePool
var C = struct {
	int                            func(any) int32
	size_t                         func(int) uintptr
	fcuda_paged_kv_copy_terminal   func() int32
	fcuda_paged_kv_copy_blocks_f32 func(unsafe.Pointer, int32, int32, int32, int32, int32) int32
	fcuda_d2d                      func(unsafe.Pointer, unsafe.Pointer, uintptr)
	fcuda_free                     func(unsafe.Pointer)
	fcuda_discard_unsubmitted      func(unsafe.Pointer, uintptr) int32
}{
	int: func(v any) int32 {
		switch x := v.(type) {
		case int:
			return int32(x)
		case int32:
			return x
		default:
			panic("unexpected C integer argument")
		}
	},
	size_t:                       func(v int) uintptr { return uintptr(v) },
	fcuda_paged_kv_copy_terminal: func() int32 { terminalQueries++; return terminalStatus },
	fcuda_paged_kv_copy_blocks_f32: func(raw unsafe.Pointer, planes, capacity, floats, dst, src int32) int32 {
		p := expectedPool
		if planes != int32(3*p.layers) || capacity != int32(p.capBlocks) ||
			floats != int32(p.blockFloats) || src == dst || src < 0 || dst < 0 ||
			int(src) >= p.capBlocks || int(dst) >= p.capBlocks {
			panic("incorrect actual copy arguments")
		}
		got := unsafe.Slice((*unsafe.Pointer)(raw), int(planes))
		want := append(append(append([]unsafe.Pointer(nil), p.K...), p.Kraw...), p.V...)
		if !reflect.DeepEqual(got, want) {
			panic("missing/reordered plane arena")
		}
		copyCalls++
		return copyStatus
	},
	fcuda_d2d:                 func(unsafe.Pointer, unsafe.Pointer, uintptr) { d2dCalls++ },
	fcuda_free:                func(unsafe.Pointer) { freeCalls++ },
	fcuda_discard_unsubmitted: func(unsafe.Pointer, uintptr) int32 { panic("unexpected arena retirement") },
}

func setup(layers int, enabled bool) (*cudaPagedKV, *cudaPagedKV) {
	cudaPagedKVCopyQualification, copyStatus, terminalStatus = enabled, 0, 0
	copyCalls, d2dCalls, freeCalls, terminalQueries = 0, 0, 0, 0
	cudaPagedPools = map[cudaPagedPoolKey]*cudaKVPagePool{}
	c := &cudaBackend{}
	cfg := KVConfig{NumLayers: layers, NumKVHeads: 1, HeadDim: 32}
	a := c.newPagedKV(cfg)
	p := a.pool
	p.capBlocks = 4
	p.blocks.grow(4)
	// Distinct host-owned storage is only a pointer fixture: no device load/store
	// is executed. Real allocator growth remains in the extracted source.
	for _, plane := range [][]unsafe.Pointer{p.K, p.Kraw, p.V} {
		for i := range plane {
			plane[i] = unsafe.Pointer(new([8192]byte))
		}
	}
	a.table = []int32{p.allocLocked()}
	a.lens[0], a.pos = 7, []int{0, 1, 2, 3, 4, 5, 6}
	a.dirty = false
	for _, ptr := range []*unsafe.Pointer{&a.dTable, &a.dSeqLen, &a.scratch, &a.gK, &a.gV} {
		*ptr = unsafe.Pointer(new(int))
	}
	a.dTableCap, a.scratchF, a.gKF, a.gVF, a.seqLenDev = 4, 19, 13, 17, 7
	b := a.Clone().(*cudaPagedKV)
	expectedPool = p
	return a, b
}

func terminal(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		if _, ok := r.(*cudaPagedKVCopyFailure); !ok {
			t.Fatalf("want typed terminal refusal, got %T %v", r, r)
		}
	}()
	f()
}

func stable(k *cudaPagedKV) string {
	p := k.pool
	return fmt.Sprintf("%v/%v/%v/%v/%v/%d/%d/%d/%p/%p/%p/%p/%p/%d/%d/%d/%d/%d/%v",
		k.table, k.lens, k.pos, p.blocks.refs, p.blocks.free, p.blocks.live, p.capBlocks, p.copySubmissions,
		k.dTable, k.dSeqLen, k.scratch, k.gK, k.gV,
		k.dTableCap, k.scratchF, k.gKF, k.gVF, k.seqLenDev, k.dirty)
}

func retained(t *testing.T, p *cudaKVPagePool, want ...*cudaPagedKV) {
	t.Helper()
	seen := map[*cudaPagedKV]bool{}
	for k := p.copyOwners; k != nil; k = k.copyNext {
		if seen[k] {
			t.Fatal("duplicate/cyclic retained owner")
		}
		seen[k] = true
		if !k.copyRetained {
			t.Fatal("retained owner missing marker")
		}
	}
	if len(seen) != len(want) {
		t.Fatalf("retained %d owners, want %d", len(seen), len(want))
	}
	for _, k := range want {
		if !seen[k] {
			t.Fatal("lost store/private-buffer owner")
		}
	}
}

func TestPagedCopyHost(t *testing.T) {
	t.Run("failed-copy-retains-and-stops-all-owners", func(t *testing.T) {
		a, b := setup(3, true)
		p := a.pool
		old := a.table[0]
		copyStatus = 17
		terminal(t, func() { a.writablePageLocked(0) })
		if a.table[0] != old || b.table[0] != old || p.blocks.refs[old] != 2 ||
			p.blocks.live != 2 || p.blocks.refs[1] != 1 || len(p.blocks.free) != 2 || copyCalls != 1 ||
			d2dCalls != 0 || p.copySubmissions != 0 || freeCalls != 0 || a.dirty {
			t.Fatal("copy failure published/released/retried or lost the quarantined page")
		}
		retained(t, p, a)
		beforeA, beforeB := stable(a), stable(b)
		queries := terminalQueries
		for i := 0; i < 2; i++ {
			terminal(t, func() { a.writablePageLocked(0) })
			terminal(t, func() { p.allocLocked() })
			terminal(t, func() { p.growLocked(5) })
			terminal(t, func() { a.Clone() })
			terminal(t, a.Free)
			terminal(t, a.releaseLocked)
			terminal(t, b.Free)
			terminal(t, func() { p.be.newPagedKV(a.cfg) })
			if out, err := p.be.restorePagedKVFromHost(KVHostSnapshot{Config: a.cfg}); out != nil || err == nil {
				t.Fatal("terminal restore must return error without minting an owner")
			}
		}
		retained(t, p, a, b)
		if stable(a) != beforeA || stable(b) != beforeB || copyCalls != 1 || d2dCalls != 0 || freeCalls != 0 || terminalQueries != queries {
			t.Fatal("terminal pool changed page, private storage, accounting or submissions")
		}
		if err := p.stopCopyLocked(99, b); err.(*cudaPagedKVCopyFailure).Status != 17 {
			t.Fatal("lost first error")
		}
		empty := &cudaPagedKV{pool: p}
		terminal(t, empty.Free)
		retained(t, p, a, b) // empty refusal must not mint an unbounded retained list
	})
	t.Run("default-shared-stop-retains-before-release", func(t *testing.T) {
		a, b := setup(2, false) // no qualification and no Go admission fault
		p := a.pool
		for _, ptr := range []*unsafe.Pointer{&b.dTable, &b.dSeqLen, &b.scratch, &b.gK, &b.gV} {
			*ptr = unsafe.Pointer(new(int))
		}
		b.dTableCap, b.scratchF, b.gKF, b.gVF = 3, 11, 5, 9
		beforeA, beforeB := stable(a), stable(b)
		arenas := append(append(append([]unsafe.Pointer(nil), p.K...), p.Kraw...), p.V...)
		queries := terminalQueries
		terminalStatus = 1
		terminal(t, a.Free)
		if p.copyFailure.Load() != -75701 || terminalQueries != queries+1 {
			t.Fatal("default archive did not latch shared terminal status")
		}
		queries = terminalQueries
		for i := 0; i < 2; i++ {
			terminal(t, a.Free)
			terminal(t, a.releaseLocked)
			terminal(t, b.Free)
			terminal(t, func() { p.allocLocked() })
			terminal(t, func() { p.growLocked(5) })
			terminal(t, func() { a.writablePageLocked(0) })
			terminal(t, func() { a.Clone() })
			terminal(t, func() { p.be.newPagedKV(a.cfg) })
			if out, err := p.be.restorePagedKVFromHost(KVHostSnapshot{Config: a.cfg}); out != nil || err == nil {
				t.Fatal("shared-stop restore minted an owner")
			}
		}
		retained(t, p, a, b)
		if stable(a) != beforeA || stable(b) != beforeB ||
			!reflect.DeepEqual(arenas, append(append(append([]unsafe.Pointer(nil), p.K...), p.Kraw...), p.V...)) ||
			copyCalls != 0 || d2dCalls != 0 || freeCalls != 0 || terminalQueries != queries {
			t.Fatal("shared stop changed pages, arenas, private owners, accounting or submissions")
		}
		if err := p.stopCopyLocked(99, b); err.(*cudaPagedKVCopyFailure).Status != -75701 {
			t.Fatal("shared stop lost first-error identity")
		}
		empty := &cudaPagedKV{pool: p}
		terminal(t, empty.Free)
		retained(t, p, a, b)
	})
	t.Run("terminal-reservation-retains-unwinding-owner", func(t *testing.T) {
		a, b := setup(3, true)
		p := a.pool
		// Exercise the real selected capacity-overflow refusal before any
		// fresh page or native copy exists. Existing owners still must survive.
		p.blocks.free = nil
		p.capBlocks = int(^uint32(0) >> 1)
		beforeA, beforeB := stable(a), stable(b)
		terminal(t, func() { a.writablePageLocked(len(a.table)) })
		if p.copyFailure.Load() != -75702 || stable(a) != beforeA || stable(b) != beforeB ||
			copyCalls != 0 || d2dCalls != 0 || freeCalls != 0 {
			t.Fatal("reservation refusal changed existing owners")
		}
		retained(t, p, a)
		terminal(t, b.Free)
		retained(t, p, a, b)
	})
	for _, native := range []bool{false, true} {
		name := "backend-fault"
		if native {
			name = "native-uncertainty"
		}
		t.Run(name, func(t *testing.T) {
			a, b := setup(2, true)
			if native {
				terminalStatus = 1
			} else {
				a.pool.be.fault = true
			}
			before := stable(a)
			terminal(t, a.Free)
			terminal(t, b.Free)
			if stable(a) != before || copyCalls != 0 || d2dCalls != 0 || freeCalls != 0 {
				t.Fatal("uncertainty released or submitted")
			}
			retained(t, a.pool, a, b)
		})
	}
	for _, tc := range []struct {
		name           string
		layers         int
		enabled        bool
		copies, legacy int
	}{
		{"selected", 3, true, 1, 0}, {"default", 3, false, 0, 9}, {"ineligible", 65, true, 0, 195},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, b := setup(tc.layers, tc.enabled)
			p, old := a.pool, a.table[0]
			fresh := a.writablePageLocked(0)
			if fresh == old || b.table[0] != old || p.blocks.refs[old] != 1 || p.blocks.refs[fresh] != 1 ||
				copyCalls != tc.copies || d2dCalls != tc.legacy || p.copySubmissions != uint64(tc.copies) {
				t.Fatal("healthy selection/refcount behavior changed")
			}
			a.Free()
			b.Free()
			if terminalQueries == 0 {
				t.Fatal("healthy pool skipped shared native health checks")
			}
			if p.blocks.live != 0 || len(p.blocks.free) != p.capBlocks || freeCalls != 5 || p.copyOwners != nil {
				t.Fatal("healthy cleanup changed")
			}
		})
	}
	t.Run("bounded-geometry", func(t *testing.T) {
		cudaPagedKVCopyQualification = true
		defer func() {
			if recover() == nil {
				t.Fatal("overflowing page geometry admitted")
			}
		}()
		(&cudaBackend{}).newPagedKV(KVConfig{NumLayers: 1, NumKVHeads: int(^uint32(0) >> 1), HeadDim: 256})
	})
}
`
