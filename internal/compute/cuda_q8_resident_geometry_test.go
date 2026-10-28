package compute

import (
	"context"
	"crypto/sha256"
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
// The estimate is the subprocess deadline, not a measured warm-cache duration.
// This host witness executes the complete production admission method, replacing
// only its allocation, transfer and cache dependencies. It does not execute CUDA
// or establish device numerical correctness. In particular, accepted geometries
// are controls for the guard, not a claim of GPU support or performance.
func TestCUDAQ8ResidentGeometryHost(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_Q8_RESIDENT_GEOMETRY_HOST") != "1" {
		t.Skip("set FAK_TEST_CUDA_Q8_RESIDENT_GEOMETRY_HOST=1 for the host admission witness")
	}
	runCUDAQ8ResidentHostProbe(t, "^TestAdmission$", "")
}

// The optional owner declaration comes from production, not a mirrored cache
// implementation. Its absence on the baseline must still compile and exercise
// the original method so a stale cache hit, not a missing symbol, is the RED.
func runCUDAQ8ResidentHostProbe(t *testing.T, run, extra string) {
	t.Helper()
	// go test runs in the package directory. runtime.Caller paths can be module-
	// relative under -trimpath, so they are not reliable filesystem locations.
	path := "cuda_upload.go"
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, path, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var method, owner []byte
	for _, decl := range f.Decls {
		if group, ok := decl.(*ast.GenDecl); ok && group.Tok == token.TYPE {
			for _, spec := range group.Specs {
				if typ, ok := spec.(*ast.TypeSpec); ok && typ.Name.Name == "cudaQ8UploadSource" {
					if owner != nil {
						t.Fatal("duplicate production Q8 source owner")
					}
					owner = append([]byte("type "), src[fs.Position(typ.Pos()).Offset:fs.Position(typ.End()).Offset]...)
				}
			}
		}
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "uploadQ8Resident" || fn.Recv == nil || len(fn.Recv.List) != 1 {
			continue
		}
		ptr, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		name, ok := ptr.X.(*ast.Ident)
		if !ok || name.Name != "cudaBackend" {
			continue
		}
		if method != nil {
			t.Fatal("duplicate production uploadQ8Resident method")
		}
		method = src[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset]
	}
	if method == nil {
		t.Fatal("production uploadQ8Resident method not found")
	}
	t.Logf("production method SHA256=%x", sha256.Sum256(method))
	t.Logf("production optional owner bytes=%d SHA256=%x", len(owner), sha256.Sum256(owner))
	dir := t.TempDir()
	probe := filepath.Join(dir, "admission_test.go")
	if err := os.WriteFile(probe, []byte(cudaQ8AdmissionHostShim+"\n"+string(owner)+"\n"+string(method)+"\n"+extra), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	var goExe string
	if root := runtime.GOROOT(); root != "" {
		goExe = filepath.Join(root, "bin", "go")
		if runtime.GOOS == "windows" {
			goExe += ".exe"
		}
	} else {
		// -trimpath can remove the compiled-in GOROOT. In that case use the
		// explicitly selected Go on PATH, never a relative "bin/go" guess.
		goExe, err = exec.LookPath("go")
		if err != nil {
			t.Fatalf("host witness needs Go on PATH when GOROOT is empty: %v", err)
		}
	}
	t.Logf("host witness Go executable: %s", goExe)
	cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", run, probe)
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
	t.Logf("host admission witness:\n%s", out)
	if ctx.Err() != nil {
		t.Fatalf("host admission witness deadline: %v", ctx.Err())
	}
	if err != nil {
		t.Fatalf("host admission witness failed: %v", err)
	}
}

const cudaQ8AdmissionHostShim = `package compute

import (
	"fmt"
	"strconv"
	"testing"
	"unsafe"
)

// These dependencies preserve the production method's integer, shape, typed
// refusal and cache-key contracts; only external effects are replaced by spies.
type Dtype uint8
const (
	F32 Dtype = iota
	F16
	BF16
	Q8_0
)
func (d Dtype) Bytes() int { if d == F32 { return 4 }; return 1 }
type Layout uint8
const RowMajor Layout = 0
type QuantSpec struct { Block int; Scale []float32 }
type Buffer interface { Ready() bool }
type HostBuffer interface { Buffer; F32() []float32; I8() []int8 }
type Tensor struct { Dtype Dtype; Layout Layout; Shape []int; Quant *QuantSpec; buf Buffer }
func (t Tensor) Numel() int { n := 1; for _, d := range t.Shape { n *= d }; return n }
type hostBuf struct { codes []int8; reads int }
func (*hostBuf) Ready() bool { return true }
func (*hostBuf) F32() []float32 { panic("unexpected host widening") }
func (h *hostBuf) I8() []int8 { h.reads++; return h.codes }
type MemoryClass string
const MemoryWeights MemoryClass = "weights"
type CUDAOpError struct { Op, Site, Msg string; Class MemoryClass }
func (e *CUDAOpError) Error() string { return e.Op + ": " + e.Site + ": " + e.Msg }
func itoaC(n int) string { return strconv.Itoa(n) }
type ucKey struct { host uintptr; dt Dtype; layout Layout }
var uploadCache map[ucKey]Tensor
type cudaBackend struct { allocs, uploads, accounts, bytes int; lastBlock, lastScales int }
type cudaBuf struct { owner *cudaBackend; host uintptr; hostKeep HostBuffer; hostDt Dtype; hostLo Layout; codes []int8; scales []float32 }
func (*cudaBuf) Ready() bool { return true }
func (c *cudaBackend) devQ8(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	c.allocs++; c.lastBlock = block; c.lastScales = nScales
	b := &cudaBuf{owner: c}
	return Tensor{Dtype: Q8_0, Layout: RowMajor, Shape: append([]int(nil), shape...), Quant: &QuantSpec{Block: block}, buf: b}, b
}
func cudaUploadQuantPayload(b *cudaBuf, codes []int8, scales []float32) {
	b.owner.uploads++; b.codes = append([]int8(nil), codes...); b.scales = append([]float32(nil), scales...)
}
func (c *cudaBackend) accountImmutableWeightUpload(n int, b *cudaBuf) {
	if b.owner != c { panic("wrong allocation owner") }; c.accounts++; c.bytes += n
}

func TestAdmission(t *testing.T) {
	cases := []struct { name string; block, width int; reject bool }{
		{"partial2", 2, 12, true}, {"partial3", 3, 12, true}, {"partial6", 6, 12, true},
		{"zero", 0, 12, true}, {"negative", -4, 12, true}, {"nondividing", 8, 12, true},
		{"aligned4", 4, 12, false}, {"aligned8", 8, 16, false}, {"aligned32", 32, 64, false},
	}
	for _, tc := range cases {
		for _, cached := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/cache=%t", tc.name, cached), func(t *testing.T) {
				c := &cudaBackend{}
				h := &hostBuf{codes: make([]int8, 2*tc.width)}
				for i := range h.codes { h.codes[i] = int8(i%7-3) }
				nScales := 2
				if tc.block > 0 { nScales = 2*(tc.width/tc.block) }
				scales := make([]float32, nScales)
				for i := range scales { scales[i] = float32(i+1)/8 }
				input := Tensor{Dtype: Q8_0, Layout: RowMajor, Shape: []int{2, tc.width}, Quant: &QuantSpec{Block: tc.block, Scale: scales}, buf: h}
				key := ucKey{uintptr(unsafe.Pointer(&h.codes[0])), Q8_0, RowMajor}
				uploadCache = make(map[ucKey]Tensor)
				seed := &cudaBuf{owner: c}
				if cached {
					if tc.reject {
						uploadCache[key] = Tensor{Dtype: Q8_0, Layout: RowMajor, Shape: []int{2, tc.width}, buf: seed}
					} else {
						seeded := c.uploadQ8Resident(input, h)
						seed = seeded.buf.(*cudaBuf)
						*c = cudaBackend{}
						h.reads = 0
					}
				}
				var got Tensor
				var recovered any
				func() { defer func() { recovered = recover() }(); got = c.uploadQ8Resident(input, h) }()
				if tc.reject {
					e, ok := recovered.(*CUDAOpError)
					if !ok || e == nil || e.Op != "Upload" || e.Site != "upload-q8-resident" || e.Class != MemoryWeights {
						t.Errorf("invalid geometry must produce typed upload refusal, got %#v", recovered)
					}
					if c.allocs != 0 || c.uploads != 0 || c.accounts != 0 || c.bytes != 0 || h.reads != 0 {
						t.Errorf("invalid geometry caused side effects: backend=%+v host reads=%d", c, h.reads)
					}
					wantEntries := 0; if cached { wantEntries = 1 }
					if len(uploadCache) != wantEntries || (cached && uploadCache[key].buf != seed) {
						t.Error("invalid geometry changed upload cache")
					}
					return
				}
				if recovered != nil { t.Fatalf("aligned geometry refused: %v", recovered) }
				if cached {
					if got.buf != seed || c.allocs != 0 || c.uploads != 0 || c.accounts != 0 || c.bytes != 0 || len(uploadCache) != 1 {
						t.Fatalf("valid cache hit did not preserve resident: backend=%+v", c)
					}
					return
				}
				buf, ok := got.buf.(*cudaBuf)
				if !ok || buf == nil || c.allocs != 1 || c.uploads != 1 || c.accounts != 1 || c.bytes != len(h.codes)+4*len(scales) || c.lastBlock != tc.block || c.lastScales != len(scales) {
					t.Fatalf("valid fresh geometry did not upload exact payload: backend=%+v", c)
				}
				if got.Numel() != input.Numel() || got.Quant.Block != tc.block || buf.hostKeep == nil || &buf.hostKeep.I8()[0] != &h.codes[0] || buf.host != key.host || buf.hostDt != Q8_0 || buf.hostLo != RowMajor || len(uploadCache) != 1 || uploadCache[key].buf != buf {
					t.Fatal("valid upload lost ownership, geometry or cache publication")
				}
				if len(buf.codes) != len(h.codes) || len(buf.scales) != len(scales) { t.Fatal("payload lengths changed") }
				for i, v := range h.codes { if buf.codes[i] != v { t.Fatal("code payload changed") } }
				for i, v := range scales { if buf.scales[i] != v { t.Fatal("scale payload changed") } }
			})
		}
	}
}
`
