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
// The estimate is the subprocess deadline, not a measured duration. This host
// witness executes the real constructor/copy helpers and complete public Upload
// and Free chains, including cache eviction, plus complete Recycle and targeted
// transient release. Allocation, transfer, driver calls, accounting, and the
// named Q56 terminal guard are explicit host fixtures. Retention cases exercise
// real cleanup control flow, not CUDA fault injection, fences or poison recovery.
// Device numerics and serving-model behavior are not exercised.
func TestCUDAUploadCacheOwnerHost(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_UPLOAD_CACHE_OWNER_HOST") != "1" {
		t.Skip("set FAK_TEST_CUDA_UPLOAD_CACHE_OWNER_HOST=1 for the host upload-cache owner witness")
	}
	var program strings.Builder
	program.WriteString(cudaUploadCacheOwnerHostFixture)
	wanted := map[string]bool{
		"QuantSpec": false, "Buffer": false, "HostBuffer": false, "Tensor": false,
		"cudaQ8UploadSource": false, "makeTensor": false, "Tensor.Numel": false,
		"NewQ2": false, "bytesAsI8": false, "*cudaBackend.uploadQ2Resident": false,
		"*cudaBackend.Upload": false, "*cudaBackend.uploadClass": false,
		"*cudaBackend.Free": false, "*cudaBackend.cudaFreeLocked": false,
		"*cudaBackend.cudaFreeStorageLocked": false,
		"*cudaBackend.Recycle":               false, "*cudaBackend.releaseTransientBuffers": false,
	}
	for _, path := range []string{"compute.go", "quant_q2.go", "cuda_upload.go", "cuda.go"} {
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, path, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		emit := func(key string, node ast.Node, prefix string) {
			seen, ok := wanted[key]
			if !ok {
				return
			}
			if seen {
				t.Fatalf("duplicate production declaration %s", key)
			}
			wanted[key] = true
			body := src[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset]
			t.Logf("%s/%s SHA256=%x", path, key, sha256.Sum256(body))
			program.WriteString(prefix)
			program.Write(body)
			program.WriteByte('\n')
		}
		for _, decl := range file.Decls {
			switch node := decl.(type) {
			case *ast.GenDecl:
				if node.Tok != token.TYPE {
					continue
				}
				for _, spec := range node.Specs {
					if typ, ok := spec.(*ast.TypeSpec); ok {
						emit(typ.Name.Name, typ, "type ")
					}
				}
			case *ast.FuncDecl:
				if node.Body == nil {
					continue
				}
				key := node.Name.Name
				if node.Recv != nil {
					if len(node.Recv.List) != 1 {
						continue
					}
					r := node.Recv.List[0].Type
					prefix := ""
					if ptr, ok := r.(*ast.StarExpr); ok {
						prefix = "*"
						r = ptr.X
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
			t.Fatalf("missing production declaration %s", key)
		}
	}
	dir := t.TempDir()
	probe := filepath.Join(dir, "cache_owner_test.go")
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
			t.Fatalf("host upload-cache owner witness needs Go on PATH: %v", err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", "^TestUploadCacheOwner$", probe)
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
	t.Logf("host upload-cache owner witness (%s):\n%s", goExe, out)
	if err != nil {
		t.Fatalf("host upload-cache owner witness: %v", err)
	}
}

const cudaUploadCacheOwnerHostFixture = `package compute

import (
	"slices"
	"strconv"
	"sync"
	"testing"
	"unsafe"
)

// Only host/driver dependencies are fixtures. The public Upload and Free
// methods, all intermediate branches, cache deletion and both transient cleanup
// methods are extracted intact. The named terminal predicate below is a fake
// boundary; no production statement is rewritten or conditionally omitted.
type Dtype uint8

const (
	F32 Dtype = iota
	F16
	Q8_0
	Q2_0
	NVFP4
	Q3_K
	Q4_K
	Q5_K
	Q6_K
	Q2_K
	IQ2_XXS
	IQ2_XS
	IQ1_M
	IQ4_NL
	IQ4_XS
)

func (d Dtype) Bytes() int {
	if d != F32 {
		panic("unexpected fixture byte width")
	}
	return 4
}
func (d Dtype) String() string { return strconv.Itoa(int(d)) }

type Layout uint8

const RowMajor Layout = 0

type Backend interface{}
type hostBuf struct{ i8 []int8 }

func (*hostBuf) Ready() bool    { return true }
func (*hostBuf) F32() []float32 { panic("unexpected F32 fixture route") }
func (h *hostBuf) I8() []int8   { return h.i8 }

type MemoryClass string

const MemoryWeights MemoryClass = "weights"
const q8DeviceBlock = 32

type CUDAOpError struct {
	Op, Site, Msg string
	Class         MemoryClass
}

func (e *CUDAOpError) Error() string { return e.Op + ": " + e.Site + ": " + e.Msg }
func itoaC(n int) string             { return strconv.Itoa(n) }

type ucKey struct {
	hp uintptr
	dt Dtype
	lo Layout
}

var uploadCache map[ucKey]Tensor
var cudaMu sync.Mutex

type uploadCounts struct{ allocs, transfers, accounts, bytes int }
type cudaBackend struct {
	uploadCounts
	dlUsed, managedN     int
	dependencies, drains int
	compressorDrains     int
	transient            []*cudaBuf
}
type cudaBuf struct {
	owner                          *cudaBackend
	ptr, scales                    unsafe.Pointer
	device                         int
	host                           uintptr
	hostKeep                       HostBuffer
	hostDt                         Dtype
	hostLo                         Layout
	graphOncePinned, managedWeight bool
	budgetedWeightBytes            int
	codes                          []int8
	scaleValues                    []float32
}

func (*cudaBuf) Ready() bool { return true }
func (c *cudaBackend) devQ2(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	if nScales != shape[0]*(shape[1]/block) {
		panic("wrong scale allocation geometry")
	}
	c.allocs++
	b := &cudaBuf{owner: c, codes: make([]int8, shape[0]*shape[1]/4), scaleValues: make([]float32, nScales)}
	b.ptr, b.scales = unsafe.Pointer(&b.codes[0]), unsafe.Pointer(&b.scaleValues[0])
	return makeTensor(c, Q2_0, RowMajor, slices.Clone(shape), &QuantSpec{Block: block}, b), b
}
func cudaUploadQuantPayload(b *cudaBuf, codes []int8, scales []float32) {
	if len(codes) != len(b.codes) || len(scales) != len(b.scaleValues) {
		panic("wrong transfer extent")
	}
	b.owner.transfers++
	copy(b.codes, codes)
	copy(b.scaleValues, scales)
}
func (c *cudaBackend) accountImmutableWeightUpload(n int, b *cudaBuf) {
	if b.owner != c {
		panic("wrong allocation owner")
	}
	c.accounts++
	c.bytes += n
	c.dlUsed += n
	b.budgetedWeightBytes = n
}

// Unexercised upload alternatives fail closed if the extracted dispatch changes.
func (*cudaBackend) uploadNVFP4(Tensor, HostBuffer) Tensor { panic("unexpected NVFP4 route") }
func (*cudaBackend) uploadRawKQuantClass(Tensor, HostBuffer, MemoryClass, string) Tensor {
	panic("unexpected raw quant route")
}
func (*cudaBackend) uploadQ8Resident(Tensor, HostBuffer) Tensor {
	panic("unexpected resident Q8 route")
}
func (*cudaBackend) uploadF16(Tensor, HostBuffer, []float32, uintptr) Tensor {
	panic("unexpected F16 route")
}
func (*cudaBackend) uploadQ8(Tensor, HostBuffer, []float32, uintptr) Tensor {
	panic("unexpected narrowing Q8 route")
}
func (*cudaBackend) dallocClass(int, MemoryClass, string) *cudaBuf {
	panic("unexpected classed allocation")
}
func (*cudaBackend) dev([]int, Dtype) (Tensor, *cudaBuf)        { panic("unexpected F32 allocation") }
func makeF32TensorLike(*cudaBackend, Tensor, *cudaBuf) Tensor   { panic("unexpected F32 descriptor") }
func finishF32Upload(Tensor, []float32, func([]float32)) Tensor { panic("unexpected F32 transfer") }

// These inactive dependencies are explicit fixture boundaries, not evidence of
// capture, graph-pin, pending-owner, or native asynchronous error correctness.
// Q56 is an explicit simulated predicate for the actual extracted release entry
// points, not a fallback for a missing method and not a production reset API.
var q56ReleaseBlocked bool
var q56ReleaseChecks int
var cudaKernelProfileOn = true
var profileBoundaries int

func cudaQ56KMMVQReleaseBlocked() bool {
	q56ReleaseChecks++
	return q56ReleaseBlocked
}
func cudaV41CompressorDrainLocked(c *cudaBackend)                  { c.compressorDrains++ }
func cudaV41CompressorPendingIndex(*cudaBuf) int                   { return -1 }
func cudaQ4KAsyncReleaseBlockedLocked() bool                       { return false }
func (c *cudaBackend) graphOnceFreeDependencyLocked(*cudaBuf)      { c.dependencies++ }
func cudaV41CompressorDeferFreeLocked(*cudaBackend, *cudaBuf) bool { return false }
func cudaQ4KAsyncRequireDrainLocked(c *cudaBackend)                { c.drains++ }

var awaited []unsafe.Pointer

func awaitAsyncUploadLocked(p unsafe.Pointer) { awaited = append(awaited, p) }

var freed map[unsafe.Pointer]int

func fakeFree(p unsafe.Pointer) {
	if p == nil {
		panic("nil fixture free")
	}
	freed[p]++
}

var C = struct {
	fcuda_ts_boundary func()
	fcuda_free        func(unsafe.Pointer)
	fcuda_free_on     func(int, unsafe.Pointer)
	fcuda_h2d         func(unsafe.Pointer, unsafe.Pointer, int)
	int               func(int) int
	size_t            func(int) int
}{
	fcuda_ts_boundary: func() { profileBoundaries++ },
	fcuda_free:        fakeFree,
	fcuda_free_on:     func(int, unsafe.Pointer) { panic("unexpected cross-device free") },
	fcuda_h2d:         func(unsafe.Pointer, unsafe.Pointer, int) { panic("unexpected F32 H2D") },
	int:               func(n int) int { return n }, size_t: func(n int) int { return n },
}

func TestUploadCacheOwner(t *testing.T) {
	q56ReleaseBlocked, q56ReleaseChecks, profileBoundaries = false, 0, 0
	c := &cudaBackend{}
	uploadCache = make(map[ucKey]Tensor)
	freed = make(map[unsafe.Pointer]int)
	awaited = nil
	packed := []byte{0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa}
	first := NewQ2(c, []int{1, 32}, packed, []float32{1}, 32)
	independent := NewQ2(c, []int{1, 32}, packed, first.Quant.Scale, 32)
	codes := first.buf.(HostBuffer).I8()
	if &codes[0] == &independent.buf.(HostBuffer).I8()[0] {
		t.Fatal("separate constructors must not alias packed codes")
	}
	// Tensor is a value. Give its copy NEW immutable quantization metadata,
	// without changing the original descriptor, codes, or scale allocation.
	second := first
	q2 := *first.Quant
	q2.Scale = []float32{2}
	second.Quant = &q2
	if second.buf != first.buf || second.Quant == first.Quant {
		t.Fatal("invalid immutable descriptor-copy fixture")
	}
	a := c.Upload(first, Q2_0)
	b := c.Upload(second, Q2_0)
	ab, bb := a.buf.(*cudaBuf), b.buf.(*cudaBuf)
	if ab == bb || c.uploadCounts != (uploadCounts{2, 2, 2, 24}) || c.dlUsed != 24 {
		t.Fatalf("expected distinct A/B residents: counts=%+v budget=%d", c.uploadCounts, c.dlUsed)
	}
	if !slices.Equal(ab.codes, codes) || !slices.Equal(bb.codes, codes) || !slices.Equal(ab.scaleValues, []float32{1}) || !slices.Equal(bb.scaleValues, []float32{2}) {
		t.Fatal("wrong resident payload")
	}
	key := ucKey{uintptr(unsafe.Pointer(&codes[0])), Q2_0, RowMajor}
	if uploadCache[key].buf != bb {
		t.Fatal("newer B did not replace the cache entry")
	}
	ap, as, bp, bs := ab.ptr, ab.scales, bb.ptr, bb.scales
	counts := c.uploadCounts
	c.Free(a)
	if uploadCache[key].buf != bb {
		t.Fatal("freeing displaced A evicted newer B")
	}
	if ab.ptr != nil || ab.scales != nil || ab.hostKeep != nil || ab.budgetedWeightBytes != 0 || c.dlUsed != 12 || c.managedN != 0 {
		t.Fatal("A storage or accounting was not released normally")
	}
	if freed[ap] != 1 || freed[as] != 1 || freed[bp] != 0 || freed[bs] != 0 || len(freed) != 2 || !slices.Equal(awaited, []unsafe.Pointer{ap}) || c.dependencies != 1 || c.drains != 1 {
		t.Fatal("displaced free changed release/drain dependencies or freed B")
	}
	if got := c.Upload(second, Q2_0); got.buf != bb || c.uploadCounts != counts || c.dlUsed != 12 {
		t.Fatal("B cache hit repeated allocation, transfer, or accounting")
	}
	c.Free(a)
	if uploadCache[key].buf != bb || len(freed) != 2 || freed[ap] != 1 || freed[as] != 1 || len(awaited) != 1 || c.dependencies != 1 || c.drains != 2 {
		t.Fatal("repeated Free(A) changed storage or the newer cache owner")
	}
	if got := c.Upload(second, Q2_0); got.buf != bb || c.uploadCounts != counts {
		t.Fatal("repeat-free lost B reuse")
	}
	// The CURRENT owner still evicts, so a later upload must stage anew.
	c.Free(b)
	if _, ok := uploadCache[key]; ok {
		t.Fatal("current B owner was not evicted")
	}
	if bb.ptr != nil || bb.scales != nil || bb.hostKeep != nil || c.dlUsed != 0 || freed[bp] != 1 || freed[bs] != 1 {
		t.Fatal("B did not release normally")
	}
	again := c.Upload(second, Q2_0)
	if again.buf == bb || c.uploadCounts != (uploadCounts{3, 3, 3, 36}) || c.dlUsed != 12 {
		t.Fatal("current-owner free failed to force re-upload")
	}
	if first.Quant.Block != 32 || second.Quant.Block != 32 || !slices.Equal(first.Quant.Scale, []float32{1}) || !slices.Equal(second.Quant.Scale, []float32{2}) || !slices.Equal(codes, []int8{-86, -86, -86, -86, -86, -86, -86, -86}) || !slices.Equal(packed, []byte{0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa}) {
		t.Fatal("Upload/Free mutated immutable host inputs")
	}
	c.Free(again)
	if len(uploadCache) != 0 || c.dlUsed != 0 {
		t.Fatal("final current-owner cleanup failed")
	}
	t.Run("Q56TerminalRetention", testUploadOwnerTerminalRetention)
}

// Every state below is a fresh HOST fixture. The healthy controls do not reset
// a failed native stream or assert that production quarantine can be recovered.
func testUploadOwnerTerminalRetention(t *testing.T) {
	for _, name := range []string{"Free", "Storage", "Recycle", "Transient"} {
		for _, blocked := range []bool{false, true} {
			t.Run(name+"/blocked="+strconv.FormatBool(blocked), func(t *testing.T) {
				c := &cudaBackend{}
				uploadCache = make(map[ucKey]Tensor)
				freed = make(map[unsafe.Pointer]int)
				awaited = nil
				q56ReleaseBlocked, q56ReleaseChecks, profileBoundaries = false, 0, 0
				defer func() { q56ReleaseBlocked = false }() // fake isolation, not native recovery
				codes := []byte{0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa}
				host := NewQ2(c, []int{1, 32}, codes, []float32{2}, 32)
				w := c.Upload(host, Q2_0)
				wb := w.buf.(*cudaBuf)
				// Exercise the existing positive accounting fields at the fixture
				// boundary; this is not a claim that Q2 upload selects managed RAM.
				wb.managedWeight, c.managedN = true, 1
				wp, ws, keep := wb.ptr, wb.scales, wb.hostKeep.(cudaQ8UploadSource)
				key := ucKey{wb.host, wb.hostDt, wb.hostLo}
				tb := &cudaBuf{owner: c, codes: []int8{1, 2, 3, 4}}
				tb.ptr = unsafe.Pointer(&tb.codes[0])
				tp := tb.ptr
				c.transient = []*cudaBuf{tb}
				counts := c.uploadCounts
				q56ReleaseBlocked = blocked
				act := func() {
					switch name {
					case "Free":
						c.Free(w)
					case "Recycle":
						c.Recycle()
					default:
						cudaMu.Lock()
						defer cudaMu.Unlock()
						if name == "Storage" {
							c.cudaFreeStorageLocked(w)
						} else {
							c.releaseTransientBuffers([]*cudaBuf{tb})
						}
					}
				}
				weightRetained := func() bool {
					got, ok := wb.hostKeep.(cudaQ8UploadSource)
					return ok && got.HostBuffer == keep.HostBuffer && len(got.scales) == len(keep.scales) && &got.scales[0] == &keep.scales[0] && got.rows == keep.rows && got.cols == keep.cols && got.block == keep.block && wb.ptr == wp && wb.scales == ws && wb.budgetedWeightBytes == 12 && wb.managedWeight && c.dlUsed == 12 && c.managedN == 1 && len(uploadCache) == 1 && uploadCache[key].buf == wb
				}
				act()
				if blocked {
					act() // a repeated release must keep every possible owner reachable
					if q56ReleaseChecks != 2 || !weightRetained() || tb.ptr != tp || len(c.transient) != 1 || c.transient[0] != tb || tb.owner != c {
						t.Fatal("terminal release lost a live pointer, source, cache, transient or accounting owner")
					}
					if len(freed) != 0 || len(awaited) != 0 || c.dependencies != 0 || c.drains != 0 || c.compressorDrains != 0 || profileBoundaries != 0 || c.uploadCounts != counts {
						t.Fatal("terminal release reached a drain, driver call, dependency or accounting effect")
					}
					return
				}
				if name == "Free" || name == "Storage" {
					if wb.ptr != nil || wb.scales != nil || wb.hostKeep != nil || wb.budgetedWeightBytes != 0 || wb.managedWeight || c.dlUsed != 0 || c.managedN != 0 || len(uploadCache) != 0 || freed[wp] != 1 || freed[ws] != 1 || len(freed) != 2 || !slices.Equal(awaited, []unsafe.Pointer{wp}) {
						t.Fatal("healthy weight release did not retain its existing storage/cache/accounting behavior")
					}
					want := 0
					if name == "Free" {
						want = 1
					}
					if c.dependencies != want || c.drains != want || c.compressorDrains != 0 || profileBoundaries != 0 || tb.ptr != tp || len(c.transient) != 1 || c.transient[0] != tb {
						t.Fatal("healthy weight release changed unrelated transient or drain behavior")
					}
				} else {
					if tb.ptr != nil || len(c.transient) != 0 || freed[tp] != 1 || len(freed) != 1 || len(awaited) != 0 || !weightRetained() || c.dependencies != 1 || c.drains != 1 {
						t.Fatal("healthy transient release changed cache owners or failed to release its target")
					}
					want := 0
					if name == "Recycle" {
						want = 1
					}
					if c.compressorDrains != want || profileBoundaries != want {
						t.Fatal("healthy transient release changed compressor/profile boundaries")
					}
				}
				if c.uploadCounts != counts {
					t.Fatal("cleanup changed upload work counters")
				}
			})
		}
	}
}
`
