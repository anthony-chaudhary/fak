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
// witness executes the real constructor/copy helpers and complete Q2 upload
// method; allocation, transfer and accounting are host fixtures. CUDA, Free,
// device numerics, and serving-model behavior are not exercised.
func TestCUDAQ2ResidentIdentityHost(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_Q2_RESIDENT_IDENTITY_HOST") != "1" {
		t.Skip("set FAK_TEST_CUDA_Q2_RESIDENT_IDENTITY_HOST=1 for the host Q2 identity witness")
	}
	var program strings.Builder
	program.WriteString(cudaQ2IdentityHostFixture)
	wanted := map[string]bool{
		"QuantSpec": false, "Buffer": false, "HostBuffer": false, "Tensor": false,
		"cudaQ8UploadSource": false, "makeTensor": false, "Tensor.Numel": false,
		"NewQ2": false, "bytesAsI8": false, "*cudaBackend.uploadQ2Resident": false,
	}
	for _, path := range []string{"compute.go", "quant_q2.go", "cuda_upload.go"} {
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
	probe := filepath.Join(dir, "q2_identity_test.go")
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
			t.Fatalf("host Q2 witness needs Go on PATH: %v", err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", "^TestQ2Identity$", probe)
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
	t.Logf("host Q2 identity witness (%s):\n%s", goExe, out)
	if err != nil {
		t.Fatalf("host Q2 identity witness: %v", err)
	}
}

const cudaQ2IdentityHostFixture = `package compute

import (
	"slices"
	"strconv"
	"testing"
	"unsafe"
)

type Dtype uint8
const (F32 Dtype = iota; Q2_0)
func (d Dtype) Bytes() int { if d != F32 { panic("unexpected fixture byte width") }; return 4 }
type Layout uint8
const RowMajor Layout = 0
type Backend interface{}
type hostBuf struct { i8 []int8 }
func (*hostBuf) Ready() bool { return true }
func (*hostBuf) F32() []float32 { return nil }
func (h *hostBuf) I8() []int8 { return h.i8 }
type MemoryClass string
const MemoryWeights MemoryClass = "weights"
type CUDAOpError struct { Op, Site, Msg string; Class MemoryClass }
func (e *CUDAOpError) Error() string { return e.Op + ": " + e.Site + ": " + e.Msg }
func itoaC(n int) string { return strconv.Itoa(n) }
type ucKey struct { hp uintptr; dt Dtype; lo Layout }
var uploadCache map[ucKey]Tensor
type cudaBackend struct { allocs, uploads, accounts, bytes int }
type cudaBuf struct { owner *cudaBackend; host uintptr; hostKeep HostBuffer; hostDt Dtype; hostLo Layout; codes []int8; scales []float32 }
func (*cudaBuf) Ready() bool { return true }
func (c *cudaBackend) devQ2(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	if nScales != shape[0]*(shape[1]/block) { panic("wrong scale allocation geometry") }
	c.allocs++
	b := &cudaBuf{owner:c}
	return makeTensor(c, Q2_0, RowMajor, slices.Clone(shape), &QuantSpec{Block:block}, b), b
}
func cudaUploadQuantPayload(b *cudaBuf, codes []int8, scales []float32) {
	b.owner.uploads++
	b.codes = slices.Clone(codes)
	b.scales = slices.Clone(scales)
}
func (c *cudaBackend) accountImmutableWeightUpload(n int, b *cudaBuf) {
	if b.owner != c { panic("wrong allocation owner") }; c.accounts++; c.bytes += n
}
func uploadQ2ForWitness(c *cudaBackend, h Tensor) Tensor {
	return c.uploadQ2Resident(h, h.buf.(HostBuffer))
}
func TestQ2Identity(t *testing.T) {
	c := &cudaBackend{}
	uploadCache = make(map[ucKey]Tensor)
	packed := []byte{0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa}
	first := NewQ2(c, []int{1,32}, packed, []float32{1}, 32)
	independent := NewQ2(c, []int{1,32}, packed, first.Quant.Scale, 32)
	codes := first.buf.(HostBuffer).I8()
	if &codes[0] == &independent.buf.(HostBuffer).I8()[0] {
		t.Fatal("false producer premise: separate NewQ2 constructors must copy codes")
	}
	r1 := uploadQ2ForWitness(c,first)
	b1 := r1.buf.(*cudaBuf)
	if c.allocs != 1 || c.uploads != 1 || c.accounts != 1 || c.bytes != 12 || !slices.Equal(b1.codes,codes) || !slices.Equal(b1.scales,[]float32{1}) {
		t.Fatal("first Q2 upload did not stage exact codes/scales")
	}
	before := *c
	if got := uploadQ2ForWitness(c,first); got.buf != r1.buf || *c != before {
		t.Fatal("identical operands lost a zero-side-effect cache hit")
	}
	// Legal Tensor value copy: only a NEW QuantSpec points at a NEW immutable
	// scale allocation. Neither original metadata nor either payload is mutated.
	second := first
	q2 := *first.Quant
	q2.Scale = []float32{2}
	second.Quant = &q2
	if second.buf != first.buf || second.Quant == first.Quant { t.Fatal("descriptor-copy fixture is invalid") }
	r2 := uploadQ2ForWitness(c,second)
	b2 := r2.buf.(*cudaBuf)
	if r2.buf == r1.buf || c.allocs != 2 || c.uploads != 2 || c.accounts != 2 || c.bytes != 24 || !slices.Equal(b2.scales,q2.Scale) {
		t.Fatal("different immutable scale owner falsely reused the prior Q2 resident")
	}
	checkOwner := func(h, resident Tensor) {
		t.Helper()
		b := resident.buf.(*cudaBuf)
		source, ok := b.hostKeep.(*cudaQ8UploadSource)
		if !ok || source == nil || source.HostBuffer != h.buf || source.rows != h.Shape[0] || source.cols != h.Shape[1] || source.block != h.Quant.Block || len(source.scales) != len(h.Quant.Scale) || &source.scales[0] != &h.Quant.Scale[0] {
			t.Fatal("Q2 resident did not retain both exact immutable operands and geometry")
		}
		if resident.Quant.Block != h.Quant.Block || !slices.Equal(resident.Shape,h.Shape) || !slices.Equal(b.codes,codes) || !slices.Equal(b.scales,h.Quant.Scale) {
			t.Fatal("Q2 resident metadata/payload differs from accepted input")
		}
	}
	checkOwner(second,r2)
	before = *c
	if got := uploadQ2ForWitness(c,second); got.buf != r2.buf || *c != before { t.Fatal("new exact scale owner did not hit") }
	third := first
	q3 := *first.Quant
	q3.Block, q3.Scale = 16, []float32{3,4}
	third.Quant = &q3
	r3 := uploadQ2ForWitness(c,third)
	if r3.buf == r2.buf || c.allocs != 3 || c.uploads != 3 || c.accounts != 3 || c.bytes != 40 { t.Fatal("valid changed block reused old Q2 metadata") }
	checkOwner(third,r3)
	reshaped := third
	reshaped.Shape = []int{2,16} // same Numel, complete valid scale groups
	r4 := uploadQ2ForWitness(c,reshaped)
	if r4.buf == r3.buf || c.allocs != 4 || c.uploads != 4 || c.accounts != 4 || c.bytes != 56 { t.Fatal("same-Numel geometry reused a stale Q2 resident") }
	checkOwner(reshaped,r4)
	key := ucKey{uintptr(unsafe.Pointer(&codes[0])),Q2_0,RowMajor}
	if len(uploadCache) != 1 || uploadCache[key].buf != r4.buf { t.Fatal("Q2 miss changed cache key/publication semantics") }
	for _, bad := range []QuantSpec{{Block:16,Scale:[]float32{9}}, {Block:0,Scale:[]float32{1,2}}} {
		input := reshaped
		input.Quant = &bad
		before = *c
		var recovered any
		func(){ defer func(){ recovered = recover() }(); uploadQ2ForWitness(c,input) }()
		e, ok := recovered.(*CUDAOpError)
		if !ok || e == nil || e.Op != "Upload" || e.Site != "upload-q2-resident" || e.Class != MemoryWeights || *c != before || len(uploadCache) != 1 || uploadCache[key].buf != r4.buf {
			t.Fatalf("malformed Q2 metadata did not refuse before side effects: %#v",recovered)
		}
	}
	if first.Quant.Block != 32 || !slices.Equal(first.Quant.Scale,[]float32{1}) || !slices.Equal(b1.scales,[]float32{1}) || !slices.Equal(b1.codes,codes) || r1.Quant.Block != 32 || !slices.Equal(r1.Shape,[]int{1,32}) {
		t.Fatal("a later upload mutated the original descriptor, operands or resident")
	}
}
`
