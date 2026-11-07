package compute

import (
	"bytes"
	"context"
	"crypto/sha256"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// fak-test:runtime medium est=45s lane=default
// Executes the intact production metadata query with explicit host allocation
// doubles. Compilation/setup dominates this bounded fixture; no CUDA runtime,
// kernels, numerical parity, device completion or performance is witnessed.
func TestCUDAQwen35IndependentDecodeCohortOwnership(t *testing.T) {
	src, err := os.ReadFile("cuda_qwen35_decode_cohort.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "cuda_qwen35_decode_cohort.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var code bytes.Buffer
	code.WriteString("package main\nimport (\"fmt\";\"sort\";\"sync\";\"unsafe\")\n")
	found := false
	for _, decl := range f.Decls {
		if gen, ok := decl.(*ast.GenDecl); ok && gen.Tok == token.IMPORT {
			continue
		}
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "Qwen35IndependentDecodeCohortOwned" {
			found = true
		}
		if err := format.Node(&code, fs, decl); err != nil {
			t.Fatal(err)
		}
		code.WriteByte('\n')
	}
	if !found {
		t.Fatal("missing real CUDA whole-cohort ownership method")
	}
	code.WriteString(cudaQwen35CohortOwnershipFixture)
	t.Logf("production metadata source sha256=%x; derived host program sha256=%x", sha256.Sum256(src), sha256.Sum256(code.Bytes()))
	dir := t.TempDir()
	path := filepath.Join(dir, "main.go")
	if err := os.WriteFile(path, code.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Fatal("required host Go compiler unavailable: ", err)
	}
	deadline := time.Now().Add(90 * time.Second)
	if suiteDeadline, ok := t.Deadline(); ok && suiteDeadline.Add(-2*time.Second).Before(deadline) {
		deadline = suiteDeadline.Add(-2 * time.Second)
	}
	if !time.Now().Before(deadline) {
		t.Fatal("insufficient test budget for ownership controls and subprocess drain")
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	// Only this synthetic child is path independent. Keep the parent test's
	// source paths intact, and do not inherit unbounded package compilation.
	cmd := exec.CommandContext(ctx, goTool, "run", "-p=1", "-trimpath", path)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOTOOLCHAIN=local", "GOFLAGS=", "CGO_ENABLED=0")
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = time.Second
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("source-bound ownership controls: %v\n%s", err, out)
	}
	const complete = "PASS CUDA metadata ownership controls; no device execution"
	if !bytes.Equal(bytes.TrimSpace(out), []byte(complete)) {
		t.Fatalf("source-bound ownership controls did not report completion:\n%s", out)
	}
	t.Logf("%s", bytes.TrimSpace(out))
}

const cudaQwen35CohortOwnershipFixture = `
// Only allocation/storage boundaries are doubled. The query and all its
// arithmetic/alias helpers above are unmodified production declarations.
type KVStore interface{}
type MemoryClass int
const MemoryKVCache MemoryClass = 1
type Dtype int
const (F32 Dtype = iota; F16; BF16)
func (d Dtype) Bytes() int { if d == F32 { return 4 }; if d == F16 || d == BF16 { return 2 }; return 0 }
type Layout int
const RowMajor Layout = 0
type KVPrecision int
const (KVPrecisionF32 KVPrecision = iota; KVPrecisionF16; KVPrecisionBF16)
type KVConfig struct { NumLayers, NumKVHeads, HeadDim int; Precision KVPrecision }
type dslice struct { ptr unsafe.Pointer; len, cap int; dtype Dtype }
type cudaKV struct { be *cudaBackend; cfg KVConfig; maxPos int; K, Kraw, V []dslice; pos []int; releaseRetained bool; releaseNext *cudaKV; snapshotCopy *int }
type cudaBuf struct {
 ptr unsafe.Pointer; n int; class MemoryClass; device int; be *cudaBackend; invalid uint32
 managed, graphOncePinned, managedWeight bool; host uintptr; hostKeep any; scales unsafe.Pointer; scalesN int; budgetedWeightBytes int64
}
func (b *cudaBuf) invalidForSubmit() bool { return b != nil && b.invalid != 0 }
type Tensor struct { buf any; be *cudaBackend; Dtype Dtype; Layout Layout; Shape []int; Quant *int }
func (t Tensor) Backend() *cudaBackend { return t.be }
type Qwen35SequenceState struct { Conv, Recurrent Tensor }
const DeviceHealthy = 0
type DeviceFaultLatch struct { health int }
func (l *DeviceFaultLatch) Health() int {
 if cudaMu.TryLock() { cudaMu.Unlock(); panic("metadata read escaped cudaMu") }
 if l == nil { return DeviceHealthy }; return l.health
}
type cudaBackend struct { capturing bool; graphOnce *int; faultLatch *DeviceFaultLatch; transient []*cudaBuf }
var cudaMu sync.Mutex
var owners [][]byte
func allocate(n int) unsafe.Pointer { b := make([]byte, n); owners = append(owners, b); return unsafe.Pointer(&b[0]) }
func must(ok bool, why string) { if !ok { panic(why) } }
func emptyKV(k *cudaKV) { k.pos=nil;for _, p:=range [][]dslice{k.K,k.Kraw,k.V}{for i:=range p{p[i]=dslice{}}} }
func fixture(n int) (*cudaBackend, []KVStore, [][]Qwen35SequenceState) {
 c := &cudaBackend{faultLatch: &DeviceFaultLatch{}}
 kvs := make([]KVStore, n); states := make([][]Qwen35SequenceState, len(kvs))
 for i := range kvs {
  k := &cudaKV{be:c, cfg:KVConfig{NumLayers:2,NumKVHeads:1,HeadDim:4},pos:[]int{0}, K:make([]dslice,2),Kraw:make([]dslice,2),V:make([]dslice,2)}
  for _, plane := range [][]dslice{k.K,k.Kraw,k.V} { plane[0]=dslice{ptr:allocate(64),len:4,cap:16} }
  kvs[i]=k; states[i]=make([]Qwen35SequenceState,2)
  tensor := func(shape []int) Tensor { return Tensor{be:c,Shape:shape,buf:&cudaBuf{ptr:allocate(64),n:64,class:MemoryKVCache}} }
  states[i][0]=Qwen35SequenceState{Conv:tensor([]int{2,4}),Recurrent:tensor([]int{2,2,2})}
 }
 return c,kvs,states
}
func snapshot(c *cudaBackend,kvs []KVStore,states [][]Qwen35SequenceState) string {
 s:=fmt.Sprintf("%#v",*c)
 for _, raw:=range kvs { if k,ok:=raw.(*cudaKV);ok&&k!=nil{s+=fmt.Sprintf("%#v",*k)}else{s+=fmt.Sprintf("%#v",raw)} }
 for _, row:=range states { for _, st:=range row { for _, x:=range []Tensor{st.Conv,st.Recurrent} { s+=fmt.Sprintf("%#v",x);if b,ok:=x.buf.(*cudaBuf);ok&&b!=nil{s+=fmt.Sprintf("%#v",*b)} } } }
 return s
}
func main() {
 for _, n := range []int{2,8,9,16,17} {
  c,k,s:=fixture(n);must(c.Qwen35IndependentDecodeCohortOwned(k,s),fmt.Sprint("disjoint cohort ",n))
 }
 cases:=[]struct{name string; mutate func(*cudaBackend,[]KVStore,[][]Qwen35SequenceState); want bool}{
  {"disjoint-full-capacities",nil,true},
  {"fresh-empty-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ for _, raw:=range k { kv:=raw.(*cudaKV);kv.pos=nil;for _, plane:=range [][]dslice{kv.K,kv.Kraw,kv.V}{for i:=range plane{plane[i]=dslice{}}} } },true},
  {"narrowed-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ for _, raw:=range k { kv:=raw.(*cudaKV);kv.cfg.Precision=KVPrecisionF16;for _, plane:=range [][]dslice{kv.K,kv.V}{for i:=range plane{plane[i].dtype=F16}} } },true},
  {"bf16-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ for _, raw:=range k { kv:=raw.(*cudaKV);kv.cfg.Precision=KVPrecisionBF16;for _, plane:=range [][]dslice{kv.K,kv.V}{for i:=range plane{plane[i].dtype=BF16}} } },true},
  {"adjacent-state-ranges",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ p:=allocate(128);s[0][0].Conv.buf.(*cudaBuf).ptr=p;s[8][0].Conv.buf.(*cudaBuf).ptr=unsafe.Add(p,64) },true},
  {"unrelated-transient",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ c.transient=[]*cudaBuf{{ptr:allocate(64),n:64}} },true},
  {"gdngdn-cross-group",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ b:=*s[0][0].Conv.buf.(*cudaBuf);s[8][0].Conv.buf=&b },false},
  {"gdn-interior-spare-capacity",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).ptr=unsafe.Add(s[0][0].Conv.buf.(*cudaBuf).ptr,40) },false},
  {"kvkv-cross-group",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).K[0].ptr=k[0].(*cudaKV).V[0].ptr },false},
  {"kv-interior-cross-group",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).K[0].ptr=unsafe.Add(k[0].(*cudaKV).Kraw[0].ptr,4) },false},
  {"kvgdn-spare-capacity",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).ptr=unsafe.Add(k[0].(*cudaKV).K[0].ptr,32) },false},
  {"same-lane-plane-alias",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[0].(*cudaKV).Kraw[0].ptr=k[0].(*cudaKV).K[0].ptr },false},
  {"unknown-paged-owner",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8]=struct{}{} },false},
  {"typed-nil-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8]=(*cudaKV)(nil) },false},
  {"shared-empty-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ kv:=k[0].(*cudaKV);kv.pos=nil;for _, p:=range [][]dslice{kv.K,kv.Kraw,kv.V}{for i:=range p{p[i]=dslice{}}};k[8]=kv },false},
  {"shallow-empty-kv-cross-group",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ a,b:=k[0].(*cudaKV),k[8].(*cudaKV);emptyKV(a);*b=*a },false},
  {"interior-empty-kv-metadata",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ a,b:=k[0].(*cudaKV),k[8].(*cudaKV);emptyKV(a);emptyKV(b);p:=make([]dslice,3);a.K=p[:2:3];b.K=p[1:3:3] },false},
  {"same-owner-metadata-planes",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ a:=k[0].(*cudaKV);emptyKV(a);a.Kraw=a.K },false},
  {"shared-position-backing",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).pos=k[0].(*cudaKV).pos },false},
  {"shared-empty-position-capacity",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ a,b:=k[0].(*cudaKV),k[8].(*cudaKV);emptyKV(a);emptyKV(b);p:=make([]int,0,8);a.pos=p;b.pos=p },false},
  {"retired-kv-planes",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).K=nil },false},
  {"foreign-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).be=&cudaBackend{} },false},
  {"retained-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).releaseRetained=true },false},
  {"snapshot-kv",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).snapshotCopy=new(int) },false},
  {"invalid-kv-bounds",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).K[0].len=17 },false},
  {"kv-byte-overflow",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).K[0].cap=int(^uint(0)>>1) },false},
  {"invalid-kv-position",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).pos[0]=3 },false},
  {"missing-kv-plane",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ k[8].(*cudaKV).V=nil },false},
  {"poisoned-state",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).invalid=1 },false},
  {"foreign-state",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.be=&cudaBackend{} },false},
  {"retired-state",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).ptr=nil },false},
  {"managed-state",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).managed=true },false},
  {"cached-state",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).hostKeep=new(int) },false},
  {"graph-pinned-state",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).graphOncePinned=true },false},
  {"foreign-device-state",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).device=1 },false},
  {"state-shape-overflow",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.Shape=[]int{int(^uint(0)>>1),8} },false},
  {"state-short-capacity",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).n=4 },false},
  {"state-wrong-class",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv.buf.(*cudaBuf).class=0 },false},
  {"state-layer-missing",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0]=Qwen35SequenceState{} },false},
  {"state-half-missing",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ s[8][0].Conv=Tensor{} },false},
  {"transient-state-alias",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ b:=*s[8][0].Conv.buf.(*cudaBuf);c.transient=[]*cudaBuf{&b} },false},
  {"capturing",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ c.capturing=true },false},
  {"graph-owner",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ c.graphOnce=new(int) },false},
  {"device-fault",func(c *cudaBackend,k []KVStore,s [][]Qwen35SequenceState){ c.faultLatch.health=1 },false},
 }
 for _, tc:=range cases {
  c,k,s:=fixture(17);if tc.mutate!=nil{tc.mutate(c,k,s)};before:=snapshot(c,k,s)
  must(c.Qwen35IndependentDecodeCohortOwned(k,s)==tc.want,tc.name)
  must(snapshot(c,k,s)==before,"query mutated "+tc.name)
  must(cudaMu.TryLock(),"query retained mutex");cudaMu.Unlock()
 }
 c,k,s:=fixture(17);must(!c.Qwen35IndependentDecodeCohortOwned(k,s[:1]),"mismatched lane count")
 must(!c.Qwen35IndependentDecodeCohortOwned(k[:1],s[:1]),"singleton")
 must(!(*cudaBackend)(nil).Qwen35IndependentDecodeCohortOwned(k,s),"nil backend")
 fmt.Println("PASS CUDA metadata ownership controls; no device execution")
}
`
