package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This executes source-extracted Go allocation/retirement logic with a modeled
// native submission. It is not a CUDA, numerical, or performance witness.
// fak-test:runtime medium est=3s lane=default
func TestCUDAGDNMixedScratchOwnership(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	function := func(source, signature string) string {
		t.Helper()
		start := strings.Index(source, signature)
		if start < 0 {
			t.Fatalf("missing production function %q", signature)
		}
		open := strings.Index(source[start:], "{")
		if open < 0 {
			t.Fatal("missing function body")
		}
		depth := 0
		for i := start + open; i < len(source); i++ {
			switch source[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : i+1]
				}
			}
		}
		t.Fatal("incomplete function body")
		return ""
	}
	// Only the native sequence ABI uses this alias. Its final mixed read is the
	// history update, ordered on g_stream before either Q/K-consuming recurrence.
	native := function(read("cuda/qwen35_gdn.cuh"), `extern "C" int fcuda_qwen35_gdn_sequence_f32(`)
	flat := strings.Join(strings.Fields(native), " ")
	history := strings.Index(flat, "k_qwen35_gdn_conv_panel_history<<<convBlocks, 256, 0, g_stream>>>( dMixed, dConvState")
	qk := strings.Index(flat, "k_qwen35_gdn_qk_norm_panel<<<dim3(nK, tokens), qThreads, (size_t)2 * qThreads * sizeof(float), g_stream>>>( dConvOut, dQNorm, dKNorm")
	lastMixed := strings.LastIndex(flat, "dMixed")
	if history < 0 || qk <= history || lastMixed >= qk || lastMixed < history {
		t.Fatal("mixed storage must be dead on the same stream before Q/K writes")
	}
	for _, contract := range []string{
		"const long long valueDim64 = (long long)nV * vHd;",
		"const long long convDim64 = 2LL * nK * kHd + valueDim64;",
		"const int convDim = (int)convDim64, valueDim = (int)valueDim64, fusedRows = (int)fusedRows64;",
		"k_qwen35_gdn_recurrent_chunks<<<nV * chunkCols, 128, 0, g_stream>>>( dConvOut, dQNorm, dKNorm",
		"k_qwen35_gdn_recurrent_panel_cols<<<nV * colBlocks, kQwen35GDNPanelCols, 0, g_stream>>>( dConvOut, dQNorm, dKNorm",
		"fak_gdn_vllm::dispatch(dConvOut, dA, dB, dALog, dDtBias, dRecurrentState, dZ, dNorm, dCore",
	} {
		if !strings.Contains(flat, contract) {
			t.Fatalf("native alias dependency changed: %s", contract)
		}
	}

	// Preserve actual Go control flow. The explicit C-to-Go scalar substitutions
	// replace only the native ABI with the modeled call below, not owner logic.
	replace := strings.NewReplacer("C.fcuda_qwen35_gdn_sequence_f32", "testGDN", "C.fcuda_free", "testFree", "C.int", "int32", "C.float", "float32")
	program := cudaGDNMixedScratchHarness
	for _, item := range []struct{ path, signature string }{
		{"cuda_kernels.go", "func (c *cudaBackend) qwen35SequenceGDNLocked("},
		{"qwen35_gdn.go", "func qwen35GDNAllocations("},
		{"cuda_specialized.go", "func (c *cudaBackend) failQwen35GDN("},
		{"cuda_qwen35_gdn.go", "func qwen35GDNKernelStage("},
		{"cuda.go", "func (c *cudaBackend) releaseTransientBuffers("},
		{"cuda.go", "func (c *cudaBackend) cf("},
		{"cuda_kv.go", "func offsetF("},
	} {
		program += "\n" + replace.Replace(function(read(item.path), item.signature)) + "\n"
	}
	program += cudaGDNMixedScratchMain
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "run", "main.go")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GO111MODULE=off", "GOWORK=off", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("source-extracted GDN ownership: %v\n%s", err, out)
	}
}

const cudaGDNMixedScratchHarness = `package main
import("fmt";"sync/atomic";"unsafe")
const F32=0
const Q8_0=1
const cudaScratchRetireFailedBase=83000
type Tensor struct { Shape []int; Dtype int; buf any }
type cudaBuf struct { ptr unsafe.Pointer; data []float32; invalid uint32; graphOncePinned bool }
type qwen35GDNAllocation struct { name string; shape []int }
type Qwen35SequenceState struct { Conv,Recurrent Tensor }
type Qwen35SequenceLayer struct { GDNInQKV,GDNInZ,GDNInB,GDNInA,GDNOut,GDNConv,GDNALog,GDNDTBias,GDNNorm Tensor }
type Qwen35SequencePrefillRequest struct { Hidden,NumKeyHeads,NumValueHeads,KeyHeadDim,ValueHeadDim,ConvKernel int; RMSNormEpsilon float32 }
type Qwen35SequenceError struct { Stage string; Layer int; Cause error }
func(e *Qwen35SequenceError)Error()string{return fmt.Sprint(e.Stage,": ",e.Cause)}
type Qwen35GDNKernelError struct { Stage string; Code int }
func(e *Qwen35GDNKernelError)Error()string{return fmt.Sprint(e.Stage,": ",e.Code)}
type latch struct{ count int }
func(l *latch)ObserveError(_ error,_ string){l.count++}
type cudaBackend struct { transient []*cudaBuf; names []string; attempts,failAt int; fenceGen uint64; faultLatch latch }
var terminal bool
var submitted,status int
var expectedKind int32
var seenMixed,seenQ,seenK unsafe.Pointer
var freed map[unsafe.Pointer]bool
func makeTensor(shape []int)Tensor{n:=1;for _,v:=range shape{n*=v};b:=&cudaBuf{data:make([]float32,n)};b.ptr=unsafe.Pointer(&b.data[0]);return Tensor{Shape:shape,buf:b}}
func(c *cudaBackend)qwen35SequenceAllocLocked(shape []int,name string)(Tensor,*cudaBuf,error){c.attempts++;if c.attempts==c.failAt{return Tensor{},nil,fmt.Errorf("allocation %s",name)};t:=makeTensor(shape);b:=t.buf.(*cudaBuf);c.transient=append(c.transient,b);c.names=append(c.names,name);return t,b,nil}
func(c *cudaBackend)cudaBufForSubmit(t Tensor)*cudaBuf{b:=t.buf.(*cudaBuf);if b.ptr==nil||atomic.LoadUint32(&b.invalid)!=0{panic("invalid checked submit")};return b}
func(c *cudaBackend)qwen35GDNQ8Args(t Tensor)(unsafe.Pointer,*float32,int32){return c.cudaBufForSubmit(t).ptr,nil,int32(t.Dtype)}
func cudaQ56KMMVQReleaseBlocked()bool{return terminal}
func cudaQ4KAsyncReleaseBlockedLocked()bool{return false}
func cudaQ4KAsyncRequireDrainLocked(*cudaBackend){}
func cudaV41CompressorDeferFreeLocked(*cudaBackend,*cudaBuf)bool{return false}
func cudaV41CompressorPendingIndex(*cudaBuf)int{return -1}
func(c *cudaBackend)graphOnceFreeDependencyLocked(*cudaBuf){}
func testFree(p unsafe.Pointer){if freed[p]{panic("duplicate owner retirement")};freed[p]=true}
func testGDN(x *float32,tokens int32,qkv unsafe.Pointer,qs *float32,qkind int32,z unsafe.Pointer,zs *float32,zkind int32,b unsafe.Pointer,bs *float32,bkind int32,a unsafe.Pointer,as *float32,akind int32,conv,alog,dt,norm *float32,outw unsafe.Pointer,outs *float32,outkind int32,convState,recurrentState,out,mixed,zbuf,bbuf,abuf,convout,qnorm,knorm,core *float32,hidden,nk,nv,khd,vhd,kernel int32,eps float32)int32{
 check(qkind==expectedKind&&zkind==expectedKind&&bkind==expectedKind&&akind==expectedKind&&outkind==expectedKind,"projection kind plumbing changed")
 submitted++;seenMixed=unsafe.Pointer(mixed);seenQ=unsafe.Pointer(qnorm);seenK=unsafe.Pointer(knorm)
 // Model only the order and region extents, not native arithmetic. The full
 // mixed panel is read before Q/K outputs overwrite either disjoint region.
 key:=int(tokens*nk*khd);value:=int(tokens*nv*vhd)
 for i:=range unsafe.Slice(mixed,2*key+value){unsafe.Slice(mixed,2*key+value)[i]=17}
 for i:=0;i<key;i++{unsafe.Slice(qnorm,key)[i]=2;unsafe.Slice(knorm,key)[i]=3}
 *convState+=1;*recurrentState+=1
 if status/10000==7{terminal=true}
 return int32(status)
}
func check(ok bool,message string){if !ok{panic(message)}}
func reset(){terminal=false;submitted=0;status=0;seenMixed=nil;seenQ=nil;seenK=nil;freed=make(map[unsafe.Pointer]bool)}
func fixture(kind int)(*cudaBackend,Tensor,Qwen35SequenceLayer,Qwen35SequenceState,Qwen35SequencePrefillRequest){
 expectedKind=int32(kind) // modeled submission plumbing only, not native selector qualification
 c:=&cudaBackend{};x:=makeTensor([]int{64});w:=makeTensor([]int{1});w.Dtype=kind;aux:=makeTensor([]int{1})
 layer:=Qwen35SequenceLayer{w,w,w,w,w,aux,aux,aux,aux};state:=Qwen35SequenceState{makeTensor([]int{1}),makeTensor([]int{1})}
 req:=Qwen35SequencePrefillRequest{64,2,4,8,8,4,1e-6};return c,x,layer,state,req
}
`

const cudaGDNMixedScratchMain = `
func main(){
 for _,tokens:=range []int{1,16,128,2048}{for _,kind:=range []int{F32,Q8_0}{
  reset();c,x,layer,state,req:=fixture(kind);out,owner,err:=c.qwen35SequenceGDNLocked(x,layer,state,req,tokens,3)
  check(err==nil,"unexpected success-path error")
  check(c.attempts==7,fmt.Sprintf("GDN physical allocations=%d want7",c.attempts))
  check(submitted==1&&c.fenceGen==1,"submission/fence changed")
  check(owner!=nil&&out.buf==owner&&len(c.transient)==1&&c.transient[0]==owner,"output owner lost")
  check(len(freed)==6,"scratch owners not retired exactly once")
  key:=tokens*req.NumKeyHeads*req.KeyHeadDim;value:=tokens*req.NumValueHeads*req.ValueHeadDim
  check(seenMixed==seenQ&&uintptr(seenK)-uintptr(seenMixed)==uintptr(4*key),"Q/K did not use disjoint mixed regions")
  all:=unsafe.Slice((*float32)(seenMixed),2*key+value)
  for _,v:=range all[:key]{check(v==2,"Q region overwritten")};for _,v:=range all[key:2*key]{check(v==3,"K region overwritten")};for _,v:=range all[2*key:]{check(v==17,"mixed tail overwritten")}
  for _,name:=range c.names{check(name!="gdn-q-norm"&&name!="gdn-k-norm","aliased region allocated an owner")}
 }}
 for fail:=1;fail<=7;fail++{
  reset();c,x,layer,state,req:=fixture(F32);c.failAt=fail;out,owner,err:=c.qwen35SequenceGDNLocked(x,layer,state,req,128,3)
  check(err!=nil&&owner==nil&&out.buf==nil,"allocation failure published output")
  check(submitted==0&&c.fenceGen==0&&len(c.transient)==0&&len(freed)==fail-1,"allocation failure lost or duplicated an owner")
 }
 for _,failure:=range []int{20001,70700}{
  reset();c,x,layer,state,req:=fixture(F32);status=failure;out,owner,err:=c.qwen35SequenceGDNLocked(x,layer,state,req,128,3)
  check(err!=nil&&out.buf==nil&&owner==nil&&c.fenceGen==0,"failed submission published output")
  check(c.attempts==7&&submitted==1&&c.faultLatch.count==1,"failure path did not keep seven dense owners")
  check(state.Conv.buf.(*cudaBuf).invalid==1&&state.Recurrent.buf.(*cudaBuf).invalid==1,"partial states not invalidated")
  if failure==70700{check(len(c.transient)==7&&len(freed)==0,"quarantined owner detached");for _,b:=range c.transient{check(b!=nil&&b.ptr!=nil&&b.invalid==1,"invalid quarantine owner")}}else{check(len(c.transient)==0&&len(freed)==7,"drained owners not released exactly once")}
 }
 fmt.Println("PASS: source-extracted GDN mixed scratch ownership and controls")
}
`
