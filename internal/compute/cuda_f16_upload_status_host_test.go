package compute

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Executes the literal upload owner and native submission wrappers with host API
// stubs. This witnesses rejection plumbing, not CUDA compilation, completion,
// numerical accuracy, or performance. The successful path stays asynchronous.
// fak-test:runtime medium est=2s lane=default
func TestCUDAF16RejectedUploadPublication(t *testing.T) {
	t.Run("go-upload-owner", func(t *testing.T) {
		functions := map[string]string{}
		for _, path := range []string{"cuda.go", "cuda_tensor_allocation.go", "cuda_backend_state.go"} {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			fs := token.NewFileSet()
			file, err := parser.ParseFile(fs, path, raw, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, decl := range file.Decls {
				if fn, ok := decl.(*ast.FuncDecl); ok {
					functions[fn.Name.Name] = string(raw[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
				}
			}
		}
		var source strings.Builder
		source.WriteString(cudaF16UploadGoPrefix)
		for _, name := range []string{"uploadF16", "allocateF16UploadStage", "devF16", "accountWeightPlacement", "accountImmutableWeightUpload", "residentBytes"} {
			body := functions[name]
			if body == "" {
				t.Fatalf("missing production method %s", name)
			}
			if name == "uploadF16" {
				// Both spellings allow the same witness to execute the unfixed
				// void-ABI baseline without manufacturing a second upload body.
				checked := strings.Contains(body, "C.fak_cuda_upload_h2d(")
				prefix := "C.fcuda_"
				if checked {
					prefix = "C.fak_cuda_"
				}
				copyName := prefix + "h2d("
				if checked {
					copyName = prefix + "upload_h2d("
				}
				for _, binding := range []struct {
					old, replacement string
					count            int
				}{
					{copyName, "hostCopy(", 1},
					{prefix + "f32_to_f16_T(", "hostConvertT(", 1},
					{prefix + "f32_to_f16(", "hostConvert(", 1},
					{"C.size_t(", "int(", 1}, {"C.int(", "int(", 3}, {"C.float", "float32", 2},
				} {
					if strings.Count(body, binding.old) != binding.count {
						t.Fatalf("unexpected ABI binding count: %s", binding.old)
					}
					body = strings.ReplaceAll(body, binding.old, binding.replacement)
				}
				freeCount := 2
				if checked {
					freeCount = 3
				}
				if strings.Count(body, "C.fcuda_free(") != freeCount {
					t.Fatal("unexpected release binding count")
				}
				body = strings.ReplaceAll(body, "C.fcuda_free(", "hostFree(")
			}
			if strings.Contains(body, "C.") {
				t.Fatalf("untranslated ABI in %s", name)
			}
			source.WriteString(body + "\n")
		}
		source.WriteString(cudaF16UploadGoMain)
		dir := t.TempDir()
		path := filepath.Join(dir, "upload.go")
		if err := os.WriteFile(path, []byte(source.String()), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, "go", "run", path).CombinedOutput(); err != nil {
			t.Fatalf("literal upload: %v\n%s", err, out)
		}
	})
	t.Run("native-submission", func(t *testing.T) {
		compiler, err := exec.LookPath("c++")
		if err != nil {
			t.Skip("host C++ compiler unavailable")
		}
		raw, err := os.ReadFile("cuda_kernels.cu")
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		extract := func(signature string) string {
			start := strings.Index(source, signature)
			if start < 0 {
				t.Fatalf("missing native wrapper %s", signature)
			}
			brace := strings.Index(source[start:], "{") + start
			depth := 0
			for i := brace; i < len(source); i++ {
				if source[i] == '{' {
					depth++
				}
				if source[i] == '}' {
					depth--
					if depth == 0 {
						return source[start:i+1] + "\n"
					}
				}
			}
			t.Fatalf("unterminated wrapper %s", signature)
			return ""
		}
		body := extract(`extern "C" int fak_cuda_upload_h2d(`) + extract(`extern "C" void fcuda_h2d(`)
		for _, suffix := range []string{"", "_T"} {
			checked := strings.Contains(source, `extern "C" int fak_cuda_f32_to_f16`+suffix+`(`)
			if checked {
				body += extract(`extern "C" int fak_cuda_f32_to_f16` + suffix + `(`)
			}
			body += extract(`extern "C" void fcuda_f32_to_f16` + suffix + `(`)
			if !checked {
				args, call := "int n", "n"
				if suffix != "" {
					args, call = "int out, int in", "out, in"
				}
				body += fmt.Sprintf("int fak_cuda_f32_to_f16%s(void *d, const float *s, %s) { fcuda_f32_to_f16%s(d,s,%s); return 0; }\n", suffix, args, suffix, call)
			}
		}
		for _, launch := range []struct{ old, replacement string }{
			{"k_f32_to_f16<<<(n + 255) / 256, 256, 0, g_stream>>>(dSrc, (__half *)dDstHalf, n);", "launchRow((n + 255) / 256, 256, 0, g_stream, dSrc, (__half *)dDstHalf, n);"},
			{"k_f32_to_f16_T<<<(total + 255) / 256, 256, 0, g_stream>>>(dSrc, (__half *)dDstHalf, out, in);", "launchColumn((total + 255) / 256, 256, 0, g_stream, dSrc, (__half *)dDstHalf, out, in);"},
		} {
			if strings.Count(body, launch.old) != 1 {
				t.Fatal("native launch expression drift")
			}
			body = strings.Replace(body, launch.old, launch.replacement, 1)
		}
		dir := t.TempDir()
		cpp, bin := filepath.Join(dir, "submission.cpp"), filepath.Join(dir, "submission")
		if err := os.WriteFile(cpp, []byte(cudaF16UploadNativePrefix+body+cudaF16UploadNativeMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("host compile: %v\n%s", err, out)
		}
		if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("native submission: %v\n%s", err, out)
		}
	})
}

const cudaF16UploadGoPrefix = `package main
import ("fmt"; "reflect"; "unsafe")
type Dtype int
const (F32 Dtype = 1; F16 Dtype = 2)
func (d Dtype) Bytes() int { if d==F32 {return 4}; return 2 }
type Layout int
const (RowMajor Layout = iota; ColMajor)
const (MemoryWeights = "weights"; MemoryOffload = "offload"; MemoryScratchpad = "scratch"; DeviceFaultUnknown = 0)
type HostBuffer interface{}
type Tensor struct { Shape []int; Layout Layout; Dtype Dtype; buf *cudaBuf }
func makeTensor(_ *cudaBackend,d Dtype,l Layout,s []int,_ any,b *cudaBuf) Tensor { return Tensor{s,l,d,b} }
type ucKey struct { hp uintptr; dt Dtype; layout Layout }
var uploadCache map[ucKey]Tensor
type cudaBuf struct {ptr unsafe.Pointer; n,scalesN int; budgetedWeightBytes int64; managedWeight,managed bool; class string; host uintptr; hostKeep HostBuffer; hostDt Dtype; hostLo Layout}
type cudaBackend struct {budgetBytes,dlUsed int64; managedN int; faultLatch *latch; fenceGen uint64; immutableWeightUploadCalls,immutableWeightUploadTransferBytes,immutableWeightUploadResidentBytes uint64}
type latch struct { code int; site string; calls int }
func (l *latch) Observe(_ int,site string,code int) {if l!=nil {l.code=code;l.site=site;l.calls++}}
type CUDAOpError struct {Op,Site,Msg,Class string; Code int}
var c *cudaBackend
var dst,stage *cudaBuf
var destination,staging [16]byte
var budget,managed,failStage,unknown,panicStage,panicDst,normalStagePanic,cleanupOK bool
var copyStatus,convertStatus,copyCalls,rowCalls,colCalls,stageReleases,dstReleases,rawBytes int
var label string
var original = &struct{ message string }{"primary unexpected failure"}
func require(ok bool,msg string) {if !ok {panic(label+": "+msg)}}
func (c *cudaBackend) dallocWeight(n int) *cudaBuf { require(n==8,"destination size");class:=MemoryWeights;if managed {class=MemoryOffload};dst=&cudaBuf{ptr:unsafe.Pointer(&destination[0]),n:n,managed:managed,class:class}; c.accountWeightPlacement(dst,n);return dst }
func (c *cudaBackend) dallocClass(n int,class,site string) *cudaBuf { require(n==16 && class==MemoryScratchpad && site=="f16-stage","scratch contract"); if failStage {panic(original)};stage=&cudaBuf{ptr:unsafe.Pointer(&staging[0]),n:n,class:class};return stage }
func hostCopy(d,h unsafe.Pointer,n int) int {copyCalls++;require(d==stage.ptr && h!=nil && n==16,"copy arguments");if copyStatus==0 {rawBytes+=n}; return copyStatus}
func hostConvert(d unsafe.Pointer,s *float32,n int) int {rowCalls++;require(d==dst.ptr && unsafe.Pointer(s)==stage.ptr && n==4,"row arguments"); if unknown {panic(original)};return convertStatus}
func hostConvertT(d unsafe.Pointer,s *float32,out,in int) int {colCalls++;require(d==dst.ptr && unsafe.Pointer(s)==stage.ptr && out==2 && in==2,"column arguments"); if unknown {panic(original)};return convertStatus}
func hostFree(p unsafe.Pointer) {
 if p==unsafe.Pointer(&staging[0]) {stageReleases++; require(stage.ptr==nil,"stage must detach before release"); if normalStagePanic {panic(original)}} else if p==unsafe.Pointer(&destination[0]) {dstReleases++} else {panic("unowned release")}
 if copyStatus!=0 || convertStatus!=0 || unknown || failStage || normalStagePanic {
  cleanupOK = cleanupOK && dst.ptr==nil && dst.budgetedWeightBytes==0 && !dst.managedWeight && c.dlUsed==77 && c.managedN==3
  cleanupOK = cleanupOK && (stage==nil || stage.ptr==nil)
 }
 if (p==unsafe.Pointer(&staging[0]) && panicStage) || (p==unsafe.Pointer(&destination[0]) && panicDst) {panic("secondary cleanup failure")}
}
`

const cudaF16UploadGoMain = `
func main() {
 for _,mode:=range []string{"h2d","convert","success","allocate","unexpected","stage-release"} {for _,layout:=range []Layout{RowMajor,ColMajor} {for _,b:=range []bool{false,true} {for _,m:=range []bool{false,true} {for _,withLatch:=range []bool{false,true} {for cleanup:=0;cleanup<4;cleanup++ {
  label=fmt.Sprintf("%s/layout%d/budget%v/managed%v/latch%v/cleanup%d",mode,layout,b,m,withLatch,cleanup)
  budget,managed=b,m;failStage=mode=="allocate";unknown=mode=="unexpected";normalStagePanic=mode=="stage-release";panicStage=cleanup&1!=0 && mode!="success";panicDst=cleanup&2!=0
  copyStatus,convertStatus=0,0;if mode=="h2d" {copyStatus=77};if mode=="convert" {convertStatus=98}
  copyCalls,rowCalls,colCalls,stageReleases,dstReleases,rawBytes=0,0,0,0,0,0;dst,stage=nil,nil;cleanupOK=true
  c=&cudaBackend{dlUsed:77,managedN:3,fenceGen:11,immutableWeightUploadCalls:5,immutableWeightUploadTransferBytes:101,immutableWeightUploadResidentBytes:103}
  if budget {c.budgetBytes=1000};if withLatch {c.faultLatch=&latch{}}
  f:=[]float32{1,2,3,4};hb:=&f;hp:=uintptr(unsafe.Pointer(&f[0]));key:=ucKey{hp,F16,layout};sentinel:=&cudaBuf{n:91};stale:=Tensor{Shape:[]int{1,1},buf:sentinel}
  uploadCache=map[ucKey]Tensor{key:stale,{hp+100,F16,layout}:stale};before:=map[ucKey]Tensor{key:stale,{hp+100,F16,layout}:stale}
  var got Tensor;var caught any
  func(){defer func(){caught=recover()}();got=c.uploadF16(Tensor{Shape:[]int{2,2},Layout:layout,Dtype:F32},hb,f,hp)}()
  expectedClass:=MemoryWeights;if managed {expectedClass=MemoryOffload};require(cleanupOK,"cleanup observed attached owners or unrestored charges");require(c.fenceGen==11,"fence generation changed");require(dst.n==8 && dst.managed==managed && dst.class==expectedClass,"unrelated metadata changed")
  if mode=="success" {
   require(caught==nil && got.buf==dst && dst.ptr!=nil && stage.ptr==nil && stageReleases==1 && dstReleases==0,"success ownership")
   require(got.Dtype==F16 && got.Layout==layout && reflect.DeepEqual(got.Shape,[]int{2,2}) && uploadCache[key].buf==dst && uploadCache[ucKey{hp+100,F16,layout}].buf==sentinel,"success publication")
   require(dst.host==hp && dst.hostKeep==hb && dst.hostDt==F16 && dst.hostLo==layout,"host retention")
   dl,mn:=int64(77),3;if budget {if managed {mn++} else {dl+=8}};require(c.dlUsed==dl && c.managedN==mn,"success charges")
   require(c.immutableWeightUploadCalls==6 && c.immutableWeightUploadTransferBytes==117 && c.immutableWeightUploadResidentBytes==111,"success counters")
  } else {
   require(caught!=nil && got.buf==nil,"failure returned a tensor")
   if mode=="h2d" || mode=="convert" {e,ok:=caught.(*CUDAOpError);site,code:="upload-f16-h2d",77;if mode=="convert" {site,code="upload-f16-convert",98};require(ok && e.Op=="Upload" && e.Site==site && e.Code==code && e.Class==expectedClass,"typed primary error")} else {require(caught==original,"unexpected primary replaced")}
   require(dst.ptr==nil && dst.budgetedWeightBytes==0 && !dst.managedWeight && dstReleases==1 && c.dlUsed==77 && c.managedN==3,"failure destination cleanup")
   expectedStage:=1;if failStage {expectedStage=0};require(stageReleases==expectedStage && (stage==nil || stage.ptr==nil),"failure stage cleanup")
   require(dst.host==0 && dst.hostKeep==nil && dst.hostDt==0 && reflect.DeepEqual(uploadCache,before),"failure published host/cache")
   require(c.immutableWeightUploadCalls==5 && c.immutableWeightUploadTransferBytes==101 && c.immutableWeightUploadResidentBytes==103,"failure counted immutable upload")
  }
  expectedCopy,expectedConvert,expectedRaw:=1,1,16;if failStage {expectedCopy=0};if failStage || copyStatus!=0 {expectedConvert=0;expectedRaw=0}
  require(copyCalls==expectedCopy && rowCalls+colCalls==expectedConvert && rawBytes==expectedRaw,"copy/conversion/traffic counts")
  require((layout==RowMajor && colCalls==0) || (layout==ColMajor && rowCalls==0),"wrong conversion route")
  if withLatch {code,calls,site:=0,0,"";if mode=="h2d" {code,calls,site=77,1,"upload-f16-h2d"};if mode=="convert" {code,calls,site=98,1,"upload-f16-convert"};require(c.faultLatch.code==code && c.faultLatch.calls==calls && c.faultLatch.site==site,"fault latch")}
 }}}}}}
}
`

const cudaF16UploadNativePrefix = `
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using cudaError_t = int;
using __half = unsigned short;
enum {cudaSuccess=0,cudaMemcpyHostToDevice=1};
static size_t g_h2d_bytes=0;
static int g_stream=19,copyStatus=0,launchStatus=0,checks=0,rows=0,columns=0;
static bool captureOpen=true;
static void require(bool ok) {if(!ok) {fprintf(stderr,"submission assertion failed\n");exit(1);}}
static int cudaMemcpy(void *d,const void *h,size_t n,int kind) {require(kind==cudaMemcpyHostToDevice);if(copyStatus) return copyStatus;memcpy(d,h,n);return 0;}
static const char *cudaGetErrorString(int) {return "injected copy failure";}
int cudaGetLastError() {++checks;return launchStatus;}
static void launchRow(int grid,int block,int shared,int stream,const float *s,__half *d,int n) {++rows;require(grid==2 && block==256 && shared==0 && stream==19 && s && d && n==300 && captureOpen);}
static void launchColumn(int grid,int block,int shared,int stream,const float *s,__half *d,int out,int in) {++columns;require(grid==2 && block==256 && shared==0 && stream==19 && s && d && out==3 && in==100 && captureOpen);}
`

const cudaF16UploadNativeMain = `
int main() {
 float source[300]={}; __half destination[300]={};
 copyStatus=77;require(fak_cuda_upload_h2d(destination,source,16)==77 && g_h2d_bytes==0);
 copyStatus=0;require(fak_cuda_upload_h2d(destination,source,16)==0 && g_h2d_bytes==16);
 int statuses[]={98,0};for(int status: statuses) {
  launchStatus=status;int before=checks;
  require(fak_cuda_f32_to_f16(destination,source,300)==status);require(checks==before+1);
  require(fak_cuda_f32_to_f16_T(destination,source,3,100)==status);require(checks==before+2);
  fcuda_f32_to_f16(destination,source,300);fcuda_f32_to_f16_T(destination,source,3,100);require(checks==before+4);
  require(g_h2d_bytes==16 && captureOpen);
 }
 require(rows==4 && columns==4 && checks==8);
 copyStatus=77;fcuda_h2d(destination,source,16);require(g_h2d_bytes==16);
 copyStatus=0;fcuda_h2d(destination,source,16);require(g_h2d_bytes==32);
}
`
