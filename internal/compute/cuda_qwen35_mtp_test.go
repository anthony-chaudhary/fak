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

// Required source-derived SOFTWARE witness, not a CUDA/device qualification.
// fak-test:runtime slow est=80s lane=default
func TestCUDAMTPPrelayerSource(t *testing.T) {
	read := func(path string) string {
		b, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		return string(b)
	}
	section := func(src, start, end string) string {
		a, b := strings.Index(src, start), strings.Index(src, end)
		if a < 0 || b <= a || strings.Count(src, start) != 1 || strings.Count(src, end) != 1 {
			t.Fatal("ambiguous section", start)
		}
		text := src[a+len(start) : b]
		t.Logf("%s sha256=%x", start, sha256.Sum256([]byte(text)))
		return text
	}
	goSource := read("cuda_qwen35_mtp.go")
	compact := strings.Join(strings.Fields(goSource), "")
	for _, want := range []string{
		"os.Getenv(\"FAK_CUDA_QWEN35_MTP\")==\"1\"",
		"func(c*cudaBackend)Qwen35MTPFuse(reqQwen35MTPFuseRequest)(Tensor,error){cudaMu.Lock()defercudaMu.Unlock()",
		"C.fcuda_qwen35_mtp_admit(", "C.fcuda_qwen35_mtp_f32(", "C.fcuda_qwen35_mtp_stop(",
		"returnuint64(C.fcuda_h2dxfer_bytes()),uint64(C.fcuda_hostxfer_bytes())",
	} {
		if !strings.Contains(compact, want) {
			t.Fatal("missing adapter boundary", want)
		}
	}
	if strings.Contains(goSource, "cudaForwardExecutionMu.Lock") || strings.Contains(goSource, "SupportsQ6KMatMul") {
		t.Fatal("unexpected outer lock or Q6 capability")
	}
	if !strings.Contains(read("cuda_kernels.cu"), "#include \"cuda/qwen35_mtp.cuh\"") {
		t.Fatal("native fragment not linked")
	}
	for _, sym := range []string{"fcuda_qwen35_mtp_admit", "fcuda_qwen35_mtp_f32", "fcuda_qwen35_mtp_stop"} {
		if !strings.Contains(read("cuda_backend.h"), sym+"(") {
			t.Fatal("missing canonical declaration", sym)
		}
	}
	native := section(read("cuda/qwen35_mtp.cuh"), "// BEGIN checked MTP native core.", "// END checked MTP native core.")
	core := section(goSource, "// BEGIN MTP prelayer owner core.", "// END MTP prelayer owner core.")
	t.Run("native-control", func(t *testing.T) {
		compiler := ""
		for _, name := range []string{"c++", "g++", "clang++"} {
			if p, e := exec.LookPath(name); e == nil {
				compiler = p
				break
			}
		}
		if compiler == "" {
			t.Fatal("required host C++ compiler unavailable")
		}
		dir := t.TempDir()
		src := filepath.Join(dir, "mtp.cc")
		exe := filepath.Join(dir, "mtp")
		if runtime.GOOS == "windows" {
			exe += ".exe"
		}
		program := cudaMTPNativePreamble + native + cudaMTPNativeFixture + cudaMTPNativeControls
		t.Logf("derived native sha256=%x", sha256.Sum256([]byte(program)))
		if e := os.WriteFile(src, []byte(program), 0600); e != nil {
			t.Fatal(e)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
		defer cancel()
		if out, e := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", src, "-o", exe).CombinedOutput(); e != nil {
			t.Fatalf("host compile: %v\n%s", e, out)
		}
		if out, e := exec.CommandContext(ctx, exe).CombinedOutput(); e != nil {
			t.Fatalf("native controls: %v\n%s", e, out)
		}
	})
	t.Run("go-owner", func(t *testing.T) {
		wanted := map[string]bool{"QuantSpec": false, "Buffer": false, "HostBuffer": false, "Tensor": false, "makeTensor": false, "Tensor.Backend": false,
			"cudaBuf": false, "*cudaBuf.Ready": false, "*cudaBuf.invalidForSubmit": false, "Qwen35MTPFuseRequest": false,
			"Qwen35MTPDraftPath": false, "UnsupportedQwen35MTPDraftError": false, "*UnsupportedQwen35MTPDraftError.Error": false}
		var program strings.Builder
		program.WriteString(cudaMTPGoFixture)
		for _, path := range []string{"compute.go", "cuda_backend_state.go", "qwen35_mtp_contract.go"} {
			src := read(path)
			fs := token.NewFileSet()
			f, e := parser.ParseFile(fs, path, src, 0)
			if e != nil {
				t.Fatal(e)
			}
			emit := func(name string, n ast.Node, prefix string) {
				seen, ok := wanted[name]
				if !ok {
					return
				}
				if seen {
					t.Fatal("duplicate", name)
				}
				wanted[name] = true
				text := src[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset]
				t.Logf("%s/%s sha256=%x", path, name, sha256.Sum256([]byte(text)))
				program.WriteString(prefix + text + "\n")
			}
			for _, d := range f.Decls {
				switch n := d.(type) {
				case *ast.GenDecl:
					for _, s := range n.Specs {
						switch v := s.(type) {
						case *ast.TypeSpec:
							emit(v.Name.Name, v, "type ")
						case *ast.ValueSpec:
							if len(v.Names) == 1 && (n.Tok == token.CONST || n.Tok == token.VAR) {
								emit(v.Names[0].Name, v, n.Tok.String()+" ")
							}
						}
					}
				case *ast.FuncDecl:
					key := n.Name.Name
					if n.Recv != nil {
						r := n.Recv.List[0].Type
						prefix := ""
						if p, ok := r.(*ast.StarExpr); ok {
							r = p.X
							prefix = "*"
						}
						id, ok := r.(*ast.Ident)
						if !ok {
							continue
						}
						key = prefix + id.Name + "." + key
					}
					emit(key, n, "")
				}
			}
		}
		for key, seen := range wanted {
			if !seen {
				t.Fatal("missing production declaration", key)
			}
		}
		program.WriteString(core)
		program.WriteString(cudaMTPGoControls)
		text := program.String()
		t.Logf("derived Go sha256=%x", sha256.Sum256([]byte(text)))
		dir := t.TempDir()
		if e := os.WriteFile(filepath.Join(dir, "owner_test.go"), []byte(text), 0600); e != nil {
			t.Fatal(e)
		}
		exe := filepath.Join(runtime.GOROOT(), "bin", "go")
		if runtime.GOOS == "windows" {
			exe += ".exe"
		}
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "test", "-run", "^TestOwner$", "-count=1", "-timeout", "45s", ".")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GO111MODULE=off")
		if out, e := cmd.CombinedOutput(); e != nil {
			t.Fatalf("Go owner controls: %v\n%s", e, out)
		}
	})
}

const cudaMTPNativePreamble = `#include <cassert>
#include <cmath>
#include <cstdint>
#include <climits>
#include <unordered_map>
#include <vector>
using std::isfinite;
using cudaError_t=int;
using cudaStream_t=void*;
using cublasStatus_t=int;
using cublasPointerMode_t=int;
using cudaStreamCaptureStatus=int;
const int cudaSuccess=0,cudaMemoryTypeDevice=2,cudaStreamCaptureStatusNone=0;
const int CUBLAS_STATUS_SUCCESS=0,CUBLAS_POINTER_MODE_HOST=0,CUBLAS_OP_T=1,CUBLAS_OP_N=0;
struct cudaPointerAttributes{int type=2,device=0;};
struct cudaDeviceProp{int maxThreadsPerBlock=256,maxThreadsDim[3]={256,1,1};size_t sharedMemPerBlock=1024;int maxGridSize[3]={1,1,1};};
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
static void *g_stream=(void*)1,*g_blas=(void*)2;
static bool g_capture_open=false,g_pool_graph_deep=false,g_ts_on=false,peer=false;
static int g_cuda_cleanup_status=0,peek=0,capture=0,device=0,prop_bad=0,attr_bad=0,driver_error=0,blas_state_error=0,blas_error=0;
static int norms=0,gemms=0,fences=0,driver_calls=0,fail_norm=0,sticky_norm=0,fence_error=0;
static bool wrong_stream=false,wrong_mode=false,delayed=false;
static float *late_out=nullptr;
static std::unordered_map<void*,size_t>g_live;
static bool v41_candidates_stream_uncertain(){return peer||g_cuda_cleanup_status!=0;}
static int cudaPeekAtLastError(){return peek;}
static int cudaGetDevice(int *p){++driver_calls;*p=device;return driver_error;}
static int cudaStreamIsCapturing(void*s,int*p){assert(s==g_stream);++driver_calls;*p=capture;return driver_error;}
static int cudaGetDeviceProperties(cudaDeviceProp*p,int d){++driver_calls;assert(d==0);*p=cudaDeviceProp{};if(prop_bad)p->maxThreadsPerBlock=255;return driver_error;}
static int cudaPointerGetAttributes(cudaPointerAttributes*p,const void*){++driver_calls;*p=cudaPointerAttributes{};if(attr_bad)p->type=1;return driver_error;}
static int cublasGetStream(void*b,void**s){assert(b==g_blas);*s=wrong_stream?(void*)3:g_stream;return blas_state_error;}
static int cublasGetPointerMode(void*b,int*p){assert(b==g_blas);*p=wrong_mode?1:0;return blas_state_error;}
#define FCUDA_TS(name,grid,...) do { assert(!g_ts_on);__VA_ARGS__; } while(0)
// Arithmetic here is an explicit host shim, not execution of k_rmsnorm.
static void fcuda_rmsnorm_f32(const float*x,const float*w,float*y,int rows,int n,float eps){
 assert(rows==1);++norms;float r[256]={};
 for(int t=0;t<256;t++)for(int i=t;i<n;i+=256)r[t]+=x[i]*x[i];
 for(int s=128;s;s/=2)for(int t=0;t<s;t++)r[t]+=r[t+s];
 float inv=1.f/std::sqrt(r[0]/n+eps);for(int i=0;i<n;i++)y[i]=x[i]*inv*w[i];
 if(norms==fail_norm)peek=17;
 if(norms==sticky_norm)g_cuda_cleanup_status=71;
}
static int cublasSgemm(void*b,int ta,int tb,int m,int n,int k,const float*a,const float*w,int lda,const float*x,int ldb,const float*beta,float*y,int ldc){
 assert(b==g_blas&&ta==CUBLAS_OP_T&&tb==CUBLAS_OP_N&&n==1&&k==2*m&&lda==k&&ldb==k&&ldc==m&&*a==1&&*beta==0);
 ++gemms;if(blas_error)return blas_error;
 if(delayed){late_out=y;return 0;}
 for(int i=0;i<m;i++){float z=0;for(int j=0;j<k;j++)z+=w[i*k+j]*x[j];y[i]=z;}return 0;
}
static int cudaStreamSynchronize(void*s){assert(s==g_stream);++fences;if(fence_error)return fence_error;if(late_out){*late_out=123;late_out=nullptr;}return 0;}
static void reset(){
 g_live.clear();g_stream=(void*)1;g_blas=(void*)2;g_capture_open=g_pool_graph_deep=g_ts_on=peer=false;
 g_cuda_cleanup_status=peek=capture=device=prop_bad=attr_bad=driver_error=blas_state_error=blas_error=0;
 norms=gemms=fences=driver_calls=fail_norm=sticky_norm=fence_error=0;wrong_stream=wrong_mode=delayed=false;late_out=nullptr;
}
`

const cudaMTPNativeFixture = `struct fixture{
 int h=3;std::vector<float>p,e,hn,en,w,j,o;
 fixture(int n=3):h(n),p(n,2),e(n,1),hn(n,3),en(n,4),w(2*n*n),j(2*n,-99),o(n,-99){
  for(int i=0;i<n;i++){w[i*2*n+i]=1;w[i*2*n+n+i]=2;}
  for(auto*v:{&p,&e,&hn,&en,&w,&j,&o})g_live[v->data()]=v->size()*4;
 }
 int admit(){return fcuda_qwen35_mtp_admit(p.data(),e.data(),hn.data(),en.data(),w.data(),h,1e-6f);}
 int run(){return fcuda_qwen35_mtp_f32(p.data(),e.data(),hn.data(),en.data(),w.data(),j.data(),o.data(),h,1e-6f);}
};
`

const cudaMTPNativeControls = `
int main(){
 for(int h:{1,255,257,5120}){reset();fixture f(h);assert(f.admit()==0&&norms==0&&gemms==0&&fences==0);assert(f.run()==0&&norms==2&&gemms==1&&fences==1);
  for(int i=0;i<h;i++){assert(f.j[i]>3.99f&&f.j[h+i]>2.99f&&std::fabs(f.o[i]-10.f)<0.001f);assert(f.p[i]==2&&f.e[i]==1&&f.hn[i]==3&&f.en[i]==4);}
 }
 for(int mode=0;mode<11;mode++){reset();fixture f;
  if(mode==0)g_ts_on=true;if(mode==1)g_capture_open=true;if(mode==2)g_pool_graph_deep=true;
  if(mode==3)device=1;if(mode==4)capture=1;if(mode==5)prop_bad=1;
  if(mode==6)wrong_stream=true;if(mode==7)wrong_mode=true;if(mode==8)attr_bad=1;
  if(mode==9)g_live[f.w.data()]--;if(mode==10)f.h=5121;
  assert(f.admit()<0&&norms==0&&gemms==0&&fences==0&&g_cuda_cleanup_status==0);
 }
 reset();fixture a;int before=driver_calls;assert(fcuda_qwen35_mtp_admit(a.p.data(),a.e.data(),a.hn.data(),a.en.data(),a.w.data(),INT_MAX,1e-6f)<0&&driver_calls==before);
 assert(fcuda_qwen35_mtp_admit(a.p.data(),a.e.data(),a.hn.data(),a.en.data(),a.w.data(),a.h,NAN)<0);
 for(int mode=0;mode<3;mode++){reset();fixture f;float *j=f.j.data(),*o=f.o.data();
  if(mode==0)o=j;if(mode==1)o=f.p.data();if(mode==2)g_live[j]--;
  assert(fcuda_qwen35_mtp_f32(f.p.data(),f.e.data(),f.hn.data(),f.en.data(),f.w.data(),j,o,f.h,1e-6f)<0&&norms==0&&gemms==0&&fences==0);
 }
 for(int which:{1,2})for(bool sticky:{false,true}){reset();fixture f;if(sticky)sticky_norm=which;else fail_norm=which;
  int status=f.run();assert(status==(sticky?71:92017)&&norms==which&&gemms==0&&fences==0&&g_cuda_cleanup_status==status);
  assert(f.run()==status&&norms==which);fcuda_qwen35_mtp_stop();assert(g_cuda_cleanup_status==status);
 }
 reset();fixture b;blas_error=8;assert(b.run()==93008&&norms==2&&gemms==1&&fences==0&&b.o[0]==-99);
 reset();fixture c;delayed=true;fence_error=19;assert(c.run()==92019&&fences==1&&c.o[0]==-99&&late_out==c.o.data());
 int oldn=norms;assert(c.run()==92019&&norms==oldn);*late_out=123;assert(c.o[0]==123&&g_cuda_cleanup_status==92019);late_out=nullptr;
 reset();fixture d;driver_error=11;assert(d.admit()==92011&&norms==0);reset();fixture e;peer=true;assert(e.admit()==95002&&norms==0);
 reset();fixture f;blas_state_error=7;assert(f.admit()==93007&&norms==0);
 reset();fcuda_qwen35_mtp_stop();assert(g_cuda_cleanup_status==95001);return 0;
}
`

const cudaMTPGoFixture = `package compute
import("fmt";"math";"runtime";"sync/atomic";"testing";"time";"unsafe")
type Backend=any
type MemoryClass string
type Dtype int
const(F32 Dtype=iota;Q8_0)
type Layout int
const(RowMajor Layout=iota;ColMajor)
type health bool
func(h health)Refusing()bool{return bool(h)}
type DeviceFaultLatch struct{bad bool;observed int}
func(f *DeviceFaultLatch)Health()health{return health(f!=nil&&f.bad)}
func(f *DeviceFaultLatch)Observe(_ string,_ string,_ int){if f!=nil{f.bad=true;f.observed++}}
const DeviceFaultExecution="execution"
type cudaBackend struct{graphOnce any;capturing bool;faultLatch *DeviceFaultLatch;transient []*cudaBuf;fenceGen uint64}
type cudaUploadFence struct{}
var cudaDev *cudaBackend
var cudaAsyncPending=map[unsafe.Pointer]*cudaUploadFence{}
`

const cudaMTPGoControls = `
func fixtureMTP()(c *cudaBackend,req Qwen35MTPFuseRequest,ops cudaMTPPrelayerOps,counts *[5]int){
 cudaMTPPrelayerActive=nil;cudaAsyncPending=map[unsafe.Pointer]*cudaUploadFence{}
 c=&cudaBackend{faultLatch:&DeviceFaultLatch{}};cudaDev=c;counts=&[5]int{}
 tensor:=func(n int,shape []int,async bool)Tensor{v:=make([]float32,n);for i:=range v{v[i]=float32(i+1)};b:=&cudaBuf{ptr:unsafe.Pointer(&v[0]),n:n*4};if async{b.be=c;b.bornGen=c.fenceGen;c.transient=append(c.transient,b)};return makeTensor(c,F32,RowMajor,shape,nil,b)}
 req=Qwen35MTPFuseRequest{PriorHidden:tensor(3,[]int{3},false),CurrentEmbedding:tensor(3,[]int{3},false),HiddenNorm:tensor(3,[]int{3},false),EmbeddingNorm:tensor(3,[]int{3},false),FC:tensor(18,[]int{3,6},false),Epsilon:1e-6}
 ops.admit=func(p [5]unsafe.Pointer,h int,e float32)int{counts[0]++;if h!=3||e!=req.Epsilon||p[0]!=req.PriorHidden.buf.(*cudaBuf).ptr{panic("wrong intake")};return 0}
 ops.allocate=func(n int)(Tensor,*cudaBuf,error){counts[1]++;x:=tensor(n,[]int{n},true);return x,x.buf.(*cudaBuf),nil}
 ops.run=func(p [5]unsafe.Pointer,j,o unsafe.Pointer,h int,e float32)int{counts[2]++;if j==o||h!=3||e!=req.Epsilon{panic("bad owner operands")};*(*float32)(o)=42;return 0}
 ops.stop=func(){counts[3]++}
 return
}
func TestOwner(t *testing.T){
 t.Run("success",func(t *testing.T){c,r,o,n:=fixtureMTP();out,e:=c.mtpPrelayerWithOps(r,true,o);if e!=nil||out.buf==nil||!out.buf.Ready()||*(*float32)(out.buf.(*cudaBuf).ptr)!=42||*n!=[5]int{1,2,1,0,0}||cudaMTPPrelayerActive!=nil||c.faultLatch.bad||c.fenceGen!=1{t.Fatalf("bad success %v %v",e,*n)}})
 for mode:=0;mode<17;mode++{t.Run(fmt.Sprintf("refuse-%d",mode),func(t *testing.T){
  c,r,o,n:=fixtureMTP();enabled:=true;b:=r.PriorHidden.buf.(*cudaBuf)
  switch mode{case 0:enabled=false;case 1:c.capturing=true;case 2:c.graphOnce=1;case 3:c.faultLatch.bad=true;case 4:r.PriorHidden.Dtype=Q8_0;case 5:r.FC.Shape=[]int{6,3};case 6:r.HiddenNorm.Shape=[]int{5121};case 7:r.Epsilon=float32(math.NaN());case 8:r.Epsilon=0;case 9:b.n--;case 10:b.managed=true;case 11:b.device=1;case 12:b.invalid=1;case 13:b.be=c;b.bornGen=c.fenceGen;case 14:cudaAsyncPending[b.ptr]=&cudaUploadFence{};case 15:r.FC.Quant=&QuantSpec{};case 16:b.graphOncePinned=true}
  out,e:=c.mtpPrelayerWithOps(r,enabled,o);if e==nil||out.buf!=nil||*n!=[5]int{}||cudaMTPPrelayerActive!=nil{t.Fatalf("effects on refusal %v %v",e,*n)}
 })}
 t.Run("native-known-refusal",func(t *testing.T){c,r,o,n:=fixtureMTP();o.admit=func([5]unsafe.Pointer,int,float32)int{n[0]++;return -1};_,e:=c.mtpPrelayerWithOps(r,true,o);if e==nil||*n!=[5]int{1,0,0,0,0}||cudaMTPPrelayerActive!=nil||c.faultLatch.bad{t.Fatalf("bad known refusal %v",*n)}})
 for mode:=0;mode<5;mode++{t.Run(fmt.Sprintf("fail-%d",mode),func(t *testing.T){
  c,r,o,n:=fixtureMTP();alloc:=o.allocate
  switch mode{
  case 0:o.admit=func([5]unsafe.Pointer,int,float32)int{n[0]++;return 92001}
  case 1,2:o.allocate=func(k int)(Tensor,*cudaBuf,error){if n[1]+1==mode{n[1]++;return Tensor{},nil,fmt.Errorf("allocation%d",mode)};return alloc(k)}
  case 3:o.run=func([5]unsafe.Pointer,unsafe.Pointer,unsafe.Pointer,int,float32)int{n[2]++;return 93001}
  case 4:o.allocate=func(k int)(Tensor,*cudaBuf,error){x,b,e:=alloc(k);b.n--;return x,b,e}
  }
  out,e:=c.mtpPrelayerWithOps(r,true,o);a:=cudaMTPPrelayerActive
  if e==nil||out.buf!=nil||a==nil||a.failure==nil||a.request.PriorHidden.buf!=r.PriorHidden.buf||a.request.FC.buf!=r.FC.buf||n[3]!=1||n[4]!=0||!c.faultLatch.bad{t.Fatalf("bad failure mode%d %v %v",mode,e,*n)}
  for _,x:=range []Tensor{a.joined,a.output}{if b,ok:=x.buf.(*cudaBuf);ok&&(!b.invalidForSubmit()||b.ptr==nil){t.Fatal("lost exact allocation")}}
  prior:=*n;_,e=c.mtpPrelayerWithOps(r,true,o);if e==nil||*n!=prior||cudaMTPPrelayerActive!=a{t.Fatal("retry effects")}
 })}
 for _,phase:=range []string{"admit","allocate","run"}{for _,kind:=range []string{"panic","goexit"}{t.Run(phase+"-"+kind,func(t *testing.T){
  c,r,o,n:=fixtureMTP();normal:=false;var recovered any;done:=make(chan struct{})
  die:=func(){if kind=="panic"{panic("controlled")};runtime.Goexit()}
  switch phase{
  case "admit":o.admit=func([5]unsafe.Pointer,int,float32)int{die();return 0}
  case "allocate":o.allocate=func(int)(Tensor,*cudaBuf,error){die();return Tensor{},nil,nil}
  case "run":o.run=func([5]unsafe.Pointer,unsafe.Pointer,unsafe.Pointer,int,float32)int{die();return 0}
  }
  go func(){defer close(done);defer func(){recovered=recover()}();_,_=c.mtpPrelayerWithOps(r,true,o);normal=true}()
  select{case <-done:case <-time.After(5*time.Second):t.Fatal("unwind timed out")}
  a:=cudaMTPPrelayerActive
  if normal||(kind=="panic"&&recovered!="controlled")||(kind=="goexit"&&recovered!=nil)||a==nil||a.request.PriorHidden.buf!=r.PriorHidden.buf||a.request.FC.buf!=r.FC.buf||a.failure==nil||n[3]!=1||!c.faultLatch.bad{t.Fatalf("bad unwind normal%v recovered%v counts%v",normal,recovered,*n)}
  prior:=*n;_,e:=c.mtpPrelayerWithOps(r,true,o);if e==nil||*n!=prior||cudaMTPPrelayerActive!=a{t.Fatal("unwind retry escaped")}
 })}}
}
`
