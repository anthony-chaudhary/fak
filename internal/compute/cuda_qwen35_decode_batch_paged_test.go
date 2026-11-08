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

// The real StepBatch selector/consumer, CUDA entry, NewKV/Clone, and page-owner
// bodies run together. CUDA allocation and scalar math are explicit host doubles;
// this is neither device qualification nor whole-model numerical parity.
// fak-test:runtime medium est=12s lane=default
func TestCUDAQwen35PagedIndependentDecodeBatch(t *testing.T) {
	read := func(path string) string {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	section := func(s, a, b string) string {
		i, j := strings.Index(s, a), strings.Index(s, b)
		if i < 0 || j <= i || strings.Count(s, a) != 1 || strings.Count(s, b) != 1 {
			t.Fatalf("ambiguous section %s", a)
		}
		return s[i:j]
	}
	wanted := map[string]bool{}
	selectNames := func(names ...string) {
		for _, n := range names {
			wanted[n] = false
		}
	}
	selectNames("cudaKVPagePool", "cudaPagedPoolKey", "cudaPagedPoolsMu", "cudaPagedKV", "cudaPagedKVCopyFailure", "*cudaPagedKVCopyFailure.Error", "*cudaKVPagePool.retainCopyOwnerLocked", "*cudaKVPagePool.stopCopyLocked", "*cudaKVPagePool.copyErrorLocked", "*cudaPagedKV.requireCopyHealthyLocked", "*cudaBackend.pagedPool", "*cudaBackend.newPagedKV", "*cudaPagedKV.Len", "*cudaPagedKV.stride", "*cudaPagedKV.Clone", "*cudaPagedKV.releaseLocked", "*cudaPagedKV.Free", "*cudaBackend.NewKV")
	selectNames("Qwen35SequenceState", "Qwen35SequenceLayer", "Qwen35SequencePrefillRequest", "Qwen35SequencePrefillResult", "Qwen35SequenceTransferCounters", "Qwen35SequenceFeatureRows", "Qwen35SequencePrefixReplay", "Qwen35SequenceError", "*Qwen35SequenceError.Error", "*Qwen35SequenceError.Unwrap", "qwen35SequenceEmbeddingContract")
	selectNames("qwen35SequenceFailure", "validateQwen35SequenceGeometry", "qwen35SequenceSameShape", "qwen35SequenceMatrixBytes", "*cudaBackend.validateQwen35SequenceTensor", "*cudaBackend.validateQwen35SequenceRequestLocked", "*cudaBackend.validateQwen35SequenceStoreRequestLocked", "*cudaBackend.Qwen35SequencePrefill", "*cudaBackend.qwen35SequenceReleaseSinceLocked", "*cudaBackend.qwen35SequencePoisonStates", "*cudaBackend.Qwen35SequenceRawHiddenPath")
	selectNames("*cudaBackend.Qwen35IndependentDecodeBatchPath", "*cudaBackend.cudaQwen35LaneRowView", "*cudaBackend.Qwen35IndependentDecodeBatch", "qwen35DecodeBatchKVChunk")
	selectNames("*BatchSession.StepBatch", "*BatchSession.stepBatchSerial", "qwen35HybridQ4KBatchStep", "*BatchSession.qwen35HybridQ4KBatchOK", "*BatchSession.stepBatchQwen35HybridQ4K", "qwen35IndependentDecodeBatchBackend", "qwen35IndependentDecodeBatchFor", "failQwen35IndependentDecodeBatchSessions", "stepBatchQwen35HybridQ4KIndependent")
	selectNames("batchDiagEnabled", "qwen35BatchDiagEnabled")
	preamble := strings.NewReplacer(
		"\"errors\";", "\"errors\";\"reflect\";",
		"type Backend interface{}", "type Backend interface{Read(Tensor)[]float32;Free(Tensor)}",
		"type cudaBuf struct{", "type cudaBuf struct{be *cudaBackend;bornGen uint64;",
		"type cudaKV struct{", "type cudaKV struct{maxPos int;",
		"type latch struct{bad bool}", "type latch struct{bad bool;calls,stopAt int}",
		"func(l *latch)Admit(string)error{", "func(l *latch)Admit(string)error{l.calls++;if l.stopAt>0&&l.calls==l.stopAt{l.bad=true};",
	).Replace(cudaPagedSequenceGoPreamble)
	var body strings.Builder
	body.WriteString(preamble)
	paths := []string{"cuda_qwen35_sequence_paged.go", "cuda_paged_kv_blocks.go", "cuda_paged_kv.go", "cuda_kv.go", "qwen35_sequence_contract.go", "cuda_qwen35_sequence.go", "qwen35_decode_batch_contract.go", "cuda_qwen35_decode_batch.go", "../model/batch_step.go", "../model/qwen35_batch_decode.go"}
	for _, path := range paths {
		source := read(path)
		fs := token.NewFileSet()
		f, err := parser.ParseFile(fs, path, source, 0)
		if err != nil {
			t.Fatal(err)
		}
		all := path == "cuda_qwen35_sequence_paged.go" || path == "cuda_paged_kv_blocks.go" || path == "qwen35_decode_batch_contract.go"
		emit := func(name string, n ast.Node, prefix string, whole bool) {
			if !whole {
				seen, ok := wanted[name]
				if !ok {
					return
				}
				if seen {
					t.Fatalf("duplicate %s", name)
				}
				wanted[name] = true
			}
			text := source[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset]
			t.Logf("%s/%s %x", path, name, sha256.Sum256([]byte(text)))
			// Mechanical package flattening only. The model control flow is unchanged.
			if strings.HasPrefix(path, "../model/") {
				text = strings.ReplaceAll(text, "compute.", "")
			}
			body.WriteString(prefix + text + "\n")
		}
		for _, decl := range f.Decls {
			switch n := decl.(type) {
			case *ast.FuncDecl:
				name := n.Name.Name
				if n.Recv != nil {
					r := n.Recv.List[0].Type
					star := ""
					if p, ok := r.(*ast.StarExpr); ok {
						star = "*"
						r = p.X
					}
					name = star + r.(*ast.Ident).Name + "." + name
				}
				emit(name, n, "", all)
			case *ast.GenDecl:
				if n.Tok == token.IMPORT {
					continue
				}
				if all {
					emit("declaration", n, "", true)
					continue
				}
				if path == "cuda_paged_kv.go" && n.Tok == token.VAR && strings.Contains(source[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset], "cudaPagedPoolsMu") {
					emit("cudaPagedPoolsMu", n, "", false)
					continue
				}
				for _, spec := range n.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						emit(s.Name.Name, s, "type ", false)
					case *ast.ValueSpec:
						if len(s.Names) == 1 {
							emit(s.Names[0].Name, s, n.Tok.String()+" ", false)
						}
					}
				}
			}
		}
	}
	for name, seen := range wanted {
		if !seen {
			t.Fatalf("missing actual declaration %s", name)
		}
	}
	// Extend the existing explicit doubles without changing the production bodies.
	shims := strings.NewReplacer(
		"return &cudaKV{be:c,cfg:cfg}", "return &cudaKV{be:c,cfg:cfg,K:make([]dslice,cfg.NumLayers),Kraw:make([]dslice,cfg.NumLayers),V:make([]dslice,cfg.NumLayers)}",
		"site==\"attention-v\"", "strings.HasSuffix(site,\"attention-v\")",
		"if c.panicAlloc{", "if site==failSite{failMatches++;if failMatches==failAt{failureTransients=append([]*cudaBuf(nil),c.transient...);failurePointers=nil;for _,b:=range failureTransients{failurePointers=append(failurePointers,b.ptr)};switch failKind{case 1:return nil,errors.New(\"selected allocation\");case 2:panic(\"selected panic\");case 3:runtime.Goexit()}}};if c.panicAlloc{",
		"C.test_counter(5)>=2", "C.test_counter(5)>=cleanupAfter",
		"func(c *cudaBackend)qwen35SequenceReserveKVLocked(*cudaKV,int,int)error{panic(\"unexpected contiguous reserve\")}", "func(c *cudaBackend)qwen35SequenceReserveKVLocked(k *cudaKV,l,n int)error{return reserveContiguous(c,k,l,n)}",
		"func(c *cudaBackend)qwen35SequenceAppendKVLocked(*cudaKV,int,Tensor,Tensor,Tensor,int,int)error{panic(\"unexpected contiguous append\")}", "func(c *cudaBackend)qwen35SequenceAppendKVLocked(k *cudaKV,l int,r,v,w Tensor,s,t int)error{return appendContiguous(k,l,r,v,w,s,t)}",
	).Replace(cudaPagedSequenceGoShims)
	body.WriteString(shims)
	body.WriteString(cudaQwenPagedBatchGoShims)
	body.WriteString(cudaQwenPagedBatchControls)
	kv := read("cuda/kv_precision.cuh")
	native := cudaPagedSequenceNativePreamble +
		section(kv, "static int kv_clone_copy_status(", "// Snapshot-only completion handling.") +
		section(kv, "static int kv_snapshot_status(", "extern \"C\" int fcuda_kv_snapshot_load_checked(") +
		section(read("cuda/paged_attention_vllm.cuh"), "__global__ void reshape_and_cache_run_kernel(", "// Current vLLM reshape_and_cache_kernel") +
		section(read("cuda/paged_attention_vllm.cuh"), "__global__ void paged_gather_rows_kernel(", "__global__ void paged_store_int_kernel(") +
		section(kv, "// BEGIN checked detached paged-sequence owner (#13881).", "// END checked detached paged-sequence owner (#13881).") +
		strings.NewReplacer("++attention;", "++attention;if(mode==13&&attention==at)return 71;", "++fences;", "++fences;if(mode==14&&fences==at)return 72;").Replace(cudaPagedSequenceNativeShims) + "\nextern \"C\" int test_live(void*p){return g_live.count(p);}\n"
	dir := t.TempDir()
	header := strings.Replace(cudaPagedSequenceFixtureHeader, "void test_reset(void);", "void test_reset(void);\nint test_live(void*);", 1)
	for name, data := range map[string]string{"main.go": body.String(), "fixture.h": header, "fixture.cc": native, "cuda_kv_precision.h": read("cuda_kv_precision.h")} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("connected host program %x; native body fixture %x", sha256.Sum256([]byte(body.String())), sha256.Sum256([]byte(native)))
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"), "run", ".")
	command.Dir = dir
	command.Env = append(os.Environ(), "CGO_ENABLED=1", "GO111MODULE=off", "GOWORK=off")
	out, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("connected paged batch host witness: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "actual StepBatch/paged independent decode controls PASS") {
		t.Fatalf("missing complete witness: %s", out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

const cudaQwenPagedBatchGoShims = `
var failSite string
var failKind int
var failAt,failMatches int
var failureTransients []*cudaBuf
var failurePointers []unsafe.Pointer
var cleanupAfter C.int = 2147483647
func makeTensor(c *cudaBackend,d Dtype,l Layout,s []int,q *QuantSpec,b Buffer)Tensor{return Tensor{Dtype:d,Layout:l,Shape:s,Quant:q,buf:b,be:c}}
func(c *cudaBackend)cudaBufForSubmit(t Tensor)*cudaBuf{return t.buf.(*cudaBuf)}
func(c *cudaBackend)Read(t Tensor)[]float32{return append([]float32(nil),unsafe.Slice((*float32)(t.buf.(*cudaBuf).ptr),t.Numel())...)}
func(c *cudaBackend)Free(t Tensor){c.releaseTransientBuffers([]*cudaBuf{t.buf.(*cudaBuf)})}
// Explicit scalar GDN stand-in. Only ownership and model dispatch are witnessed.
func(c *cudaBackend)qwen35BatchGDNLocked(x Tensor,l Qwen35SequenceLayer,r Qwen35IndependentDecodeBatchRequest,index int)(Tensor,*cudaBuf,error){for _,lane:=range r.Lanes{s:=lane.States[index];c.gdn++;*(*float32)(s.Conv.buf.(*cudaBuf).ptr)+=1;*(*float32)(s.Recurrent.buf.(*cudaBuf).ptr)+=1};return c.qwen35SequenceRMSNormLocked(x,Tensor{},len(r.Lanes),0,"gdn-batch-standin")}
func reserveContiguous(c *cudaBackend,k *cudaKV,l,n int)error{if len(k.K)==0{k.K=make([]dslice,l);k.Kraw=make([]dslice,l);k.V=make([]dslice,l)};for _,p:=range [][]dslice{k.K,k.Kraw,k.V}{for i:=range p{if p[i].cap>=n{continue};b,e:=c.dallocDeviceOnlyClass(n*4,MemoryKVCache,"contiguous-control");if e!=nil{return e};if p[i].len>0{C.fcuda_d2d(b.ptr,p[i].ptr,C.size_t(p[i].len*4))};p[i].ptr=b.ptr;p[i].cap=n;p[i].dtype=F32}};return nil}
func appendContiguous(k *cudaKV,l int,r,v,w Tensor,s,t int)error{for i,p:=range [][]dslice{k.Kraw,k.K,k.V}{src:=[]Tensor{r,v,w}[i];C.fcuda_d2d(offsetF(p[l].ptr,s*k.stride()),src.buf.(*cudaBuf).ptr,C.size_t(t*k.stride()*4));p[l].len=(s+t)*k.stride()};if l==0{for i:=0;i<t;i++{k.pos=append(k.pos,s+i)}};return nil}
func(c *cudaBackend)qwen35BatchAttentionIntoLocked(q Tensor,k *cudaKV,l,p,h int,s float32,o Tensor)error{n:=C.fak_qwen35_causal_attention_panel_f32(c.cf(q),(*C.float)(k.K[l].ptr),(*C.float)(k.V[l].ptr),c.cf(o),1,C.int(p),C.int(h),C.int(k.cfg.NumKVHeads),C.int(k.cfg.HeadDim),C.float(s));if n!=0{return errors.New("attention")};return nil}
const PreNorm=0
type config struct{BlockTopology,VocabSize int}
func(config)IsQwen35Hybrid()bool{return true}
func(config)IsMoE()bool{return false}
type model struct{Cfg config}
type fixtureHAL struct{states []Qwen35SequenceState}
func(h *fixtureHAL)mutateSequence(_ Backend,f func([]Qwen35SequenceState)(Qwen35SequencePrefillResult,error))(Qwen35SequencePrefillResult,error){return f(h.states)}
type Session struct{M *model;Backend Backend;qwen35HAL *fixtureHAL;halKV KVStore;halFailure error;halClosed bool;req Qwen35SequencePrefillRequest}
func(s *Session)ensureOpenBackendSession(){if s!=nil&&(s.halClosed||s.halFailure!=nil){panic("closed session")}}
func(s *Session)requireGLM5NextSessionUnsupported(string){}
func(s *Session)closeBackendOwned(){s.halClosed=true}
func(s *Session)beginHALTokenLineageWrite([]int)func(){return func(){}}
func(s *Session)qwen35DecodeBatchLaneRequest(id int)(Qwen35SequencePrefillRequest,func(),bool){r:=s.req;r.TokenIDs=[]int{id};return r,nil,true}
var serialFallbacks int
func(s *Session)Step(id int)[]float32{serialFallbacks++;r:=s.req;r.TokenIDs=[]int{id};v,e:=s.Backend.(*cudaBackend).Qwen35SequencePrefill(r);if e!=nil{panic(e)};out:=s.Backend.Read(v.Logits);s.Backend.Free(v.Logits);return out}
func(s *Session)StepGreedy(id int)(int,[]float32){return 0,s.Step(id)}
type BatchSession struct{M *model;Seqs []*Session;Quant bool;greedy []bool;greedyNext []int;lastStepSharedPanels int;lastStepMACs int64;ObserveSerialStep func(string,time.Duration)}
func batchDecodeFastPathOK(config,bool)bool{return false}
func(*BatchSession)stepBatchF32([]int)[][]float32{panic("unexpected f32")}
func(*BatchSession)stepBatchQ([]int)[][]float32{panic("unexpected quant")}
func splitScaledLogits(_ [][]float32,f []float32,n,v int,_ config)[][]float32{out:=make([][]float32,n);for i:=range out{out[i]=append([]float32(nil),f[i*v:(i+1)*v]...)};return out}
func batchFor(c *cudaBackend,r []Qwen35SequencePrefillRequest)*BatchSession{m:=&model{config{VocabSize:r[0].Output.Shape[0]}};b:=&BatchSession{M:m};for _,v:=range r{b.Seqs=append(b.Seqs,&Session{M:m,Backend:c,qwen35HAL:&fixtureHAL{v.States},halKV:v.KV,req:v})};return b}
func cohort(growth bool)(*cudaBackend,[]Qwen35SequencePrefillRequest,[]*cudaPagedKV){failSite="";failKind=0;failAt=1;failMatches=0;cleanupAfter=2147483647;serialFallbacks=0;c,a:=reset();b:=c.NewKV(a.cfg).(*cudaPagedKV);n,m:=3,14;if growth{n,m=15,991};preparePrefix(c,a,n);preparePrefix(c,b,m);x,y:=a.Clone().(*cudaPagedKV),b.Clone().(*cudaPagedKV);r:=reqFor(c,x,n,1);q:=reqFor(c,y,m,1);q.TokenEmbedding=r.TokenEmbedding;q.OutputNorm=r.OutputNorm;q.Output=r.Output;q.Layers=r.Layers;return c,[]Qwen35SequencePrefillRequest{r,q},[]*cudaPagedKV{a,b,x,y}}
func batchReq(r []Qwen35SequencePrefillRequest)Qwen35IndependentDecodeBatchRequest{return Qwen35IndependentDecodeBatchRequest{Path:Qwen35IndependentDecodeBatchPath,Lanes:r}}
func statePoisoned(r []Qwen35SequencePrefillRequest)bool{for _,v:=range r{for i,l:=range v.Layers{if l.Linear{if v.States[i].Conv.buf.(*cudaBuf).invalid==0||v.States[i].Recurrent.buf.(*cudaBuf).invalid==0{return false}}}};return true}
func unchanged(r []Qwen35SequencePrefillRequest){for _,v:=range r{k:=v.KV.(*cudaPagedKV);must(k.Len()==v.StartPos,"partial logical publication");for _,n:=range k.lens{must(n==v.StartPos,"partial layer publication")}}}
func retained(r []Qwen35SequencePrefillRequest){unchanged(r);must(statePoisoned(r),"incomplete group did not poison all states");count:=0;for p:=cudaQwen35PagedOwners;p!=nil;p=p.next{count++;must(p.started&&!p.completed&&p.k.pool.copyFailure.Load()!=0,"retained owner is reusable")};must(count>0,"lost failed owners");runtime.GC()}
func snapshot(c *cudaBackend,r Qwen35SequencePrefillRequest)[]float32{k:=r.KV.(*cudaPagedKV);var out []float32;for i,planes:=range [][]unsafe.Pointer{k.pool.Kraw,k.pool.K,k.pool.V}{layout:=1;if i==2{layout=0};for _,p:=range planes{t,b,e:=c.tensor([]int{k.Len(),k.stride()},MemoryScratchpad,"oracle-gather");must(e==nil,"snapshot allocate");must(C.fcuda_paged_kv_snapshot_gather_checked((*C.float)(b.ptr),(*C.float)(p),(*C.int)(k.dTable),C.int(k.Len()),4,256,16,C.int(layout))==0,"snapshot gather");out=append(out,c.Read(t)...)} };for i,l:=range r.Layers{if l.Linear{out=append(out,c.Read(r.States[i].Conv)...);out=append(out,c.Read(r.States[i].Recurrent)...)} };return out}
`

const cudaQwenPagedBatchControls = `
func main(){
 // Compare actual independent consumption with ordinary single-sequence entry,
 // including unequal cloned prefixes and growth caused by the later lane.
 for _,growth:=range []bool{false,true}{
  c,r,k:=cohort(growth);old:=k[0].pool.K[0];if growth{must(k[0].pool.capBlocks==64&&k[0].pool.blocks.live==63&&len(k[0].pool.blocks.free)==1,"later-lane growth setup")};oldTables:=[][]int32{append([]int32(nil),k[0].table...),append([]int32(nil),k[1].table...)};start:=int(C.test_counter(6));h2d:=uint64(C.fcuda_h2dxfer_bytes());b:=batchFor(c,r);out:=b.StepBatch([]int{0,1})
  must(serialFallbacks==0&&b.lastStepSharedPanels>0&&b.lastStepMACs>0&&len(out)==2,"actual StepBatch fell back instead of paged batching")
  panels,macs:=Qwen35IndependentDecodeBatchProjectionWork(r[0],2);must(b.lastStepSharedPanels==panels&&b.lastStepMACs==macs,"projection receipt")
  must(c.gdn==12&&int(C.test_counter(6))-start==4,"lane state/attention coverage")
  attention:=make([]float32,4);for i:=range attention{attention[i]=float32(C.test_attention_value(C.int(start+i),0))}
  for i:=range r{must(k[i].Len()==r[i].StartPos&&k[i+2].Len()==r[i].StartPos+1,"owner frontiers");must(reflect.DeepEqual(k[i].table,oldTables[i])&&k[i+2].table[len(k[i+2].table)-1]!=oldTables[i][len(oldTables[i])-1],"partial tail COW");for j,l:=range r[i].Layers{if l.Linear{must(*(*float32)(r[i].States[j].Conv.buf.(*cudaBuf).ptr)==1,"per-lane GDN state")}}}
  if growth{must(k[0].pool.K[0]!=old,"later lane did not grow pool")};must(cudaQwen35PagedOwners==nil&&C.test_counter(7)==0,"successful owner retirement")
  must(uint64(C.fcuda_h2dxfer_bytes())>h2d,"metadata upload was hidden")
  states:=[][]float32{snapshot(c,r[0]),snapshot(c,r[1])};c,r,_=cohort(growth);start=int(C.test_counter(6));for i:=range r{v,e:=c.Qwen35SequencePrefill(r[i]);must(e==nil,"single sequence control");must(reflect.DeepEqual(c.Read(v.Logits),out[i]),"scalar logit control");must(reflect.DeepEqual(snapshot(c,r[i]),states[i]),"serial KV/GDN state differs");for l:=0;l<2;l++{got:=float32(C.test_attention_value(C.int(start+i*2+l),0));want:=attention[l*2+i];must(math.Abs(float64(got-want))<1e-6,"batched gather differs from scalar prefix attention")}}
 }
 // The returned receipt includes the page tables as control bytes, separately
 // from activation traffic. It is not a zero-total-transfer claim.
 c,r,_:=cohort(false);before:=uint64(C.fcuda_h2dxfer_bytes());retireBefore:=C.test_counter(9);v,e:=c.Qwen35IndependentDecodeBatch(batchReq(r));retireDelta:=C.test_counter(9)-retireBefore;must(e==nil&&v.Transfers.H2DBytes==16&&uint64(C.fcuda_h2dxfer_bytes())-before==16&&v.Transfers.ActivationH2DBytes==0&&v.Transfers.ActivationD2HBytes==0,"paged control accounting");must(retireDelta>=2,"missing group retirements")
 // Pure all-lane validation must precede allocation, preparation, and GDN.
 c,r,_=cohort(false);c.faultLatch.calls=0;c.faultLatch.stopAt=2;alloc:=C.test_counter(0);_,e=c.Qwen35IndependentDecodeBatch(batchReq(r));must(e!=nil&&!Qwen35IndependentDecodeBatchMutated(e)&&c.faultLatch.calls==2&&C.test_counter(0)==alloc&&cudaQwen35PagedOwners==nil,"missing post-lock fault admission")
 for _,bad:=range []string{"final-frontier","duplicate-owner","duplicate-state","mixed"}{c,r,_:=cohort(false);switch bad{case "final-frontier":r[1].KV.(*cudaPagedKV).lens[1]++;case "duplicate-owner":r[1].KV=r[0].KV;case "duplicate-state":r[1].States[0]=r[0].States[0];case "mixed":r[1].KV=c.newContiguousKV(r[0].KV.(*cudaPagedKV).cfg)};alloc:=C.test_counter(0);_,e:=c.Qwen35IndependentDecodeBatch(batchReq(r));must(e!=nil&&!Qwen35IndependentDecodeBatchMutated(e)&&C.test_counter(0)==alloc&&c.gdn==0&&cudaQwen35PagedOwners==nil,"preflight effects: "+bad)}
 // Failures after preparation are never replayable, including cleanup,
 // native attention, and a final fence after every attention layer ran.
 for _,fault:=range []string{"reserve","copy","retire","group-retire","attention","late-ffn","fence","cleanup"}{c,r,_:=cohort(false);switch fault{case "reserve":C.test_mode(1,C.test_counter(0)+1);case "copy":C.test_mode(3,0);case "retire":C.test_mode(7,C.test_counter(9)+1);case "group-retire":C.test_mode(7,C.test_counter(9)+retireDelta);case "attention":C.test_mode(13,C.test_counter(6)+2);case "late-ffn":failSite="batch-ffn-down";failKind=1;failAt=8;case "fence":C.test_mode(14,C.test_counter(5)+3);case "cleanup":cleanupAfter=C.test_counter(5)+3;c.cleanupMode=3};v,e:=c.Qwen35IndependentDecodeBatch(batchReq(r));must(e!=nil&&Qwen35IndependentDecodeBatchMutated(e)&&v.Lanes==0,"failure was replayable: "+fault);retained(r);if fault=="attention"{must(strings.Contains(e.Error(),"71"),"first native attention cause lost")};if fault=="fence"{must(strings.Contains(e.Error(),"72"),"first native fence cause lost")};if fault=="group-retire"{n:=0;for p:=cudaQwen35PagedOwners;p!=nil;p=p.next{field:=reflect.ValueOf(p).Elem().FieldByName("retired");if field.IsValid()&&field.Bool(){n++}};must(n==1,"group retirement did not fail after the first lane retired")};if fault=="copy"{must(cudaQwen35PagedOwners.pinned&&C.test_counter(8)>0,"uncertain copy lost pinned root")}}
 c,r,_=cohort(true);C.test_mode(7,C.test_counter(9)+1);_,e=c.Qwen35IndependentDecodeBatch(batchReq(r));must(Qwen35IndependentDecodeBatchMutated(e)&&strings.Contains(e.Error(),"57"),"old arena retirement failure lost");retained(r)
 // Panic becomes a mutated batch error; Goexit is not recoverable but must
 // still run the unconditional poison/owner-retention defer.
 for _,kind:=range []int{1,2,3}{c,r,_:=cohort(false);failSite="batch-ffn-down";failKind=kind;done:=make(chan error,1);go func(){var err error;defer func(){done<-err}();_,err=c.Qwen35IndependentDecodeBatch(batchReq(r))}();select{case e:=<-done:if kind!=3{must(Qwen35IndependentDecodeBatchMutated(e),"early failure not mutated")}else{must(e==nil,"early Goexit became return")};case<-time.After(2*time.Second):panic("early unwind hung")};must(C.test_counter(7)!=0,"early unwind left native cleanup open");must(len(failureTransients)>0,"early unwind captured no active transients");for i,b:=range failureTransients{must(failurePointers[i]!=nil&&b.ptr==failurePointers[i]&&C.test_live(b.ptr)!=0,"early unwind freed submitted transient")};retained(r)}
 for _,kind:=range []int{2,3}{c,r,_:=cohort(false);failSite="batch-output-head";failKind=kind;done:=make(chan error,1);go func(){var err error;defer func(){done<-err}();_,err=c.Qwen35IndependentDecodeBatch(batchReq(r))}();select{case e:=<-done:if kind==2{must(Qwen35IndependentDecodeBatchMutated(e)&&strings.Contains(e.Error(),"selected panic"),"panic not mutated")}else{must(e==nil,"Goexit became return")};case<-time.After(2*time.Second):panic("abnormal batch hung")};retained(r)}
 for _,kind:=range []int{1,2}{c,r,_:=cohort(false);cleanupAfter=C.test_counter(5)+3;c.cleanupMode=kind;done:=make(chan error,1);go func(){var err error;defer func(){done<-err}();_,err=c.Qwen35IndependentDecodeBatch(batchReq(r))}();select{case e:=<-done:if kind==1{must(Qwen35IndependentDecodeBatchMutated(e)&&strings.Contains(e.Error(),"final cleanup panic"),"cleanup panic not mutated")}else{must(e==nil,"cleanup Goexit became return")};case<-time.After(2*time.Second):panic("cleanup unwind hung")};retained(r)}
 // Connected consumer closes every participating session, with no serial
 // replay or published receipt, when a late native error is returned.
 c,r,_=cohort(false);b:=batchFor(c,r);C.test_mode(13,C.test_counter(6)+2);var recovered interface{};func(){defer func(){recovered=recover()}();b.StepBatch([]int{0,1})}();must(recovered!=nil&&serialFallbacks==0&&b.lastStepSharedPanels==0,"consumer replayed mutation");for _,s:=range b.Seqs{must(s.halClosed&&s.halFailure!=nil,"consumer left failed lane open")};retained(r)
 // Existing contiguous batch remains selectable (explicit contiguous math
 // doubles), and the single paged entry remains covered above.
 c,r,_=cohort(false);for i:=range r{r[i].KV=c.newContiguousKV(r[i].KV.(*cudaPagedKV).cfg);r[i].StartPos=0};v,e=c.Qwen35IndependentDecodeBatch(batchReq(r));must(e==nil&&v.Lanes==2&&v.SharedProjectionPanels>0,fmt.Sprintf("contiguous batch regression: %v",e))
 fmt.Println("actual StepBatch/paged independent decode controls PASS")
}
`
