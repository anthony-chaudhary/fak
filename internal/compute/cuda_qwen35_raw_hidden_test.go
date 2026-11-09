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

// The extracted production tail and cleanup execute with explicit arithmetic,
// allocation and device shims. This is not CUDA or whole-model qualification.
// fak-test:runtime medium est=2s lane=default
func TestCUDARawHiddenSourceLifetime(t *testing.T) {
	source, err := os.ReadFile("cuda_qwen35_sequence.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	fileSet := token.NewFileSet()
	file, err := parser.ParseFile(fileSet, "cuda_qwen35_sequence.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	functions := map[string]string{}
	for _, declaration := range file.Decls {
		fn, ok := declaration.(*ast.FuncDecl)
		if !ok {
			continue
		}
		functions[fn.Name.Name] = text[fileSet.Position(fn.Pos()).Offset:fileSet.Position(fn.End()).Offset]
	}
	entry := functions["Qwen35SequencePrefill"]
	between := func(s, first, last string) string {
		t.Helper()
		i := strings.Index(s, first)
		if i < 0 {
			t.Fatalf("missing extraction start %q", first)
		}
		j := strings.Index(s[i:], last)
		if j < 0 {
			t.Fatalf("missing extraction end %q", last)
		}
		return s[i : i+j]
	}
	prefix := between(entry, "transientStart := len(c.transient)", "kv, err := c.validateQwen35SequenceStoreRequestLocked(req, paged)")
	tail := between(entry, "// BEGIN CUDA raw-hidden publication tail.", "// END CUDA raw-hidden publication tail.")
	split := between(entry, "finishSplit := func() error {", "// BEGIN sequence KV reservation.")
	release := functions["qwen35SequenceReleaseSinceLocked"]
	capability := functions["Qwen35SequenceRawHiddenPath"]
	preflight := functions["validateQwen35SequenceStoreRequestLocked"]
	gate := between(preflight[strings.Index(preflight, "{")+1:], "if req.CaptureRawHidden", "if err := validateQwen35SequenceGeometry(req)")
	if release == "" || capability == "" {
		t.Fatal("missing real lifetime/capability methods")
	}
	if !strings.Contains(capability, "c == cudaDev") || !strings.Contains(capability, `os.Getenv("FAK_CUDA_QWEN35_MTP") == "1"`) {
		t.Fatal("raw hidden must be explicit and registered-backend scoped")
	}
	for _, refused := range []string{"c.capturing", "c.graphOnce != nil", "req.CapturePrefixReplay", "len(req.FeatureLayers) != 0"} {
		if !strings.Contains(gate, refused) {
			t.Fatalf("missing pre-effect refusal %s", refused)
		}
	}
	if strings.Contains(capability, "PrefixReplayPath") {
		t.Fatal("raw capability must not advertise prefix replay")
	}
	if strings.Count(tail, "rawHiddenRows, rawHiddenBuf = x, xBuf") != 1 ||
		strings.Index(tail, "rawHiddenRows, rawHiddenBuf = x, xBuf") > strings.Index(tail, "qwen35SequenceRMSNormLocked") {
		t.Fatal("raw capture must bind the original residual before output normalization")
	}
	if !cudaRawHiddenPublicationOrdered(tail) {
		t.Fatal("unique raw publication must follow returning completion and transfer checks")
	}
	cudaRawHiddenPublicationOrderControls(t)
	bridgeSource, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	bridgeSet := token.NewFileSet()
	bridgeFile, err := parser.ParseFile(bridgeSet, "cuda.go", bridgeSource, 0)
	if err != nil {
		t.Fatal(err)
	}
	bridge := ""
	for _, declaration := range bridgeFile.Decls {
		if fn, ok := declaration.(*ast.FuncDecl); ok && fn.Name.Name == "cudaQwen35IQ2SequenceRetirementFailedLocked" {
			bridge = string(bridgeSource[bridgeSet.Position(fn.Pos()).Offset:bridgeSet.Position(fn.End()).Offset])
		}
	}
	if bridge == "" {
		t.Fatal("missing actual always-linked shared-stop bridge")
	}
	native, err := os.ReadFile("cuda/iq_gemv.cuh")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(native), "if (first != 0 && !g_cuda_cleanup_status) g_cuda_cleanup_status = first;") {
		t.Fatal("review first-cause bridge semantics before changing host model")
	}
	bridge = strings.NewReplacer("C.fcuda_qwen35_iq2_sequence_retirement_failed", "testRetirementFailed", "C.int", "int").Replace(bridge)
	split = strings.NewReplacer("C.fcuda_sequence_split_status", "testSplitStatus", "C.fcuda_sequence_split_end", "testSplitEnd").Replace(split)
	// The only substitutions are the existing C boundary names/types, not Go
	// ownership branches, failure cleanup, or release/keep logic.
	tail = strings.NewReplacer("C.fcuda_d2d", "testD2D", "C.size_t", "uintptr",
		"C.fcuda_rmsnorm_f32", "testRMSNorm", "C.float", "float32", "C.int", "int",
		"C.fcuda_h2dxfer_bytes", "testH2D", "C.fcuda_hostxfer_bytes", "testD2H").Replace(tail)
	program := cudaRawHiddenFixture + capability + "\n" + release + "\n" + bridge +
		"\nfunc (c *cudaBackend) preflight(req Qwen35SequencePrefillRequest) (*cudaKV,error) {\n" + gate + "\nreturn nil,nil\n}\n" +
		"func (c *cudaBackend) sequenceTail(req Qwen35SequencePrefillRequest) (result Qwen35SequencePrefillResult, err error) {\n" +
		"var paged *rawHiddenUnselectedPaged\n" + prefix + cudaRawHiddenTailSetup + split + tail + "\n}\n" + cudaRawHiddenControls
	t.Logf("production source %x; extracted host fixture %x", sha256.Sum256(source), sha256.Sum256([]byte(program)))
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "owner_test.go"), []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"), "test", "-run", "^TestRawLifetime$", "-count=1", "-timeout=45s")
	command.Dir = dir
	command.Env = append(os.Environ(), "GOWORK=off", "GO111MODULE=off")
	out, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("actual raw-hidden source control failed: %v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

// Empty result resets on failure are not publication. Check the syntax of the
// actual tail rather than the first textual mention of its result type. This is
// a source guard, not a substitute for the extracted lifetime fixture below.
func cudaRawHiddenPublicationOrdered(tail string) bool {
	file, err := parser.ParseFile(token.NewFileSet(), "tail.go", "package p\nfunc check() {\n"+tail+"\n}", 0)
	if err != nil || len(file.Decls) != 1 {
		return false
	}
	fn, ok := file.Decls[0].(*ast.FuncDecl)
	if !ok || fn.Body == nil {
		return false
	}
	ident := func(e ast.Expr, name string) bool { id, ok := e.(*ast.Ident); return ok && id.Name == name }
	returnsFailure := func(body *ast.BlockStmt) bool {
		if body == nil || len(body.List) == 0 {
			return false
		}
		ret, ok := body.List[len(body.List)-1].(*ast.ReturnStmt)
		return ok && len(ret.Results) == 2 && ident(ret.Results[0], "result") && !ident(ret.Results[1], "nil")
	}
	transferNonzero := func(e ast.Expr, field string) bool {
		cmp, ok := e.(*ast.BinaryExpr)
		if !ok || cmp.Op != token.NEQ {
			return false
		}
		sel, ok := cmp.X.(*ast.SelectorExpr)
		zero, zok := cmp.Y.(*ast.BasicLit)
		return ok && ident(sel.X, "transfers") && sel.Sel.Name == field && zok && zero.Kind == token.INT && zero.Value == "0"
	}
	top := make(map[ast.Stmt]bool)
	var fence, transfer token.Pos
	fenceCount, transferCount := 0, 0
	for _, stmt := range fn.Body.List {
		top[stmt] = true
		gate, ok := stmt.(*ast.IfStmt)
		if !ok || gate.Else != nil || !returnsFailure(gate.Body) {
			continue
		}
		if init, ok := gate.Init.(*ast.AssignStmt); ok && init.Tok == token.DEFINE && len(init.Lhs) == 1 && len(init.Rhs) == 1 && ident(init.Lhs[0], "err") && ident(init.Rhs[0], "finalFenceErr") {
			if cmp, ok := gate.Cond.(*ast.BinaryExpr); ok && cmp.Op == token.NEQ && ident(cmp.X, "err") && ident(cmp.Y, "nil") {
				fence, fenceCount = gate.End(), fenceCount+1
			}
		}
		if either, ok := gate.Cond.(*ast.BinaryExpr); ok && gate.Init == nil && either.Op == token.LOR && transferNonzero(either.X, "ActivationH2DBytes") && transferNonzero(either.Y, "ActivationD2HBytes") {
			transfer, transferCount = gate.End(), transferCount+1
		}
	}
	var publication token.Pos
	publications, valid := 0, true
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		assignment, ok := node.(*ast.AssignStmt)
		if !ok {
			return true
		}
		writesResult := false
		for _, lhs := range assignment.Lhs {
			writesResult = writesResult || ident(lhs, "result")
		}
		if !writesResult {
			return true
		}
		if assignment.Tok != token.ASSIGN || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
			valid = false
			return true
		}
		literal, ok := assignment.Rhs[0].(*ast.CompositeLit)
		if !ok || !ident(literal.Type, "Qwen35SequencePrefillResult") {
			valid = false
			return true
		}
		if len(literal.Elts) == 0 {
			return true
		}
		publications++
		publication = assignment.Pos()
		rawFields := 0
		for _, element := range literal.Elts {
			field, ok := element.(*ast.KeyValueExpr)
			if ok && ident(field.Key, "RawHiddenRows") && ident(field.Value, "rawHiddenRows") {
				rawFields++
			}
		}
		valid = valid && top[assignment] && rawFields == 1
		return true
	})
	return valid && publications == 1 && fenceCount == 1 && transferCount == 1 && fence < publication && transfer < publication
}

func cudaRawHiddenPublicationOrderControls(t *testing.T) {
	t.Helper()
	reset := "if splitErr != nil { result = Qwen35SequencePrefillResult{}; return result, splitErr }\n"
	fence := "if err := finalFenceErr; err != nil { return result, err }\n"
	transfer := "if transfers.ActivationH2DBytes != 0 || transfers.ActivationD2HBytes != 0 { return result, transferError() }\n"
	publish := "result = Qwen35SequencePrefillResult{LastHidden: last, RawHiddenRows: rawHiddenRows}\n"
	for _, control := range []struct {
		name, source string
		want         bool
	}{
		{"empty-reset-before-checks", reset + fence + transfer + publish, true},
		{"no-reset", fence + transfer + publish, true},
		{"before-fence", reset + publish + fence + transfer, false},
		{"before-transfer", reset + fence + publish + transfer, false},
		{"missing-fence", reset + transfer + publish, false},
		{"missing-transfer", reset + fence + publish, false},
		{"missing-publication", reset + fence + transfer, false},
		{"duplicate-publication", reset + fence + transfer + publish + publish, false},
		{"nested-publication", reset + fence + transfer + "if choose {\n" + publish + "}\n", false},
		{"wrong-raw-value", reset + fence + transfer + strings.Replace(publish, "RawHiddenRows: rawHiddenRows", "RawHiddenRows: last", 1), false},
		{"nonreturning-fence", reset + strings.Replace(fence, "return result, err", "observe(err)", 1) + transfer + publish, false},
		{"successful-fence-return", reset + strings.Replace(fence, "return result, err", "return result, nil", 1) + transfer + publish, false},
		{"nonreturning-transfer", reset + fence + strings.Replace(transfer, "return result, transferError()", "observe(transfers)", 1) + publish, false},
		{"duplicate-fence", reset + fence + fence + transfer + publish, false},
		{"syntax-error", reset + fence + transfer + publish + "{", false},
	} {
		t.Run("publication-order/"+control.name, func(t *testing.T) {
			if got := cudaRawHiddenPublicationOrdered(control.source); got != control.want {
				t.Fatalf("publication order = %v, want %v", got, control.want)
			}
		})
	}
}

const cudaRawHiddenFixture = `package compute

import ("errors";"fmt";"os";"runtime";"sync/atomic";"strings";"testing";"time";"unsafe")
type Tensor struct { Shape []int; buf *cudaBuf }
type cudaBuf struct { ptr unsafe.Pointer; data []float32; invalid uint32; id string }
type cudaKV struct{}
type cudaBackend struct {
	transient []*cudaBuf; capturing bool; graphOnce *int; faultLatch *latch
	fail string; block,privateStop bool; allocs, copies, fences, poisons int
	fenceGen uint64; bridgeCalls,terminalCode,normRows int; originalRaw *cudaBuf
}
type latch struct{ observed int }
func(l *latch) ObserveError(error,string){l.observed++}
func(l *latch) Observe(int,string,int){l.observed++}
const DeviceFaultExecution=1
type CUDAOpError struct{Op string}
func(e *CUDAOpError)Error()string{return e.Op}
const cudaQwen35IQ2SequenceOp="iq2"
const cudaQ56KMMQOp="q56"
const Qwen35SequenceRawHiddenPath="qwen35-hybrid-sequence-raw-hidden-v1"
var cudaDev *cudaBackend
var activeBackend *cudaBackend
var qwen35SequenceIQ2Qualification, cudaQ56KMMQQualification bool
type testSplitStatus struct{code int}
func testSplitEnd(token uint64,status *testSplitStatus)int{
	c:=activeBackend
	if token==0{panic("retired zero token")}
	c.fences++
	if c.fail=="split-ok"{return 0}
	// Private native quarantine is deliberately NOT the generic stop latch.
	c.privateStop=true;status.code=61;return 61
}
func testRetirementFailed(code int){
	c:=activeBackend;c.bridgeCalls++
	if code!=0&&c.terminalCode==0{c.terminalCode=code;c.block=true}
}
func cudaSequenceSplitStatusError(status testSplitStatus)error{return fmt.Errorf("split code %d",status.code)}

type dtype int
const F32 dtype=0
func(dtype)Bytes()int{return 4}
type Qwen35SequencePrefillRequest struct{
	CaptureRawHidden,CapturePrefixReplay,NeedAllLogits,NeedLogits bool
	FeatureLayers []int; Hidden int; TokenIDs []int
	OutputNorm,Output Tensor; RMSNormEpsilon float32
}
type Qwen35SequenceTransferCounters struct{H2DBytes,D2HBytes,ActivationH2DBytes,ActivationD2HBytes uint64}
type Qwen35SequencePrefillResult struct{LastHidden,RawHiddenRows,Logits,LogitsRows Tensor;Tokens int;Transfers Qwen35SequenceTransferCounters}
func qwen35SequenceFailure(stage string,layer int,reason string)error{return fmt.Errorf("%s/%d: %s",stage,layer,reason)}
func(c *cudaBackend) qwen35SequencePoisonStates(Qwen35SequencePrefillRequest){c.poisons++}
func(c *cudaBackend) make(shape []int,id string)(Tensor,*cudaBuf){
	n:=1;for _,v:=range shape{n*=v};b:=&cudaBuf{data:make([]float32,n),id:id};b.ptr=unsafe.Pointer(&b.data[0]);c.transient=append(c.transient,b);c.allocs++
	if id=="raw"{c.originalRaw=b}
	return Tensor{append([]int(nil),shape...),b},b
}
func(c *cudaBackend) maybe(site string)error{
	if c.fail=="panic-"+site{panic("injected")}
	if c.fail=="goexit-"+site{runtime.Goexit()}
	if c.fail==site{return errors.New("injected "+site)}
	return nil
}
func(c *cudaBackend) qwen35SequenceAllocLocked(s []int,site string)(Tensor,*cudaBuf,error){
	if err:=c.maybe(site);err!=nil{return Tensor{},nil,err}
	t,b:=c.make(s,site);return t,b,nil
}
func(c *cudaBackend) qwen35SequenceRMSNormLocked(x,w Tensor,tokens int,eps float32,site string)(Tensor,*cudaBuf,error){
	if c.fail=="split-defer"{return Tensor{},nil,errors.New("primary norm allocation")}
	if err:=c.maybe(site);err!=nil{return Tensor{},nil,err}
	c.normRows+=tokens;t,b:=c.make(x.Shape,site);for i,v:=range x.buf.data{b.data[i]=v*3};return t,b,nil
}
func(c *cudaBackend)cf(t Tensor)*float32{if t.buf==nil{return nil};if t.buf.id=="raw"{if err:=c.maybe("raw-submit");err!=nil{panic(err)}};return (*float32)(t.buf.ptr)}
// Explicit same arithmetic model as the full-panel fixture, not CUDA parity.
func testRMSNorm(x,w,y *float32,rows,n int,eps float32){activeBackend.normRows+=rows;in,out:=unsafe.Slice(x,rows*n),unsafe.Slice(y,rows*n);for i,v:=range in{out[i]=v*3}}
func(c *cudaBackend) qwen35SequenceMatMulLocked(w,x Tensor,tokens int,site string)(Tensor,*cudaBuf,error){
	if err:=c.maybe(site);err!=nil{return Tensor{},nil,err}
	t,b:=c.make([]int{tokens,w.Shape[0]},site);for i:=range b.data{b.data[i]=float32(i+11)};return t,b,nil
}
var hostH2D,hostD2H uint64
func testH2D()uint64{return hostH2D}
func testD2H()uint64{return hostD2H}
func offsetF(p unsafe.Pointer,n int)unsafe.Pointer{return unsafe.Add(p,4*n)}
func testD2D(d,s unsafe.Pointer,n uintptr){copy(unsafe.Slice((*float32)(d),int(n)/4),unsafe.Slice((*float32)(s),int(n)/4))}
func(c *cudaBackend) qwen35SequenceSyncLocked()error{
	c.fences++
	if c.fail=="transfer-h"{hostH2D++};if c.fail=="transfer-d"{hostD2H++}
	if err:=c.maybe("fence");err!=nil{c.block=true;return err}
	return nil
}
func(c *cudaBackend) releaseTransientBuffers(buffers []*cudaBuf){
	if c.block{return}
	for _,b:=range buffers{if b!=nil{b.ptr=nil}}
	kept:=c.transient[:0];for _,b:=range c.transient{if b.ptr!=nil{kept=append(kept,b)}};c.transient=kept
}
type rawHiddenUnselectedPaged struct { executing,drained bool }
func(*rawHiddenUnselectedPaged)abortLocked(){panic("raw-hidden fixture unexpectedly selected paged")}
func(*rawHiddenUnselectedPaged)completeLocked()error{panic("raw-hidden fixture unexpectedly selected paged")}
`

// Paged is deliberately unselected in this contiguous raw-hidden witness. The
// separate paged source witness executes the real selected ownership adapter.
const cudaRawHiddenTailSetup = `
var pagedControlH2D uint64
tokens:=len(req.TokenIDs)
x,xBuf:=c.make([]int{tokens,req.Hidden},"raw")
for i:=range xBuf.data{xBuf.data[i]=float32(i+1)}
statesMayHaveAdvanced=true
h2dStart,d2hStart:=hostH2D,hostD2H
h2dAfterGather,d2hAfterGather:=hostH2D,hostD2H
var splitToken uint64
if c.fail=="split-head"||c.fail=="split-final"||c.fail=="split-defer"||c.fail=="split-ok"{splitToken=1}
`

const cudaRawHiddenControls = `
func fixture()(*cudaBackend,Qwen35SequencePrefillRequest){
	c:=&cudaBackend{faultLatch:&latch{}};cudaDev=c;activeBackend=c;hostH2D=0;hostD2H=0;qwen35SequenceIQ2Qualification=false;cudaQ56KMMQQualification=false
	return c,Qwen35SequencePrefillRequest{Hidden:4,TokenIDs:[]int{0,1,2},CaptureRawHidden:true,NeedAllLogits:true,NeedLogits:true,Output:Tensor{Shape:[]int{5,4}},RMSNormEpsilon:1e-6}
}
func TestRawLifetime(t *testing.T){
	t.Setenv("FAK_CUDA_QWEN35_MTP","1")
	for _,capture:=range []bool{false,true}{for _,all:=range []bool{false,true}{for _,last:=range []bool{false,true}{
		c,req:=fixture();req.CaptureRawHidden=capture;req.NeedAllLogits=all;req.NeedLogits=last
		older,_:=c.make([]int{1},"prior-owner")
		got,err:=c.sequenceTail(req)
		if err!=nil||got.Tokens!=3||got.LastHidden.buf==nil||got.LastHidden.buf.ptr==nil{t.Fatalf("success: %+v %v",got,err)}
		if c.fences!=1||c.poisons!=0||older.buf.ptr==nil{t.Fatal("fence/old owner changed")}
		want:=2; if all{want++};if last{want++};if capture{want++}
		if len(c.transient)!=want{t.Fatalf("survivors=%d want%d",len(c.transient),want)}
		wantAlloc:=3;if all{wantAlloc+=2};if last{wantAlloc++};if c.allocs!=wantAlloc{t.Fatalf("output-tail allocations=%d want%d",c.allocs,wantAlloc)}
		wantNormRows:=1;if all{wantNormRows=len(req.TokenIDs)};if c.normRows!=wantNormRows{t.Fatalf("normalized rows=%d want%d",c.normRows,wantNormRows)}
		for i,v:=range got.LastHidden.buf.data{if v!=float32((len(req.TokenIDs)-1)*req.Hidden+i+1)*3{t.Fatal("normalized last row changed")}}
		if capture{
			raw:=got.RawHiddenRows
			if raw.buf==nil||raw.buf.id!="raw"||raw.buf.ptr==nil||raw.buf.invalid!=0||len(raw.Shape)!=2||raw.Shape[0]!=3||raw.Shape[1]!=4{t.Fatal("not original live raw panel")}
			for i,v:=range raw.buf.data{if v!=float32(i+1){t.Fatal("raw was normalized")}}
			if raw.buf==got.LastHidden.buf||got.LastHidden.buf.data[0]!=27{t.Fatal("normalized row identity/value changed")}
		}else if got.RawHiddenRows.buf!=nil{t.Fatal("unrequested raw published")}
		c.releaseTransientBuffers(append([]*cudaBuf(nil),c.transient...))
		if got.LastHidden.buf.ptr!=nil||(capture&&got.RawHiddenRows.buf.ptr!=nil){t.Fatal("request lifetime not retired")}
	}}}
	for _,kind:=range []string{"output-norm","last-hidden","output-all-head","output-head","fence","split-head","split-final","split-defer","transfer-h","transfer-d"}{
		t.Run(kind,func(t *testing.T){
			c,req:=fixture();c.fail=kind;if kind=="split-final"{req.NeedAllLogits=false}
			got,err:=c.sequenceTail(req)
			if err==nil||got.RawHiddenRows.buf!=nil||got.LastHidden.buf!=nil||c.poisons==0{t.Fatalf("failure published: %+v %v",got,err)}
			for _,b:=range c.transient{if b.id=="raw"&&b.invalid==0{t.Fatal("retained failed raw not invalid")}}
			if c.originalRaw==nil||c.originalRaw.invalid==0{t.Fatal("exact failed raw was not invalidated")}
			if kind=="split-head"||kind=="split-final"||kind=="split-defer"{
				if qwen35SequenceIQ2Qualification||cudaQ56KMMQQualification{t.Fatal("flags-off control changed")}
				if c.bridgeCalls!=1||c.terminalCode!=61||!c.privateStop||!c.block{t.Fatal("private split failure did not reach actual shared bridge")}
				if kind=="split-defer"&&!strings.Contains(err.Error(),"primary norm allocation"){t.Fatal("deferred retirement lost primary cause")}
			}
			if c.block{found:=false;for _,b:=range c.transient{if b==c.originalRaw&&b.ptr!=nil{found=true}};if !found{t.Fatal("uncertain completion lost raw owner")}}
		})
	}

	// The last-row branch owns no normalized panel, including on failure.
	for _,kind:=range []string{"last-hidden","output-head","fence","transfer-h","transfer-d"}{
		t.Run("last-row-"+kind,func(t *testing.T){
			c,req:=fixture();req.NeedAllLogits=false;c.fail=kind
			got,err:=c.sequenceTail(req)
			if err==nil||got.RawHiddenRows.buf!=nil||got.LastHidden.buf!=nil||c.poisons==0{t.Fatalf("last-row failure published: %+v %v",got,err)}
			if c.originalRaw==nil||c.originalRaw.invalid==0{t.Fatal("last-row failure did not invalidate original residual")}
		})
	}
	for _,all:=range []bool{false,true}{
		c,req:=fixture();c.fail="split-ok";req.NeedAllLogits=all
		got,err:=c.sequenceTail(req)
		wantFences:=1;if all{wantFences=2}
		if err!=nil||got.RawHiddenRows.buf!=c.originalRaw||c.originalRaw.ptr==nil||c.fences!=wantFences||c.fenceGen!=1||c.bridgeCalls!=0||c.block||c.privateStop{t.Fatal("successful actual split closure changed")}
	}
	for _,kind:=range []string{"panic-output-norm","goexit-output-norm","panic-output-head","goexit-output-head","panic-fence","goexit-fence","panic-raw-submit","goexit-raw-submit"}{
		t.Run(kind,func(t *testing.T){
			c,req:=fixture();c.fail=kind;c.block=true;if strings.Contains(kind,"raw-submit"){req.NeedAllLogits=false}
			done:=make(chan struct{});var recovered any;returned:=false
			go func(){defer close(done);defer func(){recovered=recover()}();_,_=c.sequenceTail(req);returned=true}()
			select{case<-done:case<-time.After(5*time.Second):t.Fatal("unwind hung")}
			if returned||c.poisons==0{t.Fatal("abnormal success")}
			if kind[:5]=="panic"&&recovered!="injected"{t.Fatalf("panic changed: %v",recovered)}
			if kind[:6]=="goexit"&&recovered!=nil{t.Fatalf("Goexit changed: %v",recovered)}
			found:=false;for _,b:=range c.transient{if b.id=="raw"{found=true;if b.ptr==nil||b.invalid==0{t.Fatal("unwind root invalidity")}}};if !found{t.Fatal("lost raw")}
		})
	}
	for _,kind:=range []string{"off","other","nil","capture","graph","prefix","features"}{
		t.Run("refuse-"+kind,func(t *testing.T){
			c,req:=fixture()
			switch kind{case "off":t.Setenv("FAK_CUDA_QWEN35_MTP","0");case "other":cudaDev=&cudaBackend{};case "nil":c=nil;case "capture":c.capturing=true;case "graph":c.graphOnce=new(int);case "prefix":req.CapturePrefixReplay=true;case "features":req.FeatureLayers=[]int{1}}
			if _,err:=c.preflight(req);err==nil{t.Fatal("unsupported raw admitted")}
			if c!=nil&&(c.allocs!=0||len(c.transient)!=0||c.fences!=0){t.Fatal("refusal had effects")}
		})
	}
	c,req:=fixture();if _,err:=c.preflight(req);err!=nil{t.Fatal(err)}
	req.CaptureRawHidden=false;req.CapturePrefixReplay=true;req.FeatureLayers=[]int{1}
	if _,err:=c.preflight(req);err!=nil{t.Fatal("changed unrequested path")}
}
`
