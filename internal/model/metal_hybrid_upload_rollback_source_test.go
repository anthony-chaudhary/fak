package model

import (
	"context"
	"crypto/sha256"
	"fmt"
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

// fak-test:runtime host est=10s lane=standard
// TestMetalHybridUploadRollbackActualSource executes the complete production
// construction body and its existing unwind helper. Only Q8 metadata and the
// native upload/free boundary are controlled fixtures; this is not a Metal,
// complete Model, or serving-survival qualification.
func TestMetalHybridUploadRollbackActualSource(t *testing.T) {
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Fatal("Go compiler required for actual-source ownership regression:", err)
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate actual source")
	}
	root := filepath.Dir(here)
	method := metalRollbackSourceDecl(t, filepath.Join(root, "metal_prefill_hybrid.go"), "metalWeightsQwen35Hybrid")
	helper := metalRollbackSourceDecl(t, filepath.Join(root, "cleanup_siblings.go"), "suppressBackendCleanupPanic")
	dir := t.TempDir()
	write := func(name, data string) {
		t.Helper()
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module rollbackfixture\n\ngo 1.26.0\n")
	write("metalgemm/native.go", metalRollbackNativeFixture)
	write("owner.go", "package model\nimport (\"fmt\";\"sync\";\"rollbackfixture/metalgemm\")\n"+metalRollbackModelFixture+"\n"+method+"\n"+helper)
	write("owner_test.go", metalRollbackCases)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goTool, "test", "-vet=off", "-count=1", "-timeout=20s", "-v", ".")
	cmd.WaitDelay = 2 * time.Second
	// Resource-qualified runs additionally own the complete process group; a
	// command timeout is failure, never proof that native cleanup completed.
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off")
	b, err := cmd.CombinedOutput()
	t.Logf("actual construction SHA256=%x; unwind helper SHA256=%x\n%s", sha256.Sum256([]byte(method)), sha256.Sum256([]byte(helper)), b)
	if err != nil {
		t.Fatalf("actual-source rollback regression: %v", err)
	}
	for _, name := range []string{"TestLateNilRollsBack", "TestUploadPanicRollsBack", "TestSuccessAndCacheHit", "TestPrimaryGoexitDrains", "TestCleanupPanicPreservesPrimary", "TestCleanupGoexitDrains", "TestInvalidLayerCountPrecedesUpload", "TestMoESkipsDenseMLPUpload"} {
		if !strings.Contains(string(b), "--- PASS: "+name) {
			t.Fatalf("required ownership control did not pass: %s", name)
		}
	}
}

func metalRollbackSourceDecl(t *testing.T, path, name string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, path, b, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == name {
			return string(b[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
		}
	}
	t.Fatal(fmt.Sprintf("actual declaration %s missing from %s", name, path))
	return ""
}

const metalRollbackModelFixture = `
type Config struct { NumLayers int; Linear bool; MoE bool }
func (c Config) isLinearAttnLayer(int) bool { return c.Linear }
func (c Config) IsMoE() bool { return c.MoE }
type Model struct { Cfg Config }
type q8Tensor struct { out,in int }
func (*Model) q8(string) *q8Tensor { return &q8Tensor{1,1} }
func dequantQ8(*q8Tensor) []float32 { return []float32{1} }
func layerName(l int,s string) string { return fmt.Sprintf("model.layers.%d.%s",l,s) }
var metalHybridMu sync.Mutex
var metalHybridWt=map[*Model]map[string]*metalgemm.Weight{}
`

const metalRollbackNativeFixture = `
package metalgemm
import ("runtime";"strings")
type Weight struct { ID int }
var Calls int
var FailAt, PanicAt, ExitAt int
var FreePanicID, FreeExitID int
var Acquired []*Weight
var Frees []int
var Retired map[int]bool
var Owner string
var WrongOwner bool
func Goroutine() string { var b [96]byte;n:=runtime.Stack(b[:],false);return strings.Fields(string(b[:n]))[1] }
func checkOwner(){if Goroutine()!=Owner {WrongOwner=true}}
func Reset(){Calls=0;FailAt=0;PanicAt=0;ExitAt=0;FreePanicID=-1;FreeExitID=-1;Acquired=nil;Frees=nil;Retired=map[int]bool{};Owner=Goroutine();WrongOwner=false}
func Upload([]float32,int,int)*Weight {
 checkOwner();Calls++
 if Calls==PanicAt {panic("upload-primary")};if Calls==ExitAt {runtime.Goexit()};if Calls==FailAt {return nil}
 w:=&Weight{ID:len(Acquired)};Acquired=append(Acquired,w);return w
}
func(w *Weight)Free(){checkOwner();Frees=append(Frees,w.ID);if w.ID==FreePanicID {panic("cleanup-secondary")};if w.ID==FreeExitID {runtime.Goexit()};Retired[w.ID]=true}
`

const metalRollbackCases = `
package model
import ("reflect";"runtime";"strings";"testing";"time";"rollbackfixture/metalgemm")
// Abort this fixture process on a stuck owner; do not continue tests against
// a live failed goroutine or report it as cleaned. Outer process supervision
// must still verify all descendants are reaped before another phase.
func awaitExit(done <-chan struct{}){select{case <-done:case <-time.After(5*time.Second):panic("rollback owner did not terminate within 5s")}}
func reset(){metalgemm.Reset();metalHybridWt=map[*Model]map[string]*metalgemm.Weight{}}
func caught(f func())(p any){defer func(){p=recover()}();f();return}
func checkRollback(t *testing.T,m *Model,n int){t.Helper();if _,ok:=metalHybridWt[m];ok {t.Fatal("failed attempt published cache")};if len(metalgemm.Acquired)!=n {t.Fatalf("acquired %d, want %d",len(metalgemm.Acquired),n)};want:=make([]int,n);for i:=range want {want[i]=n-1-i};if !reflect.DeepEqual(metalgemm.Frees,want){t.Fatalf("retirement attempts %v want %v",metalgemm.Frees,want)};if metalgemm.WrongOwner {t.Fatal("native operation left owner goroutine")}}
func TestInvalidLayerCountPrecedesUpload(t *testing.T){reset();m:=&Model{Cfg:Config{NumLayers:-1}};p:=caught(func(){m.metalWeightsQwen35Hybrid()});if p!="model: invalid metal hybrid layer count"||metalgemm.Calls!=0{t.Fatalf("invalid geometry reached upload: panic=%v calls=%d",p,metalgemm.Calls)};if _,ok:=metalHybridWt[m];ok{t.Fatal("invalid geometry published")}}
func TestLateNilRollsBack(t *testing.T){
 for _,linear:=range []bool{false,true}{reset();runtime.LockOSThread();m:=&Model{Cfg:Config{NumLayers:2,Linear:linear}};prior:=&Model{};borrowed:=&metalgemm.Weight{ID:99};metalHybridWt[prior]=map[string]*metalgemm.Weight{"prior":borrowed};metalgemm.FailAt=6;p:=caught(func(){m.metalWeightsQwen35Hybrid()});runtime.UnlockOSThread();if s,ok:=p.(string);!ok||!strings.HasPrefix(s,"model: metal hybrid weight upload failed for "){t.Fatalf("primary panic %v",p)};checkRollback(t,m,5);for i:=0;i<5;i++ {if !metalgemm.Retired[i]{t.Fatalf("handle%d not retired",i)}};if metalHybridWt[prior]["prior"]!=borrowed {t.Fatal("prior cache changed")}}
}
func TestMoESkipsDenseMLPUpload(t *testing.T){reset();m:=&Model{Cfg:Config{NumLayers:2,MoE:true}};w:=m.metalWeightsQwen35Hybrid();if metalgemm.Calls!=8||len(w)!=8 {t.Fatalf("MoE uploads=%d published=%d, want 8 attention weights",metalgemm.Calls,len(w))};for name:=range w {if strings.Contains(name,".mlp."){t.Fatalf("MoE uploaded dense FFN weight %s",name)}}}
func TestUploadPanicRollsBack(t *testing.T){reset();m:=&Model{Cfg:Config{NumLayers:1}};metalgemm.PanicAt=4;p:=caught(func(){m.metalWeightsQwen35Hybrid()});if p!="upload-primary"{t.Fatalf("primary %v",p)};checkRollback(t,m,3)}
func TestSuccessAndCacheHit(t *testing.T){for _,linear:=range []bool{false,true}{reset();m:=&Model{Cfg:Config{NumLayers:2,Linear:linear}};a:=m.metalWeightsQwen35Hybrid();want:=14;if linear {want=16};if len(a)!=want||len(metalgemm.Acquired)!=want||len(metalgemm.Frees)!=0{t.Fatalf("bad success len=%d acquired=%d frees=%v",len(a),len(metalgemm.Acquired),metalgemm.Frees)};b:=m.metalWeightsQwen35Hybrid();if reflect.ValueOf(a).Pointer()!=reflect.ValueOf(b).Pointer()||metalgemm.Calls!=want||len(metalgemm.Frees)!=0{t.Fatal("cache hit acquired or retired handles")};if metalgemm.WrongOwner{t.Fatal("wrong owner")}}}
func TestPrimaryGoexitDrains(t *testing.T){reset();m:=&Model{Cfg:Config{NumLayers:1}};metalgemm.ExitAt=4;metalgemm.FreePanicID=1;done:=make(chan struct{});returned:=false;go func(){defer close(done);metalgemm.Owner=metalgemm.Goroutine();m.metalWeightsQwen35Hybrid();returned=true}();awaitExit(done);if returned{t.Fatal("Goexit became success")};checkRollback(t,m,3);if metalgemm.Retired[1]{t.Fatal("failed retirement claimed success")}}
func TestCleanupPanicPreservesPrimary(t *testing.T){reset();m:=&Model{Cfg:Config{NumLayers:1}};metalgemm.PanicAt=4;metalgemm.FreePanicID=1;p:=caught(func(){m.metalWeightsQwen35Hybrid()});if p!="upload-primary"{t.Fatalf("cleanup replaced primary %v",p)};checkRollback(t,m,3);if metalgemm.Retired[1]||!metalgemm.Retired[0]||!metalgemm.Retired[2]{t.Fatal("retirement completion misreported")}}
func TestCleanupGoexitDrains(t *testing.T){reset();m:=&Model{Cfg:Config{NumLayers:1}};metalgemm.FailAt=4;metalgemm.FreeExitID=1;done:=make(chan struct{});returned:=false;go func(){defer close(done);metalgemm.Owner=metalgemm.Goroutine();m.metalWeightsQwen35Hybrid();returned=true}();awaitExit(done);if returned{t.Fatal("cleanup Goexit became success")};checkRollback(t,m,3);if metalgemm.Retired[1]||!metalgemm.Retired[0]||!metalgemm.Retired[2]{t.Fatal("cleanup Goexit skipped sibling or claimed success")}}
`
