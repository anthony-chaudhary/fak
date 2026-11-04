package compute

import (
	"bytes"
	"context"
	"go/ast"
	"go/format"
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

// Compile the actual production wrappers and cleanup defers with injected
// operation/runtime results. This verifies host ordering, not CUDA execution.
// fak-test:runtime medium est=2s lane=default
func TestCUDAPrefillLockedCleanupOrder(t *testing.T) {
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "cuda.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	methods := map[string]*ast.FuncDecl{}
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok {
			methods[fn.Name.Name] = fn
		}
	}
	for _, name := range []string{"Read", "BatchedMatMul", "readLocked", "batchedMatMulLocked"} {
		if methods[name] == nil {
			t.Fatalf("missing production method %s", name)
		}
	}
	render := func(n any) string {
		var b bytes.Buffer
		if err := format.Node(&b, fs, n); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	read := methods["readLocked"]
	var readDefer string
	for _, s := range read.Body.List {
		if d, ok := s.(*ast.DeferStmt); ok {
			readDefer = render(d)
		}
	}
	if !strings.Contains(readDefer, "cudaV41CompressorReadEndLocked") {
		t.Fatal("read cleanup absent")
	}
	batch := methods["batchedMatMulLocked"]
	var batchDefer string
	for _, s := range batch.Body.List {
		if d, ok := s.(*ast.DeferStmt); ok {
			batchDefer = render(d)
		}
	}
	if !strings.Contains(batchDefer, "fcuda_event_elapsed_ms_end") {
		t.Fatal("trace cleanup absent")
	}
	fixture := `package cleanup
import("sync";"time";"testing";"reflect")
type Tensor struct{}
type cudaBackend struct{}
func(*cudaBackend)Name()string{return "cuda"}
func(Tensor)cudaMatMulTraceEvent(string,Tensor,Tensor,int,time.Time,float64,bool)(int,bool){events=append(events,"event");return 1,true}
type trackedMutex struct{mu sync.Mutex;held bool}
func(m *trackedMutex)Lock(){m.mu.Lock();m.held=true;events=append(events,"lock")}
func(m *trackedMutex)Unlock(){if !m.held{panic("unlock without lock")};events=append(events,"unlock");m.held=false;m.mu.Unlock()}
var cudaMu trackedMutex
var events []string
var operationPanic,cleanupPanic bool
func operation(){if !cudaMu.held{panic("operation unlocked")};events=append(events,"operation");if operationPanic{panic("operation failure")}}
var C=struct{fcuda_event_elapsed_ms_end func()float64}{func()float64{if !cudaMu.held{panic("trace outside lock")};events=append(events,"trace");if cleanupPanic{panic("trace failure")};return 1}}
var computetrace=struct{Record func(int)}{func(int){events=append(events,"record")}}
func cudaV41CompressorReadEndLocked(c *cudaBackend,pending,readOK bool){if !cudaMu.held{panic("read cleanup outside lock")};events=append(events,"read-cleanup");if readOK==operationPanic{panic("bad read completion flag")};if cleanupPanic{panic("read cleanup failure")}}
` + render(methods["Read"]) + "\n" + render(methods["BatchedMatMul"]) + `
func(c *cudaBackend)readLocked(t Tensor)[]float32{
 pending:=true;readOK:=false
` + readDefer + `
 operation();readOK=true;return []float32{1}
}
func(c *cudaBackend)batchedMatMulLocked(w,X Tensor,P int)Tensor{
 trace:=true;var started time.Time;var y Tensor;completed:=false
` + batchDefer + `
 operation();completed=true;return y
}
func TestOrders(t *testing.T){
 for _,read:=range []bool{false,true}{for _,opPanic:=range []bool{false,true}{for _,cleanPanic:=range []bool{false,true}{
 events=nil;operationPanic=opPanic;cleanupPanic=cleanPanic;c:=&cudaBackend{};var caught any
 func(){defer func(){caught=recover()}();if read{c.Read(Tensor{})}else{c.BatchedMatMul(Tensor{},Tensor{},1)}}()
 if cudaMu.held{t.Fatal("mutex stranded")}
 want:=[]string{"lock","operation"};if read{want=append(want,"read-cleanup")}else{want=append(want,"trace");if !opPanic&&!cleanPanic{want=append(want,"event","record")}};want=append(want,"unlock")
 if !reflect.DeepEqual(events,want){t.Fatalf("order got %v want %v",events,want)}
 var wantPanic any;if opPanic{wantPanic="operation failure"};if cleanPanic{if read{wantPanic="read cleanup failure"}else{wantPanic="trace failure"}}
 if caught!=wantPanic{t.Fatalf("panic got %v want %v",caught,wantPanic)}
 // Prove the same public wrapper is usable after recovery.
 operationPanic=false;cleanupPanic=false;if read{c.Read(Tensor{})}else{c.BatchedMatMul(Tensor{},Tensor{},1)}
 }}}
}
`
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "cleanup_test.go"), []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module cleanup\ngo 1.26\n"), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	goBin := filepath.Join(runtime.GOROOT(), "bin", "go")
	cmd := exec.CommandContext(ctx, goBin, "test", "-timeout=10s", "-count=1", "-v", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("executed source cleanup: %v\n%s", err, out)
	} else {
		t.Log(string(out))
	}
}
