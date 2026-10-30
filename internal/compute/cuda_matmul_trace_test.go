package compute

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:justify why=contract when=changed:internal/compute/cuda*
// fak-test:runtime fast est=20ms lane=default
func TestCUDAMatMulTracePayloadContract(t *testing.T) {
	// This local interface keeps the parent runnable: missing wiring is a
	// behavioral failure, not an undefined-symbol compiler failure. The candidate
	// calls the same production method the two CUDA dispatch sites consume.
	type eventBuilder interface {
		cudaMatMulTraceEvent(string, Tensor, Tensor, int, time.Time, float64, bool) (computetrace.Event, bool)
	}
	started := time.Unix(123, 0)
	x := Tensor{Dtype: F32, Shape: []int{256}}
	y := Tensor{Dtype: F32, Shape: []int{2}}
	recorder, stop := computetrace.Enable(20, "cpu-payload-micro", "simulated-not-gpu")
	t.Cleanup(stop)
	var last computetrace.Event
	for _, tc := range []struct {
		dtype Dtype
		bytes int64
	}{
		{F32, 2048}, {F16, 1024}, {Q8_0, 576}, {Q2_0, 192},
		{Q2_K, 168}, {Q3_K, 220}, {Q4_K, 288}, {Q5_K, 352}, {Q6_K, 420},
		{IQ2_XXS, 132}, {IQ2_XS, 148}, {IQ1_M, 112}, {IQ4_NL, 288}, {NVFP4, 288},
	} {
		w := Tensor{Dtype: tc.dtype, Shape: []int{2, 256}}
		if tc.dtype == Q8_0 || tc.dtype == Q2_0 {
			w.Quant = &QuantSpec{Block: 32}
		}
		if IsRawIQ(tc.dtype) {
			w = NewRawIQ(Default(), tc.dtype, w.Shape, make([]byte, tc.bytes))
		}
		builder, wired := any(w).(eventBuilder)
		if !wired {
			t.Fatal("CUDA MatMul operand-payload event constructor is not wired")
		}
		event, ok := builder.cudaMatMulTraceEvent("cuda", x, y, 1, started, 2.5, false)
		if !ok || event.BytesRead != tc.bytes+1024 || event.Bytes != event.BytesRead || event.BytesWritten != 8 ||
			event.InputDType != "f32" || event.WeightDType != tc.dtype.String() || event.OutputDType != "f32" ||
			event.DurationNS != 2500000 || event.DeviceDurationNS != event.DurationNS || event.TimerDomain != "cuda_event" ||
			event.EstimatedFLOPs != 1024 || event.Status != "ok" || !event.StartedAt.Equal(started) {
			t.Fatalf("%s modeled operand payload/timing contract: ok=%t event=%+v", tc.dtype, ok, event)
		}
		computetrace.Record(event)
		last = event
	}
	w := NewRawIQ(Default(), IQ2_XXS, []int{2, 256}, make([]byte, 132))
	// The dispatched P=3 consumes only three rows, not this view's capacity.
	batch, ok := any(w).(eventBuilder).cudaMatMulTraceEvent("cuda",
		Tensor{Dtype: F32, Shape: []int{8, 256}}, Tensor{Dtype: F32, Shape: []int{8, 2}}, 3, started, 4, true)
	if !ok || batch.BytesRead != 132+3*256*4 || batch.BytesWritten != 3*2*4 ||
		batch.EstimatedFLOPs != 3*1024 || len(batch.Shapes) != 2 || len(batch.Shapes[1]) != 2 ||
		batch.Shapes[1][0] != 3 || batch.Kernel != "iq2_xxs_batched_matmul" {
		t.Fatalf("batch capacity was confused with consumed payload: ok=%t event=%+v", ok, batch)
	}
	computetrace.Record(batch)
	for _, bad := range []Tensor{
		{Dtype: IQ3_S, Shape: []int{2, 256}}, // Not a CUDA MatMul dtype.
		{Dtype: IQ2_XXS, Shape: []int{2, 255}},
		{Dtype: F32, Shape: []int{int(^uint(0) >> 1), 256}},
	} {
		if _, ok := any(bad).(eventBuilder).cudaMatMulTraceEvent("cuda", x, y, 1, started, 1, false); ok {
			t.Fatalf("unsupported/malformed/overflow payload accepted: %+v", bad)
		}
	}
	artifact := recorder.Artifact()
	if len(artifact.Events) != 15 || artifact.Dropped != 0 {
		t.Fatalf("real trace sink lost modeled events: %+v", artifact)
	}
	wire, err := json.Marshal(artifact)
	if err != nil || !strings.Contains(string(wire), `"bytes_read":1156`) || !strings.Contains(string(wire), `"weight_dtype":"iq1_m"`) {
		t.Fatalf("typed payload missing from actual trace wire: err=%v wire=%s", err, wire)
	}
	stop()
	computetrace.Record(last)
	if len(recorder.Artifact().Events) != 15 {
		t.Fatal("disabled recorder retained an event")
	}
	assertCUDAMatMulTraceWiring(t)
}

func assertCUDAMatMulTraceWiring(t *testing.T) {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), "cuda.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"MatMul", "BatchedMatMul"} {
		var fn *ast.FuncDecl
		for _, decl := range file.Decls {
			if f, ok := decl.(*ast.FuncDecl); ok && f.Name.Name == name && f.Recv != nil {
				fn = f
			}
		}
		if fn == nil {
			t.Fatalf("missing actual CUDA %s", name)
		}
		counts := map[string]int{}
		var stack []ast.Node
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if n == nil {
				stack = stack[:len(stack)-1]
				return true
			}
			stack = append(stack, n)
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			receiver, ok := selector.X.(*ast.Ident)
			if !ok {
				return true
			}
			key := receiver.Name + "." + selector.Sel.Name
			if key == "computetrace.Enabled" {
				for _, parent := range stack {
					if assignment, ok := parent.(*ast.AssignStmt); ok && len(assignment.Lhs) == 1 && len(assignment.Rhs) == 1 && assignment.Rhs[0] == call {
						if variable, ok := assignment.Lhs[0].(*ast.Ident); ok && variable.Name == "trace" {
							counts[key]++
						}
					}
				}
				return true
			}
			if key == "w.cudaMatMulTraceEvent" {
				wantInput, wantBatch := "x", "false"
				if name == "BatchedMatMul" {
					wantInput, wantBatch = "X", "true"
				}
				if len(call.Args) != 7 {
					t.Fatalf("%s builder lost actual operand/row arguments", name)
				}
				for index, want := range map[int]string{2: "y", 4: "started", 5: "ms"} {
					value, ok := call.Args[index].(*ast.Ident)
					if !ok || value.Name != want {
						t.Fatalf("%s builder lost actual %s", name, want)
					}
				}
				input, inputOK := call.Args[1].(*ast.Ident)
				batch, batchOK := call.Args[6].(*ast.Ident)
				if !inputOK || !batchOK || input.Name != wantInput || batch.Name != wantBatch {
					t.Fatalf("%s builder does not consume its actual input/mode", name)
				}
				if name == "BatchedMatMul" {
					rows, ok := call.Args[3].(*ast.Ident)
					if !ok || rows.Name != "P" {
						t.Fatal("batch trace lost dispatched P")
					}
				} else if rows, ok := call.Args[3].(*ast.BasicLit); !ok || rows.Value != "1" {
					t.Fatal("single trace lost its one dispatched row")
				}
			}
			if key != "C.fcuda_event_elapsed_ms_start" && key != "C.fcuda_event_elapsed_ms_end" && key != "w.cudaMatMulTraceEvent" && key != "computetrace.Record" {
				return true
			}
			counts[key]++
			guarded, completed, fed := false, false, false
			for _, parent := range stack {
				branch, ok := parent.(*ast.IfStmt)
				if !ok || call.Pos() < branch.Body.Pos() || call.End() > branch.Body.End() {
					continue
				}
				condition, ok := branch.Cond.(*ast.Ident)
				if !ok {
					continue
				}
				guarded = guarded || condition.Name == "trace"
				completed = completed || condition.Name == "completed"
				assignment, ok := branch.Init.(*ast.AssignStmt)
				if condition.Name != "ok" || !ok || len(assignment.Rhs) != 1 || len(assignment.Lhs) != 2 {
					continue
				}
				builder, ok := assignment.Rhs[0].(*ast.CallExpr)
				if !ok {
					continue
				}
				method, ok := builder.Fun.(*ast.SelectorExpr)
				if !ok || method.Sel.Name != "cudaMatMulTraceEvent" || len(builder.Args) != 7 {
					continue
				}
				produced, pOK := assignment.Lhs[0].(*ast.Ident)
				if len(call.Args) == 1 {
					consumed, cOK := call.Args[0].(*ast.Ident)
					fed = pOK && cOK && produced.Name == consumed.Name
				}
			}
			if !guarded || (key == "w.cudaMatMulTraceEvent" && !completed) || (key == "computetrace.Record" && (!fed || !completed)) {
				t.Fatalf("%s %s lacks trace guard or production builder-to-sink flow", name, key)
			}
			return true
		})
		for _, key := range []string{"computetrace.Enabled", "C.fcuda_event_elapsed_ms_start", "C.fcuda_event_elapsed_ms_end", "w.cudaMatMulTraceEvent", "computetrace.Record"} {
			if counts[key] != 1 {
				t.Fatalf("%s %s count=%d, want one real call", name, key, counts[key])
			}
		}
	}
}
