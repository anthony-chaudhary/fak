package compute

import (
	"slices"
	"strings"
	"testing"
	"time"
)

func plannedQ8Fixture() (*cpuBackend, Tensor, Tensor) {
	be := &cpuBackend{}
	weights := make([]float32, 64)
	for i := range weights {
		weights[i] = float32((i%11)-5) / 7
	}
	input := make([]float32, 32)
	for i := range input {
		input[i] = float32((i%7)-3) / 5
	}
	return be, QuantizeQ8(be, []int{2, 32}, weights, 32), NewF32(be, []int{32}, input)
}

func TestMatMulPlanPrepareOnceRunTwiceMatchesDirect(t *testing.T) {
	be, w, x := plannedQ8Fixture()
	planner, ok := Backend(be).(Planner)
	if !ok {
		t.Fatal("cpu-ref does not expose optional Planner capability")
	}
	plan, err := planner.PrepareMatMul(w, x)
	if err != nil {
		t.Fatal(err)
	}
	desc := plan.Descriptor()
	if desc.Operation != "matmul" || desc.Backend != "cpu-ref" || desc.Device != "host" {
		t.Fatalf("unexpected descriptor: %+v", desc)
	}
	wantWorkspace := x.Numel() + 4*(x.Numel()/w.Quant.Block)
	if desc.WorkspaceBytes != wantWorkspace {
		t.Fatalf("workspace=%d want exactly %d", desc.WorkspaceBytes, wantWorkspace)
	}
	workspace := make([]byte, desc.WorkspaceBytes)
	want := be.Read(be.MatMul(w, x))

	got1, run1, err := plan.Run(be, w, x, workspace)
	if err != nil {
		t.Fatal(err)
	}
	got2, run2, err := plan.Run(be, w, x, workspace)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(be.Read(got1), want) || !slices.Equal(be.Read(got2), want) {
		t.Fatalf("planned output mismatch: direct=%v run1=%v run2=%v", want, be.Read(got1), be.Read(got2))
	}
	for i, run := range []PlanRunDiagnostic{run1, run2} {
		if run.Operation != "matmul" || run.Backend != "cpu-ref" || run.Device != "host" || run.WorkspaceBytes != wantWorkspace || run.Duration < 0 {
			t.Fatalf("run %d diagnostic: %+v", i+1, run)
		}
	}

	// The descriptor is a copy: mutating diagnostics cannot stale the prepared key.
	desc.WeightShape[0] = 99
	if plan.Descriptor().WeightShape[0] != 2 {
		t.Fatal("plan descriptor is mutable through returned shape")
	}
}

func TestMatMulPlanWorkspaceAndAllocationBehavior(t *testing.T) {
	be, w, x := plannedQ8Fixture()
	plan, err := be.PrepareMatMul(w, x)
	if err != nil {
		t.Fatal(err)
	}
	n := plan.Descriptor().WorkspaceBytes
	for _, size := range []int{n - 1, n + 1} {
		_, _, err := plan.Run(be, w, x, make([]byte, size))
		if err == nil || !strings.Contains(err.Error(), "need exactly") || !strings.Contains(err.Error(), "Descriptor().WorkspaceBytes") {
			t.Fatalf("workspace size %d error=%v", size, err)
		}
	}
	workspace := make([]byte, n)
	plannedAllocs := testing.AllocsPerRun(20, func() {
		out, _, err := plan.Run(be, w, x, workspace)
		if err != nil {
			panic(err)
		}
		be.Free(out)
	})
	directAllocs := testing.AllocsPerRun(20, func() { be.Free(be.MatMul(w, x)) })
	if plannedAllocs >= directAllocs {
		t.Fatalf("planned run allocations=%g, direct=%g; caller workspace must remove temporary quantization allocations", plannedAllocs, directAllocs)
	}
}

func TestMatMulPlanRejectsCompatibilityMismatchAndStalePlan(t *testing.T) {
	be, w, x := plannedQ8Fixture()
	plan, err := be.PrepareMatMul(w, x)
	if err != nil {
		t.Fatal(err)
	}
	workspace := make([]byte, plan.Descriptor().WorkspaceBytes)

	other := &cpuBackend{}
	cases := []struct {
		name string
		be   Backend
		w, x Tensor
		want string
	}{
		{"backend ownership", other, w, x, "tensor backend mismatch"},
		{"dtype", be, NewF32(be, w.Shape, make([]float32, w.Numel())), x, "dtype mismatch"},
		{"shape", be, w, NewF32(be, []int{16}, make([]float32, 16)), "shape mismatch"},
		{"operation parameter", be, QuantizeQ8(be, w.Shape, make([]float32, w.Numel()), 16), x, "operation parameter mismatch"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := plan.Run(tc.be, tc.w, tc.x, workspace)
			if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "prepare again") {
				t.Fatalf("error=%v, want actionable %q", err, tc.want)
			}
		})
	}

	wrongDevice := *plan
	wrongDevice.desc = plan.Descriptor()
	wrongDevice.desc.Device = "cuda:0"
	_, _, err = wrongDevice.Run(be, w, x, workspace)
	if err == nil || !strings.Contains(err.Error(), "backend/device mismatch") || !strings.Contains(err.Error(), "prepare again") {
		t.Fatalf("device mismatch error=%v", err)
	}

	stale := *plan
	stale.revision--
	_, _, err = stale.Run(be, w, x, workspace)
	if err == nil || !strings.Contains(err.Error(), "stale") || !strings.Contains(err.Error(), "prepare again") {
		t.Fatalf("stale error=%v", err)
	}
}

func TestMatMulPlanUnsupportedEnvelopeKeepsDirectFallback(t *testing.T) {
	be := &cpuBackend{}
	w := NewF32(be, []int{2, 2}, []float32{1, 2, 3, 4})
	x := NewF32(be, []int{2}, []float32{5, 6})
	if _, err := be.PrepareMatMul(w, x); err == nil || !strings.Contains(err.Error(), "use Backend.MatMul directly") {
		t.Fatalf("prepare error=%v", err)
	}
	if got := be.Read(Backend(be).MatMul(w, x)); !slices.Equal(got, []float32{17, 39}) {
		t.Fatalf("direct fallback output=%v", got)
	}
}

func TestMatMulPlanDiagnosticsReportReuseWithoutSpeedupClaim(t *testing.T) {
	be, w, x := plannedQ8Fixture()
	plan, err := be.PrepareMatMul(w, x)
	if err != nil {
		t.Fatal(err)
	}
	plan.desc.Preparation = 17 * time.Microsecond // deterministic captured preparation observation
	runs := []PlanRunDiagnostic{{Duration: 2 * time.Microsecond}, {Duration: 4 * time.Microsecond}}
	d := plan.Diagnostics(8*time.Microsecond, runs...)
	if d.Preparation != 17*time.Microsecond || !slices.Equal(d.RunTimes, []time.Duration{2 * time.Microsecond, 4 * time.Microsecond}) || d.WorkspaceBytes != plan.Descriptor().WorkspaceBytes {
		t.Fatalf("incomplete diagnostics: %+v", d)
	}
	if !d.BreakEvenObserved || d.BreakEvenReuse != 4 {
		t.Fatalf("break-even diagnostics: %+v", d)
	}
	if d.SpeedupClaimed {
		t.Fatal("diagnostics made an unsupported speedup claim")
	}

	noGain := plan.Diagnostics(3*time.Microsecond, runs...)
	if noGain.BreakEvenObserved || noGain.BreakEvenReuse != 0 || noGain.SpeedupClaimed {
		t.Fatalf("non-positive net result must not claim break-even or speedup: %+v", noGain)
	}
}

func BenchmarkMatMulPlanRunCallerWorkspace(b *testing.B) {
	be, w, x := plannedQ8Fixture()
	plan, err := be.PrepareMatMul(w, x)
	if err != nil {
		b.Fatal(err)
	}
	workspace := make([]byte, plan.Descriptor().WorkspaceBytes)
	b.ReportMetric(float64(len(workspace)), "workspace_bytes")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _, err := plan.Run(be, w, x, workspace)
		if err != nil {
			b.Fatal(err)
		}
		be.Free(out)
	}
}
