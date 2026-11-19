package computetrace

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestBreakdownOf_ClassifiesAndAccounts(t *testing.T) {
	base := time.Unix(1_000_000, 0).UTC()
	a := Artifact{
		Schema: Schema,
		Events: []Event{
			// kernel: [0,100)
			{Sequence: 1, Operation: "matmul", Phase: "kernel", Backend: "cuda", Device: "cuda:0", Kernel: "sgemm", StartedAt: base, DurationNS: 100, TimerDomain: "cuda_event"},
			// copy: [100,150)
			{Sequence: 2, Operation: "copy", Phase: "copy", Backend: "cuda", Device: "cuda:0", Kernel: "h2d_upload", StartedAt: base.Add(100 * time.Nanosecond), DurationNS: 50, TimerDomain: "cuda_event"},
			// weight: [150,200)
			{Sequence: 3, Operation: "weight_load", Phase: "weight", Backend: "cuda", Device: "cuda:0", StartedAt: base.Add(150 * time.Nanosecond), DurationNS: 50, TimerDomain: "cuda_event"},
			// sync: [200,220)
			{Sequence: 4, Operation: "sync", Phase: "sync", Backend: "cuda", Device: "cuda:0", StartedAt: base.Add(200 * time.Nanosecond), DurationNS: 20, TimerDomain: "cuda_event"},
			// cpu: [220,300)
			{Sequence: 5, Operation: "matmul", Phase: "kernel", Backend: "cpu-ref", Device: "host", Kernel: "f32_matmul", StartedAt: base.Add(220 * time.Nanosecond), DurationNS: 80, TimerDomain: "host_monotonic"},
		},
	}
	b := BreakdownOf(a)
	if b.SpanNS != 300 {
		t.Fatalf("span = %d, want 300", b.SpanNS)
	}
	if b.CoveredNS != 300 {
		t.Fatalf("covered = %d, want 300", b.CoveredNS)
	}
	if b.UnexplainedNS != 0 {
		t.Fatalf("unexplained = %d, want 0", b.UnexplainedNS)
	}
	want := map[string]int64{
		ClassKernel: 100,
		ClassCopy:   50,
		ClassWeight: 50,
		ClassSync:   20,
		ClassCPU:    80,
	}
	for c, w := range want {
		if got := b.ClassNS[c]; got != w {
			t.Fatalf("class %s = %d, want %d", c, got, w)
		}
	}
	if b.Events != 5 {
		t.Fatalf("events = %d, want 5", b.Events)
	}
}

// fak-test:runtime fast est=1s
func TestBreakdownOf_UnexplainedGap(t *testing.T) {
	base := time.Unix(1_000_000, 0).UTC()
	a := Artifact{
		Schema: Schema,
		Events: []Event{
			// [0,100) then a 200ns gap then [300,400)
			{Sequence: 1, Operation: "matmul", Phase: "kernel", Backend: "cuda", Device: "cuda:0", StartedAt: base, DurationNS: 100, TimerDomain: "cuda_event"},
			{Sequence: 2, Operation: "matmul", Phase: "kernel", Backend: "cuda", Device: "cuda:0", StartedAt: base.Add(300 * time.Nanosecond), DurationNS: 100, TimerDomain: "cuda_event"},
		},
	}
	b := BreakdownOf(a)
	if b.SpanNS != 400 {
		t.Fatalf("span = %d, want 400", b.SpanNS)
	}
	if b.CoveredNS != 200 {
		t.Fatalf("covered = %d, want 200", b.CoveredNS)
	}
	if b.UnexplainedNS != 200 {
		t.Fatalf("unexplained = %d, want 200", b.UnexplainedNS)
	}
	if b.ClassNS[ClassKernel] != 200 {
		t.Fatalf("kernel = %d, want 200", b.ClassNS[ClassKernel])
	}
}

// fak-test:runtime fast est=1s
func TestBreakdownOf_OverlappingClasses(t *testing.T) {
	base := time.Unix(1_000_000, 0).UTC()
	a := Artifact{
		Schema: Schema,
		Events: []Event{
			// kernel [0,100) and a concurrent host cpu event [50,150): they overlap.
			{Sequence: 1, Operation: "matmul", Phase: "kernel", Backend: "cuda", Device: "cuda:0", StartedAt: base, DurationNS: 100, TimerDomain: "cuda_event"},
			{Sequence: 2, Operation: "matmul", Phase: "kernel", Backend: "cpu-ref", Device: "host", StartedAt: base.Add(50 * time.Nanosecond), DurationNS: 100, TimerDomain: "host_monotonic"},
		},
	}
	b := BreakdownOf(a)
	// span [0,150) = 150; covered union [0,150) = 150; unexplained 0.
	if b.SpanNS != 150 || b.CoveredNS != 150 || b.UnexplainedNS != 0 {
		t.Fatalf("span/covered/unexplained = %d/%d/%d, want 150/150/0", b.SpanNS, b.CoveredNS, b.UnexplainedNS)
	}
	// Class sums may exceed covered when classes overlap: kernel 100 + cpu 100 = 200.
	if b.ClassNS[ClassKernel] != 100 || b.ClassNS[ClassCPU] != 100 {
		t.Fatalf("kernel/cpu = %d/%d, want 100/100", b.ClassNS[ClassKernel], b.ClassNS[ClassCPU])
	}
}

// fak-test:runtime fast est=1s
func TestBreakdownOf_Empty(t *testing.T) {
	b := BreakdownOf(Artifact{Schema: Schema})
	if b.SpanNS != 0 || b.CoveredNS != 0 || b.UnexplainedNS != 0 || b.Events != 0 {
		t.Fatalf("empty breakdown = %+v, want all zero", b)
	}
}
