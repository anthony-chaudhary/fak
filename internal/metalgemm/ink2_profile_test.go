//go:build darwin && arm64 && cgo

package metalgemm

import (
	"slices"
	"strings"
	"testing"
	"time"
)

const ink2ProfileTestKernels = `#include <metal_stdlib>
using namespace metal;

struct Spin { float scale, bias; uint rounds; };

kernel void test_spin(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      constant Spin &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float v = x[i];
    for (uint r = 0; r < p.rounds; r++) v = v * p.scale + p.bias;
    y[i] = v;
}

kernel void test_copy(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    y[i] = x[i];
}
`

type ink2ProfileTestRun struct {
	out     []float32
	stats   Ink2Stats
	profile Ink2Profile
}

// ink2ProfileTestSequence runs two wide spins under scope "wide", three narrow
// ones under "narrow" and one copy, each launch reading the previous output.
func ink2ProfileTestSequence(t *testing.T, mode Ink2ProfileMode) ink2ProfileTestRun {
	t.Helper()
	if !Available() {
		t.Skip("requires a Metal device")
	}
	d, err := newInk2Device(mode)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	const wide, narrow = 1 << 16, 1 << 10
	buf := ink2TestAlloc(t, d, 8*wide)
	x := make([]float32, wide)
	for i := range x {
		x[i] = float32(i%97)/97 - .5
	}
	if err := d.Write(buf, x); err != nil {
		t.Fatal(err)
	}
	spin, err := d.PipelinePrecise(ink2ProfileTestKernels, "test_spin")
	if err != nil {
		t.Fatal(err)
	}
	cp, err := d.PipelinePrecise(ink2ProfileTestKernels, "test_copy")
	if err != nil {
		t.Fatal(err)
	}
	type shape struct {
		scale, bias float32
		rounds      uint32
	}
	slot := 0
	launch := func(p Ink2Pipeline, n int, args ...Ink2Arg) {
		t.Helper()
		in, out := buf.At(slot*wide), buf.At((slot+1)*wide)
		slot++
		all := append([]Ink2Arg{in.Span(n), out.Span(n)}, args...)
		if err := d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, p.MaxThreads()), 1, 1}, all...); err != nil {
			t.Fatal(err)
		}
	}
	before := d.Stats()
	d.SetProfileScope("wide")
	launch(spin, wide, Ink2Const(shape{.999, .001, 4000}))
	launch(spin, wide, Ink2Const(shape{1.001, -.002, 4000}))
	d.SetProfileScope("narrow")
	for range 3 {
		launch(spin, narrow, Ink2Const(shape{.5, .25, 8}))
	}
	launch(cp, narrow)
	out, err := d.Read(buf.At(slot*wide), narrow)
	if err != nil {
		t.Fatal(err)
	}
	return ink2ProfileTestRun{out: out, stats: d.Stats().Sub(before), profile: d.Profile()}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2ProfileOffCollectsNothingButTimesTheFence(t *testing.T) {
	r := ink2ProfileTestSequence(t, Ink2ProfileOff)
	if r.profile.Enabled() || len(r.profile.Rows) != 0 {
		t.Fatalf("profile off collected %+v", r.profile)
	}
	s := r.stats
	if s.Submissions != 1 || s.Dispatches != 6 {
		t.Fatalf("stats %+v, want 6 dispatches in one submission", s)
	}
	if s.EncodeSeconds <= 0 || s.GPUSeconds <= 0 || s.WaitSeconds < s.GPUSeconds || s.QueueSeconds < 0 || s.TailSeconds < 0 {
		t.Fatalf("stats %+v, want positive encode and GPU time inside the wait", s)
	}
	if got := s.QueueSeconds + s.GPUSeconds + s.TailSeconds; got > s.WaitSeconds*1.01+1e-4 {
		t.Fatalf("queue+gpu+tail = %v exceeds the wait %v", got, s.WaitSeconds)
	}
	text := r.profile.Format(1, time.Second, s)
	if !strings.HasPrefix(text, "totals: tokens=1 ") || strings.Count(text, "\n") != 1 {
		t.Fatalf("profile-off report is not the totals line alone:\n%s", text)
	}
}

func ink2ProfileTestCheck(t *testing.T, mode Ink2ProfileMode) {
	want := ink2ProfileTestSequence(t, Ink2ProfileOff)
	r := ink2ProfileTestSequence(t, mode)
	if !slices.Equal(r.out, want.out) {
		t.Fatalf("%s mode changed the result", mode)
	}
	p := r.profile
	if p.Requested != mode {
		t.Fatalf("requested mode %q, want %q", p.Requested, mode)
	}
	if p.Mode != mode {
		if mode != Ink2ProfilePass || p.Mode != Ink2ProfileCommit {
			t.Fatalf("mode %q for a %q request", p.Mode, mode)
		}
		t.Logf("pass sampling unsupported here: fell back to commit")
	}
	if wantSub := map[Ink2ProfileMode]uint64{Ink2ProfilePass: 1, Ink2ProfileCommit: 6}[p.Mode]; r.stats.Submissions != wantSub {
		t.Fatalf("%s mode made %d submissions, want %d", p.Mode, r.stats.Submissions, wantSub)
	}
	type key struct {
		scope, kernel string
		n             int
	}
	got := map[key]Ink2ProfileRow{}
	for _, row := range p.Rows {
		if !row.Threads || row.Grid[1] != 1 || row.Grid[2] != 1 {
			t.Fatalf("row %+v is not a 1-D thread launch", row)
		}
		got[key{row.Scope, row.Kernel, row.Grid[0]}] = row
	}
	wantRows := map[key]uint64{{"wide", "test_spin", 1 << 16}: 2, {"narrow", "test_spin", 1 << 10}: 3, {"narrow", "test_copy", 1 << 10}: 1}
	if len(got) != len(wantRows) || len(p.Rows) != len(wantRows) {
		t.Fatalf("rows %+v, want keys %v", p.Rows, wantRows)
	}
	sum := 0.0
	for k, n := range wantRows {
		row, ok := got[k]
		if !ok || row.Dispatches != n || row.GPUSeconds <= 0 || row.EncodeSeconds <= 0 {
			t.Fatalf("row %v = %+v (present %v), want %d dispatches with GPU and encode time", k, row, ok, n)
		}
		sum += row.GPUSeconds
	}
	if p.Unsampled != 0 {
		t.Fatalf("%d dispatches went unsampled", p.Unsampled)
	}
	// The wide spins dominate; a wrong timestamp unit would miss by 40x.
	heavy := got[key{"wide", "test_spin", 1 << 16}].GPUSeconds
	if p.Rows[0].Scope != "wide" || heavy < .5*sum {
		t.Fatalf("the wide spins took %v of %v profiled seconds, first row %+v", heavy, sum, p.Rows[0])
	}
	if p.BufferGPUSeconds <= 0 || sum > 1.25*p.BufferGPUSeconds || sum < .5*p.BufferGPUSeconds {
		t.Fatalf("profiled %v s against %v s of command-buffer GPU time", sum, p.BufferGPUSeconds)
	}
	text := p.Format(2, 10*time.Millisecond, r.stats)
	for _, part := range []string{"mode=" + string(p.Mode), "by scope:", "by kernel:", "test_spin", "65536x1x1t", "totals: tokens=2 "} {
		if !strings.Contains(text, part) {
			t.Fatalf("report lacks %q:\n%s", part, text)
		}
	}
	t.Logf("\n%s", text)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2ProfilePassTimesEachDispatchInOneSubmission(t *testing.T) {
	ink2ProfileTestCheck(t, Ink2ProfilePass)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2ProfileCommitTimesEachDispatch(t *testing.T) {
	ink2ProfileTestCheck(t, Ink2ProfileCommit)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2ProfilePassSpansSampleBuffersAndResets(t *testing.T) {
	if !Available() {
		t.Skip("requires a Metal device")
	}
	d, err := newInk2Device(Ink2ProfilePass)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if d.ProfileMode() != Ink2ProfilePass {
		t.Skip("pass sampling unsupported on this device")
	}
	const n, launches = 256, 1300
	buf := ink2TestAlloc(t, d, 2*n)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i)
	}
	if err := d.Write(buf, x); err != nil {
		t.Fatal(err)
	}
	cp, err := d.PipelinePrecise(ink2ProfileTestKernels, "test_copy")
	if err != nil {
		t.Fatal(err)
	}
	run := func() {
		t.Helper()
		for i := range launches {
			in, out := buf.At(i%2*n), buf.At((i+1)%2*n)
			if err := d.DispatchThreads(cp, [3]int{n, 1, 1}, [3]int{n, 1, 1}, in.Span(n), out.Span(n)); err != nil {
				t.Fatal(err)
			}
		}
	}
	before := d.Stats()
	run()
	if p := d.Profile(); len(p.Rows) != 1 || p.Rows[0].Dispatches != launches || p.Rows[0].GPUSeconds != 0 {
		t.Fatalf("pending profile %+v, want %d untimed dispatches", p.Rows, launches)
	}
	got, err := d.Read(buf, n)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, x) {
		t.Fatal("the copy chain changed the data")
	}
	p := d.Profile()
	if s := d.Stats().Sub(before); s.Submissions != 1 || len(p.Rows) != 1 || p.Rows[0].Dispatches != launches || p.Rows[0].GPUSeconds <= 0 || p.Unsampled != 0 {
		t.Fatalf("stats %+v profile %+v, want %d timed dispatches in one submission", s, p, launches)
	}
	run()
	if err := d.ResetProfile(); err != nil {
		t.Fatal(err)
	}
	if p := d.Profile(); len(p.Rows) != 0 || p.BufferGPUSeconds != 0 || d.Stats().Pending != 0 {
		t.Fatalf("after a reset: %+v", p)
	}
	run()
	if err := d.Flush(); err != nil {
		t.Fatal(err)
	}
	if p := d.Profile(); len(p.Rows) != 1 || p.Rows[0].Dispatches != launches || p.Unsampled != 0 {
		t.Fatalf("after a reset and one more run: %+v", p)
	}
}

func TestInk2ProfileModeParses(t *testing.T) {
	for in, want := range map[string]Ink2ProfileMode{"": Ink2ProfileOff, "off": Ink2ProfileOff, "0": Ink2ProfileOff, " Pass ": Ink2ProfilePass, "commit": Ink2ProfileCommit} {
		if got, err := parseInk2ProfileMode(in); err != nil || got != want {
			t.Fatalf("mode %q parsed as %q, %v", in, got, err)
		}
	}
	if _, err := parseInk2ProfileMode("everything"); err == nil {
		t.Fatal("an unknown mode parsed")
	}
}
