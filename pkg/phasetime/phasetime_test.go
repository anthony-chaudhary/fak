package phasetime

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeClock is a deterministic, goroutine-safe clock for newWithClock.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// skipIfEnvRecorderActive guards tests that assert the no-recorder path: the
// environment recorder is a process-wide sync.Once, so a host that exports
// FAK_FLOW_TIMINGS makes FromContext non-nil by design.
func skipIfEnvRecorderActive(t *testing.T) {
	t.Helper()
	if Enabled() {
		t.Skipf("%s is set in the host environment; the env recorder is active by design", EnvVar)
	}
}

func TestPhasetimeStartRecordsOffsetDurationAndNote(t *testing.T) {
	clk := newFakeClock()
	r := newWithClock(clk.now, nil)

	clk.advance(1 * time.Second)
	end := r.Start("census")
	clk.advance(2 * time.Second)
	end("  12 refs", "bounded  ")

	got := r.Phases()
	if len(got) != 1 {
		t.Fatalf("Phases() len = %d, want 1: %+v", len(got), got)
	}
	want := Phase{Name: "census", Offset: 1 * time.Second, Duration: 2 * time.Second, Note: "12 refs bounded"}
	if got[0] != want {
		t.Fatalf("phase = %+v, want %+v", got[0], want)
	}
	if e := r.Elapsed(); e != 3*time.Second {
		t.Fatalf("Elapsed() = %s, want 3s", e)
	}

	// A phase ended with no note records an empty note.
	end2 := r.Start("bare")
	end2()
	if ph := r.Phases(); len(ph) != 2 || ph[1].Name != "bare" || ph[1].Note != "" || ph[1].Duration != 0 {
		t.Fatalf("bare phase = %+v", ph)
	}
}

func TestPhasetimeEndTwiceRecordsOnce(t *testing.T) {
	clk := newFakeClock()
	r := newWithClock(clk.now, nil)

	end := r.Start("status.probe")
	clk.advance(500 * time.Millisecond)
	end("first")
	clk.advance(5 * time.Second)
	end("second")
	end()

	got := r.Phases()
	if len(got) != 1 {
		t.Fatalf("double end recorded %d phases, want 1: %+v", len(got), got)
	}
	if got[0].Note != "first" || got[0].Duration != 500*time.Millisecond {
		t.Fatalf("double end kept the wrong call: %+v (want note=first dur=500ms)", got[0])
	}
}

func TestPhasetimeEndTwiceConcurrentlyRecordsOnce(t *testing.T) {
	r := New()
	end := r.Start("racy")
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			end(fmt.Sprintf("n%d", i))
		}(i)
	}
	wg.Wait()
	if got := r.Phases(); len(got) != 1 {
		t.Fatalf("concurrent double end recorded %d phases, want 1", len(got))
	}
}

func TestPhasetimeNilRecorderIsNoop(t *testing.T) {
	var r *Recorder

	end := r.Start("anything")
	if end == nil {
		t.Fatal("nil recorder Start returned a nil end func")
	}
	end()
	end("note")

	if got := r.Phases(); got != nil {
		t.Fatalf("nil recorder Phases() = %+v, want nil", got)
	}
	if e := r.Elapsed(); e != 0 {
		t.Fatalf("nil recorder Elapsed() = %s, want 0", e)
	}
	var buf bytes.Buffer
	r.WriteTable(&buf)
	if buf.Len() != 0 {
		t.Fatalf("nil recorder WriteTable wrote %q, want nothing", buf.String())
	}
}

func TestPhasetimeWithRecorderAndFromContext(t *testing.T) {
	base := context.Background()
	r1 := New()
	r2 := New()

	ctx1 := WithRecorder(base, r1)
	if got := FromContext(ctx1); got != r1 {
		t.Fatalf("FromContext(ctx1) = %p, want r1 %p", got, r1)
	}

	// A nil recorder leaves the context unchanged.
	if got := WithRecorder(base, nil); got != base {
		t.Fatal("WithRecorder(ctx, nil) did not return ctx unchanged")
	}
	if got := FromContext(WithRecorder(ctx1, nil)); got != r1 {
		t.Fatalf("WithRecorder(ctx1, nil) lost r1: got %p", got)
	}

	// The innermost carried recorder wins.
	ctx2 := WithRecorder(ctx1, r2)
	if got := FromContext(ctx2); got != r2 {
		t.Fatalf("FromContext(ctx2) = %p, want r2 %p", got, r2)
	}
	if got := FromContext(ctx1); got != r1 {
		t.Fatalf("parent context changed: got %p, want r1 %p", got, r1)
	}
}

func TestPhasetimeFromContextNilWithoutEnv(t *testing.T) {
	skipIfEnvRecorderActive(t)

	if got := FromContext(context.Background()); got != nil {
		t.Fatalf("FromContext(background) = %p, want nil with %s unset", got, EnvVar)
	}
	var nilCtx context.Context
	if got := FromContext(nilCtx); got != nil {
		t.Fatalf("FromContext(nil) = %p, want nil with %s unset", got, EnvVar)
	}

	// Track and Mark on the no-recorder path are no-ops that never panic.
	end := Track(context.Background(), "wip.census")
	end("ignored")
	end()
	Mark("wip.identity")("ignored")
}

func TestPhasetimeTrackUsesContextRecorder(t *testing.T) {
	clk := newFakeClock()
	r := newWithClock(clk.now, nil)
	ctx := WithRecorder(context.Background(), r)

	clk.advance(250 * time.Millisecond)
	end := Track(ctx, "wip.worktree_status")
	clk.advance(750 * time.Millisecond)
	end("3 worktree(s)")

	got := r.Phases()
	if len(got) != 1 {
		t.Fatalf("Track recorded %d phases on the ctx recorder, want 1", len(got))
	}
	want := Phase{Name: "wip.worktree_status", Offset: 250 * time.Millisecond, Duration: 750 * time.Millisecond, Note: "3 worktree(s)"}
	if got[0] != want {
		t.Fatalf("Track phase = %+v, want %+v", got[0], want)
	}
}

func TestPhasetimeEnabledParsesEnv(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"", false},
		{"0", false},
		{"false", false},
		{"FALSE", false},
		{"False", false},
		{"off", false},
		{"OFF", false},
		{"no", false},
		{"No", false},
		{"  off  ", false},
		{"1", true},
		{"true", true},
		{"TRUE", true},
		{"on", true},
		{"yes", true},
		{"stderr", true},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%q", tc.val), func(t *testing.T) {
			t.Setenv(EnvVar, tc.val)
			if got := Enabled(); got != tc.want {
				t.Fatalf("Enabled() with %s=%q = %t, want %t", EnvVar, tc.val, got, tc.want)
			}
		})
	}

	// Enabled re-reads the environment on every call.
	t.Run("rereads", func(t *testing.T) {
		t.Setenv(EnvVar, "1")
		if !Enabled() {
			t.Fatal("Enabled() = false after setting 1")
		}
		t.Setenv(EnvVar, "off")
		if Enabled() {
			t.Fatal("Enabled() = true after resetting to off; it must re-read the env")
		}
	})
}

func TestPhasetimePhasesSortedByOffsetThenNameAndCopied(t *testing.T) {
	clk := newFakeClock()
	r := newWithClock(clk.now, nil)

	// Two phases share offset 0 ("b" started first), one starts at 1s. They
	// end in reverse order so insertion order differs from the sorted order.
	endB := r.Start("b")
	endA := r.Start("a")
	clk.advance(1 * time.Second)
	endC := r.Start("c")
	clk.advance(1 * time.Second)
	endC()
	endB()
	endA()

	got := r.Phases()
	var names []string
	for _, p := range got {
		names = append(names, p.Name)
	}
	if strings.Join(names, ",") != "a,b,c" {
		t.Fatalf("Phases() order = %v, want [a b c] (offset, then name)", names)
	}

	// The returned slice is a copy: mutating it never changes the recorder.
	got[0].Name = "mutated"
	again := r.Phases()
	if len(again) != 3 || again[0].Name != "a" {
		t.Fatalf("Phases() is not a copy; after mutation got %+v", again)
	}
}

func TestPhasetimeConcurrentStartEndRaceFree(t *testing.T) {
	r := New()
	const n = 50
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			end := r.Start(fmt.Sprintf("worker.%02d", i))
			_ = r.Phases() // concurrent readers alongside writers
			_ = r.Elapsed()
			end(fmt.Sprintf("i=%d", i))
			end("again") // idempotent under contention too
		}(i)
	}
	close(start)
	wg.Wait()

	got := r.Phases()
	if len(got) != n {
		t.Fatalf("recorded %d phases, want %d", len(got), n)
	}
	seen := map[string]bool{}
	for _, p := range got {
		if seen[p.Name] {
			t.Fatalf("phase %q recorded twice", p.Name)
		}
		seen[p.Name] = true
		if p.Duration < 0 || p.Offset < 0 {
			t.Fatalf("phase %q has negative offset/duration: %+v", p.Name, p)
		}
		if !strings.HasPrefix(p.Note, "i=") {
			t.Fatalf("phase %q kept note %q, want the first end call's note", p.Name, p.Note)
		}
	}
	if !sort.SliceIsSorted(got, func(i, j int) bool {
		if got[i].Offset != got[j].Offset {
			return got[i].Offset < got[j].Offset
		}
		return got[i].Name < got[j].Name
	}) {
		t.Fatalf("Phases() not sorted by offset then name: %+v", got)
	}
}

func TestPhasetimeStreamingWritesOneLinePerPhase(t *testing.T) {
	var buf bytes.Buffer
	r := NewStreaming(&buf)
	r.Start("wip.identity")()
	r.Start("wip.census")("40 refs, partial=false")

	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("streaming wrote %d lines, want 2:\n%s", len(lines), buf.String())
	}
	if !strings.HasPrefix(lines[0], "timing:") || !strings.Contains(lines[0], "wip.identity") {
		t.Fatalf("line 0 = %q, want a timing: line naming wip.identity", lines[0])
	}
	if !strings.HasPrefix(lines[1], "timing:") || !strings.Contains(lines[1], "wip.census") || !strings.Contains(lines[1], "40 refs, partial=false") {
		t.Fatalf("line 1 = %q, want a timing: line naming wip.census with its note", lines[1])
	}
	if got := r.Phases(); len(got) != 2 {
		t.Fatalf("streaming recorder must also keep phases; got %d", len(got))
	}

	// A double end streams exactly once.
	buf.Reset()
	end := r.Start("once")
	end()
	end()
	if c := strings.Count(buf.String(), "timing:"); c != 1 {
		t.Fatalf("double end streamed %d lines, want 1: %q", c, buf.String())
	}
}

func TestPhasetimeStreamingDeterministicFormat(t *testing.T) {
	clk := newFakeClock()
	var buf bytes.Buffer
	r := newWithClock(clk.now, &buf)

	clk.advance(1500 * time.Millisecond)
	end := r.Start("status.probe")
	clk.advance(2250 * time.Millisecond)
	end("8 probed")

	line := strings.TrimRight(buf.String(), "\n")
	for _, want := range []string{"timing:", "status.probe", "+", "1.50s", "2.25s", "8 probed"} {
		if !strings.Contains(line, want) {
			t.Fatalf("streamed line %q missing %q", line, want)
		}
	}
	if strings.Contains(buf.String(), "\n\n") || strings.Count(buf.String(), "\n") != 1 {
		t.Fatalf("streamed output is not exactly one line: %q", buf.String())
	}
}

func TestPhasetimeWriteTable(t *testing.T) {
	clk := newFakeClock()
	r := newWithClock(clk.now, nil)

	endWip := r.Start("wip")
	endCensus := r.Start("wip.census")
	clk.advance(1 * time.Second)
	endRefs := r.Start("wip.census.refs")
	clk.advance(1 * time.Second)
	endRefs()
	endCensus("12 refs")
	clk.advance(2 * time.Second)
	endWip()

	var buf bytes.Buffer
	r.WriteTable(&buf)
	out := buf.String()
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) != 2+3 {
		t.Fatalf("table has %d lines, want header+columns+3 rows:\n%s", len(lines), out)
	}
	if !strings.Contains(lines[0], "PHASE TIMINGS") || !strings.Contains(lines[0], "4.00s") {
		t.Fatalf("header = %q, want PHASE TIMINGS with wall 4.00s", lines[0])
	}
	if !strings.Contains(lines[1], "wall%") || !strings.Contains(lines[1], "phase") {
		t.Fatalf("column header = %q, want phase/wall%% columns", lines[1])
	}

	rows := map[string]string{}
	for _, l := range lines[2:] {
		f := strings.Fields(l)
		if len(f) == 0 {
			t.Fatalf("blank row in table:\n%s", out)
		}
		rows[f[0]] = l
	}
	cases := []struct {
		name  string
		depth int
		share string
		note  string
	}{
		{"wip", 0, "100.0%", ""},
		{"wip.census", 1, "50.0%", "12 refs"},
		{"wip.census.refs", 2, "25.0%", ""},
	}
	for _, tc := range cases {
		row, ok := rows[tc.name]
		if !ok {
			t.Fatalf("no row for %q:\n%s", tc.name, out)
		}
		indent := len(row) - len(strings.TrimLeft(row, " "))
		if want := 2 + 2*tc.depth; indent != want {
			t.Errorf("row %q indent = %d, want %d (2 + 2 per dot):\n%q", tc.name, indent, want, row)
		}
		if !strings.Contains(row, tc.share) {
			t.Errorf("row %q missing wall share %s: %q", tc.name, tc.share, row)
		}
		if tc.note != "" && !strings.HasSuffix(strings.TrimRight(row, " "), tc.note) {
			t.Errorf("row %q missing note %q: %q", tc.name, tc.note, row)
		}
	}
	// Rows render in Phases() order (offset, then name).
	var order []string
	for _, l := range lines[2:] {
		order = append(order, strings.Fields(l)[0])
	}
	if strings.Join(order, ",") != "wip,wip.census,wip.census.refs" {
		t.Fatalf("row order = %v", order)
	}
}

func TestPhasetimeWriteTableZeroWall(t *testing.T) {
	clk := newFakeClock()
	r := newWithClock(clk.now, nil)
	r.Start("instant")()

	var buf bytes.Buffer
	r.WriteTable(&buf)
	out := buf.String()
	if !strings.Contains(out, "PHASE TIMINGS") || !strings.Contains(out, "instant") {
		t.Fatalf("zero-wall table missing header or row:\n%s", out)
	}
	if !strings.Contains(out, "0.0%") || strings.Contains(out, "NaN") || strings.Contains(out, "Inf") {
		t.Fatalf("zero-wall table must render a 0.0%% share, not NaN/Inf:\n%s", out)
	}
}
