//go:build darwin && arm64 && cgo

package inklingmetal

import (
	"os"
	"testing"
	"time"
)

// fak-test:runtime slow est=8s
func TestRejectedRowsPreserveInklingContinuation(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s, err := m.NewSession(128)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	prompt := []int32{371, 151451, 149879, 151194, 149884, 152717, 13, 149917, 151451, 149879, 153067, 149884, 154328, 13, 149917, 151451, 149879, 153630, 149884}
	head, err := s.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	best := 0
	for i, x := range head {
		if x > head[best] {
			best = i
		}
	}
	if best != 182416 {
		t.Fatalf("prefill ID %d, want 182416", best)
	}
	for _, step := range []struct {
		ids  []int32
		keep int
		want int32
	}{
		{[]int32{int32(best), 111, 222}, 1, 13},
		{[]int32{1234, 5678}, 0, -1},
		{[]int32{13}, 1, 149917},
	} {
		got, err := s.BeginBlockIDs(step.ids)
		if err != nil {
			t.Fatal(err)
		}
		if step.want >= 0 && got[0] != step.want {
			t.Fatalf("after commit, got %d want %d", got[0], step.want)
		}
		if err := s.Commit(step.keep); err != nil {
			t.Fatal(err)
		}
	}
}

// fak-test:runtime slow est=12s
func TestReusedArenaStartsAtUnrelatedPrompt(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	caps := []int32{371, 151451, 149879, 151194, 149884, 152717, 13, 149917, 151451, 149879, 153067, 149884, 154328, 13, 149917, 151451, 149879, 153630, 149884}
	fib := []int32{1719, 220, 46389, 7, 77, 8, 25, 198, 149862, 172706, 72358, 149876, 149934, 12, 258, 181713, 150107, 13, 70646, 198}
	for i := 0; i < 6; i++ {
		ids, want := caps, int32(182416)
		if i%2 == 1 {
			ids, want = fib, 149862
		}
		before := time.Now()
		idle := m.idle
		s, err := m.NewSession(128)
		if err != nil {
			t.Fatal(err)
		}
		if i > 0 && s.ptr != idle {
			t.Fatal("serial request failed to reuse the idle native arena")
		}
		acquire := time.Since(before)
		got, err := s.PrefillID(ids)
		first := time.Since(before)
		if err != nil || got != want {
			t.Fatalf("request %d got %d want %d: %v", i, got, want, err)
		}
		if _, err = s.BeginBlockIDs([]int32{got, 111, 222}); err != nil {
			t.Fatal(err)
		}
		if err = s.Commit(i % 3); err != nil {
			t.Fatal(err)
		}
		startClose := time.Now()
		if err = s.Close(); err != nil {
			t.Fatal(err)
		}
		t.Logf("request %d acquire=%s first_token=%s close=%s", i, acquire, first, time.Since(startClose))
	}
	first, err := m.NewSession(64)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := m.NewSession(64)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if first.ptr == second.ptr {
		t.Fatal("overlapping requests shared mutable arenas")
	}
	if _, err = first.PrefillID(caps); err != nil {
		t.Fatal(err)
	}
	if got, err := second.PrefillID(fib); err != nil || got != 149862 {
		t.Fatalf("concurrent Fibonacci got %d: %v", got, err)
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if got, err := first.BeginBlockIDs([]int32{182416}); err != nil || got[0] != 13 {
		t.Fatalf("live session lost model ownership: %v %v", got, err)
	}
}

// fak-test:runtime slow est=1s
func TestInklingMetalMemoryPlan(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to plan the local model")
	}
	small, err := Estimate(path, 128)
	if err != nil {
		t.Fatal(err)
	}
	large, err := Estimate(path, 8192)
	if err != nil {
		t.Fatal(err)
	}
	if small.TargetImages < 1<<30 || small.PrefillScratch == 0 || small.DecodeScratch == 0 ||
		small.State == 0 || large.TargetImages != small.TargetImages || large.KV <= small.KV {
		t.Fatalf("inconsistent native allocation plan: short=%+v long=%+v", small, large)
	}
	t.Logf("cap128=%+v total=%d cap8192=%d", small, small.Bytes(), large.Bytes())
}

// fak-test:runtime slow est=8s
func TestFusedCommitLimitsAndContinues(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	draft, err := adjacentDraft(path)
	if err != nil {
		t.Fatal(err)
	}
	if draft == "" {
		t.Skip("a matching DFlash2 checkpoint is required")
	}
	m, err := OpenWithDraft(path, draft)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := []int32{371, 151451, 149879, 151194, 149884, 152717, 13, 149917, 151451, 149879, 153067, 149884, 154328, 13, 149917, 151451, 149879, 153630, 149884}
	for _, tc := range []struct {
		maximum int
		eos     int32
	}{{1, -1}, {8, 13}} {
		s, err := m.NewSession(64)
		if err != nil {
			t.Fatal(err)
		}
		func() {
			defer s.Close()
			anchor, err := s.PrefillID(prompt)
			if err != nil || anchor != 182416 {
				t.Fatalf("prefill returned %d: %v", anchor, err)
			}
			rows, scores, retained, err := s.ProposeVerifyCommit(anchor, 7, 4, tc.maximum, tc.eos)
			if err != nil || len(rows) == 0 || len(scores) != len(rows) || rows[0][0] != anchor || scores[0] != 13 || retained != 1 {
				t.Fatalf("bounded fused verification retained %#x over %d rows: %v", retained, len(rows), err)
			}
			next, err := s.BeginBlockIDs([]int32{13})
			if err != nil || len(next) != 1 || next[0] != 149917 {
				t.Fatalf("continuation after fused commit: %v, %v", next, err)
			}
		}()
	}
}
