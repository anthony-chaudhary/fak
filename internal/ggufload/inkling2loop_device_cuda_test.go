//go:build cuda && cgo

package ggufload

import (
	"context"
	"reflect"
	"runtime"
	"testing"
)

// TestInkling2CUDAScalarCohortCoexistence alternates scalar prefix-reuse
// requests with cohort batch requests on the same CUDA device, verifying
// byte-equal greedy output and clean Inkling2Device.Reset() handoffs.
// fak-test:runtime integration est=60s lane=optin
func TestInkling2CUDAScalarCohortCoexistence(t *testing.T) {
	t.Setenv("FAK_INKLING2_CTX", "8192")
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	if !inkling2ResidentEligible(m, be) {
		t.Skip("model does not run on the resident device forward")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var capitals, fib []int
	for _, c := range inkling2OracleCases {
		switch c.name {
		case "capitals":
			capitals = c.prompt
		case "fibonacci":
			fib = c.prompt
		}
	}
	if capitals == nil || fib == nil {
		t.Fatal("oracle cases missing")
	}

	const gen = 16
	dev := inkling2DeviceFor(be.(Inkling2GraphResetter))

	sess, err := NewInkling2LoopSession(m, be)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	d := &ink2ReuseDriver{s: sess}

	bt := NewInkling2LoopCohorts(m, be)
	defer bt.Close()

	// 1. Scalar turn 1 (cold) and turn 2 (prefix reuse).
	p1 := append([]int(nil), capitals...)
	out1, reused1, err := d.turn(p1, gen)
	if err != nil {
		t.Fatalf("scalar turn 1: %v", err)
	}
	if reused1 != 0 {
		t.Fatalf("scalar turn 1 reused=%d, want 0", reused1)
	}

	p2 := catTokens(p1, out1[:4], fib[:8])
	out2, reused2, err := d.turn(p2, gen)
	if err != nil {
		t.Fatalf("scalar turn 2: %v", err)
	}
	if reused2 == 0 {
		t.Fatalf("scalar turn 2 reused=0, want >0")
	}

	// 2. Cohort batch request, then ReleaseArenas -> Inkling2Device.Reset() handoff.
	c1 := runCohort(t, bt, [][]int{capitals, fib}, gen)
	if dev.Owner() != bt {
		t.Fatalf("device owner during cohort = %v, want cohort engine", dev.Owner())
	}
	bt.ReleaseArenas()
	if dev.Owner() != nil || dev.ExecCount() != 0 {
		t.Fatalf("after ReleaseArenas: owner=%v execs=%d, want nil/0", dev.Owner(), dev.ExecCount())
	}

	// 3. Scalar turn 3 on the same resident session after cohort Reset().
	p3 := catTokens(p2, out2[:4], capitals[:6])
	out3, reused3, err := d.turn(p3, gen)
	if err != nil {
		t.Fatalf("scalar turn 3 after cohort: %v", err)
	}
	if reused3 == 0 {
		t.Fatalf("scalar turn 3 reused=0, want >0")
	}

	// 4. Second cohort batch request, then Close -> Inkling2Device.Reset() handoff.
	c2 := runCohort(t, bt, [][]int{fib, capitals}, gen)
	bt.Close()
	if dev.Owner() != nil || dev.ExecCount() != 0 {
		t.Fatalf("after cohort Close: owner=%v execs=%d, want nil/0", dev.Owner(), dev.ExecCount())
	}
	sess.Close()

	// 5. Verify byte-equal greedy output against fresh single-sequence sessions.
	freshGreedy := func(prompt []int) []int {
		s, err := NewInkling2LoopSession(m, be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if _, err := s.Prefill(context.Background(), prompt[:len(prompt)-1]); err != nil {
			t.Fatal(err)
		}
		nx, err := s.Prefill(context.Background(), prompt[len(prompt)-1:])
		if err != nil {
			t.Fatal(err)
		}
		got, err := ink2DecodeFrom(s, nx, gen)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}

	if want := freshGreedy(p1); !reflect.DeepEqual(out1, want) {
		t.Fatalf("scalar turn 1 mismatch: got %v, want %v", out1, want)
	}
	if want := freshGreedy(p2); !reflect.DeepEqual(out2, want) {
		t.Fatalf("scalar turn 2 mismatch: got %v, want %v", out2, want)
	}
	if want := freshGreedy(p3); !reflect.DeepEqual(out3, want) {
		t.Fatalf("scalar turn 3 mismatch: got %v, want %v", out3, want)
	}
	wantCap := freshGreedy(capitals)
	wantFib := freshGreedy(fib)
	if !reflect.DeepEqual(c1.tokens[0], wantCap) || !reflect.DeepEqual(c1.tokens[1], wantFib) {
		t.Fatalf("cohort 1 mismatch: got (%v, %v), want (%v, %v)", c1.tokens[0], c1.tokens[1], wantCap, wantFib)
	}
	if !reflect.DeepEqual(c2.tokens[0], wantFib) || !reflect.DeepEqual(c2.tokens[1], wantCap) {
		t.Fatalf("cohort 2 mismatch: got (%v, %v), want (%v, %v)", c2.tokens[0], c2.tokens[1], wantFib, wantCap)
	}
}
