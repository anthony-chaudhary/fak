//go:build cuda && cgo

package ggufload

import (
	"context"
	"runtime"
	"slices"
	"testing"
)

// A session whose first prefill sized the batched arenas for 8 tokens replaces them
// for a following 512-token chunk, frees the old ones, and still continues exactly as
// the whole prompt prefilled cold.
func TestInkling2CUDAPrefillArenaRegrow(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	const n = 16
	prompt := inkling2PrefillPrompt(8 + 512)
	want, _, err := inkling2ResidentGreedy(m, be, prompt, n)
	if err != nil {
		t.Fatal(err)
	}
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	if _, err := s.prefill(context.Background(), prompt[:8]); err != nil {
		t.Fatal(err)
	}
	small, owned := s.prefillCap, len(s.owned)
	next, err := s.prefill(context.Background(), prompt[8:])
	if err != nil {
		t.Fatal(err)
	}
	if small != 8 || s.prefillCap != 512 {
		t.Fatalf("prefill arenas sized %d then %d tokens, want 8 then 512", small, s.prefillCap)
	}
	if len(s.owned) != owned {
		t.Fatalf("regrow left %d owned buffers, want %d: the replaced arenas were not freed", len(s.owned), owned)
	}
	got := []int{next}
	for len(got) < n {
		if next, err = s.step(next); err != nil {
			t.Fatal(err)
		}
		got = append(got, next)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("8+512 prefill:\n got %v\nwant %v", got, want)
	}
}
