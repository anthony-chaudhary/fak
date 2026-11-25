//go:build cuda && cgo

package ggufload

import (
	"reflect"
	"runtime"
	"testing"
	"time"
)

// A resident session that keeps its cache across a multi-turn transcript and
// rewinds to the previous prompt's checkpoint decodes exactly what a fresh
// session per turn decodes. The turns are long enough to wrap the local window
// ring, and a turn diverges from the history to exercise the reset path.
func TestInkling2CUDAPrefixReuseMatchesFresh(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "")
	defer m.Close()
	if !inkling2ResidentEligible(m, be) {
		t.Skip("model does not run on the resident device forward")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	const gen = 24
	window := m.HP.SlidingWindow
	// a system prompt past the window wraps every local ring
	system := inkling2PrefillPrompt(min(window+200, inkling2ResidentCtx-512))
	user := func(seed, n int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = system[(seed*37+i*11)%len(system)]
		}
		return out
	}
	for _, batched := range []string{"0", "1"} {
		t.Run("batched="+batched, func(t *testing.T) {
			t.Setenv("FAK_INKLING2_BATCHED_PREFILL", batched)
			sess, err := NewInkling2LoopSession(m, be)
			if err != nil {
				t.Fatal(err)
			}
			d := &ink2ReuseDriver{s: sess}
			var prompts [][]int
			var reuseOut [][]int
			var reuseTokens int
			var reuseWall time.Duration
			var prev, prevGen []int
			next := func(i int) []int {
				switch i {
				case 0:
					return ink2Cat(system, user(1, 40))
				case 1:
					return ink2Cat(prev, prevGen, user(2, 30))
				case 2:
					return ink2Cat(prev, ink2Rerender(prevGen), user(3, 25))
				case 3:
					return ink2Cat(system[:64], user(4, 50))
				default:
					return append([]int(nil), prev...)
				}
			}
			for i := 0; i < 5; i++ {
				p := next(i)
				start := time.Now()
				got, reused, err := d.turn(p, gen)
				if err != nil {
					sess.Close()
					t.Fatalf("turn %d: %v", i, err)
				}
				wall := time.Since(start)
				reuseWall += wall
				reuseTokens += len(p) - reused
				t.Logf("reuse turn %d: prompt %d, reused %d, prefilled %d, wall %s", i, len(p), reused, len(p)-reused, wall)
				prompts, reuseOut = append(prompts, p), append(reuseOut, got)
				prev, prevGen = p, got
			}
			sess.Close()
			var freshTokens int
			var freshWall time.Duration
			for i, p := range prompts {
				s, err := NewInkling2LoopSession(m, be)
				if err != nil {
					t.Fatal(err)
				}
				start := time.Now()
				// the engine's reuse-off mode: fresh session, same last-token split
				var want []int
				_, err = s.Prefill(p[:len(p)-1])
				if err == nil {
					var nx int
					if nx, err = s.Prefill(p[len(p)-1:]); err == nil {
						want, err = ink2DecodeFrom(s, nx, gen)
					}
				}
				s.Close()
				if err != nil {
					t.Fatalf("fresh turn %d: %v", i, err)
				}
				freshWall += time.Since(start)
				freshTokens += len(p)
				if !reflect.DeepEqual(reuseOut[i], want) {
					t.Fatalf("turn %d: reuse decoded %v, fresh %v", i, reuseOut[i], want)
				}
			}
			t.Logf("prefill tokens: reuse %d vs fresh %d; wall: reuse %s vs fresh %s", reuseTokens, freshTokens, reuseWall, freshWall)
		})
	}
}
