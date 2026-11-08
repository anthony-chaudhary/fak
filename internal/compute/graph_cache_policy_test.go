package compute

import (
	"errors"
	"fmt"
	"testing"
)

// fakeGraphCacheEngine records which engine path the policy chose so tests
// assert closed behavior (counts, outcomes, releases), not error prose.
type fakeGraphCacheEngine struct {
	begins, ends, aborts, replays, eagres int
	beginErr, endErr, replayErr, eagerErr error
	released                              []string
	execs                                 map[string]*fakeGraphCacheExecutable
}

func newFakeGraphCacheEngine() *fakeGraphCacheEngine {
	return &fakeGraphCacheEngine{execs: make(map[string]*fakeGraphCacheExecutable)}
}

func (f *fakeGraphCacheEngine) BeginCapture() error {
	f.begins++
	return f.beginErr
}

func (f *fakeGraphCacheEngine) EndCapture() (GraphCacheExecutable, error) {
	f.ends++
	if f.endErr != nil {
		return nil, f.endErr
	}
	key := fmt.Sprintf("exec-%d", f.ends)
	ex := &fakeGraphCacheExecutable{key: key, eng: f}
	f.execs[key] = ex
	return ex, nil
}

func (f *fakeGraphCacheEngine) AbortCapture() { f.aborts++ }

func (f *fakeGraphCacheEngine) Replay(GraphCacheExecutable) error {
	f.replays++
	return f.replayErr
}

func (f *fakeGraphCacheEngine) RunEager() error {
	f.eagres++
	return f.eagerErr
}

type fakeGraphCacheExecutable struct {
	key string
	eng *fakeGraphCacheEngine
}

func (e *fakeGraphCacheExecutable) Release() {
	e.eng.released = append(e.eng.released, e.key)
}

var errGraphCacheBodyFailed = errors.New("graph cache policy test: body failed")

// fak-test:runtime fast est=1ms lane=default
func TestGraphCachePolicyVoterOwnership(t *testing.T) {
	for _, allow := range []bool{false, true} {
		t.Run(fmt.Sprintf("allow=%v", allow), func(t *testing.T) {
			eng := newFakeGraphCacheEngine()
			votes := 0
			voters := []func() bool{func() bool { votes++; return allow }}
			p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 1, Voters: voters})
			if err != nil {
				t.Fatal(err)
			}
			if allow {
				voters[0] = nil // Must not turn an admitted callback into a panic.
			} else {
				voters[0] = func() bool { return true } // Must not bypass the declining rank.
			}
			bodies := 0
			out, err := p.Run("a", 1, func() error { bodies++; return nil })
			if err != nil || votes != 1 {
				t.Fatalf("vote lost: outcome=%v votes=%d error=%v", out, votes, err)
			}
			if allow {
				if out != GraphRunCaptured || bodies != 1 || eng.begins != 1 || p.Len() != 1 {
					t.Fatal("owned allowing vote failed to capture")
				}
				p.RecordPeerFailure() // Release the fake executable through its owner.
			} else if out != GraphRunEager || bodies != 0 || eng.begins != 0 || eng.eagres != 1 || p.Len() != 0 {
				t.Fatal("caller replacement bypassed declining vote")
			}
		})
	}
	for _, on := range []bool{false, true} {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: on, Cap: 1, Voters: []func() bool{func() bool { return true }, nil}})
		if p != nil || !errors.Is(err, ErrGraphCacheNilVoter) || eng.begins != 0 || eng.eagres != 0 {
			t.Fatalf("malformed voter list accepted: policy=%v error=%v", p, err)
		}
	}
}

// TestGraphCachePolicyCaptureReplayEvict pins the TensorFold graph-cache
// contract adapted in graph_cache_policy.go (zig/src/cuda/graph_cache.zig:178
// @9e8c368b0b4c2c7892a0c9f0f302d7c164016209): capture-once/replay-on-hit,
// LRU eviction under cap, fingerprint drop-all, room-miss shed to min_keep,
// hold evicts nothing, vote-gated misses, eager fallbacks, and peer-failure
// disable.
func TestGraphCachePolicyCaptureReplayEvict(t *testing.T) {
	t.Run("capture then replay", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		var votes int
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{
			On: true, Cap: 4, MinKeep: 1,
			Voters: []func() bool{func() bool { votes++; return true }, func() bool { votes++; return true }},
		})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		out, err := p.Run("a", 1, func() error { return nil })
		if err != nil || out != GraphRunCaptured {
			t.Fatalf("first run = (%s, %v), want captured", out, err)
		}
		if eng.begins != 1 || eng.ends != 1 || votes != 2 {
			t.Fatalf("after capture begins=%d ends=%d votes=%d, want 1/1/2", eng.begins, eng.ends, votes)
		}
		out, err = p.Run("a", 1, func() error { t.Fatal("body must not run on a hit"); return nil })
		if err != nil || out != GraphRunReplayed {
			t.Fatalf("second run = (%s, %v), want replayed", out, err)
		}
		if eng.replays != 1 || eng.begins != 1 || votes != 2 {
			t.Fatalf("after replay replays=%d begins=%d votes=%d, want 1/1/2 (hits never recapture or re-vote)", eng.replays, eng.begins, votes)
		}
		if p.Len() != 1 {
			t.Fatalf("Len = %d, want 1", p.Len())
		}
	})

	t.Run("cap evicts LRU", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		// MinKeep 2 makes the room-miss shed exactly one entry, isolating the
		// LRU-order assertion from the shed-to-floor behavior covered below.
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 3, MinKeep: 2})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		for _, k := range []string{"a", "b", "c"} {
			if out, _ := p.Run(k, 1, func() error { return nil }); out != GraphRunCaptured {
				t.Fatalf("insert %s = %s, want captured", k, out)
			}
		}
		if out, _ := p.Run("a", 1, func() error { return nil }); out != GraphRunReplayed {
			t.Fatalf("touch a = %s, want replayed", out)
		}
		if out, _ := p.Run("d", 1, func() error { return nil }); out != GraphRunCaptured {
			t.Fatalf("insert d at cap = %s, want captured", out)
		}
		if p.Len() != 3 {
			t.Fatalf("Len = %d, want 3", p.Len())
		}
		// b was least recently used: its executable is the only release.
		if len(eng.released) != 1 || eng.released[0] != "exec-2" {
			t.Fatalf("released = %v, want [exec-2] (b evicted)", eng.released)
		}
		// b missed again after eviction, so it captures rather than replays.
		if out, _ := p.Run("b", 1, func() error { return nil }); out != GraphRunCaptured {
			t.Fatalf("re-run evicted b = %s, want captured", out)
		}
	})

	t.Run("new fingerprint drops all", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 4, MinKeep: 1})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		if out, _ := p.Run("a", 1, func() error { return nil }); out != GraphRunCaptured {
			t.Fatalf("a@fp1 = %s, want captured", out)
		}
		if out, _ := p.Run("b", 1, func() error { return nil }); out != GraphRunCaptured {
			t.Fatalf("b@fp1 = %s, want captured", out)
		}
		out, err := p.Run("a", 2, func() error { return nil })
		if err != nil || out != GraphRunCaptured {
			t.Fatalf("a@fp2 = (%s, %v), want captured (old graphs dropped)", out, err)
		}
		if p.Len() != 1 {
			t.Fatalf("Len = %d, want 1 (only the fresh a)", p.Len())
		}
		if len(eng.released) != 2 || eng.released[0] != "exec-1" || eng.released[1] != "exec-2" {
			t.Fatalf("released = %v, want [exec-1 exec-2]", eng.released)
		}
	})

	t.Run("room-miss sheds to min_keep", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 5, MinKeep: 2})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		for _, k := range []string{"a", "b", "c", "d", "e"} {
			if out, _ := p.Run(k, 1, func() error { return nil }); out != GraphRunCaptured {
				t.Fatalf("insert %s = %s, want captured", k, out)
			}
		}
		if out, _ := p.Run("f", 1, func() error { return nil }); out != GraphRunCaptured {
			t.Fatalf("room-miss f = %s, want captured", out)
		}
		// Shed down to min_keep=2 (a, b, c gone; d, e kept), then insert f.
		if p.Len() != 3 {
			t.Fatalf("Len = %d, want 3 (min_keep + new)", p.Len())
		}
		if len(eng.released) != 3 || eng.released[0] != "exec-1" || eng.released[1] != "exec-2" || eng.released[2] != "exec-3" {
			t.Fatalf("released = %v, want [exec-1 exec-2 exec-3]", eng.released)
		}
		if out, _ := p.Run("d", 1, func() error { return nil }); out != GraphRunReplayed {
			t.Fatalf("survivor d = %s, want replayed", out)
		}
	})

	t.Run("hold evicts nothing", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 2, MinKeep: 1, Hold: true})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		for _, k := range []string{"a", "b", "c"} {
			if out, _ := p.Run(k, 1, func() error { return nil }); out != GraphRunCaptured {
				t.Fatalf("insert %s = %s, want captured", k, out)
			}
		}
		if p.Len() != 3 {
			t.Fatalf("Len = %d, want 3 (hold grows past cap)", p.Len())
		}
		if len(eng.released) != 0 {
			t.Fatalf("released = %v, want none", eng.released)
		}
	})

	t.Run("declined vote runs eager", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{
			On: true, Cap: 4, MinKeep: 1,
			Voters: []func() bool{func() bool { return true }, func() bool { return false }},
		})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		out, err := p.Run("a", 1, func() error { t.Fatal("body must not run when the vote declines"); return nil })
		if err != nil || out != GraphRunEager {
			t.Fatalf("declined miss = (%s, %v), want eager", out, err)
		}
		if eng.begins != 0 || eng.eagres != 1 || p.Len() != 0 {
			t.Fatalf("begins=%d eagres=%d Len=%d, want 0/1/0", eng.begins, eng.eagres, p.Len())
		}
		if out, _ = p.Run("a", 1, func() error { return nil }); out != GraphRunEager {
			t.Fatalf("second declined miss = %s, want eager (nothing stored)", out)
		}
	})

	t.Run("peer failure disables", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 4, MinKeep: 1})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		if out, _ := p.Run("a", 1, func() error { return nil }); out != GraphRunCaptured {
			t.Fatalf("a = %s, want captured", out)
		}
		p.RecordPeerFailure()
		if !p.Disabled() || p.Len() != 0 {
			t.Fatalf("Disabled=%v Len=%d, want true/0", p.Disabled(), p.Len())
		}
		if len(eng.released) != 1 {
			t.Fatalf("released = %v, want the one stored executable", eng.released)
		}
		out, err := p.Run("a", 1, func() error { return nil })
		if err != nil || out != GraphRunEager {
			t.Fatalf("post-failure run = (%s, %v), want eager", out, err)
		}
		if eng.eagres != 1 || eng.begins != 1 {
			t.Fatalf("eagres=%d begins=%d, want 1/1 (no recapture once disabled)", eng.eagres, eng.begins)
		}
	})

	t.Run("env off runs eager", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: false, Cap: 4, MinKeep: 1})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		out, err := p.Run("a", 1, func() error { t.Fatal("body must not run while off"); return nil })
		if err != nil || out != GraphRunEager {
			t.Fatalf("off run = (%s, %v), want eager", out, err)
		}
		if eng.begins != 0 || eng.eagres != 1 || p.Len() != 0 {
			t.Fatalf("begins=%d eagres=%d Len=%d, want 0/1/0", eng.begins, eng.eagres, p.Len())
		}
	})

	t.Run("begin capture failure falls back to eager", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		eng.beginErr = errors.New("out of device memory")
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 4, MinKeep: 1})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		out, err := p.Run("a", 1, func() error { return nil })
		if err != nil || out != GraphRunEager {
			t.Fatalf("OOM miss = (%s, %v), want eager fallback", out, err)
		}
		if eng.eagres != 1 || p.Len() != 0 || eng.aborts != 0 {
			t.Fatalf("eagres=%d Len=%d aborts=%d, want 1/0/0", eng.eagres, p.Len(), eng.aborts)
		}
	})

	t.Run("failed body aborts and stores nothing", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 4, MinKeep: 1})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		out, err := p.Run("a", 1, func() error { return errGraphCacheBodyFailed })
		if !errors.Is(err, errGraphCacheBodyFailed) || out != GraphRunCaptureFailed {
			t.Fatalf("failed body = (%s, %v), want capture-failed with the body error", out, err)
		}
		if eng.aborts != 1 || p.Len() != 0 {
			t.Fatalf("aborts=%d Len=%d, want 1/0 (uninserted executable released)", eng.aborts, p.Len())
		}
	})

	t.Run("constructor sentinels", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		if _, err := NewGraphCachePolicy(nil, GraphCachePolicyOptions{On: true, Cap: 1}); !errors.Is(err, ErrGraphCacheNilEngine) {
			t.Fatalf("nil engine error = %v, want ErrGraphCacheNilEngine", err)
		}
		if _, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 0}); !errors.Is(err, ErrGraphCacheBadCap) {
			t.Fatalf("zero cap error = %v, want ErrGraphCacheBadCap", err)
		}
		if _, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 2, MinKeep: 2}); !errors.Is(err, ErrGraphCacheBadMinKeep) {
			t.Fatalf("min_keep==cap error = %v, want ErrGraphCacheBadMinKeep", err)
		}
		if _, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 2, MinKeep: -1}); !errors.Is(err, ErrGraphCacheBadMinKeep) {
			t.Fatalf("negative min_keep error = %v, want ErrGraphCacheBadMinKeep", err)
		}
	})
}
