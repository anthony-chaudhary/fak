package model

import (
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

// ep_decode_coord_test.go — the CPU regression witness for #4835's coordinated EP decode
// (ep_decode_coord.go). It proves the rank-0-rooted, token-broadcast protocol produces
// EXACTLY the tokens the scalar pure-fak monolith produces, while the followers own no
// tokenization/sampling/history — i.e. the coordinated path is a drop-in for the redundant
// HTTP mirror, minus the N-way redundant work. The end-to-end tok/s gain is GPU-gated and
// witnessed separately on the 8-GPU checkpoint; this gate discharges DoD item 3 (token
// agreement vs the scalar reference for representative routed experts) on pure CPU.

func TestEPDecodeFrameRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		op   epDecodeOp
		pos  int
		ids  []int
	}{
		{"prefill", epOpPrefill, 0, []int{1, 2, 3, 1 << 20}},
		{"prefill-empty", epOpPrefill, 0, nil},
		{"prefill-resumed", epOpPrefill, 4096, []int{5}},
		{"step", epOpStep, 29, []int{7}},
		{"shutdown", epOpShutdown, 0, nil},
	}
	for _, c := range cases {
		op, pos, ids, err := decodeEPDecodeFrame(encodeEPDecodeFrame(c.op, c.pos, c.ids))
		if err != nil {
			t.Fatalf("%s: decode round-trip: %v", c.name, err)
		}
		if op != c.op {
			t.Fatalf("%s: op = %d, want %d", c.name, op, c.op)
		}
		if pos != c.pos {
			t.Fatalf("%s: pos = %d, want %d", c.name, pos, c.pos)
		}
		if len(ids) != len(c.ids) {
			t.Fatalf("%s: got %d ids, want %d", c.name, len(ids), len(c.ids))
		}
		for i := range ids {
			if ids[i] != c.ids[i] {
				t.Fatalf("%s: id[%d] = %d, want %d", c.name, i, ids[i], c.ids[i])
			}
		}
	}

	// Fail-closed: truncation, an op/count mismatch, an unknown op, and a negative position
	// must all be rejected — a desynced follower must error, never mis-drive its forward.
	if _, _, _, err := decodeEPDecodeFrame([]byte{byte(epOpStep), 2, 0}); err == nil {
		t.Fatal("truncated frame was not rejected")
	}
	if _, _, _, err := decodeEPDecodeFrame(append(encodeEPDecodeFrame(epOpStep, 0, []int{9}), 0, 0, 0, 0)); err == nil {
		t.Fatal("STEP frame with a trailing id was not rejected")
	}
	if _, _, _, err := decodeEPDecodeFrame(encodeEPDecodeFrame(epOpShutdown, 0, []int{1})); err == nil {
		t.Fatal("SHUTDOWN frame with an id was not rejected")
	}
	if _, _, _, err := decodeEPDecodeFrame([]byte{0x7f, 0, 0, 0, 0, 0, 0, 0, 0}); err == nil {
		t.Fatal("unknown op was not rejected")
	}
	if _, _, _, err := decodeEPDecodeFrame(encodeEPDecodeFrame(epOpPrefill, -1, []int{1})); err == nil {
		t.Fatal("negative position was not rejected")
	}
}

// runEPGroupTimeout is runGroup (dist_collective_test.go) with a wall-clock guard so a
// protocol desync surfaces as a test failure instead of hanging the suite on a follower
// stuck in BroadcastFromRoot.
func runEPGroupTimeout(t *testing.T, size int, timeout time.Duration, fn func(g *DistComm) ([]float32, error)) ([][]float32, []error) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	addr := ln.Addr().String()

	results := make([][]float32, size)
	errs := make([]error, size)
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		g, cerr := Coordinate(ln, size)
		if cerr != nil {
			errs[0] = cerr
			return
		}
		defer g.Close()
		results[0], errs[0] = fn(g)
	}()
	for r := 1; r < size; r++ {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			conn, derr := net.Dial("tcp", addr)
			if derr != nil {
				errs[rank] = derr
				return
			}
			g, jerr := Join(conn, rank, size)
			if jerr != nil {
				conn.Close()
				errs[rank] = jerr
				return
			}
			defer g.Close()
			results[rank], errs[rank] = fn(g)
		}(r)
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		t.Fatalf("coordinated EP decode did not complete within %s — a rank is deadlocked on a broadcast/collective", timeout)
	}
	return results, errs
}

func packToks(sets ...[]int) []float32 {
	var out []float32
	for _, s := range sets {
		out = append(out, float32(len(s)))
		for _, id := range s {
			out = append(out, float32(id))
		}
	}
	return out
}

func unpackToks(v []float32) [][]int {
	var sets [][]int
	for i := 0; i < len(v); {
		n := int(v[i])
		i++
		s := make([]int, n)
		for j := 0; j < n; j++ {
			s[j] = int(v[i])
			i++
		}
		sets = append(sets, s)
	}
	return sets
}

func TestEPCoordinatedDecodeMatchesScalarReference(t *testing.T) {
	full, cfg, _, _, _ := glmFixtureRoutedLayer(t)
	if cfg.NumExperts < 2 {
		t.Skipf("fixture NumExperts=%d has no multi-rank sharding", cfg.NumExperts)
	}

	// Two representative requests from FRESH sessions — the second pins the multi-request
	// protocol reset (a fresh PREFILL frame after the first decode drains).
	prompt1 := []int{1, 2, 3}
	prompt2 := []int{2, 1}
	const n = 6

	// Scalar pure-fak reference: the monolith (epRanks unset => glmMoeFFN) greedy-decodes.
	want1 := full.NewSession().Generate(prompt1, n)
	want2 := full.NewSession().Generate(prompt2, n)

	for ranks := 2; ranks <= cfg.NumExperts; ranks++ {
		plan, err := ExpertParallelPlan(cfg.NumExperts, ranks)
		if err != nil {
			t.Fatalf("ExpertParallelPlan(%d): %v", ranks, err)
		}
		results, errs := runEPGroupTimeout(t, ranks, 30*time.Second, func(g *DistComm) ([]float32, error) {
			s := plan.Shards[g.Rank()]
			local := modelWithExpertBandForTest(full, s.Lo, s.Hi)
			local.SetExpertParallelRanks(ranks)
			local.SetExpertParallelRank(g.Rank())
			local.SetExpertParallelCollective(NewDistCommCollective(g))
			if g.Rank() == 0 {
				// Rank 0 installs the coordinator and then decodes through the ORDINARY
				// Session.Generate loop — no bespoke EP driver. That is the whole point of the
				// Prefill/Step seam: the live serve's real decode loop is what drives the group.
				coord, err := NewEPDecodeCoordinator(g)
				if err != nil {
					return nil, err
				}
				local.SetEPDecodeCoordinator(coord)
				t1 := local.NewSession().Generate(prompt1, n)
				t2 := local.NewSession().Generate(prompt2, n)
				if err := coord.Shutdown(); err != nil {
					return nil, err
				}
				return packToks(t1, t2), nil
			}
			// Followers contribute only local expert work + collectives, then exit on SHUTDOWN.
			if err := RunEPFollower(g, local.NewSession); err != nil {
				return nil, err
			}
			return nil, nil
		})
		for r, err := range errs {
			if err != nil {
				t.Fatalf("ranks=%d rank %d: %v", ranks, r, err)
			}
		}
		got := unpackToks(results[0])
		if len(got) != 2 {
			t.Fatalf("ranks=%d: rank 0 returned %d token sets, want 2", ranks, len(got))
		}
		assertToksEqual(t, ranks, "request 1", got[0], want1)
		assertToksEqual(t, ranks, "request 2", got[1], want2)
		t.Logf("coordinated EP decode ranks=%d: rank-0-owned sampling reproduces the scalar reference exactly, followers owned no tokenization/sampling", ranks)
	}
}

// TestEPFollowerRefusesMirrorDesync pins the one failure this protocol must never take
// silently. If rank 0 resumes a session from a restored prefix-cache snapshot it prefills only
// the divergent SUFFIX, at a position the follower's fresh mirror never computed. The follower
// would then feed a hidden state built from a different context into the same per-layer
// AllReduce, and the reduce would sum a garbage partial into a plausible-looking answer. The
// follower must refuse instead — this is the guard the serve wiring depends on.
func TestEPFollowerRefusesMirrorDesync(t *testing.T) {
	full, cfg, _, _, _ := glmFixtureRoutedLayer(t)
	if cfg.NumExperts < 2 {
		t.Skipf("fixture NumExperts=%d has no multi-rank sharding", cfg.NumExperts)
	}
	const ranks = 2
	plan, err := ExpertParallelPlan(cfg.NumExperts, ranks)
	if err != nil {
		t.Fatalf("ExpertParallelPlan(%d): %v", ranks, err)
	}

	// Each case optionally runs ONE real coordinated prefill first — that opens a live mirror
	// session on the follower AND supplies rank 0's own half of its collectives — and then emits
	// the desynced frame the follower must refuse. Without the real prefill the refusal would
	// only prove the trivial "no mirror session yet" branch.
	cases := []struct {
		name      string
		openFirst bool
		op        epDecodeOp
		pos       int
		ids       []int
		want      string
	}{
		// Rank 0 resumed a session from a restored prefix-cache snapshot and prefills only the
		// divergent suffix, at a position the follower's mirror never computed.
		{"suffix-prefill-after-cache-hit", true, epOpPrefill, 9, []int{4, 5}, "mirror desync"},
		// Rank 0's step position ran ahead of the mirror (a forward it never announced).
		{"step-ahead-of-mirror", true, epOpStep, 7, []int{4}, "mirror desync"},
		// A step before any prefill opened a mirror session at all.
		{"step-before-prefill", false, epOpStep, 0, []int{3}, "before any PREFILL"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, errs := runEPGroupTimeout(t, ranks, 30*time.Second, func(g *DistComm) ([]float32, error) {
				s := plan.Shards[g.Rank()]
				local := modelWithExpertBandForTest(full, s.Lo, s.Hi)
				local.SetExpertParallelRanks(ranks)
				local.SetExpertParallelRank(g.Rank())
				local.SetExpertParallelCollective(NewDistCommCollective(g))
				if g.Rank() != 0 {
					return nil, RunEPFollower(g, local.NewSession)
				}
				if c.openFirst {
					// A real coordinated prefill: announces PREFILL at pos 0 and runs rank 0's
					// own forward, so the follower's mirrored forward finds its AllReduce peer.
					coord, err := NewEPDecodeCoordinator(g)
					if err != nil {
						return nil, err
					}
					local.SetEPDecodeCoordinator(coord)
					local.NewSession().Prefill([]int{1, 2, 3})
					local.SetEPDecodeCoordinator(nil)
				}
				// Now the desynced frame, broadcast directly — the shape rank 0 emits when its
				// session resumed from a prefix cache the follower never saw.
				if _, err := g.BroadcastFromRoot(encodeEPDecodeFrame(c.op, c.pos, c.ids)); err != nil {
					return nil, nil // the follower already bailed; that is the expected outcome
				}
				return nil, nil
			})
			if errs[0] != nil {
				t.Fatalf("rank 0: %v", errs[0])
			}
			if errs[1] == nil {
				t.Fatalf("follower accepted the desynced %s frame instead of failing closed — it would have reduced a partial computed from a different context", c.name)
			}
			if !strings.Contains(errs[1].Error(), c.want) {
				t.Fatalf("follower refused for the wrong reason: got %q, want it to contain %q", errs[1], c.want)
			}
			t.Logf("follower failed closed as required: %v", errs[1])
		})
	}
}

func assertToksEqual(t *testing.T, ranks int, label string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("ranks=%d %s: got %d tokens %v, want %d %v", ranks, label, len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ranks=%d %s: token[%d]=%d, want %d (coordinated decode diverged from the scalar reference)\n got=%v\nwant=%v", ranks, label, i, got[i], want[i], got, want)
		}
	}
}
