package model

import (
	"net"
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
		ids  []int
	}{
		{"prefill", epOpPrefill, []int{1, 2, 3, 1 << 20}},
		{"prefill-empty", epOpPrefill, nil},
		{"step", epOpStep, []int{7}},
		{"shutdown", epOpShutdown, nil},
	}
	for _, c := range cases {
		op, ids, err := decodeEPDecodeFrame(encodeEPDecodeFrame(c.op, c.ids))
		if err != nil {
			t.Fatalf("%s: decode round-trip: %v", c.name, err)
		}
		if op != c.op {
			t.Fatalf("%s: op = %d, want %d", c.name, op, c.op)
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

	// Fail-closed: truncation, an op/count mismatch, and an unknown op must all be rejected.
	if _, _, err := decodeEPDecodeFrame([]byte{byte(epOpStep), 2, 0}); err == nil {
		t.Fatal("truncated frame was not rejected")
	}
	if _, _, err := decodeEPDecodeFrame([]byte{byte(epOpStep), 2, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0}); err == nil {
		t.Fatal("STEP frame with count=2 was not rejected")
	}
	if _, _, err := decodeEPDecodeFrame([]byte{0x7f, 0, 0, 0, 0}); err == nil {
		t.Fatal("unknown op was not rejected")
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
				t1, err := epFrontDecode(g, local.NewSession(), prompt1, n)
				if err != nil {
					return nil, err
				}
				t2, err := epFrontDecode(g, local.NewSession(), prompt2, n)
				if err != nil {
					return nil, err
				}
				if err := epShutdownFollowers(g); err != nil {
					return nil, err
				}
				return packToks(t1, t2), nil
			}
			// Followers contribute only local expert work + collectives, then exit on SHUTDOWN.
			if err := epFollowerDecode(g, local); err != nil {
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
