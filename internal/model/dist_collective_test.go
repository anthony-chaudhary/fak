package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"net"
	"sync"
	"testing"
)

// dist_collective_test.go — the gates for DistComm (dist_collective.go), the first REAL
// cross-process collective. They are the distributed-collective analog of
// TestTCPTransportMatchesLocal: run the ranks as goroutines over a loopback TCP socket —
// each rank holding ONLY its own part — and prove the result every rank ends with is
// byte-for-byte (max|Δ|=0) the in-process LocalCollective / sumPartialsRankOrder result.
// That is the whole point of the seam: a real NCCL/RDMA collective swapped in later for
// the wire is correct exactly when it reproduces these bytes. The fail-closed gates prove
// a ragged reduce, a mis-width gather, and a process-group op desync are refused on EVERY
// rank without deadlocking a peer on a response that never comes.
//
// (bridgeRandVec lives in collective_bridge_test.go — same package, reused here.)

// distAssertExact fails unless got == want bit-for-bit (the max|Δ|=0 bar a reorder, drop,
// or double-count of a rank would break).
func distAssertExact(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len = %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s [%d] = %v, want %v (not bit-exact vs the in-process collective)", label, i, got[i], want[i])
		}
	}
}

// runGroup spins up a size-rank DistComm over a loopback TCP listener, runs fn on every
// rank CONCURRENTLY (rank 0 coordinates; ranks 1.. dial in and Join), and returns each
// rank's (result, error) in rank order. It mirrors TestTCPTransportMatchesLocal's loopback
// idiom — a genuine cross-process exchange on one box. t.Fatalf is never called from a
// spawned goroutine; failures are returned and asserted on the test goroutine.
func runGroup(t *testing.T, size int, fn func(g *DistComm) ([]float32, error)) ([][]float32, []error) {
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
	wg.Wait()
	return results, errs
}

// TestDistCommAllReduceSumMatchesLocal pins the cross-process all-reduce byte-for-byte
// against LocalCollective over several rank counts, each rank holding only its own part.
// A reduction that reordered, dropped, or double-counted a rank over the wire would be
// caught at max|Δ|=0. size=1 exercises the no-wire identity.
func TestDistCommAllReduceSumMatchesLocal(t *testing.T) {
	rng := rand.New(rand.NewSource(7411))
	for _, size := range []int{1, 2, 3, 5} {
		const n = 17 // a non-round length exercises fdot's tail
		parts := make([][]float32, size)
		for r := range parts {
			parts[r] = bridgeRandVec(rng, n)
		}
		want, err := LocalCollective{}.AllReduceSum(parts)
		if err != nil {
			t.Fatalf("local AllReduceSum size=%d: %v", size, err)
		}
		results, errs := runGroup(t, size, func(g *DistComm) ([]float32, error) {
			return g.AllReduceSum(parts[g.Rank()])
		})
		for r := 0; r < size; r++ {
			if errs[r] != nil {
				t.Fatalf("size=%d rank %d AllReduceSum: %v", size, r, errs[r])
			}
			distAssertExact(t, fmt.Sprintf("AllReduceSum size=%d rank=%d", size, r), results[r], want)
		}
	}
}

// TestDistCommAllGatherMatchesLocal pins the cross-process all-gather byte-for-byte against
// LocalCollective, using a real NewTPPlan so the per-rank shard widths are uneven (the
// gather's purpose). Each rank holds only its shard; every rank ends with the full
// rank-ordered concatenation.
func TestDistCommAllGatherMatchesLocal(t *testing.T) {
	rng := rand.New(rand.NewSource(919))
	for _, size := range []int{1, 2, 3, 5} {
		dim := 6*size + 1 // > size and rarely divisible, so shards are uneven
		plan, err := NewTPPlan(dim, size)
		if err != nil {
			t.Fatalf("NewTPPlan(%d,%d): %v", dim, size, err)
		}
		parts := make([][]float32, size)
		for r, s := range plan.Shards {
			parts[r] = bridgeRandVec(rng, s.Width())
		}
		want, err := LocalCollective{}.AllGather(parts, plan)
		if err != nil {
			t.Fatalf("local AllGather size=%d: %v", size, err)
		}
		results, errs := runGroup(t, size, func(g *DistComm) ([]float32, error) {
			return g.AllGather(parts[g.Rank()], plan)
		})
		for r := 0; r < size; r++ {
			if errs[r] != nil {
				t.Fatalf("size=%d rank %d AllGather: %v", size, r, errs[r])
			}
			distAssertExact(t, fmt.Sprintf("AllGather size=%d rank=%d", size, r), results[r], want)
		}
	}
}

// TestDistCommAllReduceEqualsRankOrderSpec pins the cross-process reduce against
// sumPartialsRankOrder directly — the FIXED rank-order spec, independent of LocalCollective.
// This is the invariant the row-parallel TP gate depends on: the wire must not perturb the
// reduction order.
func TestDistCommAllReduceEqualsRankOrderSpec(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const size, n = 4, 9
	parts := make([][]float32, size)
	for r := range parts {
		parts[r] = bridgeRandVec(rng, n)
	}
	want := sumPartialsRankOrder(parts)
	results, errs := runGroup(t, size, func(g *DistComm) ([]float32, error) {
		return g.AllReduceSum(parts[g.Rank()])
	})
	for r := 0; r < size; r++ {
		if errs[r] != nil {
			t.Fatalf("rank %d AllReduceSum: %v", r, errs[r])
		}
		distAssertExact(t, fmt.Sprintf("rank=%d vs sumPartialsRankOrder", r), results[r], want)
	}
}

// TestDistCommFailsClosedRaggedAllReduce proves ragged per-rank parts are refused on EVERY
// rank (the coordinator's LocalCollective reduce rejects them and broadcasts the refusal),
// not silently truncated — and that no worker deadlocks waiting on a response.
func TestDistCommFailsClosedRaggedAllReduce(t *testing.T) {
	const size = 3
	_, errs := runGroup(t, size, func(g *DistComm) ([]float32, error) {
		return g.AllReduceSum(make([]float32, g.Rank()+2)) // lengths 2,3,4 — ragged
	})
	for r := 0; r < size; r++ {
		if errs[r] == nil {
			t.Fatalf("rank %d: ragged AllReduceSum should fail closed, got nil error", r)
		}
	}
}

// TestDistCommFailsClosedMisWidthAllGather proves a gather shard whose width disagrees with
// the plan is refused on every rank, with the same fail-closed contract LocalCollective has.
func TestDistCommFailsClosedMisWidthAllGather(t *testing.T) {
	const size = 3
	plan, err := NewTPPlan(9, size) // shard widths 3,3,3
	if err != nil {
		t.Fatalf("NewTPPlan: %v", err)
	}
	_, errs := runGroup(t, size, func(g *DistComm) ([]float32, error) {
		w := plan.Shards[g.Rank()].Width()
		if g.Rank() == size-1 {
			w++ // last rank sends one element too many
		}
		return g.AllGather(make([]float32, w), plan)
	})
	for r := 0; r < size; r++ {
		if errs[r] == nil {
			t.Fatalf("rank %d: mis-width AllGather should fail closed, got nil error", r)
		}
	}
}

// TestDistCommFailsClosedOpDesync proves the coordinator refuses a process-group desync —
// a peer that calls a DIFFERENT collective in the same round — on every rank rather than
// reducing mismatched buffers. Rank 0 runs AllReduceSum; rank 1 runs AllGather.
func TestDistCommFailsClosedOpDesync(t *testing.T) {
	const size = 2
	plan, err := NewTPPlan(3, 1)
	if err != nil {
		t.Fatalf("NewTPPlan: %v", err)
	}
	_, errs := runGroup(t, size, func(g *DistComm) ([]float32, error) {
		if g.Rank() == 0 {
			return g.AllReduceSum([]float32{1, 2, 3})
		}
		return g.AllGather([]float32{4, 5, 6}, plan)
	})
	for r := 0; r < size; r++ {
		if errs[r] == nil {
			t.Fatalf("rank %d: op desync should fail closed, got nil error", r)
		}
	}
}

// distAssertBitsEqual compares via math.Float32bits so a NaN payload (NaN != NaN under ==)
// and a signed zero (-0.0 == +0.0 under ==) are still held to bit-for-bit equality — the
// exactness the random-payload gates' == comparison cannot enforce for those patterns.
func distAssertBitsEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len = %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s [%d]: bits %#x != %#x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// TestDistCommF32CodecPreservesBits pins the wire-format comment's "preserves signed zero
// and NaN and cannot perturb a single bit" claim with a max|Δ|=0 gate, over the adversarial
// bit patterns the byte-exact gates' random [-1,1) payloads never reach: ±0.0, ±Inf,
// MaxFloat32, the smallest denormal, and several NaN payloads. It covers the codec in
// isolation (incl. the empty vector) AND end-to-end through a 2-rank AllGather — the full
// writeRequest/readRequest/writeResponse/readResponse frame path — where rank-ordered
// concatenation must reproduce every rank's bytes verbatim.
func TestDistCommF32CodecPreservesBits(t *testing.T) {
	adversarial := []float32{
		0,
		float32(math.Copysign(0, -1)),    // -0.0
		float32(math.Inf(1)),             // +Inf
		float32(math.Inf(-1)),            // -Inf
		math.MaxFloat32,                  // largest finite
		math.SmallestNonzeroFloat32,      // smallest denormal
		math.Float32frombits(0x7fc00000), // quiet NaN
		math.Float32frombits(0x7fa00000), // NaN, non-canonical payload
		math.Float32frombits(0xffc00001), // negative NaN with payload
	}

	// (1) codec round-trip in isolation, plus the empty vector.
	for _, v := range [][]float32{adversarial, {}} {
		got, used, err := decodeF32(encodeF32(v))
		if err != nil {
			t.Fatalf("decodeF32(encodeF32(len=%d)): %v", len(v), err)
		}
		if want := 4 + len(v)*4; used != want {
			t.Fatalf("decodeF32 consumed %d bytes, want %d", used, want)
		}
		distAssertBitsEqual(t, fmt.Sprintf("codec round-trip len=%d", len(v)), got, v)
	}

	// (2) end-to-end through a 2-rank AllGather over the real wire: each rank holds its
	// plan-shard of the adversarial vector; the concatenation must reproduce every bit.
	plan, err := NewTPPlan(len(adversarial), 2)
	if err != nil {
		t.Fatalf("NewTPPlan: %v", err)
	}
	w0 := plan.Shards[0].Width()
	parts := [][]float32{adversarial[:w0], adversarial[w0:]}
	want, err := LocalCollective{}.AllGather(parts, plan)
	if err != nil {
		t.Fatalf("local AllGather: %v", err)
	}
	results, errs := runGroup(t, 2, func(g *DistComm) ([]float32, error) {
		return g.AllGather(parts[g.Rank()], plan)
	})
	for r := 0; r < 2; r++ {
		if errs[r] != nil {
			t.Fatalf("rank %d AllGather adversarial: %v", r, errs[r])
		}
		distAssertBitsEqual(t, fmt.Sprintf("AllGather adversarial rank=%d", r), results[r], want)
	}
}

// TestDistCommConstructionFailsClosed pins the construction contract: a bad size at the
// coordinator and a bad rank/size/nil at a worker are rejected before any wire I/O.
func TestDistCommConstructionFailsClosed(t *testing.T) {
	if _, err := Coordinate(nil, 0); err == nil {
		t.Fatal("Coordinate(size=0) should fail closed")
	}
	c1, c2 := net.Pipe()
	defer c1.Close()
	defer c2.Close()
	for _, tc := range []struct {
		name       string
		rank, size int
		conn       net.Conn
	}{
		{"rank 0 is the coordinator", 0, 2, c1},
		{"size 1 has no workers", 1, 1, c1},
		{"rank >= size", 3, 2, c1},
		{"nil connection", 1, 2, nil},
	} {
		if _, err := Join(tc.conn, tc.rank, tc.size); err == nil {
			t.Fatalf("Join(%s) should fail closed", tc.name)
		}
	}
}

// TestDistCommCoordinateRejectsBadAnnounce proves the coordinator refuses a worker that
// announces an out-of-range rank (a corrupt/buggy peer), closing it rather than indexing
// out of bounds.
func TestDistCommCoordinateRejectsBadAnnounce(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	go func() {
		conn, derr := net.Dial("tcp", ln.Addr().String())
		if derr != nil {
			return
		}
		defer conn.Close()
		var hb [4]byte
		binary.LittleEndian.PutUint32(hb[:], 5) // out of range for a size-2 group
		conn.Write(hb[:])
		readFrame(conn) // block until the coordinator closes us
	}()
	if _, err := Coordinate(ln, 2); err == nil {
		t.Fatal("Coordinate should reject an out-of-range announced rank")
	}
}
