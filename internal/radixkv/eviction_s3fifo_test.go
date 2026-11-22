package radixkv

import (
	"errors"
	"slices"
	"testing"
)

func newTestS3FIFO(t *testing.T, capacity uint64, rnd func() uint32) *S3FIFOPolicy {
	t.Helper()
	return newTestS3FIFOConfig(t, S3FIFOConfig{Capacity: capacity, Rand: rnd})
}

func newTestS3FIFOConfig(t *testing.T, cfg S3FIFOConfig) *S3FIFOPolicy {
	t.Helper()
	seed := uint64(0x5eed)
	cfg.Seed = &seed
	if cfg.Rand == nil {
		cfg.Rand = func() uint32 { return 1 }
	}
	p, err := NewS3FIFOPolicy(cfg)
	if err != nil {
		t.Fatalf("NewS3FIFOPolicy: %v", err)
	}
	return p
}

// checkS3FIFOInvariants recounts every queue and compares it with the tracked sizes.
func checkS3FIFOInvariants(t *testing.T, p *S3FIFOPolicy) {
	t.Helper()
	var total uint64
	count := 0
	sums := map[S3FIFOQueue]uint64{}
	for _, q := range []S3FIFOQueue{S3FIFOWindow, S3FIFOProbation, S3FIFOProtected} {
		d := p.queueFor(q)
		n := 0
		var prev *s3fifoNode
		for e := d.head; e != nil; e = e.next {
			if e.queue != q {
				t.Fatalf("key %d linked in %v but tagged %v", e.key, q, e.queue)
			}
			if e.prev != prev {
				t.Fatalf("broken prev link at key %d in %v", e.key, q)
			}
			if p.nodes[e.key] != e {
				t.Fatalf("key %d linked in %v but not in map", e.key, q)
			}
			prev = e
			sums[q] += e.weight
			n++
		}
		if d.tail != prev || d.n != n {
			t.Fatalf("%v: tail/len mismatch (len %d, walked %d)", q, d.n, n)
		}
		total += sums[q]
		count += n
	}
	if count != len(p.nodes) {
		t.Fatalf("linked %d entries, map has %d", count, len(p.nodes))
	}
	if total != p.weightedSize {
		t.Fatalf("weightedSize %d, recount %d", p.weightedSize, total)
	}
	if sums[S3FIFOWindow] != p.windowWeightedSize {
		t.Fatalf("windowWeightedSize %d, recount %d", p.windowWeightedSize, sums[S3FIFOWindow])
	}
	if sums[S3FIFOProtected] != p.mainProtectedWeightedSize {
		t.Fatalf("mainProtectedWeightedSize %d, recount %d", p.mainProtectedWeightedSize, sums[S3FIFOProtected])
	}
}

func TestS3FIFOEviction(t *testing.T) {
	t.Run("zero capacity is refused", func(t *testing.T) {
		if _, err := NewS3FIFOPolicy(S3FIFOConfig{}); !errors.Is(err, ErrS3FIFOZeroCapacity) {
			t.Fatalf("err = %v, want ErrS3FIFOZeroCapacity", err)
		}
	})

	t.Run("segment split follows the source percentages", func(t *testing.T) {
		p := newTestS3FIFO(t, 1000, nil)
		if p.windowMaximum != 10 || p.mainProtectedMaximum != 792 {
			t.Fatalf("window %d protected %d, want 10 and 792", p.windowMaximum, p.mainProtectedMaximum)
		}
	})

	t.Run("bounded by capacity and evicts exactly the overflow", func(t *testing.T) {
		p := newTestS3FIFO(t, 10, nil)
		for k := uint64(1); k <= 10; k++ {
			if ev := p.Add(k, 1); len(ev) != 0 {
				t.Fatalf("add %d under capacity evicted %v", k, ev)
			}
		}
		ev := p.Add(11, 1)
		if len(ev) != 1 || p.Len() != 10 || p.WeightedSize() != 10 {
			t.Fatalf("evicted %v, len %d, size %d", ev, p.Len(), p.WeightedSize())
		}
		if p.Contains(ev[0]) {
			t.Fatalf("evicted key %d still resident", ev[0])
		}
		checkS3FIFOInvariants(t, p)
	})

	t.Run("window overflow demotes to probation and an access promotes to protected", func(t *testing.T) {
		p := newTestS3FIFO(t, 100, nil) // window maximum 1
		p.Add(1, 1)
		if q := p.QueueOf(1); q != S3FIFOWindow {
			t.Fatalf("new key in %v, want window", q)
		}
		p.Add(2, 1)
		if q := p.QueueOf(1); q != S3FIFOProbation {
			t.Fatalf("window head after overflow in %v, want probation", q)
		}
		if q := p.QueueOf(2); q != S3FIFOWindow {
			t.Fatalf("newest key in %v, want window", q)
		}
		if !p.Access(1) {
			t.Fatal("resident key reported as a miss")
		}
		if q := p.QueueOf(1); q != S3FIFOProtected {
			t.Fatalf("accessed probation key in %v, want protected", q)
		}
		if p.Access(99) || p.Contains(99) {
			t.Fatal("miss must not admit the key")
		}
		if q := p.QueueOf(99); q != S3FIFONotResident || q.String() != "not-resident" {
			t.Fatalf("absent key queue %v", q)
		}
		checkS3FIFOInvariants(t, p)
	})

	t.Run("a one-shot scan cannot flush frequently used keys", func(t *testing.T) {
		p := newTestS3FIFO(t, 100, nil)
		for k := uint64(1); k <= 100; k++ {
			p.Add(k, 1)
		}
		for round := 0; round < 5; round++ {
			for k := uint64(1); k <= 100; k++ {
				p.Access(k)
			}
		}
		for k := uint64(1000); k < 3000; k++ {
			p.Add(k, 1)
			checkS3FIFOInvariants(t, p)
		}
		hot, scan := 0, 0
		for k := uint64(1); k <= 100; k++ {
			if p.Contains(k) {
				hot++
			}
		}
		for k := uint64(1000); k < 3000; k++ {
			if p.Contains(k) {
				scan++
			}
		}
		// Sketch aging during a scan 20x the capacity lets some scan keys win duels, but plain
		// LRU would keep none of the hot keys.
		if hot < 70 || hot+scan != p.Len() {
			t.Fatalf("after scan: %d hot keys, %d scan keys resident, window %d", hot, scan, p.windowMaximum)
		}
	})

	t.Run("hash-dos admission threshold", func(t *testing.T) {
		var r uint32
		p := newTestS3FIFO(t, 64, func() uint32 { return r })
		p.sketch.ensureCapacity(64)
		const warm, victim, cold = 7, 8, 9
		for i := 0; i < 10; i++ {
			p.sketch.increment(warm)
			p.sketch.increment(victim)
		}
		for i := 0; i < 3; i++ {
			p.sketch.increment(cold)
		}
		if p.Frequency(warm) < s3fifoAdmitHashdosThreshold || p.Frequency(warm) > p.Frequency(victim) {
			t.Fatalf("setup: warm %d victim %d", p.Frequency(warm), p.Frequency(victim))
		}
		r = 0
		if !p.admit(warm, victim) {
			t.Fatal("warm candidate must be randomly admittable when rand&127 == 0")
		}
		r = 1
		if p.admit(warm, victim) {
			t.Fatal("warm candidate admitted although rand&127 != 0")
		}
		r = 0
		if p.admit(cold, victim) {
			t.Fatal("candidate below the threshold admitted at random")
		}
		r = 1
		if !p.admit(victim, cold) {
			t.Fatal("a strictly more frequent candidate must always be admitted")
		}
	})

	t.Run("weights: oversized rejected, zero weight pinned, update reweighs", func(t *testing.T) {
		p := newTestS3FIFOConfig(t, S3FIFOConfig{Capacity: 10, Weighted: true})
		if ev := p.Add(1, 11); !slices.Equal(ev, []uint64{1}) || p.Contains(1) {
			t.Fatalf("oversized add evicted %v", ev)
		}
		p.Add(2, 0)
		for k := uint64(10); k < 40; k++ {
			p.Add(k, 1)
		}
		if !p.Contains(2) {
			t.Fatal("zero-weight entry was evicted for size")
		}
		p.Add(10, 5) // re-admits 10 if it already left, else reweighs it
		p.Add(39, 6)
		if p.WeightedSize() > p.Capacity() {
			t.Fatalf("size %d over capacity %d after reweigh", p.WeightedSize(), p.Capacity())
		}
		if ev := p.Add(39, 11); !slices.Contains(ev, 39) || p.Contains(39) {
			t.Fatalf("growing past capacity must evict the entry, got %v", ev)
		}
		checkS3FIFOInvariants(t, p)
	})

	t.Run("unweighted ignores weight", func(t *testing.T) {
		p := newTestS3FIFO(t, 10, nil)
		if ev := p.Add(1, 11); len(ev) != 0 || !p.Contains(1) || p.WeightedSize() != 1 {
			t.Fatalf("unweighted add must count 1, evicted %v size %d", ev, p.WeightedSize())
		}
		checkS3FIFOInvariants(t, p)
	})

	t.Run("weighted sketch is sized by resident entries, not weight capacity", func(t *testing.T) {
		p := newTestS3FIFOConfig(t, S3FIFOConfig{Capacity: 1 << 40, Weighted: true})
		for k := uint64(1); k <= 4; k++ {
			p.Add(k, 1<<31)
		}
		p.SetCapacity(1 << 41)
		if n := len(p.sketch.table); n > 64 {
			t.Fatalf("sketch table has %d words for 4 resident entries", n)
		}
		checkS3FIFOInvariants(t, p)
	})

	t.Run("remove is not an eviction", func(t *testing.T) {
		p := newTestS3FIFO(t, 10, nil)
		p.Add(1, 3)
		if !p.Remove(1) || p.Contains(1) || p.WeightedSize() != 0 {
			t.Fatal("remove did not drop the entry")
		}
		if p.Remove(1) {
			t.Fatal("second remove reported success")
		}
		checkS3FIFOInvariants(t, p)
	})

	t.Run("set capacity shrinks and ignores zero", func(t *testing.T) {
		p := newTestS3FIFO(t, 100, nil)
		for k := uint64(0); k < 100; k++ {
			p.Add(k, 1)
		}
		if ev := p.SetCapacity(0); len(ev) != 0 || p.Capacity() != 100 {
			t.Fatal("zero capacity must be ignored")
		}
		ev := p.SetCapacity(40)
		if len(ev) != 60 || p.Len() != 40 || p.Capacity() != 40 {
			t.Fatalf("shrink evicted %d, len %d", len(ev), p.Len())
		}
		checkS3FIFOInvariants(t, p)
	})

	t.Run("hill climber moves quota between window and protected", func(t *testing.T) {
		p := newTestS3FIFO(t, 1000, nil)
		for k := uint64(0); k < 1000; k++ {
			p.Add(k, 1)
		}
		budget := p.windowMaximum + p.mainProtectedMaximum
		sample := p.sketch.sampleSize

		// First sample: a rise in hit rate keeps the initial (shrinking) direction.
		p.hitsInSample, p.missesInSample = sample*8/10, sample*2/10
		p.climb()
		if p.windowMaximum != 1 {
			t.Fatalf("after shrink: window %d adjustment %d", p.windowMaximum, p.adjustment)
		}
		if p.windowMaximum+p.mainProtectedMaximum != budget {
			t.Fatal("shrink changed the window+protected budget")
		}
		checkS3FIFOInvariants(t, p)

		// A drop in hit rate reverses direction and grows the window by one step.
		p.hitsInSample, p.missesInSample = sample/2, sample/2
		p.climb()
		if p.windowMaximum != 63 {
			t.Fatalf("after grow: window %d adjustment %d", p.windowMaximum, p.adjustment)
		}
		if p.windowMaximum+p.mainProtectedMaximum != budget {
			t.Fatal("grow changed the window+protected budget")
		}
		if p.windowWeightedSize != p.windowMaximum {
			t.Fatalf("window holds %d, want filled to %d", p.windowWeightedSize, p.windowMaximum)
		}
		checkS3FIFOInvariants(t, p)

		// An undersized sample does not move the split.
		w := p.windowMaximum
		p.hitsInSample, p.missesInSample = 1, 1
		p.climb()
		if p.windowMaximum != w {
			t.Fatal("climber adapted on a partial sample")
		}
	})

	t.Run("climber is idle below half capacity", func(t *testing.T) {
		p := newTestS3FIFO(t, 1000, nil)
		p.sketch.ensureCapacity(1000)
		p.Add(1, 1)
		p.hitsInSample, p.missesInSample = p.sketch.sampleSize, 0
		p.climb()
		if p.windowMaximum != 10 || p.hitsInSample != 0 {
			t.Fatalf("window %d hits %d: sample must be discarded", p.windowMaximum, p.hitsInSample)
		}
	})

	t.Run("deterministic for a fixed seed", func(t *testing.T) {
		run := func() []uint64 {
			p := newTestS3FIFO(t, 50, nil)
			var all []uint64
			for i := uint64(0); i < 2000; i++ {
				k := (i * 2654435761) % 173
				if !p.Access(k) {
					all = append(all, p.Add(k, 1)...)
				}
			}
			checkS3FIFOInvariants(t, p)
			return all
		}
		a, b := run(), run()
		if len(a) == 0 || !slices.Equal(a, b) {
			t.Fatalf("eviction sequences differ or empty (%d vs %d)", len(a), len(b))
		}
	})
}
