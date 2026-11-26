package radixkv

import "testing"

func tinyLFUKey(i uint64) uint64 {
	z := i*0x9e3779b97f4a7c15 + 0x632be59bd9b4e5d
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func tinyLFUBump(a *TinyLFUAdmission, key uint64, n int) {
	for i := 0; i < n; i++ {
		a.Increment(key)
	}
}

func TestTinyLFUAdmission(t *testing.T) {
	t.Run("doorkeeper absorbs first sighting", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		k := tinyLFUKey(1)
		if got := a.Estimate(k); got != 0 {
			t.Fatalf("unseen estimate = %d, want 0", got)
		}
		a.Increment(k)
		if got := a.freq.estimate(k); got != 0 {
			t.Fatalf("sketch after first sighting = %d, want 0 (doorkeeper only)", got)
		}
		if got := a.Estimate(k); got != 1 {
			t.Fatalf("estimate after one = %d, want 1", got)
		}
		a.Increment(k)
		if got := a.Estimate(k); got != 2 {
			t.Fatalf("estimate after two = %d, want 2", got)
		}
		if got := a.Estimate(tinyLFUKey(2)); got != 0 {
			t.Fatalf("other key estimate = %d, want 0", got)
		}
	})

	t.Run("counters saturate at fifteen", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		k := tinyLFUKey(3)
		tinyLFUBump(a, k, 40)
		if got := a.Estimate(k); got != 16 {
			t.Fatalf("saturated estimate = %d, want 16 (15 + doorkeeper)", got)
		}
	})

	t.Run("reset halves counters and clears doorkeeper", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		k := tinyLFUKey(4)
		tinyLFUBump(a, k, 9)
		if got := a.Estimate(k); got != 9 {
			t.Fatalf("estimate = %d, want 9", got)
		}
		a.Reset()
		if got := a.Estimate(k); got != 4 {
			t.Fatalf("after reset = %d, want 4 (8 halved, doorkeeper cleared)", got)
		}
		odd := NewTinyLFUAdmission(1024, 1)
		tinyLFUBump(odd, k, 16) // sketch saturates at 15
		odd.Reset()
		if got := odd.Estimate(k); got != 7 {
			t.Fatalf("15 halved = %d, want 7 (no bleed across nibbles)", got)
		}
	})

	t.Run("periodic reset fires after numCounters increments", func(t *testing.T) {
		a := NewTinyLFUAdmission(16, 1)
		k := tinyLFUKey(5)
		tinyLFUBump(a, k, 15)
		if a.incrs != 15 || a.Estimate(k) != 15 {
			t.Fatalf("before reset incrs=%d est=%d, want 15/15", a.incrs, a.Estimate(k))
		}
		a.Increment(k)
		if a.incrs != 0 {
			t.Fatalf("incrs after 16th increment = %d, want 0", a.incrs)
		}
		if got := a.Estimate(k); got != 7 {
			t.Fatalf("after periodic reset = %d, want 7", got)
		}
	})

	t.Run("clear zeroes everything", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		k := tinyLFUKey(6)
		a.Push([]uint64{k, k, k})
		a.Clear()
		if got := a.Estimate(k); got != 0 || a.incrs != 0 {
			t.Fatalf("after clear est=%d incrs=%d, want 0/0", got, a.incrs)
		}
	})

	t.Run("seeded sketch replays", func(t *testing.T) {
		a, b := NewTinyLFUAdmission(64, 7), NewTinyLFUAdmission(64, 7)
		for i := uint64(0); i < 40; i++ {
			a.Increment(tinyLFUKey(i % 9))
			b.Increment(tinyLFUKey(i % 9))
		}
		for i := uint64(0); i < 12; i++ {
			if a.Estimate(tinyLFUKey(i)) != b.Estimate(tinyLFUKey(i)) {
				t.Fatalf("same seed diverged on key %d", i)
			}
		}
	})

	t.Run("tiny and invalid sizes", func(t *testing.T) {
		a := NewTinyLFUAdmission(1, 1)
		a.Increment(tinyLFUKey(1))
		a.Increment(tinyLFUKey(1)) // hits the reset path every increment
		defer func() {
			if recover() == nil {
				t.Fatal("numCounters=0 did not panic")
			}
		}()
		NewTinyLFUAdmission(0, 1)
	})

	residents := func(keys ...uint64) []TinyLFUCandidate {
		out := make([]TinyLFUCandidate, len(keys))
		for i, k := range keys {
			out[i] = TinyLFUCandidate{Key: k, Cost: 1}
		}
		return out
	}

	t.Run("admit without pressure", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		victims, ok := a.Admit(tinyLFUKey(10), 2, 8, 10, residents(tinyLFUKey(11)))
		if !ok || len(victims) != 0 {
			t.Fatalf("fit: ok=%v victims=%v, want admitted with none", ok, victims)
		}
	})

	t.Run("reject oversized item", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		if victims, ok := a.Admit(tinyLFUKey(10), 11, 0, 10, nil); ok || victims != nil {
			t.Fatalf("oversized: ok=%v victims=%v, want rejected", ok, victims)
		}
	})

	t.Run("hot incoming evicts coldest sampled", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		in := tinyLFUKey(20)
		hot, cold := tinyLFUKey(21), tinyLFUKey(22)
		tinyLFUBump(a, in, 6)
		tinyLFUBump(a, hot, 10)
		tinyLFUBump(a, cold, 1)
		victims, ok := a.Admit(in, 1, 2, 2, residents(hot, cold))
		if !ok || len(victims) != 1 || victims[0].Key != cold {
			t.Fatalf("ok=%v victims=%v, want admitted evicting only cold", ok, victims)
		}
	})

	t.Run("cold incoming rejected by hotter sample", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		in := tinyLFUKey(30)
		r1, r2 := tinyLFUKey(31), tinyLFUKey(32)
		tinyLFUBump(a, in, 1)
		tinyLFUBump(a, r1, 3)
		tinyLFUBump(a, r2, 4)
		victims, ok := a.Admit(in, 1, 2, 2, residents(r1, r2))
		if ok || len(victims) != 0 {
			t.Fatalf("ok=%v victims=%v, want rejected with none", ok, victims)
		}
	})

	t.Run("tie admits", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		in, r := tinyLFUKey(40), tinyLFUKey(41)
		tinyLFUBump(a, in, 3)
		tinyLFUBump(a, r, 3)
		victims, ok := a.Admit(in, 1, 1, 1, residents(r))
		if !ok || len(victims) != 1 {
			t.Fatalf("tie: ok=%v victims=%v, want admitted (incoming >= victim)", ok, victims)
		}
	})

	t.Run("partial eviction then reject keeps victims", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		in := tinyLFUKey(50)
		cold, hot := tinyLFUKey(51), tinyLFUKey(52)
		tinyLFUBump(a, in, 4)
		tinyLFUBump(a, cold, 1)
		tinyLFUBump(a, hot, 9)
		victims, ok := a.Admit(in, 2, 2, 2, residents(cold, hot))
		if ok || len(victims) != 1 || victims[0].Key != cold {
			t.Fatalf("ok=%v victims=%v, want rejected after evicting cold", ok, victims)
		}
	})

	t.Run("sample window refills past five", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		in := tinyLFUKey(60)
		tinyLFUBump(a, in, 8)
		var keys []uint64
		for i := uint64(0); i < 7; i++ {
			k := tinyLFUKey(61 + i)
			tinyLFUBump(a, k, 2)
			keys = append(keys, k)
		}
		victims, ok := a.Admit(in, 7, 7, 7, residents(keys...))
		if !ok {
			t.Fatalf("ok=%v victims=%v, want admitted after evicting every resident", ok, victims)
		}
		evicted := map[uint64]bool{}
		for _, v := range victims {
			evicted[v.Key] = true
		}
		for _, k := range keys {
			if !evicted[k] {
				t.Fatalf("resident %d survived admission; victims=%v", k, victims)
			}
		}
		if len(evicted) != len(keys) {
			t.Fatalf("evicted %d distinct keys, want exactly the %d residents", len(evicted), len(keys))
		}
	})

	t.Run("pressure with no residents rejects", func(t *testing.T) {
		a := NewTinyLFUAdmission(1024, 1)
		in := tinyLFUKey(70)
		tinyLFUBump(a, in, 15)
		if victims, ok := a.Admit(in, 2, 9, 10, nil); ok || len(victims) != 0 {
			t.Fatalf("ok=%v victims=%v, want rejected", ok, victims)
		}
	})
}
