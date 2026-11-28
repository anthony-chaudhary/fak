package kvbudget

import "testing"

// TestCappedReserveBelowCapReservesWorstCase proves the cap is inactive when the
// worst case already fits under it: the capped reservation equals the full
// worst-case reservation, and CapActive is false.
func TestCappedReserveBelowCapReservesWorstCase(t *testing.T) {
	// prompt 128 + 512 new = 640 tokens -> ceilDiv(640,128) = 5 blocks worst case.
	s := Stream{PromptTokens: 128, MaxNewTokens: 512, BlockSize: 128}
	cap := DefaultPreallocNewTokens // 1024 > 512, so the cap does not bite
	if s.CapActive(cap) {
		t.Fatalf("CapActive = true, want false when MaxNewTokens (%d) <= cap (%d)", s.MaxNewTokens, cap)
	}
	if got, want := s.CappedReserveBlocks(cap), s.ReserveBlocks(); got != want {
		t.Errorf("CappedReserveBlocks = %d, want %d (== uncapped ReserveBlocks when cap inactive)", got, want)
	}
	if got := s.CappedReserveBlocks(cap); got != 5 {
		t.Errorf("CappedReserveBlocks = %d, want 5", got)
	}
}

// TestCappedReserveAboveCapReservesCap proves the hog is stopped: a request with
// a huge MaxNewTokens reserves only the capped footprint, far below its full
// worst case.
func TestCappedReserveAboveCapReservesCap(t *testing.T) {
	// prompt 128, asking for 120000 new tokens, block size 128.
	s := Stream{PromptTokens: 128, MaxNewTokens: 120000, BlockSize: 128}
	cap := 1024
	if !s.CapActive(cap) {
		t.Fatalf("CapActive = false, want true when MaxNewTokens (%d) > cap (%d)", s.MaxNewTokens, cap)
	}
	// Uncapped worst case: ceilDiv(128+120000,128) = ceilDiv(120128,128) = 939.
	if got := s.WorstCaseBlocks(); got != 939 {
		t.Fatalf("WorstCaseBlocks = %d, want 939", got)
	}
	// Capped footprint: ceilDiv(128+1024,128) = ceilDiv(1152,128) = 9.
	if got := s.CappedReserveBlocks(cap); got != 9 {
		t.Errorf("CappedReserveBlocks = %d, want 9 (min(worst,cap))", got)
	}
	if s.CappedReserveBlocks(cap) >= s.ReserveBlocks() {
		t.Errorf("capped reservation %d not below uncapped %d", s.CappedReserveBlocks(cap), s.ReserveBlocks())
	}
}

// TestChargeTracksGenerationNotCap proves the CHARGE axis follows real generated
// tokens and is independent of the preallocation cap: the same request, reserved
// at the cap, is charged for whatever it actually generates — less than, equal
// to, or more than the cap.
func TestChargeTracksGenerationNotCap(t *testing.T) {
	s := Stream{PromptTokens: 128, MaxNewTokens: 120000, BlockSize: 128}
	cap := 1024
	// The reservation is fixed at the capped footprint regardless of generation.
	if got := s.CappedReserveBlocks(cap); got != 9 {
		t.Fatalf("CappedReserveBlocks = %d, want 9", got)
	}
	cases := []struct {
		genNewTokens int
		wantCharge   int // ceilDiv(128+gen,128)
	}{
		{genNewTokens: 0, wantCharge: 1},        // ceilDiv(128,128)=1
		{genNewTokens: 512, wantCharge: 5},      // ceilDiv(640,128)=5   (below cap)
		{genNewTokens: 896, wantCharge: 8},      // ceilDiv(1024,128)=8  (near cap)
		{genNewTokens: 5000, wantCharge: 41},    // ceilDiv(5128,128)=41 (above cap)
		{genNewTokens: 120000, wantCharge: 939}, // full run == uncapped worst case
	}
	for _, c := range cases {
		if got := s.ChargedBlocks(c.genNewTokens); got != c.wantCharge {
			t.Errorf("ChargedBlocks(%d) = %d, want %d", c.genNewTokens, got, c.wantCharge)
		}
	}
	// The charge for a real run above the cap exceeds the capped reservation —
	// the two axes are genuinely decoupled.
	if s.ChargedBlocks(5000) <= s.CappedReserveBlocks(cap) {
		t.Errorf("charge %d did not exceed capped reservation %d for a run past the cap",
			s.ChargedBlocks(5000), s.CappedReserveBlocks(cap))
	}
	// A negative generation count is treated as zero (fail-closed to the floor).
	if got := s.ChargedBlocks(-5); got != 1 {
		t.Errorf("ChargedBlocks(-5) = %d, want 1 (negative clamps to zero generation)", got)
	}
}

// TestAdmitCappedThrottlesReservation proves AdmitCapped decrements the budget by
// the capped reservation, not the full worst case, and reports the full worst
// case on the verdict so a caller sees the cap bit.
func TestAdmitCappedThrottlesReservation(t *testing.T) {
	r := Reservation{FreeBlocks: 100}
	s := Stream{PromptTokens: 128, MaxNewTokens: 120000, BlockSize: 128}
	v := r.AdmitCapped(s, 1024)
	if !v.Admitted {
		t.Fatalf("AdmitCapped refused a stream that fits under the cap: %+v", v)
	}
	if v.ReservedBlocks != 9 {
		t.Errorf("verdict ReservedBlocks = %d, want 9 (capped)", v.ReservedBlocks)
	}
	if v.WorstCaseBlocks != 939 {
		t.Errorf("verdict WorstCaseBlocks = %d, want 939 (full, uncapped)", v.WorstCaseBlocks)
	}
	if r.ReservedBlocks != 9 {
		t.Errorf("budget decremented by %d, want 9 (the capped reservation)", r.ReservedBlocks)
	}
	// Uncapped Admit on a fresh budget would reserve the whole 939.
	r2 := Reservation{FreeBlocks: 1000}
	if v2 := r2.Admit(s); v2.ReservedBlocks != 939 {
		t.Errorf("uncapped Admit reserved %d, want 939 (control)", v2.ReservedBlocks)
	}
}

// TestAdmitCappedInvalidCapFailsClosed proves a non-positive preallocation cap
// refuses with the typed ReasonInvalidCap and never mutates the budget.
func TestAdmitCappedInvalidCapFailsClosed(t *testing.T) {
	s := Stream{PromptTokens: 128, MaxNewTokens: 512, BlockSize: 128}
	for _, cap := range []int{0, -1, -1024} {
		r := Reservation{FreeBlocks: 100}
		v := r.AdmitCapped(s, cap)
		if v.Admitted {
			t.Errorf("cap %d admitted, want fail-closed refusal", cap)
		}
		if v.Reason != ReasonInvalidCap {
			t.Errorf("cap %d Reason = %q, want %q", cap, v.Reason, ReasonInvalidCap)
		}
		if r.ReservedBlocks != 0 {
			t.Errorf("cap %d mutated the budget: ReservedBlocks = %d, want 0", cap, r.ReservedBlocks)
		}
	}
	// A malformed stream still fails closed with the stream reason even under a
	// valid cap.
	r := Reservation{FreeBlocks: 100}
	if v := r.AdmitCapped(Stream{BlockSize: 0}, 1024); v.Reason != ReasonInvalidRequest {
		t.Errorf("malformed stream Reason = %q, want %q", v.Reason, ReasonInvalidRequest)
	}
}

// TestAntiHogCoAdmit is the anti-hog property end to end: two requests each
// declaring a huge MaxNewTokens cannot co-admit under their full worst cases
// (the first hogs the pool and sheds the second), yet BOTH co-admit once each is
// capped — one greedy max_tokens can no longer reserve the whole pool.
func TestAntiHogCoAdmit(t *testing.T) {
	// prompt 128, 120000 new tokens each -> 939 worst-case blocks each.
	hog := Stream{PromptTokens: 128, MaxNewTokens: 120000, BlockSize: 128}
	streams := []Stream{hog, hog}

	// Uncapped: a pool of 1000 blocks seats the first (939) and sheds the second
	// (needs 939, only 61 free) — the hog.
	uncapped, ru := AdmitAll(1000, streams)
	if !uncapped[0].Admitted {
		t.Fatalf("uncapped first should admit: %+v", uncapped[0])
	}
	if uncapped[1].Admitted {
		t.Fatalf("uncapped second should be shed by the hog, but admitted: %+v", uncapped[1])
	}
	if uncapped[1].Reason != ReasonNoRoomToRetain {
		t.Errorf("uncapped second Reason = %q, want %q", uncapped[1].Reason, ReasonNoRoomToRetain)
	}
	if ru.ReservedBlocks != 939 {
		t.Errorf("uncapped reserved %d, want 939 (one hog took it all)", ru.ReservedBlocks)
	}

	// Capped at 1024 new tokens: each reserves only 9 blocks, so both co-admit in
	// the SAME 1000-block pool the uncapped pair could not share.
	capped, rc := AdmitAllCapped(1000, 1024, streams)
	for i, v := range capped {
		if !v.Admitted {
			t.Errorf("capped[%d] refused, want admitted under the cap: %+v", i, v)
		}
		if v.ReservedBlocks != 9 {
			t.Errorf("capped[%d] ReservedBlocks = %d, want 9", i, v.ReservedBlocks)
		}
	}
	if rc.ReservedBlocks != 18 {
		t.Errorf("capped total reserved = %d, want 18 (9+9, both seated)", rc.ReservedBlocks)
	}

	// Tighter proof: a pool that fits exactly the two CAPPED footprints (18) but
	// not even one full worst case (939) still co-admits both under the cap.
	tight, rt := AdmitAllCapped(18, 1024, streams)
	if !tight[0].Admitted || !tight[1].Admitted {
		t.Errorf("both should co-admit in an 18-block pool under the cap: %+v", tight)
	}
	if rt.Available() != 0 {
		t.Errorf("tight pool Available() = %d, want 0 (exact co-admit)", rt.Available())
	}
	// The same 18-block pool cannot admit even ONE uncapped worst case.
	if v := (&Reservation{FreeBlocks: 18}).Admit(hog); v.Admitted {
		t.Errorf("18-block pool admitted a 939-block worst case uncapped: %+v", v)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestQuestPageBudgetSelect(t *testing.T) {
	// tokenBudget=48, pageSize=16 -> pageBudget=3 (top 2 early pages + last page 4).
	scores := [][]float32{
		{0.1, 0.9, 0.3, 0.8, 0.05}, // last page index 4 has lowest score, must still be kept
		{0.7, 0.2, 0.6, 0.1, 0.01},
	}
	sel, ok := QuestPageBudgetSelect(48, 16, scores)
	if !ok || sel.PageBudget != 3 {
		t.Fatalf("QuestPageBudgetSelect = (%+v, %v), want PageBudget=3", sel, ok)
	}
	want0 := []int{1, 3, 4}
	want1 := []int{0, 2, 4}
	for i := range want0 {
		if sel.LayerSelected[0][i] != want0[i] || sel.LayerSelected[1][i] != want1[i] {
			t.Fatalf("LayerSelected = %v, want [%v %v]", sel.LayerSelected, want0, want1)
		}
	}
}
