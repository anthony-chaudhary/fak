package kvbudget

import "testing"

// TestReserveBlocksWorstCase pins the worst-case block math: blocks are sized
// against prompt+MaxNewTokens (the maximum length), rounded up per block, with
// already-held and ref-held reusable blocks subtracted from the reservation.
func TestReserveBlocksWorstCase(t *testing.T) {
	cases := []struct {
		name        string
		s           Stream
		wantWorst   int
		wantReserve int
	}{
		{
			name:        "exact multiple",
			s:           Stream{PromptTokens: 100, MaxNewTokens: 156, BlockSize: 128},
			wantWorst:   2, // 256/128
			wantReserve: 2,
		},
		{
			name:        "rounds up partial block",
			s:           Stream{PromptTokens: 100, MaxNewTokens: 1, BlockSize: 128},
			wantWorst:   1, // 101 -> 1 block
			wantReserve: 1,
		},
		{
			name:        "held blocks subtract",
			s:           Stream{PromptTokens: 512, MaxNewTokens: 0, BlockSize: 128, HeldBlocks: 3},
			wantWorst:   4,
			wantReserve: 1,
		},
		{
			name:        "retained reusable subtract",
			s:           Stream{PromptTokens: 512, MaxNewTokens: 0, BlockSize: 128, RetainedBlocks: 4},
			wantWorst:   4,
			wantReserve: 0,
		},
		{
			name:        "over-held clamps to zero",
			s:           Stream{PromptTokens: 128, MaxNewTokens: 0, BlockSize: 128, HeldBlocks: 9},
			wantWorst:   1,
			wantReserve: 0,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.s.WorstCaseBlocks(); got != c.wantWorst {
				t.Errorf("WorstCaseBlocks() = %d, want %d", got, c.wantWorst)
			}
			if got := c.s.ReserveBlocks(); got != c.wantReserve {
				t.Errorf("ReserveBlocks() = %d, want %d", got, c.wantReserve)
			}
		})
	}
}

// TestAdmitFitsDecrementsBudget proves an arrival whose worst case fits is
// admitted and the budget is decremented by exactly that reservation.
func TestAdmitFitsDecrementsBudget(t *testing.T) {
	r := Reservation{FreeBlocks: 10}
	s := Stream{PromptTokens: 512, MaxNewTokens: 0, BlockSize: 128} // 4 blocks
	v := r.Admit(s)
	if !v.Admitted {
		t.Fatalf("Admit refused a fitting stream: %+v", v)
	}
	if v.Reason != ReasonAdmitted {
		t.Errorf("Reason = %q, want empty", v.Reason)
	}
	if v.ReservedBlocks != 4 {
		t.Errorf("ReservedBlocks = %d, want 4", v.ReservedBlocks)
	}
	if r.ReservedBlocks != 4 {
		t.Errorf("budget ReservedBlocks = %d, want 4", r.ReservedBlocks)
	}
	if r.Available() != 6 {
		t.Errorf("Available() = %d, want 6", r.Available())
	}
}

// TestAdmitRefusesLeavesBudgetUnchanged proves a refusal is typed, fail-closed,
// and does not touch the budget.
func TestAdmitRefusesLeavesBudgetUnchanged(t *testing.T) {
	r := Reservation{FreeBlocks: 3}
	s := Stream{PromptTokens: 512, MaxNewTokens: 0, BlockSize: 128} // 4 blocks > 3 free
	v := r.Admit(s)
	if v.Admitted {
		t.Fatalf("Admit accepted a stream that does not fit: %+v", v)
	}
	if v.Reason != ReasonNoRoomToRetain {
		t.Errorf("Reason = %q, want %q", v.Reason, ReasonNoRoomToRetain)
	}
	if v.ReservedBlocks != 0 {
		t.Errorf("ReservedBlocks = %d, want 0 on refusal", v.ReservedBlocks)
	}
	if r.ReservedBlocks != 0 {
		t.Errorf("budget mutated on refusal: ReservedBlocks = %d, want 0", r.ReservedBlocks)
	}
	if r.Available() != 3 {
		t.Errorf("Available() = %d, want 3 (unchanged)", r.Available())
	}
}

// TestAdmitExactFitBoundary proves the boundary admits: a reservation equal to
// the available room fits (need == avail), while need == avail+1 refuses.
func TestAdmitExactFitBoundary(t *testing.T) {
	// need == avail: 4 blocks into exactly 4 free.
	r := Reservation{FreeBlocks: 4}
	if v := r.Admit(Stream{PromptTokens: 512, MaxNewTokens: 0, BlockSize: 128}); !v.Admitted {
		t.Fatalf("exact fit refused: %+v", v)
	}
	if r.Available() != 0 {
		t.Errorf("Available() = %d, want 0 after exact fit", r.Available())
	}
	// need == avail+1: 5 blocks into 4 free refuses.
	r2 := Reservation{FreeBlocks: 4}
	if v := r2.Admit(Stream{PromptTokens: 513, MaxNewTokens: 0, BlockSize: 128}); v.Admitted {
		t.Fatalf("over-by-one admitted: %+v", v)
	}
}

// TestAdmitInvalidFailsClosed proves a self-inconsistent request refuses with a
// typed reason and never mutates the budget.
func TestAdmitInvalidFailsClosed(t *testing.T) {
	r := Reservation{FreeBlocks: 100}
	for _, s := range []Stream{
		{PromptTokens: 128, MaxNewTokens: 0, BlockSize: 0},  // zero block size
		{PromptTokens: -1, MaxNewTokens: 0, BlockSize: 128}, // negative tokens
		{PromptTokens: 128, MaxNewTokens: 0, BlockSize: 128, HeldBlocks: -1},
	} {
		v := r.Admit(s)
		if v.Admitted {
			t.Errorf("invalid stream admitted: %+v", s)
		}
		if v.Reason != ReasonInvalidRequest {
			t.Errorf("Reason = %q, want %q for %+v", v.Reason, ReasonInvalidRequest, s)
		}
	}
	if r.ReservedBlocks != 0 {
		t.Errorf("budget mutated by invalid requests: %d", r.ReservedBlocks)
	}
}

// TestReleaseReturnsReservation proves Release returns exactly the admitted
// reservation to the free pool, and a double release cannot go negative.
func TestReleaseReturnsReservation(t *testing.T) {
	r := Reservation{FreeBlocks: 10}
	v := r.Admit(Stream{PromptTokens: 512, MaxNewTokens: 0, BlockSize: 128}) // 4
	if !v.Admitted || r.ReservedBlocks != 4 {
		t.Fatalf("setup admit failed: %+v (reserved=%d)", v, r.ReservedBlocks)
	}
	r.Release(v)
	if r.ReservedBlocks != 0 {
		t.Errorf("after Release ReservedBlocks = %d, want 0", r.ReservedBlocks)
	}
	if r.Available() != 10 {
		t.Errorf("after Release Available() = %d, want 10", r.Available())
	}
	// Double release stays clamped, never negative.
	r.Release(v)
	if r.ReservedBlocks != 0 {
		t.Errorf("double Release drove ReservedBlocks to %d, want 0", r.ReservedBlocks)
	}
	// Releasing a refused verdict is a no-op.
	refused := Verdict{Admitted: false, ReservedBlocks: 5}
	r.Release(refused)
	if r.ReservedBlocks != 0 {
		t.Errorf("releasing a refused verdict mutated the budget: %d", r.ReservedBlocks)
	}
}

// TestAdmitSequenceGuarantee is the guarantee end to end: three streams that fit
// are admitted and reserve the pool; a later BIG stream is refused because the
// free room is already reserved by the earlier admits — proving no admitted
// stream can be dropped mid-decode to seat a newcomer. A small stream that still
// fits the remainder is then admitted, showing the refusal is capacity-exact,
// not a hard stop.
func TestAdmitSequenceGuarantee(t *testing.T) {
	// Pool of 10 blocks, block size 128 tokens.
	streams := []Stream{
		{PromptTokens: 384, MaxNewTokens: 0, BlockSize: 128}, // 3 blocks -> admit (reserved 3)
		{PromptTokens: 384, MaxNewTokens: 0, BlockSize: 128}, // 3 blocks -> admit (reserved 6)
		{PromptTokens: 128, MaxNewTokens: 0, BlockSize: 128}, // 1 block  -> admit (reserved 7)
		{PromptTokens: 512, MaxNewTokens: 0, BlockSize: 128}, // 4 blocks -> refuse (only 3 free)
		{PromptTokens: 256, MaxNewTokens: 0, BlockSize: 128}, // 2 blocks -> admit (reserved 9)
	}
	verdicts, r := AdmitAll(10, streams)

	wantAdmit := []bool{true, true, true, false, true}
	for i, w := range wantAdmit {
		if verdicts[i].Admitted != w {
			t.Errorf("verdict[%d].Admitted = %v, want %v (%+v)", i, verdicts[i].Admitted, w, verdicts[i])
		}
	}
	// The refused big stream must carry the no-room reason and reserve nothing.
	if verdicts[3].Reason != ReasonNoRoomToRetain {
		t.Errorf("verdict[3].Reason = %q, want %q", verdicts[3].Reason, ReasonNoRoomToRetain)
	}
	if verdicts[3].AvailableBlocks != 3 {
		t.Errorf("verdict[3].AvailableBlocks = %d, want 3 (reserved by earlier admits)", verdicts[3].AvailableBlocks)
	}
	// Running reserved total: 3+3+1+2 = 9, one block still free.
	if r.ReservedBlocks != 9 {
		t.Errorf("final ReservedBlocks = %d, want 9", r.ReservedBlocks)
	}
	if r.Available() != 1 {
		t.Errorf("final Available() = %d, want 1", r.Available())
	}
}
