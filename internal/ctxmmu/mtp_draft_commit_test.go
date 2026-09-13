package ctxmmu

import (
	"errors"
	"testing"
)

// newDraftBlock builds a PageBlock left at the zero refcount. RecordDraft itself
// retains each associated block exactly once, so a committed block ends at 1 while
// a rejected block (released once) returns to 0.
func newDraftBlock(id int64, tokenCount, capacity int) *PageBlock {
	return &PageBlock{ID: id, TokenCount: tokenCount, Capacity: capacity}
}

// TestMTPDraftCommitRollbackPageRefcounts witnesses acceptance criterion 2 of #12238:
// the context-MMU commits accepted draft token KV pages and immediately frees rejected
// draft pages (without leaks), across partial acceptance, full acceptance, rollback,
// re-record leak prevention, and out-of-range rejection.
func TestMTPDraftCommitRollbackPageRefcounts(t *testing.T) {
	t.Run("partial acceptance commits prefix and frees tail", func(t *testing.T) {
		var s MTPDraftState
		blocks := []*PageBlock{
			newDraftBlock(1, 1, 16),
			newDraftBlock(2, 1, 16),
			newDraftBlock(3, 1, 16),
			newDraftBlock(4, 1, 16),
		}
		s.RecordDraft([]int32{10, 11, 12, 13}, blocks...)
		if len(s.DraftPages) != 4 {
			t.Fatalf("len(s.DraftPages) = %d, want 4 after RecordDraft", len(s.DraftPages))
		}
		for i, b := range blocks {
			if got := b.RefCount(); got != 1 {
				t.Fatalf("recorded block[%d] RefCount = %d, want 1 (RecordDraft retains)", i, got)
			}
		}
		rejected := append([]*MTPDraftPage(nil), s.DraftPages[2:]...)

		committed, freed, err := s.CommitDraft(2)
		if err != nil {
			t.Fatalf("CommitDraft(2) error = %v", err)
		}
		if committed != 2 {
			t.Errorf("committedPages = %d, want 2", committed)
		}
		if freed != 2 {
			t.Errorf("freedPages = %d, want 2", freed)
		}
		for i := 0; i < 2; i++ {
			if got := blocks[i].RefCount(); got != 1 {
				t.Errorf("accepted block[%d] RefCount = %d, want 1 (must not be released)", i, got)
			}
		}
		for i := 2; i < 4; i++ {
			if got := blocks[i].RefCount(); got != 0 {
				t.Errorf("rejected block[%d] RefCount = %d, want 0 (must be released)", i, got)
			}
		}
		// The rejected MTPDraftPage must have its PageBlock nil'd on release.
		for i, dp := range rejected {
			if dp.PageBlock != nil {
				t.Errorf("rejected DraftPage[%d].PageBlock = %p, want nil after free", i, dp.PageBlock)
			}
		}
		if s.CommittedPages != 2 {
			t.Errorf("s.CommittedPages = %d, want 2", s.CommittedPages)
		}
		if s.FreedPages != 2 {
			t.Errorf("s.FreedPages = %d, want 2", s.FreedPages)
		}
		if s.AcceptedCount != 2 {
			t.Errorf("s.AcceptedCount = %d, want 2", s.AcceptedCount)
		}
		if !s.RollbackOccurred {
			t.Error("s.RollbackOccurred = false, want true on partial acceptance")
		}
		if len(s.DraftPages) != 2 {
			t.Errorf("len(s.DraftPages) = %d, want 2 (rejected pages dropped)", len(s.DraftPages))
		}
		if s.DraftPages[0].PageBlock != blocks[0] || s.DraftPages[1].PageBlock != blocks[1] {
			t.Error("retained DraftPages do not reference the accepted prefix blocks")
		}
		if !s.DraftPages[0].Committed || !s.DraftPages[1].Committed {
			t.Error("accepted DraftPages must be marked Committed")
		}
	})

	t.Run("full acceptance commits all and frees none", func(t *testing.T) {
		var s MTPDraftState
		blocks := []*PageBlock{
			newDraftBlock(11, 1, 16),
			newDraftBlock(12, 1, 16),
			newDraftBlock(13, 1, 16),
			newDraftBlock(14, 1, 16),
		}
		s.RecordDraft([]int32{20, 21, 22, 23}, blocks...)

		committed, freed, err := s.CommitDraft(4)
		if err != nil {
			t.Fatalf("CommitDraft(4) error = %v", err)
		}
		if committed != 4 {
			t.Errorf("committedPages = %d, want 4", committed)
		}
		if freed != 0 {
			t.Errorf("freedPages = %d, want 0", freed)
		}
		for i, b := range blocks {
			if got := b.RefCount(); got != 1 {
				t.Errorf("block[%d] RefCount = %d, want 1", i, got)
			}
		}
		if s.RollbackOccurred {
			t.Error("s.RollbackOccurred = true, want false on full acceptance")
		}
		if len(s.DraftPages) != 4 {
			t.Errorf("len(s.DraftPages) = %d, want 4", len(s.DraftPages))
		}
	})

	t.Run("rollback frees all pages", func(t *testing.T) {
		var s MTPDraftState
		blocks := []*PageBlock{
			newDraftBlock(21, 1, 16),
			newDraftBlock(22, 1, 16),
			newDraftBlock(23, 1, 16),
			newDraftBlock(24, 1, 16),
		}
		s.RecordDraft([]int32{30, 31, 32, 33}, blocks...)

		freed, err := s.RollbackDraft()
		if err != nil {
			t.Fatalf("RollbackDraft error = %v", err)
		}
		if freed != 4 {
			t.Errorf("freedPages = %d, want 4", freed)
		}
		for i, b := range blocks {
			if got := b.RefCount(); got != 0 {
				t.Errorf("block[%d] RefCount = %d, want 0 after rollback", i, got)
			}
		}
		if len(s.DraftPages) != 0 {
			t.Errorf("len(s.DraftPages) = %d, want 0 after rollback", len(s.DraftPages))
		}
	})

	t.Run("re-record frees prior uncommitted pages", func(t *testing.T) {
		var s MTPDraftState
		old := []*PageBlock{
			newDraftBlock(31, 1, 16),
			newDraftBlock(32, 1, 16),
		}
		s.RecordDraft([]int32{40, 41}, old...)

		fresh := []*PageBlock{
			newDraftBlock(33, 1, 16),
			newDraftBlock(34, 1, 16),
			newDraftBlock(35, 1, 16),
		}
		s.RecordDraft([]int32{50, 51, 52}, fresh...)

		for i, b := range old {
			if got := b.RefCount(); got != 0 {
				t.Errorf("prior block[%d] RefCount = %d, want 0 (re-record must free it)", i, got)
			}
		}
		for i, b := range fresh {
			if got := b.RefCount(); got != 1 {
				t.Errorf("fresh block[%d] RefCount = %d, want 1", i, got)
			}
		}
		if len(s.DraftPages) != 3 {
			t.Errorf("len(s.DraftPages) = %d, want 3", len(s.DraftPages))
		}
	})

	t.Run("out-of-range accepted returns ErrInvalidAcceptedCount", func(t *testing.T) {
		for _, accepted := range []int{5, -1} {
			var s MTPDraftState
			blocks := []*PageBlock{
				newDraftBlock(41, 1, 16),
				newDraftBlock(42, 1, 16),
				newDraftBlock(43, 1, 16),
				newDraftBlock(44, 1, 16),
			}
			s.RecordDraft([]int32{60, 61, 62, 63}, blocks...)

			committed, freed, err := s.CommitDraft(accepted)
			if !errors.Is(err, ErrInvalidAcceptedCount) {
				t.Errorf("CommitDraft(%d) err = %v, want ErrInvalidAcceptedCount", accepted, err)
			}
			if committed != 0 || freed != 0 {
				t.Errorf("CommitDraft(%d) = (%d,%d), want (0,0) on error", accepted, committed, freed)
			}
			for i, b := range blocks {
				if got := b.RefCount(); got != 1 {
					t.Errorf("block[%d] RefCount = %d, want 1 (invalid commit must not mutate refs)", i, got)
				}
			}
			if len(s.DraftPages) != 4 {
				t.Errorf("len(s.DraftPages) = %d, want 4 (invalid commit must not truncate)", len(s.DraftPages))
			}
		}
	})

	t.Run("session descriptor delegates commit and rollback", func(t *testing.T) {
		var d SessionDescriptor
		blocks := []*PageBlock{
			newDraftBlock(51, 1, 16),
			newDraftBlock(52, 1, 16),
			newDraftBlock(53, 1, 16),
		}
		d.RecordMTPDraft([]int32{70, 71, 72}, blocks...)

		committed, freed, err := d.CommitMTPDraft(1)
		if err != nil {
			t.Fatalf("CommitMTPDraft(1) error = %v", err)
		}
		if committed != 1 || freed != 2 {
			t.Errorf("CommitMTPDraft(1) = (%d,%d), want (1,2)", committed, freed)
		}
		if got := blocks[0].RefCount(); got != 1 {
			t.Errorf("accepted block RefCount = %d, want 1", got)
		}
		for i := 1; i < 3; i++ {
			if got := blocks[i].RefCount(); got != 0 {
				t.Errorf("rejected block[%d] RefCount = %d, want 0", i, got)
			}
		}

		more := []*PageBlock{newDraftBlock(54, 1, 16)}
		d.RecordMTPDraft([]int32{80}, more...)
		n, err := d.RollbackMTPDraft()
		if err != nil {
			t.Fatalf("RollbackMTPDraft error = %v", err)
		}
		if n != 1 {
			t.Errorf("RollbackMTPDraft freed = %d, want 1", n)
		}
		if got := more[0].RefCount(); got != 0 {
			t.Errorf("rolled-back block RefCount = %d, want 0", got)
		}
	})
}
