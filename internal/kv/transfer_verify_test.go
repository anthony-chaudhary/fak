package kv

import (
	"errors"
	"testing"
)

type transferFixture struct {
	plan          TransferGeometry
	src, dst      []uint64
	local, remote [][]int64
	rowWords      int
}

const (
	transferDstSalt int64 = 0
	transferSrcSalt int64 = 1
	transferSeed    int64 = 67
)

// newTransferFixture models a pull: the local pool (salt 0) is the
// destination, the remote pool (salt 1) the source.
func newTransferFixture(t *testing.T) *transferFixture {
	t.Helper()
	p, err := NewTransferGeometry("dsv4", 512, 256, 64, 5, 2, 2, 2.0)
	if err != nil {
		t.Fatalf("NewTransferGeometry: %v", err)
	}
	f := &transferFixture{plan: p, rowWords: p.RowBytes / 8}
	f.src = make([]uint64, p.PoolWords())
	f.dst = make([]uint64, p.PoolWords())
	FillTransferPattern(f.src, transferSrcSalt)
	FillTransferPattern(f.dst, transferDstSalt)
	seedL := TransferTableSeed(p, TransferSideLocal, transferSeed)
	seedR := TransferTableSeed(p, TransferSideRemote, transferSeed)
	for r := 0; r < p.BatchMax; r++ {
		l, err := TransferBlockTable(p, seedL, r, 0)
		if err != nil {
			t.Fatalf("local table: %v", err)
		}
		rm, err := TransferBlockTable(p, seedR, r, 0)
		if err != nil {
			t.Fatalf("remote table: %v", err)
		}
		f.local = append(f.local, l)
		f.remote = append(f.remote, rm)
	}
	return f
}

// copyRows moves words [shift, shift+n) of each source row into the
// destination row the tables pair it with.
func (f *transferFixture) copyRows(from []uint64, dstRows, srcRows []int64, shift, n int) {
	for i := range dstRows {
		d := int(dstRows[i]) * f.rowWords
		s := int(srcRows[i])*f.rowWords + shift
		copy(f.dst[d:d+n], from[s:s+n])
	}
}

func (f *transferFixture) pullAll() error {
	for r := range f.local {
		f.copyRows(f.src, f.local[r], f.remote[r], 0, f.rowWords)
	}
	return nil
}

func (f *transferFixture) verify() error {
	return VerifyTransferRequests(f.plan, TransferPull, f.dst, f.local, f.remote, transferSrcSalt, 3)
}

func TestTransferVerifySentinel(t *testing.T) {
	t.Run("pattern_words_unique_and_not_sentinel", func(t *testing.T) {
		f := newTransferFixture(t)
		sentinel := uint64(TransferSentinel) * 0x0101010101010101
		seen := make(map[uint64]bool, 2*len(f.src))
		for _, pool := range [][]uint64{f.src, f.dst} {
			for i, w := range pool {
				if seen[w] || w == sentinel {
					t.Fatalf("word %d = %#x repeats or equals the sentinel", i, w)
				}
				seen[w] = true
			}
		}
		if TransferPatternWord(7, transferSrcSalt) != f.src[7] {
			t.Fatalf("TransferPatternWord disagrees with FillTransferPattern")
		}
		WipeTransferSentinel(f.dst)
		for i, w := range f.dst {
			if w != sentinel {
				t.Fatalf("wiped word %d = %#x", i, w)
			}
		}
	})

	t.Run("tables_derived_without_exchange_and_disjoint", func(t *testing.T) {
		f := newTransferFixture(t)
		p := f.plan
		seedL := TransferTableSeed(p, TransferSideLocal, transferSeed)
		if seedL == TransferTableSeed(p, TransferSideRemote, transferSeed) {
			t.Fatalf("local and remote seeds collide")
		}
		again, _ := TransferBlockTable(p, seedL, 1, 0)
		for i := range again {
			if again[i] != f.local[1][i] {
				t.Fatalf("table not deterministic from plan+seed")
			}
		}
		used := map[int64]bool{}
		for k := 0; k < p.TableSets; k++ {
			for r := 0; r < p.BatchMax; r++ {
				rows, err := TransferBlockTable(p, seedL, r, k)
				if err != nil {
					t.Fatalf("set %d request %d: %v", k, r, err)
				}
				for _, row := range rows {
					if used[row] || row < 0 || int(row) >= p.PoolRows {
						t.Fatalf("row %d aliased or out of pool", row)
					}
					used[row] = true
				}
			}
		}
		if _, err := TransferBlockTable(p, seedL, 0, p.PoolRows); err == nil {
			t.Fatalf("over-pool table set accepted")
		}
	})

	t.Run("pass", func(t *testing.T) {
		f := newTransferFixture(t)
		if err := RunVerifiedTransfer(f.dst, f.pullAll, f.verify); err != nil {
			t.Fatalf("correct transfer failed: %v", err)
		}
	})

	t.Run("push_pass", func(t *testing.T) {
		f := newTransferFixture(t)
		// Push lands on the remote rows; reuse dst as the remote pool and src as local.
		transfer := func() error {
			for r := range f.local {
				f.copyRows(f.src, f.remote[r], f.local[r], 0, f.rowWords)
			}
			return nil
		}
		verify := func() error {
			return VerifyTransferRequests(f.plan, TransferPush, f.dst, f.local, f.remote, transferSrcSalt, 3)
		}
		if err := RunVerifiedTransfer(f.dst, transfer, verify); err != nil {
			t.Fatalf("correct push failed: %v", err)
		}
	})

	faults := []struct {
		name     string
		transfer func(f *transferFixture) error
	}{
		{"never_happened", func(f *transferFixture) error { return nil }},
		{"left_sentinel", func(f *transferFixture) error {
			_ = f.pullAll()
			d := int(f.local[1][2]) * f.rowWords
			WipeTransferSentinel(f.dst[d : d+f.rowWords])
			return nil
		}},
		{"swapped_tables", func(f *transferFixture) error {
			f.copyRows(f.src, f.local[0], f.remote[1], 0, f.rowWords)
			f.copyRows(f.src, f.local[1], f.remote[0], 0, f.rowWords)
			return nil
		}},
		{"loopback", func(f *transferFixture) error {
			self := make([]uint64, len(f.dst))
			FillTransferPattern(self, transferDstSalt)
			for r := range f.local {
				f.copyRows(self, f.local[r], f.remote[r], 0, f.rowWords)
			}
			return nil
		}},
		{"shift_8_bytes", func(f *transferFixture) error {
			// One word late on the source; the extra pattern word keeps a
			// shifted read of the last row in bounds.
			padded := append(append([]uint64(nil), f.src...),
				TransferPatternWord(uint64(len(f.src)), transferSrcSalt))
			for r := range f.local {
				f.copyRows(padded, f.local[r], f.remote[r], 1, f.rowWords)
			}
			return nil
		}},
		{"missing_tail", func(f *transferFixture) error {
			for r := range f.local {
				f.copyRows(f.src, f.local[r], f.remote[r], 0, f.rowWords-1)
			}
			return nil
		}},
	}
	for _, fc := range faults {
		t.Run("fault_"+fc.name, func(t *testing.T) {
			f := newTransferFixture(t)
			err := RunVerifiedTransfer(f.dst, func() error { return fc.transfer(f) }, f.verify)
			if !errors.Is(err, ErrTransferVerify) {
				t.Fatalf("fault %s did not fail closed: %v", fc.name, err)
			}
		})
	}

	t.Run("transfer_error_fails_before_verify", func(t *testing.T) {
		f := newTransferFixture(t)
		called := false
		err := RunVerifiedTransfer(f.dst, func() error { return errors.New("boom") },
			func() error { called = true; return nil })
		if !errors.Is(err, ErrTransferVerify) || called {
			t.Fatalf("err=%v verifyCalled=%v", err, called)
		}
	})

	t.Run("verify_panic_fails_closed", func(t *testing.T) {
		f := newTransferFixture(t)
		err := RunVerifiedTransfer(f.dst, f.pullAll, func() error { panic("index") })
		if !errors.Is(err, ErrTransferVerify) {
			t.Fatalf("panicking verify did not fail closed: %v", err)
		}
	})

	t.Run("entries_edge_cases_fail_closed", func(t *testing.T) {
		f := newTransferFixture(t)
		_ = f.pullAll()
		n := uint64(len(f.dst)) * 8
		cases := map[string]error{
			"probe_beyond_pool": VerifyTransferEntries(f.dst, []uint64{n}, []uint64{0}, []uint64{64}, transferSrcSalt, 1),
			"zero_word_entry":   VerifyTransferEntries(f.dst, []uint64{0}, []uint64{0}, []uint64{4}, transferSrcSalt, 1),
			"length_mismatch":   VerifyTransferEntries(f.dst, []uint64{0, 8}, []uint64{0}, []uint64{8}, transferSrcSalt, 1),
			"unknown_op":        VerifyTransferRequests(f.plan, "copy", f.dst, f.local, f.remote, transferSrcSalt, 1),
		}
		for name, err := range cases {
			if !errors.Is(err, ErrTransferVerify) {
				t.Errorf("%s: want ErrTransferVerify, got %v", name, err)
			}
		}
	})

	t.Run("bulk", func(t *testing.T) {
		f := newTransferFixture(t)
		nbytes := uint64(len(f.src)) * 8
		bulk := func() error { copy(f.dst, f.src); return nil }
		check := func() error { return VerifyTransferBulk(f.dst, nbytes, transferSrcSalt, 5, 64) }
		if err := RunVerifiedTransfer(f.dst, bulk, check); err != nil {
			t.Fatalf("correct bulk failed: %v", err)
		}
		tail := func() error { copy(f.dst[:len(f.dst)-1], f.src); return nil }
		if err := RunVerifiedTransfer(f.dst, tail, check); !errors.Is(err, ErrTransferVerify) {
			t.Fatalf("bulk missing tail did not fail closed: %v", err)
		}
		if err := RunVerifiedTransfer(f.dst, func() error { return nil }, check); !errors.Is(err, ErrTransferVerify) {
			t.Fatalf("bulk never-happened did not fail closed: %v", err)
		}
	})
}
