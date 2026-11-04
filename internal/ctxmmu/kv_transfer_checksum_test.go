// fak-test:runtime fast est=1ms
package ctxmmu_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestKVTransferChecksum pins the borrowed SGLang design (disaggregation/
// checksum.py @1b78083): an adler32 over the page-index vector computed at the
// transfer source and verified at the destination, default-off behind the
// transfer opt-in. The contract is closed: fixed vectors, fixed sums, sentinel
// errors via errors.Is, and a byte-exact disabled wire form.
func TestKVTransferChecksum(t *testing.T) {
	source := []uint64{7, 42, 99, 1000, 4095}
	tampered := []uint64{7, 42, 98, 1000, 4095} // one page ID swapped

	t.Run("fixed-vector sums are pinned", func(t *testing.T) {
		if got := ctxmmu.PageIndexAdler32(source); got != 667812499 {
			t.Fatalf("PageIndexAdler32(source) = %#010x, want 0x27c4d2eb", got)
		}
		if got := ctxmmu.PageIndexAdler32(tampered); got != 666239634 {
			t.Fatalf("PageIndexAdler32(tampered) = %#010x, want 0x27b5f1a2", got)
		}
	})

	t.Run("count prefix keeps distinct vectors distinct", func(t *testing.T) {
		ab := ctxmmu.PageIndexAdler32([]uint64{1, 2})
		cd := ctxmmu.PageIndexAdler32([]uint64{12})
		if ab == cd {
			t.Fatalf("[1,2] and [12] both sum to %#010x; count prefix lost", ab)
		}
		if ab != 6815750 || cd != 8388622 {
			t.Fatalf("([1,2], [12]) = (%#010x, %#010x), want (0x681572, 0x800006)", ab, cd)
		}
	})

	t.Run("destination match verifies", func(t *testing.T) {
		rec := ctxmmu.NewKVTransferChecksum(source)
		if !rec.Enabled() || rec.Sum() != 667812499 || rec.PageCount() != 5 {
			t.Fatalf("record = enabled=%v sum=%#010x pages=%d, want enabled true sum 0x27c4d2eb pages 5", rec.Enabled(), rec.Sum(), rec.PageCount())
		}
		if err := rec.Verify(source); err != nil {
			t.Fatalf("Verify(source) = %v, want nil", err)
		}
	})

	t.Run("destination mismatch refuses with sentinel", func(t *testing.T) {
		rec := ctxmmu.NewKVTransferChecksum(source)
		err := rec.Verify(tampered)
		if !errors.Is(err, ctxmmu.ErrKVTransferChecksumMismatch) {
			t.Fatalf("Verify(tampered) = %v, want ErrKVTransferChecksumMismatch", err)
		}
		short := rec.Verify(source[:4])
		if !errors.Is(short, ctxmmu.ErrKVTransferChecksumMismatch) {
			t.Fatalf("Verify(source[:4]) = %v, want ErrKVTransferChecksumMismatch", short)
		}
	})

	t.Run("default-off leaves behavior and wire unchanged", func(t *testing.T) {
		var off ctxmmu.KVTransferChecksum
		if off.Enabled() {
			t.Fatal("zero-value record reports Enabled")
		}
		if err := off.Verify(source); err != nil {
			t.Fatalf("disabled Verify(source) = %v, want nil no-op", err)
		}
		if err := off.Verify(tampered); err != nil {
			t.Fatalf("disabled Verify(tampered) = %v, want nil no-op", err)
		}
		wire, err := json.Marshal(off)
		if err != nil {
			t.Fatalf("Marshal(disabled) error: %v", err)
		}
		if !bytes.Equal(wire, []byte("{}")) {
			t.Fatalf("disabled wire = %q, want {} (pre-feature baseline)", wire)
		}
		var back ctxmmu.KVTransferChecksum
		if err := json.Unmarshal([]byte("{}"), &back); err != nil {
			t.Fatalf("Unmarshal({}) error: %v", err)
		}
		if back.Enabled() {
			t.Fatal("unmarshaled {} record reports Enabled")
		}
	})

	t.Run("enabled record round-trips the wire", func(t *testing.T) {
		rec := ctxmmu.NewKVTransferChecksum(source)
		wire, err := json.Marshal(rec)
		if err != nil {
			t.Fatalf("Marshal(enabled) error: %v", err)
		}
		want := []byte(`{"adler32":667812499,"pages":5}`)
		if !bytes.Equal(wire, want) {
			t.Fatalf("enabled wire = %s, want %s", wire, want)
		}
		var back ctxmmu.KVTransferChecksum
		if err := json.Unmarshal(wire, &back); err != nil {
			t.Fatalf("Unmarshal(enabled) error: %v", err)
		}
		if !back.Enabled() || back.Sum() != rec.Sum() || back.PageCount() != rec.PageCount() {
			t.Fatalf("round-tripped record = enabled=%v sum=%#010x pages=%d, want %v", back.Enabled(), back.Sum(), back.PageCount(), rec)
		}
		if err := back.Verify(source); err != nil {
			t.Fatalf("round-tripped Verify(source) = %v, want nil", err)
		}
		if err := back.Verify(tampered); !errors.Is(err, ctxmmu.ErrKVTransferChecksumMismatch) {
			t.Fatalf("round-tripped Verify(tampered) = %v, want ErrKVTransferChecksumMismatch", err)
		}
	})

	t.Run("malformed wire records refuse with sentinel", func(t *testing.T) {
		for _, bad := range []string{`{"adler32":667812499}`, `{"pages":5}`, `{"adler32":1,"pages":-1}`} {
			var c ctxmmu.KVTransferChecksum
			if err := json.Unmarshal([]byte(bad), &c); !errors.Is(err, ctxmmu.ErrKVTransferChecksumMalformed) {
				t.Fatalf("Unmarshal(%s) = %v, want ErrKVTransferChecksumMalformed", bad, err)
			}
		}
	})
}
