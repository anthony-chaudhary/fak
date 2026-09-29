package sessionimage

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// TestACRFenceUnreadableWitnessRefusesDispatch is the #12815 named witness: a restore
// whose completion-evidence journal is unreadable must get an explicit REFUSE, never a
// FIRE that would re-execute an effect the predecessor may already have performed.
func TestACRFenceUnreadableWitnessRefusesDispatch(t *testing.T) {
	dir := t.TempDir()
	if _, err := DumpDir(dir, Input{
		SessionID: "sess-unreadable",
		Drive:     session.DefaultState("sess-unreadable"),
		Witness: []WitnessEntry{{
			EffectID: "refund:500",
			Record:   taskmgr.WitnessRecord{VerifiedState: taskmgr.VerifiedDone, Source: "path-witness"},
		}},
		Now: 1_700_000_000,
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}

	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}

	// Replace the indexed witness part with a directory so the read fails
	// deterministically (a directory is not a file -> non-IsNotExist read error).
	wpath := filepath.Join(dir, WitnessFile)
	if err := os.Remove(wpath); err != nil {
		t.Fatalf("remove witness: %v", err)
	}
	if err := os.Mkdir(wpath, 0o755); err != nil {
		t.Fatalf("mkdir witness: %v", err)
	}

	guard, guardErr := NewACRFenceGuard(img)
	if guardErr == nil {
		t.Fatal("NewACRFenceGuard accepted unreadable completion evidence")
	}

	if got := guard.Decide("refund:500"); got != ACRFenceRefuse {
		t.Fatalf("unreadable-evidence guard Decide = %v, want REFUSE", got)
	}
	if got := (*ACRFenceGuard)(nil).Decide("refund:500"); got != ACRFenceRefuse {
		t.Fatalf("nil guard Decide = %v, want REFUSE", got)
	}
	if got := (&ACRFenceGuard{}).Decide("refund:500"); got != ACRFenceRefuse {
		t.Fatalf("zero guard Decide = %v, want REFUSE", got)
	}

	bareGuard, bareErr := NewACRFenceGuard(&Image{})
	if bareErr == nil {
		t.Fatal("NewACRFenceGuard(empty image) returned nil error")
	}
	if got := bareGuard.Decide("refund:500"); got != ACRFenceRefuse {
		t.Fatalf("bare-image guard Decide = %v, want REFUSE", got)
	}

	nilGuard, nilErr := NewACRFenceGuard(nil)
	if nilErr == nil {
		t.Fatal("NewACRFenceGuard(nil image) returned nil error")
	}
	if got := nilGuard.Decide("refund:500"); got != ACRFenceRefuse {
		t.Fatalf("nil-image guard Decide = %v, want REFUSE", got)
	}
}

// TestACRFenceGuardValidWitnessDecidesBytewise proves the guard's positive side: a valid
// keep-bit yields SKIP_DUPLICATE for the witnessed effect, FIRE for an unwitnessed one,
// and DecideAll stays parallel to its input.
func TestACRFenceGuardValidWitnessDecidesBytewise(t *testing.T) {
	dir := t.TempDir()
	if _, err := DumpDir(dir, Input{
		SessionID: "sess-valid",
		Drive:     session.DefaultState("sess-valid"),
		Witness: []WitnessEntry{{
			EffectID: "refund:500",
			Record:   taskmgr.WitnessRecord{VerifiedState: taskmgr.VerifiedDone, Source: "path-witness"},
		}},
		Now: 1_700_000_000,
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}

	guard, err := NewACRFenceGuard(img)
	if err != nil {
		t.Fatalf("NewACRFenceGuard: %v", err)
	}
	if got := guard.Decide("refund:500"); got != ACRFenceSkipDuplicate {
		t.Fatalf("witnessed effect Decide = %v, want SKIP_DUPLICATE", got)
	}
	if got := guard.Decide("refund:501"); got != ACRFenceFire {
		t.Fatalf("unwitnessed effect Decide = %v, want FIRE", got)
	}

	got := guard.DecideAll([]string{"refund:500", "refund:501", "refund:500"})
	want := []ACRFenceVerdict{ACRFenceSkipDuplicate, ACRFenceFire, ACRFenceSkipDuplicate}
	if len(got) != len(want) {
		t.Fatalf("DecideAll len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("DecideAll[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	if s := ACRFenceRefuse.String(); s != "REFUSE" {
		t.Fatalf("Refuse.String() = %q", s)
	}
	if s := ACRFenceFire.String(); s != "FIRE" {
		t.Fatalf("Fire.String() = %q", s)
	}
	if s := ACRFenceSkipDuplicate.String(); s != "SKIP_DUPLICATE" {
		t.Fatalf("SkipDuplicate.String() = %q", s)
	}
	if s := ACRFenceVerdict(99).String(); s != "UNKNOWN" {
		t.Fatalf("out-of-range String() = %q", s)
	}
}
