package cachemeta

import "testing"

// TestTierLadderOrder pins the hot->cold ordering, with CXL and NUMA-far slotted as
// first-class tiers between local DRAM and disk — the structural claim of the
// hardware-aware tier model.
func TestTierLadderOrder(t *testing.T) {
	want := []ResidencyTier{TierHBM, TierDRAM, TierNUMAFar, TierCXL, TierDisk}
	for i := 1; i < len(want); i++ {
		if TierRank(want[i-1]) >= TierRank(want[i]) {
			t.Fatalf("tier %s should rank hotter than %s", want[i-1], want[i])
		}
	}
	// Off-ladder tiers sort after the local hierarchy.
	if TierRank(TierRemote) <= TierRank(TierDisk) {
		t.Fatalf("remote should rank colder than disk")
	}
	if !IsLocalTier(TierCXL) || !IsLocalTier(TierNUMAFar) {
		t.Fatalf("CXL and NUMA-far must be local relocation tiers")
	}
	if IsLocalTier(TierRemote) || IsLocalTier(TierRecompute) {
		t.Fatalf("remote/recompute are not local relocation tiers")
	}
}

// TestNextColderWarmer walks the demote and promote ladders end to end.
func TestNextColderWarmer(t *testing.T) {
	steps := []struct{ from, colder ResidencyTier }{
		{TierHBM, TierDRAM},
		{TierDRAM, TierNUMAFar},
		{TierNUMAFar, TierCXL},
		{TierCXL, TierDisk},
		{TierDisk, TierRecompute},
		{TierRemote, TierUnknown},
	}
	for _, s := range steps {
		if got := NextColderTier(s.from); got != s.colder {
			t.Fatalf("NextColderTier(%s)=%s want %s", s.from, got, s.colder)
		}
	}
	if NextWarmerTier(TierCXL) != TierNUMAFar {
		t.Fatalf("warmer than CXL should be NUMA-far")
	}
	if NextWarmerTier(TierHBM) != TierUnknown {
		t.Fatalf("nothing is warmer than HBM")
	}
}

// TestAttendableInPlace asserts the property that makes CXL/NUMA-far demotion cheap:
// byte-addressable + coherent tiers are attendable without staging back; disk/remote
// are not.
func TestAttendableInPlace(t *testing.T) {
	p := DefaultTierProfiles()
	for _, hot := range []ResidencyTier{TierHBM, TierDRAM, TierNUMAFar, TierCXL} {
		if !p[hot].AttendableInPlace() {
			t.Fatalf("%s should be attendable in place", hot)
		}
	}
	for _, cold := range []ResidencyTier{TierDisk, TierRemote} {
		if p[cold].AttendableInPlace() {
			t.Fatalf("%s should NOT be attendable in place", cold)
		}
	}
}

// TestDefaultProfilesMonotone checks the defaults form a sane hierarchy: capacity
// grows and bandwidth falls as tiers get colder (HBM->DRAM->CXL->Disk).
func TestDefaultProfilesMonotone(t *testing.T) {
	p := DefaultTierProfiles()
	chain := []ResidencyTier{TierHBM, TierDRAM, TierCXL, TierDisk}
	for i := 1; i < len(chain); i++ {
		hi, lo := p[chain[i-1]], p[chain[i]]
		if lo.BandwidthMBPerSec >= hi.BandwidthMBPerSec {
			t.Fatalf("%s bandwidth should be below %s", lo.Tier, hi.Tier)
		}
		if lo.CapacityBytes <= hi.CapacityBytes {
			t.Fatalf("%s capacity should exceed %s", lo.Tier, hi.Tier)
		}
	}
}

// TestShareKindZeroCopy asserts the fail-safe default and the zero-copy kinds.
func TestShareKindZeroCopy(t *testing.T) {
	if ShareCopy.ZeroCopy() {
		t.Fatalf("ShareCopy (zero value) must NOT be zero-copy")
	}
	var unset ShareKind
	if unset.ZeroCopy() {
		t.Fatalf("the zero value ShareKind must default to copy")
	}
	for _, k := range []ShareKind{ShareMmap, ShareCXLHDM, ShareRDMA, ShareDmabuf} {
		if !k.ZeroCopy() {
			t.Fatalf("%s should be zero-copy", k)
		}
	}
}

// TestWithShareOnEntry threads a zero-copy descriptor through the Option/Entry seam.
func TestWithShareOnEntry(t *testing.T) {
	e := FromKVPrefix(KVPrefix{Tokens: []int{1, 2, 3}, ModelID: "m", TokenizerID: "tok"},
		WithShare(ShareDescriptor{Kind: ShareCXLHDM, Handle: "0xdeadbeef", Coherent: true}))
	if !e.Residency.ZeroCopy() {
		t.Fatalf("entry residency should advertise zero-copy")
	}
	if e.Residency.Share.Kind != ShareCXLHDM || e.Residency.Share.Handle != "0xdeadbeef" {
		t.Fatalf("share descriptor not threaded: %+v", e.Residency.Share)
	}
	// An entry with no share descriptor defaults to copy (fail-safe).
	plain := FromKVPrefix(KVPrefix{Tokens: []int{1}})
	if plain.Residency.ZeroCopy() {
		t.Fatalf("a plain entry must default to copy, not zero-copy")
	}
}
