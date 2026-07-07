package cachemeta

import "testing"

// sysprompt_fingerprint_test.go — the #1563 witness: with a fixed session (fixed
// resident spine+policy prefix), (a) an UNCHANGED sys-prompt overlay reuses the prefix,
// and (b) a CHANGED sys-prompt overlay flips the affected prefix's fingerprint and
// invalidates it (miss), never silently reused. Fails before the fingerprint tie exists
// (no ComputePrefixFingerprint / DecideReuse), passes after.

// residentPrefix is the fixed resident spine+policy span of the session — byte-identical
// across turns, since the overlay lands after the cache breakpoint.
func residentPrefix() []PromptSegment {
	return []PromptSegment{
		{Kind: SegStable, Tokens: 20, Content: []byte("fak spine: the gate, the journal, capabilities")},
		{Kind: SegStable, Tokens: 12, Content: []byte("policy floor: default-deny, versioned, resident")},
	}
}

func overlay(body, witness string) []PromptSegment {
	return []PromptSegment{
		{Kind: SegMessage, Tokens: 8, Content: []byte(body), Witness: witness},
	}
}

func TestSysPromptOverlayUnchangedReusesPrefix(t *testing.T) {
	resident := residentPrefix()
	ov := overlay("skill card: git-review v1 body", "blob-sha256:aaa")

	cached := ComputePrefixFingerprint(resident, ov)
	// Same session, same overlay identity next turn — even a freshly built segment slice.
	current := ComputePrefixFingerprint(residentPrefix(), overlay("skill card: git-review v1 body", "blob-sha256:aaa"))

	if cached.Combined != current.Combined {
		t.Fatalf("unchanged overlay must yield an identical fingerprint: cached %s current %s", cached.Combined, current.Combined)
	}
	d := cached.DecideReuse(current)
	if d.State != PrefixReuse {
		t.Fatalf("unchanged overlay must reuse the prefix, got %s (%s)", d.State, d.Reason)
	}
	if d.OverlayChanged {
		t.Fatalf("unchanged overlay must not flag OverlayChanged: %s", d.Reason)
	}
}

func TestSysPromptOverlayChangeInvalidatesAffectedPrefix(t *testing.T) {
	resident := residentPrefix()
	cached := ComputePrefixFingerprint(resident, overlay("skill card: git-review v1 body", "blob-sha256:aaa"))

	// The sys-prompt overlay changes (edited body → new witness) while the resident
	// spine+policy span stays byte-identical.
	current := ComputePrefixFingerprint(residentPrefix(), overlay("skill card: git-review v2 body EDITED", "blob-sha256:bbb"))

	if cached.ResidentDigest != current.ResidentDigest {
		t.Fatalf("resident span must be unchanged for this case; digests diverged: %s vs %s", cached.ResidentDigest, current.ResidentDigest)
	}
	if cached.Combined == current.Combined {
		t.Fatal("changed overlay must change the affected prefix's fingerprint, but Combined matched — the overlay is not folded into the fingerprint (silent reuse)")
	}

	d := cached.DecideReuse(current)
	if d.State != PrefixInvalidated {
		t.Fatalf("changed overlay must invalidate the prefix (miss), got %s (%s)", d.State, d.Reason)
	}
	if !d.OverlayChanged {
		t.Fatalf("changed overlay with an identical resident span must attribute the miss to the overlay, got OverlayChanged=false (%s)", d.Reason)
	}
}

// TestSysPromptOverlayWitnessRevokeInvalidates proves a capability-body rotation (same
// body text, new witness digest) also moves the fingerprint — the overlay identity is
// witness-sensitive, not only content-sensitive.
func TestSysPromptOverlayWitnessRevokeInvalidates(t *testing.T) {
	resident := residentPrefix()
	cached := ComputePrefixFingerprint(resident, overlay("same body", "blob-sha256:v1"))
	current := ComputePrefixFingerprint(residentPrefix(), overlay("same body", "blob-sha256:v2"))

	if cached.Combined == current.Combined {
		t.Fatal("a rotated overlay witness must change the fingerprint even when the body text matches")
	}
	if d := cached.DecideReuse(current); d.State != PrefixInvalidated || !d.OverlayChanged {
		t.Fatalf("rotated overlay witness must invalidate as an overlay change, got %s OverlayChanged=%v", d.State, d.OverlayChanged)
	}
}

// TestResidentSpanChangeInvalidatesButNotOverlayFlagged proves the orthogonal case: a
// resident-span edit invalidates too, but is NOT attributed to the overlay.
func TestResidentSpanChangeInvalidatesButNotOverlayFlagged(t *testing.T) {
	ov := overlay("skill card body", "blob-sha256:aaa")
	cached := ComputePrefixFingerprint(residentPrefix(), ov)

	edited := residentPrefix()
	edited[0].Content = []byte("fak spine: EDITED head")
	current := ComputePrefixFingerprint(edited, ov)

	d := cached.DecideReuse(current)
	if d.State != PrefixInvalidated {
		t.Fatalf("resident-span edit must invalidate, got %s", d.State)
	}
	if d.OverlayChanged {
		t.Fatalf("a resident-span edit must not be mis-attributed to the overlay: %s", d.Reason)
	}
}
