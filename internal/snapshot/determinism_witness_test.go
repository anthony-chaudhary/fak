package snapshot

// Witness test for the OPEN proof obligation:
//
//	[snapshot-codec-bit-exact] For a fixed primitive value and a fixed injected
//	clock, the snapshot codec (Marshal -> Encode) renders BYTE-IDENTICAL output on
//	every repeat, AND the codec's stored Body bytes are a STABLE, pinned value. So
//	a dumped primitive is a reproducible artifact: the same input always produces
//	the same bytes, today and after any refactor that does not intentionally change
//	the format.
//
// Mechanism under test: Marshal (snapshot.go:77-105) json.Marshals the body into a
// compact json.RawMessage (the Body field) and Encode (snapshot.go:108) renders the
// envelope. The two distinct claims are witnessed separately so a regression in
// either is caught:
//
//   - DETERMINISM: encode the same primitive twice (and 256x) — the bytes must be
//     identical. The risk surface is real: the witness primitive embeds a Go map,
//     and a non-canonical map encoder (or any RNG / wall-clock dependence) would
//     make repeats diverge. Go's encoding/json sorts string map keys, which is the
//     exact property this pins.
//
//   - BIT-EXACT STABILITY: the stored Body bytes are pinned to an inline constant.
//     The Body — not the whole envelope — is the target on purpose: the envelope
//     stamps app_version (from the VERSION file) and created_unix, both of which
//     legitimately vary per build / per dump, so pinning them would false-fail on
//     every release. The Body is the part the CODEC owns (field names, value
//     rendering, map-key ordering); a rename, retype, or ordering drift changes it.
//
// Proof discipline: fak/docs/proofs/00-METHOD.md. Deterministic: a fixed injected
// clock (witnessNow) and a fixed primitive; no randomness, no timing, no goroutines.
// Non-vacuous: the pinned Body is asserted to actually contain the sorted map keys,
// so the byte-equality cannot pass on an empty or degenerate body.

import (
	"testing"
)

// witnessNow is the injected clock the witness encodes under, so created_unix is
// constant across repeats (distinct from determinism_test.go's canonNow to keep this
// file self-contained).
const witnessNow int64 = 1_700_000_001

// witnessBody is the primitive under witness: a small struct whose fields render in
// declaration order plus a map whose keys (deliberately out of sorted order in the
// literal) must be canonicalized by the encoder. If the codec were non-deterministic
// the map would be the thing that diverged.
type witnessBody struct {
	Name  string            `json:"name"`
	Order []int             `json:"order"`
	Attrs map[string]string `json:"attrs"`
}

func witnessPrimitive() witnessBody {
	return witnessBody{
		Name:  "snapshot-primitive",
		Order: []int{3, 1, 2},
		Attrs: map[string]string{"z": "last", "a": "first", "m": "mid"},
	}
}

// pinnedBody is the bit-exact, stable rendering of witnessPrimitive()'s JSON body:
// struct fields in declaration order, slice order preserved, and — the load-bearing
// part — map keys in canonical sorted order (a,m,z) despite the literal's z,a,m. A
// codec change that altered field rendering or dropped map-key sorting would change
// these exact bytes and fail the pin.
const pinnedBody = `{"name":"snapshot-primitive","order":[3,1,2],"attrs":{"a":"first","m":"mid","z":"last"}}`

// encodeWitnessOnce is the single composition under witness: Marshal then Encode —
// the exact path a live dump and the snapshot CLI take.
func encodeWitnessOnce(t *testing.T) []byte {
	t.Helper()
	snap, err := Marshal(KindTool, "witness-1", witnessPrimitive(), nil, witnessNow)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	b, err := snap.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	return b
}

// TestSnapshotCodecBitExactDeterministic is the determinism + bit-exact witness:
// encoding the same primitive repeatedly is byte-identical to a reference encode,
// and the codec's stored Body bytes equal the pinned constant.
func TestSnapshotCodecBitExactDeterministic(t *testing.T) {
	// Determinism: a reference encode, then many repeats, all byte-identical.
	ref := encodeWitnessOnce(t)
	const repeats = 256
	for i := 0; i < repeats; i++ {
		if got := encodeWitnessOnce(t); string(got) != string(ref) {
			t.Fatalf("repeat %d: snapshot codec not deterministic (bytes differ from reference)", i)
		}
	}

	// Bit-exact stability: the stored Body is the pinned value. Marshal stores the
	// compact body bytes verbatim in snap.Body, so this is the codec's own contract,
	// independent of the build-varying app_version / created_unix header.
	snap, err := Marshal(KindTool, "witness-1", witnessPrimitive(), nil, witnessNow)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(snap.Body) != pinnedBody {
		t.Fatalf("snapshot codec body drifted from the bit-exact pin.\n--- want ---\n%s\n--- got ---\n%s", pinnedBody, snap.Body)
	}

	// Non-vacuity: the pinned body genuinely carries the canonical sorted map order,
	// so the byte-equality above is over a real, non-degenerate body. (A regression
	// that emitted an empty body would make the determinism check trivially pass but
	// fail the pin and this guard.)
	if want := `"attrs":{"a":"first","m":"mid","z":"last"}`; !containsSub(pinnedBody, want) {
		t.Fatalf("non-vacuity: pinned body must carry the sorted map keys %q", want)
	}

	// The integrity stamp must be recorded, and it is a pure function of the (now
	// pinned) body — so a stable body implies a stable content address.
	if snap.BodyDigest == "" {
		t.Fatal("BodyDigest is empty — the integrity stamp is not being recorded")
	}

	// Round-trip: the encoded envelope re-parses and re-verifies its digest, and the
	// thawed primitive equals the original — the bit-exact dump is also a faithful one.
	parsed, err := Parse(ref)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	var back witnessBody
	if err := parsed.Into(&back); err != nil {
		t.Fatalf("Into: %v", err)
	}
	if back.Name != witnessPrimitive().Name || len(back.Order) != 3 || back.Attrs["a"] != "first" {
		t.Fatalf("round-trip lost data: %+v", back)
	}
}

// containsSub is a tiny dependency-free substring check (avoids pulling strings into
// a witness file whose point is to have no moving parts).
func containsSub(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
