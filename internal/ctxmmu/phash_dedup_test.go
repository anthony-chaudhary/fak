package ctxmmu_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // CAS backend backing the dedup witness
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/wirescreen" // PhashScreen: the phash dedup proposer seam
)

// phash_dedup_test.go is the rung-3 multi-modal screenshot-triage WITNESS / acceptance
// suite (issue #571). It drives the REAL phash dedup proposer (via wirescreen.PhashScreen)
// through ctxmmu.MMU.Admit end to end, so the assertions exercise the genuine proposer ->
// ScreenDigest advisory -> digestToPointer reversible Transform path, not a stand-in.
//
// The non-negotiable honesty gate is byte-exact reversibility: a deduped frame must
// PageIn to the original pixels with zero loss. The default-inert contract (no screen
// registered when the operator has not opted in) is proven in
// internal/wirescreen/wirescreen_test.go:TestDefaultInertRegistersNoABIScreen; the cases
// here prove the one-sided fall-through a NON-duplicate frame gets when the arm IS on.

var registerPhashOnce sync.Once

// ensurePhashScreen registers the phash dedup proposer's abi.SemanticScreen once for the
// test binary. It mirrors screen_digest_test.go's ensureDigestScreen: the global registry
// has no unregister, so registration is permanent — but the proposer DECLINES (ScreenAllow)
// for every non-duplicate image and every non-image body, so it cannot perturb any other
// ctxmmu test's assertions.
func ensurePhashScreen() {
	registerPhashOnce.Do(func() { abi.RegisterSemanticScreen(wirescreen.PhashScreen()) })
}

// oversizedBase64Image renders a base64-encoded PNG screenshot body that exceeds
// OversizeBytes (so it routes through the oversize Transform branch) and survives the
// regex floor (no secret/injection/repeat). The seed varies the frame's structure so two
// distinct seeds are perceptually different (their pHashes are far apart), while the same
// seed reproduces the IDENTICAL body — the shape a re-sent screenshot takes.
func oversizedBase64Image(t *testing.T, seed uint64) []byte {
	t.Helper()
	const dim = 128
	img := image.NewGray(image.Rect(0, 0, dim, dim))
	s := seed*2862933555777941757 + 3037000493
	if s == 0 {
		s = 1
	}
	for y := 0; y < dim; y++ {
		for x := 0; x < dim; x++ {
			s = s*6364136223846793005 + 1442695040888963407 // Knuth MMIX LCG
			// Mix x/y/seed in so distinct seeds yield structurally different frames (far-apart
			// pHashes), while the same seed is byte-identical.
			img.SetGray(x, y, color.Gray{Y: uint8(s>>33) ^ uint8(x*int(seed+1)) ^ uint8(y)})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	body := []byte(base64.StdEncoding.EncodeToString(buf.Bytes()))
	if len(body) <= ctxmmu.OversizeBytes {
		t.Fatalf("test screenshot body %d bytes must exceed OversizeBytes %d", len(body), ctxmmu.OversizeBytes)
	}
	if _, floored := ctxmmu.ScreenBytes(body); floored {
		t.Fatalf("test screenshot body tripped the regex floor — pick different content")
	}
	return body
}

// transformStub materializes a Transform verdict's injected pointer to its stub map.
func transformStub(t *testing.T, ctx context.Context, v abi.Verdict) map[string]any {
	t.Helper()
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("verdict payload is not TransformPayload: %T", v.Payload)
	}
	b := resolveBody(t, ctx, tp.NewArgs)
	var stub map[string]any
	if err := json.Unmarshal(b, &stub); err != nil {
		t.Fatalf("transform stub is not valid JSON: %v (%s)", err, b)
	}
	return stub
}

// stubDigest returns the "digest" field of a Transform stub ("" when absent) — the field a
// dedup pointer rides in (the ScreenDigest path), and that the opaque oversize pointer omits.
func stubDigest(t *testing.T, ctx context.Context, v abi.Verdict) string {
	t.Helper()
	d, _ := transformStub(t, ctx, v)["digest"].(string)
	return d
}

// TestPhashDedupCollapseAndByteExactRestore is the LOAD-BEARING witness for issue #571: an
// unchanged screenshot frame admitted after a prior identical frame is collapsed to a
// sub-PointerMax "unchanged, see frame#k" pointer (VerdictTransform), and the original
// pixels page into the CAS via pageOut so a witness Clear + PageIn restores them
// BYTE-EXACT — the dedup pointer is lossy display, the CAS bytes are the witness.
func TestPhashDedupCollapseAndByteExactRestore(t *testing.T) {
	ensurePhashScreen()
	ctx := context.Background()
	m := ctxmmu.New()
	body := oversizedBase64Image(t, 1)

	// First sight: a NEW frame. The proposer declines, so the MMU opaquely pages it out —
	// NOT a dedup. This is the one-sided scope: only a redundant frame is collapsed.
	c1 := call("screenshot")
	r1 := result(c1, body)
	v1 := m.Admit(ctx, c1, r1)
	if v1.Kind != abi.VerdictTransform {
		t.Fatalf("first sight oversize screenshot: want VerdictTransform (opaque page-out), got %v", v1.Kind)
	}
	if d := stubDigest(t, ctx, v1); d != "" {
		t.Fatalf("first sight must NOT be deduped (one-sided: a new frame is not redundant), got digest %q", d)
	}

	// Second sight: the IDENTICAL frame. The proposer emits a dedup pointer; the MMU pages
	// the duplicate pixels out to a stub carrying it.
	c2 := call("screenshot")
	r2 := result(c2, body)
	digestedBefore := m.Digested()
	v2 := m.Admit(ctx, c2, r2)
	if v2.Kind != abi.VerdictTransform {
		t.Fatalf("duplicate frame: want VerdictTransform (dedup), got %v (reason %s)", v2.Kind, abi.ReasonName(v2.Reason))
	}
	digest := stubDigest(t, ctx, v2)
	if !strings.HasPrefix(digest, "unchanged, see frame#") {
		t.Fatalf("duplicate frame stub digest = %q, want an \"unchanged, see frame#k\" dedup pointer", digest)
	}
	// A counter incremented on the dedup collapse (the Digested() peer of Screened()/paged).
	if m.Digested() != digestedBefore+1 {
		t.Fatalf("Digested() = %d, want %d (the dedup collapse must be observed)", m.Digested(), digestedBefore+1)
	}
	// The dedup pointer (the whole injected stub) must fit under PointerMax.
	tp, ok := v2.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("dedup verdict payload is not TransformPayload: %T", v2.Payload)
	}
	if tp.NewArgs.Len >= ctxmmu.PointerMax {
		t.Fatalf("dedup pointer len %d >= PointerMax %d", tp.NewArgs.Len, ctxmmu.PointerMax)
	}

	// THE WITNESS: the deduped frame's original pixels page back in byte-exact AFTER a Clear.
	id := r2.Meta["pageout_id"]
	if id == "" {
		t.Fatalf("dedup page-out minted no pageout_id (the original is not recoverable)")
	}
	if _, err := m.PageIn(ctx, id); err == nil {
		t.Fatalf("PageIn before Clear must be refused (witness discipline)")
	}
	m.Clear(id)
	restored, err := m.PageIn(ctx, id)
	if err != nil {
		t.Fatalf("PageIn after Clear: unexpected error %v", err)
	}
	if !bytes.Equal(restored, body) {
		t.Fatalf("deduped frame did NOT restore byte-exact — the dedup pointer is lossy display, never the witness")
	}
}

// TestPhashDedupNewAndDifferentFramesFallThrough proves the strictly-one-sided scope: a
// perceptually DIFFERENT frame (and any first-sight frame) declines, so the MMU pages it
// out to the OPAQUE oversize pointer — never a dedup pointer, never a new refusal. This is
// the behaviour a non-duplicate sees whether the arm is on or off (the proposer declines),
// which is the default-inert guarantee for non-redundant frames.
func TestPhashDedupNewAndDifferentFramesFallThrough(t *testing.T) {
	ensurePhashScreen()
	ctx := context.Background()
	m := ctxmmu.New()

	for _, seed := range []uint64{42, 99, 197} { // distinct seeds -> perceptually different frames
		body := oversizedBase64Image(t, seed)
		c := call("screenshot")
		r := result(c, body)
		v := m.Admit(ctx, c, r)
		if v.Kind != abi.VerdictTransform {
			t.Fatalf("seed %d: a non-duplicate oversize frame must fall through to opaque page-out (VerdictTransform), got %v", seed, v.Kind)
		}
		if d := stubDigest(t, ctx, v); d != "" {
			t.Errorf("seed %d: a non-duplicate frame must NOT carry a dedup pointer, got digest %q", seed, d)
		}
		if r.Meta["pageout_id"] != "" {
			t.Errorf("seed %d: opaque page-out must not set pageout_id (no held-ledger witness), got %q", seed, r.Meta["pageout_id"])
		}
	}
}

// TestPhashDedupReSendNamesSamePriorFrame proves the dedup pointer is STABLE: repeated
// re-sends of the same frame collapse to the SAME prior-frame number, and each restores
// byte-exact — a redundant frame is never lost across multiple collapses.
func TestPhashDedupReSendNamesSamePriorFrame(t *testing.T) {
	ensurePhashScreen()
	ctx := context.Background()
	m := ctxmmu.New()
	body := oversizedBase64Image(t, 5)

	// Prime: first sight is opaque (new frame, remembered).
	r0 := result(call("screenshot"), body)
	m.Admit(ctx, call("screenshot"), r0)

	// Two re-sends must both dedup to the same prior-frame number and both restore exactly.
	var firstDigest string
	for i := 0; i < 2; i++ {
		r := result(call("screenshot"), body)
		v := m.Admit(ctx, call("screenshot"), r)
		if v.Kind != abi.VerdictTransform {
			t.Fatalf("re-send %d: want VerdictTransform (dedup), got %v", i, v.Kind)
		}
		d := stubDigest(t, ctx, v)
		if !strings.HasPrefix(d, "unchanged, see frame#") {
			t.Fatalf("re-send %d: stub digest = %q, want a dedup pointer", i, d)
		}
		if i == 0 {
			firstDigest = d
		} else if d != firstDigest {
			t.Errorf("re-send %d dedup pointer = %q, want %q (the prior frame is stable)", i, d, firstDigest)
		}
		id := r.Meta["pageout_id"]
		if id == "" {
			t.Fatalf("re-send %d: no pageout_id", i)
		}
		m.Clear(id)
		restored, err := m.PageIn(ctx, id)
		if err != nil {
			t.Fatalf("re-send %d PageIn: %v", i, err)
		}
		if !bytes.Equal(restored, body) {
			t.Fatalf("re-send %d: restored bytes differ from the original frame", i)
		}
	}
}
