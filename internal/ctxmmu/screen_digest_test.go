package ctxmmu_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // CAS backend backing the digest witness
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// digestMarkerScreen is a test-only abi.SemanticScreen that returns a ScreenDigest
// advisory for bodies carrying a unique marker and is inert otherwise. It is registered
// once per test binary (registerOnce) — the global registry has no unregister — but the
// marker guard means it NEVER fires for any other test's body, so it cannot perturb the
// floor/screen behaviour those tests assert. It proves the rung-3 wiring without depending
// on FAK_WIRE_SCREEN (an init-time env this binary does not set).
type digestMarkerScreen struct{}

func (digestMarkerScreen) ScreenResult(_ context.Context, _ *abi.ToolCall, body []byte) abi.ScreenAdvice {
	if bytes.Contains(body, []byte("digest-me-marker")) {
		return abi.ScreenAdvice{
			Disposition: abi.ScreenDigest,
			Digest:      fmt.Sprintf("DIGEST of a %d-byte marker body", len(body)),
			By:          "test:digest",
		}
	}
	return abi.ScreenAdvice{}
}

var registerOnce sync.Once

func ensureDigestScreen() {
	registerOnce.Do(func() { abi.RegisterSemanticScreen(digestMarkerScreen{}) })
}

// TestAdmitScreenDigestProducesDigestStubAndByteExactRestore is the rung-3 witness /
// acceptance test (issue #570): a ScreenDigest advisory on an oversize-benign body
// produces a Transform verdict whose stub carries the authored digest (useful page-out),
// AND the original stays pinned in CAS so a witness Clear + PageIn restores it
// byte-exact — the digest is lossy display, never the witness.
func TestAdmitScreenDigestProducesDigestStubAndByteExactRestore(t *testing.T) {
	ensureDigestScreen()
	ctx := context.Background()
	m := ctxmmu.New()

	// An oversize body that survives the floor (distinct windows, no secret/injection)
	// and carries the marker so the test screen advises a digest.
	body := append(distinctOversize(8*1024), []byte(" digest-me-marker")...)
	if len(body) <= ctxmmu.OversizeBytes {
		t.Fatalf("body must exceed OversizeBytes, got %d", len(body))
	}

	c := call("dump_table")
	r := result(c, body)
	before := m.Digested()
	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("ScreenDigest oversize: want VerdictTransform, got %v (reason %s)", v.Kind, abi.ReasonName(v.Reason))
	}
	if m.Digested() != before+1 {
		t.Fatalf("Digested() = %d, want %d (the digest page-out must be observed)", m.Digested(), before+1)
	}

	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("Transform verdict payload is not TransformPayload: %T", v.Payload)
	}
	// The stub must carry the authored digest (the useful page-out), not just an opaque pointer.
	stubBytes := resolveBody(t, ctx, tp.NewArgs)
	var stub map[string]any
	if err := json.Unmarshal(stubBytes, &stub); err != nil {
		t.Fatalf("stub is not valid JSON: %v (%s)", err, stubBytes)
	}
	if stub["_paged"] != true {
		t.Errorf("stub _paged = %v, want true", stub["_paged"])
	}
	if _, hasRef := stub["ref"].(string); !hasRef || stub["ref"] == "" {
		t.Errorf("stub missing the non-empty CAS ref the witness needs: %+v", stub["ref"])
	}
	digest, _ := stub["digest"].(string)
	if digest == "" {
		t.Fatalf("stub carries NO digest text — useful page-out did not fire: %s", stubBytes)
	}
	if want := fmt.Sprintf("DIGEST of a %d-byte marker body", len(body)); digest != want {
		t.Errorf("stub digest = %q, want %q", digest, want)
	}

	// The original is the witness: it must restore byte-exact via PageIn AFTER a Clear.
	id := r.Meta["pageout_id"]
	if id == "" {
		t.Fatalf("expected a pageout_id in result meta (held-ledger witness)")
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
		t.Fatalf("PageIn is not byte-exact — the digest is lossy display, never the witness")
	}
}

// TestScreenDigestOnSmallBodyIsAdmittedAsIs proves the one-sided scope: a ScreenDigest
// advisory on a NON-oversize body is captured but NOT applied (the full bytes are strictly
// better than a lossy digest), so the body enters context unchanged. The digest only
// upgrades the oversize page-out path.
func TestScreenDigestOnSmallBodyIsAdmittedAsIs(t *testing.T) {
	ensureDigestScreen()
	ctx := context.Background()
	m := ctxmmu.New()

	body := []byte(`{"ok":"small digest-me-marker body"}`)
	if len(body) > ctxmmu.OversizeBytes {
		t.Fatalf("body must be under OversizeBytes for this case, got %d", len(body))
	}
	c := call("tiny")
	r := result(c, body)
	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("small body with a ScreenDigest advisory: want VerdictAllow (digest ignored, full bytes kept), got %v", v.Kind)
	}
	if m.Digested() != 0 {
		t.Fatalf("Digested() = %d, want 0 (no oversize page-out occurred)", m.Digested())
	}
}

// TestOpaqueOversizeUnchangedWithoutDigest proves the ScreenDigest wiring did not perturb
// the v0.1 opaque oversize page-out: with no digest advised (no marker), an oversize body
// pages out to the opaque {_paged,ref,len} stub exactly as before, with no digest field.
func TestOpaqueOversizeUnchangedWithoutDigest(t *testing.T) {
	ensureDigestScreen()
	ctx := context.Background()
	m := ctxmmu.New()

	body := distinctOversize(8 * 1024) // no marker -> no ScreenDigest advisory
	if len(body) <= ctxmmu.OversizeBytes {
		t.Fatalf("body must exceed OversizeBytes, got %d", len(body))
	}
	c := call("dump_table")
	r := result(c, body)
	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("opaque oversize: want VerdictTransform, got %v", v.Kind)
	}
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("payload not TransformPayload: %T", v.Payload)
	}
	stubBytes := resolveBody(t, ctx, tp.NewArgs)
	var stub map[string]any
	if err := json.Unmarshal(stubBytes, &stub); err != nil {
		t.Fatalf("opaque stub not valid JSON: %v", err)
	}
	if stub["digest"] != nil {
		t.Errorf("opaque oversize stub must carry NO digest: got %+v (the v0.1 pointer must be unchanged)", stub["digest"])
	}
	if r.Meta["pageout_id"] != "" {
		t.Errorf("opaque oversize must not set pageout_id (no held-ledger witness): got %q", r.Meta["pageout_id"])
	}
	if m.Digested() != 0 {
		t.Fatalf("Digested() = %d, want 0 (opaque page-out is not a digest page-out)", m.Digested())
	}
}
