package recall

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// sentinelMarker is a BENIGN nonce word. It is deliberately invisible to BOTH
// built-in re-screen detectors — canon.Scan (de-obfuscation) does not flag it and
// the bare ctxmmu gate admits it (asserted as a precondition below). Its ONLY
// catcher is a registered ResultAdmitter folded into the re-screen. That isolates
// exactly one question: does recall's page-in re-screen fold the kernel's
// REGISTERED admitter chain, or only its own bare ctxmmu gate?
const sentinelMarker = "qzxmarker7f3a"

// sentinelBody is a benign-looking note carrying the marker — the kind of page a
// weak write-time gate admits as clean.
var sentinelBody = []byte("Quarterly planning note: roadmap " + sentinelMarker + " confirmed for next sprint.")

// sentinelAdmitter is a test-only rank-5 ResultAdmitter (the tier normgate
// occupies) that quarantines any result whose bytes contain sentinelMarker. A real
// registered detector (normgate) is exactly this shape; a synthetic one keeps the
// keep-bit deterministic and weight-free.
type sentinelAdmitter struct{}

func (sentinelAdmitter) Caps() []abi.Capability { return nil }

func (sentinelAdmitter) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	if strings.Contains(string(r.Payload.Inline), sentinelMarker) {
		return abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonTrustViolation, By: "test/sentinel"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test/sentinel"}
}

// TestRecallReScreenInheritsRegisteredAdmitters is the RSI-cycle keep-bit.
//
// THE INVARIANT (GROWTH.md, readmission-gate-strength): recall's page-in re-screen
// must enforce the kernel's REGISTERED ResultAdmitter chain — the same fold
// kvmmu.FoldedGate uses — so a rank-5+ detector (normgate today, anything added
// later) that quarantines a payload ALSO catches it on reload. A bare ctxmmu gate
// (the master construction) ignores the registry, so a payload only a registered
// detector catches sails back in.
//
// RED on master: reScreen falls back to the bare ctxmmu gate, which never consults
// the registry, so the sentinel page resolves and Resolve returns no error.
// GREEN after the fix: reScreen folds abi.ResultAdmittersFor, the sentinelAdmitter
// quarantines, and Resolve returns ErrSealed.
func TestRecallReScreenInheritsRegisteredAdmitters(t *testing.T) {
	ctx := context.Background()

	// Precondition: NEITHER built-in detector may catch the sentinel on its own,
	// or the test would pass on master for the wrong reason (the failure mode that
	// a first draft of this test actually hit — canon flagged a key-like nonce as a
	// secret). Assert both are blind to it before trusting the keep-bit.
	if canon.Scan(sentinelBody).Any() {
		t.Fatal("precondition void: canon.Scan flags the sentinel — pick a marker canon ignores")
	}
	bare := ctxmmu.New().Admit(ctx, &abi.ToolCall{Tool: "read_report"},
		&abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: sentinelBody, Len: int64(len(sentinelBody))}})
	if bare.Kind == abi.VerdictQuarantine {
		t.Fatal("precondition void: bare ctxmmu catches the sentinel — pick a marker it ignores")
	}

	// Register the rank-5 sentinel admitter (process-global registry; the marker is
	// unique so it cannot perturb other tests). Registrations are additive.
	abi.RegisterResultAdmitter(5, sentinelAdmitter{})

	// A weak recorder admitted the page as benign (the sentinel is invisible to its
	// write-time gate). Model that directly: a benign, non-quarantined page.
	d := Digest(sentinelBody)
	s := &Session{
		Manifest: Manifest{
			Version: ManifestVersion,
			Pages: []Page{{
				Step: 0, Role: "read_report", Digest: d, Len: int64(len(sentinelBody)),
				Quarantined: false, Descriptor: "read_report: benign-looking summary",
			}},
		},
		cas:     map[string][]byte{d: sentinelBody},
		cleared: map[string]bool{},
		gate:    ctxmmu.New(),
	}

	_, err := s.Resolve(ctx, 0)
	if err == nil {
		t.Fatal("readmission admitted a page a REGISTERED rank-5 admitter quarantines — " +
			"recall re-screen is using a bare gate that ignores the kernel admitter registry " +
			"(the GROWTH.md readmission-gate-strength weakening). It must fold abi.ResultAdmittersFor.")
	}
	if !errors.Is(err, ErrSealed) {
		t.Fatalf("expected ErrSealed from the folded re-screen, got %v", err)
	}
}
