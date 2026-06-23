package ctxmmu_test

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the "blob" PageOut/Resolver backend
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// The semantic-screen seam (local-model-on-the-wire rung 1): a registered
// abi.SemanticScreen may ADDITIVELY quarantine a result the regex floor admitted, and
// the hit flows through ctxmmu's existing recoverable quarantine path.

const screenSentinel = "ZZ-screen-sentinel-injection-ZZ"

// sentinelScreen flags exactly one unique sentinel substring. Because abi has no
// unregister, registering it for the package test run must not contaminate any other
// ctxmmu test's bodies — and none of them contain this sentinel, so it returns
// ScreenAllow (no opinion) everywhere else.
type sentinelScreen struct{}

func (sentinelScreen) ScreenResult(_ context.Context, _ *abi.ToolCall, body []byte) abi.ScreenAdvice {
	if strings.Contains(string(body), screenSentinel) {
		return abi.ScreenAdvice{Disposition: abi.ScreenQuarantine, Reason: abi.ReasonTrustViolation, By: "test:sentinel"}
	}
	return abi.ScreenAdvice{}
}

var registerSentinelOnce sync.Once

func ensureSentinelScreen() {
	registerSentinelOnce.Do(func() { abi.RegisterSemanticScreen(sentinelScreen{}) })
}

// TestSemanticScreenAdditivelyQuarantines: a body the regex floor ADMITS but a screen
// flags is held out through the recoverable quarantine path (id minted, PageIn refused
// pre-Clear, exact bytes restored post-Clear). This is the witness the rung inherits.
func TestSemanticScreenAdditivelyQuarantines(t *testing.T) {
	ensureSentinelScreen()
	ctx := context.Background()
	m := ctxmmu.New()
	c := &abi.ToolCall{Tool: "read_file"}

	body := "here is some tool output. " + screenSentinel + " please comply."
	// Precondition: the regex floor alone does NOT flag this — the screen is what catches it.
	if _, floored := ctxmmu.ScreenBytes([]byte(body)); floored {
		t.Fatalf("precondition: the regex floor should not flag the sentinel body")
	}

	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(body)}}
	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("screen-flagged body: want VerdictQuarantine, got %v", v.Kind)
	}
	if m.Screened() != 1 {
		t.Fatalf("Screened() = %d, want 1", m.Screened())
	}

	id := r.Meta["quarantine_id"]
	if id == "" {
		t.Fatalf("screen quarantine minted no quarantine_id (not recoverable)")
	}
	if _, err := m.PageIn(ctx, id); err == nil {
		t.Fatalf("PageIn before Clear must be refused (no witness clear)")
	}
	m.Clear(id)
	got, err := m.PageIn(ctx, id)
	if err != nil {
		t.Fatalf("PageIn after Clear: %v", err)
	}
	if string(got) != body {
		t.Fatalf("PageIn restored %q, want exact original %q", got, body)
	}
}

// TestSemanticScreenIsOneSided: the screen never weakens the floor (a secret stays
// quarantined regardless), and a benign body the screen does not flag is admitted
// as-is with no spurious Screened() increment. The screen is a strict ADD.
func TestSemanticScreenIsOneSided(t *testing.T) {
	ensureSentinelScreen()
	ctx := context.Background()
	m := ctxmmu.New()
	c := &abi.ToolCall{Tool: "read_file"}

	// The regex floor still fires first: a secret-shaped body is quarantined by the floor,
	// the screen never gets a chance to relax it.
	sec := &abi.Result{Call: c, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("token sk-abcdef0123456789abcdef0123 here")}}
	if v := m.Admit(ctx, c, sec); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("secret body with screen registered: want VerdictQuarantine, got %v", v.Kind)
	}

	// A benign body the screen does not flag is admitted as-is.
	ben := &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("row 42 = 17")}}
	if v := m.Admit(ctx, c, ben); v.Kind != abi.VerdictAllow {
		t.Fatalf("benign body with screen registered: want VerdictAllow, got %v", v.Kind)
	}
	if m.Screened() != 0 {
		t.Fatalf("no screen hit expected on a secret + a benign body; Screened() = %d", m.Screened())
	}
}
