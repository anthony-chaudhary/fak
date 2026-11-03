package abi

import "context"

// semscreen.go — the local-model-on-the-wire ADVISORY seam: a registration point for
// the "witnessed lossy proposer" family (docs/notes/RESEARCH-local-model-on-the-wire-
// 2026-06-23.md), where a small LOCAL model (or any cheap predicate) adds SEMANTIC
// judgement the context-MMU's deterministic regex floor cannot. The MMU consults the
// registered chain AFTER ScreenBytes clears a result; a screen may only make the MMU
// MORE restrictive about a floor-admitted result (never weaken/un-quarantine/delete),
// riding the CAS-pin + PageIn-after-witness-Clear machinery so a false positive is
// recoverable and a false negative degrades to floor-only (no screen = v0.1 floor).

// SemanticScreen is one advisory link in the local-model-on-the-wire chain: the MMU consults registered screens only after its regex floor admits a result, so a screen may only make admission MORE restrictive (never weaken, un-quarantine, or delete).
type SemanticScreen interface {
	// ScreenResult inspects a floor-admitted result body and returns an advisory
	// disposition (zero ScreenAdvice = "no opinion"); consulted only on results the
	// floor admitted, so its cost is paid only on the benign path. c may be nil.
	ScreenResult(ctx context.Context, c *ToolCall, body []byte) ScreenAdvice
}

// ScreenDisposition is the advisory a SemanticScreen returns: an OPEN, additive enum —
// NOT part of the closed VerdictKind/Reason/Status freeze (TestABIGoldenFreeze does not
// move) — that the MMU maps onto its own dispositions.
type ScreenDisposition uint8

// ScreenDisposition is a screen's advisory: ScreenAllow gives no opinion (the floor's decision stands); ScreenQuarantine holds the result out (rung 1, wired); ScreenDigest pages it out to a digest-bearing stub (rung 3, wired; issue #570).
const (
	ScreenAllow ScreenDisposition = iota
	ScreenQuarantine
	ScreenDigest
)

// ScreenAdvice is one screen's advisory: Reason rides a ScreenQuarantine into the MMU's
// ReasonTrustViolation/secret path (ReasonNone defaults to ReasonTrustViolation); Digest
// carries the rung-3 summary the page-out stub displays; By names the screen for audit.
type ScreenAdvice struct {
	Disposition ScreenDisposition
	Reason      ReasonCode
	Digest      string
	By          string
}

// RegisterSemanticScreen adds a screen to the chain the context-MMU consults after its
// regex floor. Screens run in registration order and the FIRST non-Allow advisory wins
// (the floor already ran, so any screen hit is a strict add). A leaf registers from init().
func RegisterSemanticScreen(s SemanticScreen) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.screens = append(reg.screens, s)
	rebuildSnapshot()
}

// SemanticScreens returns the registered screen chain as the registry's own immutable
// slice (walk, never mutate); empty when nothing registered — the inert v0.1 default.
func SemanticScreens() []SemanticScreen { return loadSnapshot().screens }
