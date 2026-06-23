package abi

import "context"

// semscreen.go — the local-model-on-the-wire ADVISORY seam.
//
// SemanticScreen is the registration point for the "witnessed lossy proposer"
// family described in docs/notes/RESEARCH-local-model-on-the-wire-2026-06-23.md: a
// small LOCAL model (or any cheap predicate) that adds SEMANTIC judgement the
// context-MMU's deterministic regex floor cannot. The context-MMU consults the
// registered screen chain AFTER ScreenBytes clears a result.
//
// The contract is strictly ADDITIVE and one-sided. A screen may only ask the MMU to
// be MORE restrictive about a result the floor already admitted; it can never weaken
// the floor, un-quarantine, or delete a fact. Whatever it proposes flows through the
// MMU's existing CAS-pin + PageIn-after-witness-Clear machinery, so a false positive
// is fully recoverable and a false negative degrades to floor-only behaviour. With no
// screen registered the MMU is exactly the v0.1 regex floor — default-absent is inert.
//
// This is additive to the frozen ABI: it adds a registry + an OPEN advisory enum,
// touching none of the closed VerdictKind/Reason/Status freeze (TestABIGoldenFreeze
// does not move).
type SemanticScreen interface {
	// ScreenResult inspects a result body that SURVIVED the regex floor and returns
	// an advisory disposition. The zero ScreenAdvice (ScreenAllow) means "no opinion".
	// It is consulted only on results the floor admitted, so its cost is paid only on
	// the benign path. c may be nil; a screen that scopes by tool reads c.Tool.
	ScreenResult(ctx context.Context, c *ToolCall, body []byte) ScreenAdvice
}

// ScreenDisposition is the advisory a SemanticScreen returns. It is an OPEN, additive
// enum — NOT part of the closed VerdictKind freeze — that the MMU maps onto its own
// dispositions. ScreenQuarantine is wired today (rung 1). ScreenDigest is RESERVED for
// the rung-2 useful-page-out upgrade (a model-authored digest replacing the opaque
// oversize pointer); the MMU accepts it in the seam but does not act on it yet, so the
// interface needs no change when rung 2 lands.
type ScreenDisposition uint8

const (
	ScreenAllow      ScreenDisposition = iota // no opinion; the floor's decision stands
	ScreenQuarantine                          // hold this result out (rung 1, wired)
	ScreenDigest                              // RESERVED: page out to a useful digest (rung 2)
)

// ScreenAdvice is one screen's advisory. Reason rides a ScreenQuarantine into the
// MMU's existing ReasonTrustViolation/secret quarantine path; Digest carries the
// rung-2 summary (reserved). By names the screen for observability/audit.
type ScreenAdvice struct {
	Disposition ScreenDisposition
	Reason      ReasonCode // for ScreenQuarantine (ReasonNone defaults to ReasonTrustViolation)
	Digest      string     // for ScreenDigest (rung 2; reserved, ignored today)
	By          string     // screen name, for observability
}

// RegisterSemanticScreen adds a screen to the chain the context-MMU consults after
// its regex floor. Additive to the frozen registry surface; screens run in
// registration order and the FIRST non-Allow advisory wins (the floor already ran, so
// any screen hit is a strict add). A leaf registers its screen from init().
func RegisterSemanticScreen(s SemanticScreen) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.screens = append(reg.screens, s)
	rebuildSnapshot()
}

// SemanticScreens returns the registered screen chain as the registry's own immutable
// slice (walk, never mutate). Empty when nothing registered — the inert v0.1 default,
// for which the MMU's screen loop ranges a nil slice and costs nothing.
func SemanticScreens() []SemanticScreen { return loadSnapshot().screens }
