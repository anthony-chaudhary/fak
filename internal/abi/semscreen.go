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
// dispositions. ScreenQuarantine is wired today (rung 1). ScreenDigest is wired as the
// rung-3 useful-page-out (issue #570): the MMU maps it onto its oversize Transform so
// the page-out stub carries the model-authored digest, with the original retained in CAS
// and a witness PageIn-after-Clear restoring it byte-exact.
type ScreenDisposition uint8

const (
	ScreenAllow      ScreenDisposition = iota // no opinion; the floor's decision stands
	ScreenQuarantine                          // hold this result out (rung 1, wired)
	ScreenDigest                              // page out to a digest-bearing stub (rung 3, wired; issue #570)
)

// ScreenAdvice is one screen's advisory. Reason rides a ScreenQuarantine into the
// MMU's existing ReasonTrustViolation/secret quarantine path; Digest carries the
// rung-3 summary that the oversize page-out stub displays. By names the screen for
// observability/audit.
type ScreenAdvice struct {
	Disposition ScreenDisposition
	Reason      ReasonCode // for ScreenQuarantine (ReasonNone defaults to ReasonTrustViolation)
	Digest      string     // for ScreenDigest (rung 3; the oversize page-out stub carries it)
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
