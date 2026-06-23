// Package wirescreen is rung 1 of the "local model on the wire" proposer spine: a
// small LOCAL model (or any cheap predicate) wired as an ADDITIVE, witnessed screen
// behind the context-MMU's deterministic regex floor. It is the first concrete member
// of the witnessed-lossy-proposer family designed in
// docs/notes/RESEARCH-local-model-on-the-wire-2026-06-23.md. See doc.go for the family
// roadmap (the digest, multi-modal, forecaster, and redactor siblings).
//
// What it does: when FAK_WIRE_SCREEN selects a Screener, this leaf registers an
// abi.SemanticScreen adapter that the context-MMU consults AFTER ScreenBytes. A
// Screener may only ASK to quarantine a result the regex floor already admitted (an
// injection-shaped payload with no literal marker). The hit flows through the MMU's
// existing CAS-pin + PageIn-after-Clear path, so the held bytes are recoverable and a
// miss degrades to floor-only behaviour.
//
// Default-inert: with FAK_WIRE_SCREEN unset, this leaf registers NOTHING with the ABI,
// so abi.SemanticScreens() stays empty and the MMU is exactly the v0.1 regex floor at
// zero added cost. The leaf is safe to blank-import in the defconfig for that reason.
package wirescreen

import (
	"context"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Screener is the extension point a concrete proposer implements. It is a LOSSY
// proposer bounded by a witness: it proposes that a result the regex floor admitted is
// injection-shaped, and the MMU enforces the recoverable quarantine. A Screener is
// NEVER trusted to be correct — a false positive is recoverable (operator Clear +
// PageIn restores the exact bytes) and a false negative degrades to the regex floor.
//
// The reference heuristicScreener here is a deterministic, dependency-free stand-in
// used to prove the wiring. The real value is a small (1-3B) local model registered
// under "model" in a gated follow-on (the build needs the model package + weights +
// a measured latency number before it can default on) — see doc.go.
type Screener interface {
	// Name identifies the screener for FAK_WIRE_SCREEN selection and audit.
	Name() string
	// Flag reports whether body (which SURVIVED the regex floor) is injection-shaped.
	// tool is the producing tool name (may be empty). why is a short human reason for
	// the audit trail. A screener that cannot decide returns (false, "").
	Flag(ctx context.Context, body []byte, tool string) (flagged bool, why string)
}

var (
	mu       sync.RWMutex
	registry = map[string]Screener{}

	active         Screener // the FAK_WIRE_SCREEN-selected screener (nil = inert)
	activeResolved bool     // active resolved from env yet? (lazy, init-order robust)

	flags int64 // lifetime count of results this leaf flagged (observability)
)

// Register adds a named Screener to the catalog. A leaf — the heuristic reference
// here, or a model-backed screener in a follow-on — registers itself from init(); the
// operator selects one with FAK_WIRE_SCREEN=<name>. Registering twice under one name
// replaces the prior entry (last wins), matching the abi RegionBackend idiom.
func Register(name string, s Screener) {
	mu.Lock()
	defer mu.Unlock()
	registry[name] = s
}

// Active returns the Screener selected by FAK_WIRE_SCREEN, or nil when unset/unknown
// (the inert default). Resolution is lazy and once-only: it runs on the first call, by
// which time every leaf's init() has registered its screener, so selection is robust to
// init() ordering across files.
func Active() Screener {
	mu.Lock()
	defer mu.Unlock()
	if !activeResolved {
		active = registry[strings.TrimSpace(os.Getenv("FAK_WIRE_SCREEN"))] // nil if unset/unknown
		activeResolved = true
	}
	return active
}

// Flags reports how many results this leaf has flagged over its lifetime — the leaf's
// own observability peer of ctxmmu.MMU.Screened().
func Flags() int64 { return atomic.LoadInt64(&flags) }

// screenAdapter bridges the selected Screener to the abi.SemanticScreen seam the MMU
// consults. It is registered ONLY when FAK_WIRE_SCREEN is set (see init), so the
// default binary registers nothing and the MMU stays the bare regex floor.
type screenAdapter struct{}

func (screenAdapter) ScreenResult(ctx context.Context, c *abi.ToolCall, body []byte) abi.ScreenAdvice {
	s := Active()
	if s == nil {
		return abi.ScreenAdvice{} // inert: no selected screener
	}
	tool := ""
	if c != nil {
		tool = c.Tool
	}
	if flagged, _ := s.Flag(ctx, body, tool); flagged {
		atomic.AddInt64(&flags, 1)
		return abi.ScreenAdvice{Disposition: abi.ScreenQuarantine, Reason: abi.ReasonTrustViolation, By: "wirescreen:" + s.Name()}
	}
	return abi.ScreenAdvice{}
}

func init() {
	// The deterministic reference screener is always in the catalog so
	// FAK_WIRE_SCREEN=heuristic works out of the box, but it is INERT unless selected.
	Register("heuristic", heuristicScreener{})
	// Register the ABI adapter only when the operator opted in. This keeps the default
	// build's abi.SemanticScreens() empty (zero MMU overhead); the adapter resolves the
	// concrete screener lazily, so a model-backed screener registered by a later init()
	// is still picked up.
	if strings.TrimSpace(os.Getenv("FAK_WIRE_SCREEN")) != "" {
		abi.RegisterSemanticScreen(screenAdapter{})
		abi.RegisterCapability("wirescreen.v1")
	}
}
