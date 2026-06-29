// Package supportmaturity defines the closed, ordered support-maturity ladder
// M0–M7 — the C1 keystone of the support-maturity epic (#1243) and the deliverable
// of #1244. "Supported" today is one fuzzy word collapsing a whole ladder: none /
// loads / runs / correct / optimized / SOTA-parity / beyond-SOTA. That fuzz misroutes
// effort (a 10,000-step optimization loop pointed at something that merely "loads").
// This package replaces the fuzz with a single TOTALLY-ORDERED vocabulary that
// UNIFIES today's scattered support scales instead of adding an (N+1)th parallel one.
//
// What it unifies (and the line drawn against each sibling — see the doctrine note
// docs/notes/support-maturity-ladder-doctrine-1244.md):
//
//   - covmatrix.Support (UNDEFINED/FENCED/PROOF-PATH-ONLY/SUPPORTED) — the LOWER band
//     "is this (family,backend) cell present, honest, and does it run?" → M0,M1,M3,M4.
//   - the ggufload preflight verdicts (READY/REFUSE_*) — the narrower "can it even
//     load on this host?" → M0–M2.
//   - compute.CorrectnessClass (Reference/Approx) — the BAR for M4, not a new rung.
//
// A rung is an ENVELOPE WITH A WITNESS, never a promise: it names the most a cell can
// honestly claim given the witness it carries — a correct-but-slow cell is honestly
// M4, not a failure. This package is VOCABULARY ONLY: it lowers the existing scales
// onto the ladder. Binding each rung to a non-author witness, the shipgate-gated
// promotion rule, and drop-on-regression is the next child (#1245, C2); this file is
// what that child stands on.
package supportmaturity

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// Rung is one position on the closed M0–M7 support-maturity ladder. The underlying
// uint8 makes the ordering TOTAL (M0 < M1 < … < M7) for free; Less is the named form.
type Rung uint8

const (
	// M0None: no honest support — either UNDEFINED (a silently-reachable wrong-result
	// path, covmatrix.UNDEFINED) or a file that is not even a parseable model.
	M0None Rung = iota
	// M1Fenced: an honest refusal. The path does not run, but it refuses out loud (a
	// requirePreNorm-style fence, an unsupported-arch refusal, a capacity refusal)
	// rather than silently diverging. A fence is honest — it is NOT debt.
	M1Fenced
	// M2Loads: the model loads / passes preflight — header parsed, arch known, fits
	// (or fit unknown) — safe to load, but not yet shown to produce output.
	M2Loads
	// M3Runs: runs and is correct on the scalar reference proof path, but the numeric
	// claim is asserted, not proven by a CI oracle (covmatrix.PROOF-PATH-ONLY).
	M3Runs
	// M4Correct: correctness is witnessed by a CI-runnable oracle. covmatrix.SUPPORTED
	// (its definition asserts a CI-runnable witness) and compute.CorrectnessClass land
	// here — Reference holds the bit-exact bar, Approx the argmax+cosine bar.
	M4Correct
	// M5Optimized: runs on the accelerated fast path AND a committed bench witnesses
	// the speedup (compute.Caps advertises the capability; the bench proves it).
	M5Optimized
	// M6Parity: matches the SOTA-local baseline (the turnbench sota-local-baseline
	// parity class).
	M6Parity
	// M7BeyondSOTA: beats the SOTA-local baseline. No current vocabulary witnesses this
	// rung; it is the open top of the ladder.
	M7BeyondSOTA
)

// rungMeta names each rung's short id and one-word doctrine label, indexed by Rung.
var rungMeta = []struct{ id, label string }{
	M0None:       {"M0", "none"},
	M1Fenced:     {"M1", "fenced"},
	M2Loads:      {"M2", "loads"},
	M3Runs:       {"M3", "runs"},
	M4Correct:    {"M4", "correct"},
	M5Optimized:  {"M5", "optimized"},
	M6Parity:     {"M6", "parity"},
	M7BeyondSOTA: {"M7", "beyond-sota"},
}

// Rungs is the closed ladder in ascending maturity order. Its order IS the total order
// Less encodes; the witness test asserts it is complete (M0..M7) and strictly rising.
var Rungs = []Rung{M0None, M1Fenced, M2Loads, M3Runs, M4Correct, M5Optimized, M6Parity, M7BeyondSOTA}

// String renders the rung as its short ladder id ("M0".."M7").
func (r Rung) String() string {
	if int(r) < len(rungMeta) {
		return rungMeta[r].id
	}
	return fmt.Sprintf("M?(%d)", uint8(r))
}

// Label renders the rung's one-word doctrine name ("none", "fenced", "runs", …).
func (r Rung) Label() string {
	if int(r) < len(rungMeta) {
		return rungMeta[r].label
	}
	return "unknown"
}

// Valid reports whether r is one of the closed M0–M7 rungs.
func (r Rung) Valid() bool { return int(r) < len(rungMeta) }

// Less reports whether r is a strictly lower rung than o — the ladder's total order.
func (r Rung) Less(o Rung) bool { return r < o }

// FromSupport lowers a covmatrix.Support value onto the ladder. covmatrix is the
// LOWER-BAND vocabulary — "is this (family,backend) cell present, honest, and does it
// run on the reference?" — so it spans M0–M4 (M2 'loads' is owned by the preflight
// verdict, below):
//
//	UNDEFINED        → M0None    (a silently-reachable wrong-result path — the debt)
//	FENCED           → M1Fenced  (an honest accelerated-path refusal)
//	PROOF-PATH-ONLY  → M3Runs    (runs on the cpu reference, no CI oracle)
//	SUPPORTED        → M4Correct (its definition asserts a CI-runnable witness)
//
// SUPPORTED maps to M4 because covmatrix's own definition (covmatrix.go) is "the cell
// runs AND has a CI-runnable witness". covmatrix.StaleCells already flags the
// accelerated-SUPPORTED cells whose witness is in fact absent; binding that per-cell
// witness so such a cell DROPS to M3 is #1245 (C2). This vocabulary map gives the rung
// the value CLAIMS; the witness binding confirms or drops it. An unrecognized value
// floors to M0None — the honest "we can witness nothing" default.
func FromSupport(s covmatrix.Support) Rung {
	switch s {
	case covmatrix.Supported:
		return M4Correct
	case covmatrix.ProofPathOnly:
		return M3Runs
	case covmatrix.Fenced:
		return M1Fenced
	case covmatrix.Undefined:
		return M0None
	default:
		return M0None
	}
}

// FromPreflightVerdict lowers a ggufload preflight verdict onto the ladder. Preflight
// answers the narrower "can this model even load on this host?" question, so it spans
// the M0–M2 sub-band:
//
//	READY             → M2Loads   (header parsed, arch known, fits — safe to load)
//	REFUSE_TOO_BIG    → M1Fenced  (honest capacity fence: fine model, wrong-size device)
//	REFUSE_BAD_ARCH   → M1Fenced  (honest refusal: the arch / required keys are absent)
//	REFUSE_BAD_HEADER → M0None    (not a parseable model — nothing is defined here)
//
// The two REFUSE_* fences are M1 (honest refusals, not debt); only an unparseable file
// is M0. An unrecognized verdict floors to M0None.
func FromPreflightVerdict(v string) Rung {
	switch v {
	case ggufload.PreflightReady:
		return M2Loads
	case ggufload.PreflightRefuseTooBig, ggufload.PreflightRefuseArch:
		return M1Fenced
	case ggufload.PreflightRefuseHeader:
		return M0None
	default:
		return M0None
	}
}

// FromCorrectnessClass places a compute.CorrectnessClass on the ladder. The class is
// the BAR for M4, not a separate rung: Reference is held to bit-identity + the HF
// argmax oracle, Approx to argmax-exact + a logit-cosine gate. Either, when its gate
// passes, witnesses M4Correct — the class says HOW correctness is judged, never WHETHER
// a higher rung is reached. (Optimization beyond correctness is M5+, witnessed by a
// bench, not by the correctness class.)
func FromCorrectnessClass(compute.CorrectnessClass) Rung { return M4Correct }
