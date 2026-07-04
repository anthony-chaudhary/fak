// Package microagent is the microagent-minimal registration set: the composable
// SUBSET of the full-kernel defconfig (internal/registrations) that an in-process
// Go microagent host (#2000 M9) blank-imports INSTEAD of the whole defconfig.
//
// The full defconfig wires ~30 subsystems for every fak process. A microagent host
// that only adjudicates and dispatches a tool call needs a strict floor of that:
// the Ref backend + resolver, the vDSO fast-path tiers, the pre-flight rung ladder,
// the DOS reference monitor, the write-time result-admission chain, and the engine
// seam. It does NOT need the opt-in storage tiers, the audit/trajectory recorders,
// the git/ship/plan gates, or the AgentDojo/toollint stewards — those stay in the
// full defconfig, which remains the DEFAULT for `fak serve`/`guard`. Importing this
// package changes nothing about the full build; it is a second, smaller entry point,
// not a replacement.
//
// Enabling or disabling an idea is exactly one blank-import line here, the same rule
// as the full defconfig — the kernel itself never imports a leaf. Every leaf below is
// also in internal/registrations, so this set is a proven, strict subset (witnessed
// structurally and at runtime by microagent_test.go).
//
// The capability floor this set MUST preserve (no adjudication rung silently dropped):
//   - blob        Ref backend + MMU page-out codec — REQUIRED so abi.ActiveResolver()
//     is non-nil; without it the kernel cannot resolve a single Ref.
//   - adjudicator the in-process DOS reference monitor (carries DefaultPolicy() — the
//     POLICY_BLOCK floor) — the authoritative tool-call adjudicator.
//   - ctxmmu      write-time result admission (the quarantine floor).
//   - normgate    normalize-and-rescan admitter (obfuscation-evasion floor, rank 5).
//   - ifc         information-flow control — provenance taint + tainted->sink refusal.
//   - grammar/preflight/ratelimit  the always-on pre-flight rung ladder.
//   - engine/modelengine  the engine seam (mock offline echo + the inkernel model).
//
// Generation intent: gen/second-next architectural exploration (issue #2009). This
// is an OPTION behind an explicit import boundary, never a change to the default
// exposure. Closing evidence for the generation frame:
//
//   - Promotion evidence: microagent_test.go proves this set is a strict, non-empty
//     subset of the full defconfig AND that, linked in isolation, it still carries a
//     live adjudication floor (resolver + DOS reference monitor + result admission +
//     engine seam). Promote it to the M9 host's default entry point once an M8
//     cold-start/RSS measurement confirms the smaller link set yields a material
//     startup/resident-set win (the runtime witness this file cannot produce without
//     a built host).
//   - Demotion / retirement criteria: retire this set (host falls back to the full
//     defconfig) if a floor rung the microagent path needs becomes always-on and is
//     not in this list, or if the M8 cold-start/RSS delta is not material — i.e. the
//     smaller link set buys nothing measurable, so the split's maintenance cost is
//     not justified.
//   - Invalidating assumption: this set assumes the microagent tool-exec path needs
//     ONLY the resolver + adjudicator + result-admission + engine seam, and none of
//     the git/ship/plan gates or the audit/trajectory recorders by default. If a
//     microagent turns out to need one of those always-on, this minimal set silently
//     under-protects — the assumption is invalid and the offending rung must move
//     into the floor list (and into the compatibility test's floor set) here.
package microagent

import (
	// Ref backend + MMU page-out codec — REQUIRED so abi.ActiveResolver() works.
	_ "github.com/anthony-chaudhary/fak/internal/blob"

	// vDSO fast-path tiers.
	_ "github.com/anthony-chaudhary/fak/internal/vdso"

	// Pre-flight rung ladder + grammar rung (always-on floor).
	_ "github.com/anthony-chaudhary/fak/internal/grammar"
	_ "github.com/anthony-chaudhary/fak/internal/preflight"
	_ "github.com/anthony-chaudhary/fak/internal/ratelimit"

	// The in-process DOS reference monitor (authoritative adjudicator, carries the
	// POLICY_BLOCK floor via DefaultPolicy()).
	_ "github.com/anthony-chaudhary/fak/internal/adjudicator"

	// Context-MMU write-time result admission (quarantine floor).
	_ "github.com/anthony-chaudhary/fak/internal/ctxmmu"

	// Normalize-and-rescan admitter (rank 5, obfuscation-evasion floor).
	_ "github.com/anthony-chaudhary/fak/internal/normgate"

	// Information-flow control (source-stamps Ref.Taint; refuses tainted->sensitive
	// sink flows) — the provenance complement to the lexical detectors.
	_ "github.com/anthony-chaudhary/fak/internal/ifc"

	// Stewards (single-invariant validators).
	_ "github.com/anthony-chaudhary/fak/internal/steward"

	// Engine seam: mock (offline echo fallback) + the inkernel model engine.
	_ "github.com/anthony-chaudhary/fak/internal/engine"
	_ "github.com/anthony-chaudhary/fak/internal/modelengine"
)
