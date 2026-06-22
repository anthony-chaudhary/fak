// mask.go — the per-replay REGISTRY MASK: a chain-level helper that returns a COPY
// of an adjudicator chain with exactly one named rung removed, so a replay driver
// can ablate a single PDP/PEP link WITHOUT mutating the process-global registry.
//
// WHY THIS IS ADDITIVE TO #500. WithAdjudicators (issue #500) lets a kernel fold an
// EXPLICIT adjudicator chain instead of the process-global registry. This file is the
// other half a lever-flip needs: given that explicit chain, produce a variant with one
// rung DISABLED. Compose the two — WithoutRung(Adjudicators(), "grammar") handed to
// kernel.WithAdjudicators — and you get a kernel that folds every rung EXCEPT grammar,
// with nothing global touched. K such masked chains fold concurrently (each owns its
// own copy; the registry's immutable slice is never mutated), which is what the
// lever-flip causal-attribution driver (turnbench.RunLeverFlip) needs.
//
// HOW A RUNG IS NAMED (generically, no special cases). A rung's identity is its
// SELF-REPORTED By string — the same forensic name the kernel records in every Verdict
// and `fak preflight --explain` prints per rung. RungName probes the adjudicator with a
// neutral sentinel call and reads the By it returns, normalizing any per-call suffix
// (e.g. "ifc-sink(off)" -> "ifc-sink"). Every registered rung reports a STABLE By even
// when it DEFERS (grammar, preflight, monitor, ifc-sink, plancfi, ratelimit, shipgate,
// engine-residency all do), so the name is a deterministic property of the rung, not of
// the call. This is the generic identity the vDSO ablation falls out of: the lever-flip
// driver disables the rung NAMED "vdso" the same way it disables "grammar" — there is no
// SetVDSO special-case in the attribution logic (the fast-path realization differs, but
// the lever is named and flipped uniformly; see turnbench.RunLeverFlip).
package abi

import "context"

// maskProbeTool is the neutral sentinel tool RungName probes a rung with to read its
// self-reported By. It is deliberately a name no real policy allow-lists, so the probe
// exercises a rung's identity (its By) without depending on a particular verdict.
const maskProbeTool = "__rungmask_probe__"

// RungName returns a rung's canonical identity: the By string it self-reports, with any
// parenthesized per-call suffix stripped ("ifc-sink(off)" -> "ifc-sink"). It probes the
// adjudicator with a neutral sentinel call (no engine, no network — a pure in-process
// Adjudicate), so the name is a deterministic property of the rung. A rung that reports
// no By at all yields "" (it cannot be addressed by name — the caller treats an unnamed
// rung as unmaskable rather than removing the wrong one).
func RungName(a Adjudicator) string {
	if a == nil {
		return ""
	}
	v := a.Adjudicate(context.Background(), &ToolCall{Tool: maskProbeTool})
	return canonicalRungName(v.By)
}

// canonicalRungName strips a parenthesized state suffix from a By so two verdicts the
// SAME rung emits in different states ("ifc-sink", "ifc-sink(off)", "ifc-sink(authorized)")
// collapse to one stable name. A By with no suffix is returned unchanged.
func canonicalRungName(by string) string {
	if i := indexByte(by, '('); i >= 0 {
		return by[:i]
	}
	return by
}

func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

// WithoutRung returns a COPY of chain with every rung whose RungName equals name
// removed. The input slice is never mutated (it may be the registry's own immutable
// view), so the result is safe to hand to kernel.WithAdjudicators and to fold
// concurrently with other masked copies. removed reports how many rungs were dropped:
// 0 means no rung in the chain is named name (the mask is a no-op — the caller can
// surface that as "this lever is not present in this chain" rather than silently
// ablating nothing). A rung that reports no name ("") is never matched by a non-empty
// name, so it is always carried through.
func WithoutRung(chain []Adjudicator, name string) (masked []Adjudicator, removed int) {
	out := make([]Adjudicator, 0, len(chain))
	for _, a := range chain {
		if name != "" && RungName(a) == name {
			removed++
			continue
		}
		out = append(out, a)
	}
	return out, removed
}

// RungNames returns the canonical name of every rung in chain, in chain order. It is the
// enumeration a lever-flip driver iterates to ablate each chain rung in turn (one masked
// replay per name). Duplicate names (two rungs reporting the same By) appear once each in
// chain order; an unnamed rung contributes "".
func RungNames(chain []Adjudicator) []string {
	out := make([]string, 0, len(chain))
	for _, a := range chain {
		out = append(out, RungName(a))
	}
	return out
}
