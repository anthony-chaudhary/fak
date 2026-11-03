// mask.go — the per-replay REGISTRY MASK: a chain-level helper returning a COPY of an
// adjudicator chain with exactly one named rung removed, so a replay driver can ablate a
// single PDP/PEP link WITHOUT mutating the process-global registry.
//
// WHY THIS IS ADDITIVE TO #500: WithAdjudicators (issue #500) lets a kernel fold an
// EXPLICIT adjudicator chain instead of the process-global registry; this file is the
// other half a lever-flip needs — given that explicit chain, produce a variant with one
// rung DISABLED. Compose the two (WithoutRung(Adjudicators(), "grammar") handed to
// kernel.WithAdjudicators) and you get a kernel folding every rung EXCEPT grammar, with
// nothing global touched — exactly what the lever-flip driver (turnbench.RunLeverFlip) needs.
package abi

import "context"

// maskProbeTool is the neutral sentinel tool RungName probes a rung with to read its
// self-reported By; deliberately a name no real policy allow-lists, so the probe
// exercises a rung's identity without depending on a particular verdict.
const maskProbeTool = "__rungmask_probe__"

// RungName returns a rung's canonical identity: the By string it self-reports, with any
// parenthesized per-call suffix stripped ("ifc-sink(off)" -> "ifc-sink"). It probes the
// adjudicator with a neutral sentinel call (a pure in-process Adjudicate, no engine or
// network), so the name is a deterministic property of the rung. A rung that reports no
// By yields "" (unaddressable by name — the caller treats it as unmaskable).
func RungName(a Adjudicator) string {
	if a == nil {
		return ""
	}
	v := a.Adjudicate(context.Background(), &ToolCall{Tool: maskProbeTool})
	return canonicalRungName(v.By)
}

// canonicalRungName strips a parenthesized state suffix from a By so two verdicts the
// SAME rung emits in different states collapse to one stable name; no suffix is unchanged.
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

// WithoutRung returns a COPY of chain with every rung whose RungName equals name removed.
// The input slice is never mutated (it may be the registry's own immutable view), so the
// result is safe to hand to kernel.WithAdjudicators and fold concurrently with other
// masked copies. removed counts rungs dropped: 0 means no rung is named name (a no-op mask
// the caller can surface as "lever not present"); a rung reporting no name is carried through.
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

// RungNames returns the canonical name of every rung in chain, in chain order — the
// enumeration a lever-flip driver iterates to ablate each rung in turn. Duplicates appear
// once each; an unnamed rung contributes "".
func RungNames(chain []Adjudicator) []string {
	out := make([]string, 0, len(chain))
	for _, a := range chain {
		out = append(out, RungName(a))
	}
	return out
}
