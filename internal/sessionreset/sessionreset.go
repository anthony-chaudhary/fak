// Package sessionreset builds the "human-like" carryover a fresh session is seeded
// with when a long-running session crosses its token budget. It is the SMARTS the
// existing budget-reset directive (session.DebitUsage mints a continuation id;
// gateway emits a SessionResetDirective) never carried: WHAT a human keeps when they
// start a chat over, and what they let go.
//
// THE MODEL. A reset is not "clear everything." It is a selective consolidation —
// the same move CONTEXT-IS-NOT-MEMORY.md makes at the durability gate. This package
// folds a set of pluggable CONTRIBUTORS over the drained session's transcript, each
// proposing one ordered Part of the seed:
//
//   - durabilityFacts — keeps the durable facts (preferences, identity, conventions),
//     drops the turn/session ephemera, by REUSING the shipped ctxmmu durability prior.
//   - taskDistill     — a compact "where we are" recap (objective + last step).
//   - warmPrefix      — a descriptor for replaying the stable prefix from the vCache
//     prefix-DAG so the fresh window doesn't re-pay for the system preamble.
//   - verbatimTail    — the last few turns kept verbatim, like a human keeping the
//     last thing on screen.
//
// THE EXTENSION SEAM. The four built-ins are just the first registrants. Register
// adds a Contributor; BuildSeed folds every registered one deterministically. New
// carryover items ("other items that come up") register without editing this core —
// the registry IS the open seam.
//
// TIER. Mechanism (2). It imports the durability classifier (ctxmmu, 2) and the
// vCache prefix-DAG decision layer (vcachechain, 2) and stdlib — NOTHING upward. It
// deliberately does NOT import the wire message type (internal/agent, tier 4); it
// operates on a minimal local Msg, and the tier-4 gateway converts at the boundary.
// Pure and deterministic: no clock, no network, no randomness, registers nothing into
// the kernel.
package sessionreset

import (
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/vcachechain"
)

// Msg is the minimal transcript line the carryover reasons over — role + text only.
// It is deliberately NOT internal/agent.Message: keeping this package off the wire
// type holds its tier low (a mechanism, not an integrator) and keeps the registry
// reusable by any caller, not just the gateway. The gateway converts agent.Message
// to Msg at the boundary.
type Msg struct {
	Role    string
	Content string
}

// Input is what every contributor sees: the drained session's trace, its transcript
// (oldest-first), and the fresh budget the reset will re-arm with (so a contributor
// can size its contribution to the new window).
type Input struct {
	Trace          string
	Messages       []Msg
	FreshBudgetTok int // the context-token budget the fresh session is re-armed with (0 = unbounded)
}

// Part is one contributor's proposal for the seed. Order fixes the deterministic
// position in the rendered seed (lower first); Text is the rendered carryover line(s);
// Meta is an auditable bag (e.g. how many facts were kept/dropped) that never affects
// the model input. A contributor that has nothing to add returns ok=false.
type Part struct {
	Name  string
	Order int
	Text  string
	Meta  map[string]string
}

// Contributor proposes one Part of the carryover seed. Implementations MUST be pure
// and deterministic over Input (the whole package's determinism rests on it): same
// transcript in, same Part out, so a reset is reproducible and auditable.
type Contributor interface {
	Name() string
	Contribute(Input) (Part, bool)
}

// Seed is the folded carryover: the ordered parts plus the single recap text the
// gateway splices ahead of the fresh session's first request. Parts is retained for
// audit/observability (which contributors fired, with what Meta).
type Seed struct {
	Parts []Part
	Recap string // the rendered system-message text that opens the fresh session

	// WarmPrefix is the #1611 warm-prefix descriptor computed over the stable prefix
	// (the system preamble) BEFORE this reset, when one was present. It lets a
	// consumer verify a later replay/rehydrate of that prefix from the vCache
	// prefix-DAG is byte-identical to the original by comparing digests, instead of
	// re-paying to prefill the part of the window that never changed. nil when the
	// drained transcript had no system preamble to describe.
	WarmPrefix *vcachechain.WarmPrefixDescriptor
}

// registry holds the registered contributors. The four built-ins register in init;
// app code (or a test) adds more via Register. A mutex guards registration so a
// concurrent init order is safe; BuildSeed snapshots under the lock.
var (
	mu       sync.RWMutex
	registry []Contributor
)

// Register adds a contributor to the carryover fold. It is the open extension seam:
// a new carryover item registers here without editing BuildSeed. Registration order
// does NOT affect the seed (BuildSeed sorts by Part.Order then Name), so registrants
// are order-independent. A nil contributor is ignored.
func Register(c Contributor) {
	if c == nil {
		return
	}
	mu.Lock()
	registry = append(registry, c)
	mu.Unlock()
}

// Registered returns the names of the currently-registered contributors (for the
// audit/observability surface and tests). Sorted for determinism.
func Registered() []string {
	mu.RLock()
	names := make([]string, 0, len(registry))
	for _, c := range registry {
		names = append(names, c.Name())
	}
	mu.RUnlock()
	sort.Strings(names)
	return names
}

// BuildSeed folds every registered contributor over in and renders the carryover
// recap deterministically. Contributors that decline (ok=false) drop out; the rest
// are sorted by Order then Name and rendered into one recap block. The fold is pure:
// no clock, no randomness — the same transcript always yields the same Seed.
func BuildSeed(in Input) Seed {
	mu.RLock()
	snapshot := make([]Contributor, len(registry))
	copy(snapshot, registry)
	mu.RUnlock()

	parts := make([]Part, 0, len(snapshot))
	for _, c := range snapshot {
		if p, ok := c.Contribute(in); ok && strings.TrimSpace(p.Text) != "" {
			if p.Name == "" {
				p.Name = c.Name()
			}
			parts = append(parts, p)
		}
	}
	sort.SliceStable(parts, func(i, j int) bool {
		if parts[i].Order != parts[j].Order {
			return parts[i].Order < parts[j].Order
		}
		return parts[i].Name < parts[j].Name
	})
	return Seed{Parts: parts, Recap: renderRecap(parts), WarmPrefix: warmPrefixDescriptorFor(in)}
}

// renderRecap stitches the ordered parts into the single carryover block the fresh
// session opens with. A header names it as a continuation so the model treats it as a
// recap of prior context, not a fresh instruction.
func renderRecap(parts []Part) string {
	if len(parts) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[continuation of a prior session, reset at its token budget — carried-over context follows]\n")
	for _, p := range parts {
		b.WriteString("\n")
		b.WriteString(strings.TrimRight(p.Text, "\n"))
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}
