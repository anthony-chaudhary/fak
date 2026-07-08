// Rung B4 (issue #1868): the baton-fidelity scorer. A baton carries pointers, never
// bytes (baton.go), and rung C3 (resolve.go) gave us a per-store Resolver that reports
// whether ONE pointer still resolves. This file lifts that to the whole baton: given a
// baton, resolve every Artifact pointer (commit/issue/memory/ledger/file) and report a
// single fidelity score plus the unresolved set — the core dogfooding metric that makes
// pointer decay measurable BEFORE any enforcement leg acts on it (Track B — Observe).
//
// Read-only by construction. The scorer resolves and tallies; it never mutates the
// baton, the stores, or any ledger (issue "Out of scope: no mutation").
//
// Score definition and its load-bearing honesty choice:
//
//   - Score = Verified / (Verified + Dangling): fidelity among the pointers the resolver
//     could actually JUDGE. A ResolveUnknown (store unreachable, or a pointer kind whose
//     store has no resolver yet — only commit is wired today) is deliberately EXCLUDED
//     from the denominator. Counting unknown against the score would let a not-yet-built
//     resolver make an honest baton look unfaithful; that is the same fail-closed
//     doctrine resolve.go encodes (unknown != dangling). Unknown is still surfaced in the
//     counts and in Unresolved so nothing is hidden.
//   - When no pointer could be judged (Verified+Dangling == 0 — an empty baton, or every
//     pointer unknown), Score is 1.0 by vacuous truth: the scorer detected no dangling
//     pointer. The counts (Total/Verified/Dangling/Unknown) carry the real story, so a
//     consumer that wants "at least N verified" gates on the counts, not the ratio alone.
package relay

// Fidelity is the read-only score of one baton's pointer set against its durable stores.
// It reports one ratio plus the full verdict tally and the concrete unresolved pointers,
// so a reader gets both the headline metric and the evidence behind it.
type Fidelity struct {
	// Total is the number of artifact pointers scored (len(baton.Artifacts)).
	Total int `json:"total"`
	// Verified is the count of pointers the resolver confirmed resolve in their store.
	Verified int `json:"verified"`
	// Dangling is the count of pointers whose store was reachable but which do not
	// resolve — the decay the fidelity metric exists to make measurable.
	Dangling int `json:"dangling"`
	// Unknown is the count of pointers no verdict could be reached for (store unreachable
	// or no resolver for the kind yet). Excluded from Score's denominator; never counted
	// as either verified or dangling.
	Unknown int `json:"unknown"`
	// Score is Verified / (Verified + Dangling): fidelity among judged pointers, in
	// [0,1]. 1.0 when nothing could be judged (see file header).
	Score float64 `json:"score"`
	// Unresolved is every Resolution whose verdict is not ResolveVerified — dangling AND
	// unknown — in artifact order. It is the "unresolved set" the done condition names:
	// the actionable list of pointers a successor cannot rely on.
	Unresolved []Resolution `json:"unresolved"`
}

// ScoreBatonFidelity resolves every pointer in b through r and reports the aggregate
// fidelity. It is pure over the injected Resolver (mirroring VerifyReload/CheckBatonStale)
// so it is hermetic and testable without live stores. It reads b; it never writes.
//
// A baton whose kinds span more than one store should be scored with a Resolver that owns
// each kind — see MultiResolver, which dispatches per kind so a commit-only resolver does
// not report every issue/memory/ledger/file pointer as unknown.
func ScoreBatonFidelity(b Baton, r Resolver) Fidelity {
	f := Fidelity{Total: len(b.Artifacts), Unresolved: []Resolution{}}
	for _, a := range b.Artifacts {
		res := r.Resolve(a)
		switch res.Verdict {
		case ResolveVerified:
			f.Verified++
		case ResolveDangling:
			f.Dangling++
			f.Unresolved = append(f.Unresolved, res)
		default: // ResolveUnknown — no verdict; excluded from the score, still surfaced.
			f.Unknown++
			f.Unresolved = append(f.Unresolved, res)
		}
	}
	judged := f.Verified + f.Dangling
	if judged == 0 {
		f.Score = 1.0 // vacuous: no dangling pointer detected (see file header).
		return f
	}
	f.Score = float64(f.Verified) / float64(judged)
	return f
}

// MultiResolver dispatches an Artifact to the first sub-resolver that returns a verdict
// other than ResolveUnknown, so a baton spanning several stores (commit/issue/memory/
// ledger/file) is resolved by whichever per-store resolver owns each kind. If every
// sub-resolver returns ResolveUnknown (no store owns the kind yet) the composite reports
// ResolveUnknown — the honest "no resolver for this kind" state, never a false positive.
type MultiResolver struct {
	resolvers []Resolver
}

// NewMultiResolver composes sub-resolvers in priority order (first non-unknown wins).
func NewMultiResolver(resolvers ...Resolver) MultiResolver {
	return MultiResolver{resolvers: resolvers}
}

// Resolve returns the first non-unknown verdict among the sub-resolvers, else a single
// ResolveUnknown for the kind. An empty MultiResolver resolves everything to unknown.
func (m MultiResolver) Resolve(a Artifact) Resolution {
	for _, r := range m.resolvers {
		res := r.Resolve(a)
		if res.Verdict != ResolveUnknown {
			return res
		}
	}
	return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: "no resolver owns kind " + a.Kind}
}
