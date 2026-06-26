package recall

import (
	"sort"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// REPLICA PARITY (#786, parent #782). The vDSO revocation path already handles ONE
// witness being refuted. The next ECC-style step is parity across INDEPENDENT evidence:
// when the same logical memory is present under several witnesses, agents, or replicas
// and they DISAGREE, reuse should become suspect before the model treats one version as
// truth — the memory-integrity analogue of a multi-replica parity check that refuses to
// serve a word two replicas disagree on without a tiebreaker.
//
// This layer is deliberately PURE and advisory/fail-closed: it compares already-assembled
// candidates and returns a finding (agree / resolved-to-one-live-witness / conflict). It
// never mutates CAS bytes and never silently picks a winner. recall does NOT invent a
// resource taxonomy it cannot honestly derive — the CALLER groups candidates by whatever
// logical-resource identity it already holds (GroupByResource), so there is no hot-path
// O(N) scan for ordinary single-page recall: the grouping happens only where recall has
// already assembled a candidate set.

// ParityVerdict is the disagreement verdict over a group of replicas of one logical
// resource.
type ParityVerdict uint8

const (
	// ParityAgree: the replicas corroborate — they share a single content address. (A lone
	// replica trivially agrees with itself.) Safe to reuse.
	ParityAgree ParityVerdict = iota
	// ParityResolved: the replicas disagree on bytes, but exactly ONE distinct digest is
	// carried by a still-live witness — a clear latest valid witness. That replica is
	// authoritative; every divergent replica (revoked, unwitnessed, or simply different) is
	// sealable by a fail-closed caller.
	ParityResolved
	// ParityConflict: the replicas disagree on bytes with no clear latest valid witness —
	// zero live witnesses, or two-or-more live witnesses on differing bytes. Refuse reuse /
	// require a witness rather than pick one. Fail-closed.
	ParityConflict
)

// String renders the verdict for audit rows and logs.
func (v ParityVerdict) String() string {
	switch v {
	case ParityAgree:
		return "agree"
	case ParityResolved:
		return "resolved"
	case ParityConflict:
		return "conflict"
	}
	return "unknown"
}

// ParityFinding is one logical resource's parity result — an audit row, never a mutation.
// SealSteps name the replicas a fail-closed caller should seal/refuse (only populated for
// ParityResolved, where a single live witness survives); RequireWitness marks a conflict
// the caller must not resolve on its own.
type ParityFinding struct {
	Resource       string        `json:"resource,omitempty"`
	Verdict        ParityVerdict `json:"verdict"`
	Replicas       int           `json:"replicas"`
	Digests        []string      `json:"digests"`            // distinct content addresses (short), sorted
	Disagree       []string      `json:"disagree,omitempty"` // fields that differ across replicas
	Authoritative  int           `json:"authoritative"`      // step of the surviving live-witnessed replica; -1 if none
	SealSteps      []int         `json:"seal_steps,omitempty"`
	RequireWitness bool          `json:"require_witness,omitempty"`
}

// ParityReport folds per-resource findings into the disagreement counts that form the
// audit rows / metrics surface.
type ParityReport struct {
	Findings   []ParityFinding `json:"findings"`
	Resources  int             `json:"resources"`
	Agreements int             `json:"agreements"`
	Resolved   int             `json:"resolved"`
	Conflicts  int             `json:"conflicts"`
}

// ParityAudit is the advisory, fail-closed check a caller runs at recall selection time
// before reusing candidates: it groups THIS loaded image's page table by a caller-supplied
// logical-resource key and reports replica/witness parity disagreements. It is read-only —
// it never mutates CAS bytes or the page table — and groups only where the caller supplies
// identity, so it adds no hot-path scan to ordinary single-page recall. recall does not
// invent the resource taxonomy: the caller, which holds the candidate set and knows what
// "the same resource" means, supplies `key`. (Auto-wiring this into the top-k Recall path
// is deferred: a recall image carries no inherent resource identity to group by — that is
// caller knowledge — so the hook is exposed on Session for the caller to invoke.)
func (s *Session) ParityAudit(key func(Page) string) ParityReport {
	return ParityScan(GroupByResource(s.Manifest.Pages, key))
}

// ParityCheck decides the parity verdict for a group of replicas the caller asserts are
// the same logical resource. It is pure: it reads metadata only and never touches bytes.
func ParityCheck(replicas []Page) ParityFinding {
	f := ParityFinding{Replicas: len(replicas), Authoritative: -1}
	digests := distinctDigests(replicas)
	for _, d := range digests {
		f.Digests = append(f.Digests, short(d))
	}
	// One distinct address (or none) ⇒ the replicas corroborate. No witness math needed: if
	// they agree on bytes there is nothing to disambiguate.
	if len(digests) <= 1 {
		f.Verdict = ParityAgree
		return f
	}

	f.Disagree = disagreements(replicas)

	// The distinct addresses carried by a STILL-LIVE witness. A revoked or unwitnessed
	// replica is not an authority: it cannot be the "clear latest valid witness" the
	// resolution rule requires, so it never resolves a disagreement on its own.
	authDigests := map[string]int{} // live-witnessed digest -> a representative step
	for _, p := range replicas {
		if replicaAuthoritative(p) {
			if _, ok := authDigests[p.Digest]; !ok {
				authDigests[p.Digest] = p.Step
			}
		}
	}

	if len(authDigests) == 1 {
		// Exactly one live-witnessed truth survives. It is authoritative; every replica whose
		// bytes differ from it is sealable (a fail-closed caller seals/refuses those and keeps
		// the witnessed one).
		var truth string
		for d, step := range authDigests {
			truth, f.Authoritative = d, step
		}
		for _, p := range replicas {
			if p.Digest != truth {
				f.SealSteps = append(f.SealSteps, p.Step)
			}
		}
		sort.Ints(f.SealSteps)
		f.Verdict = ParityResolved
		return f
	}

	// Zero or multiple live truths on differing bytes: no clear latest valid witness. Refuse
	// reuse / require a witness rather than silently pick one.
	f.Verdict = ParityConflict
	f.RequireWitness = true
	return f
}

// ParityScan folds caller-supplied resource groups into one report with disagreement
// counts. Findings are emitted in a deterministic (resource-sorted) order.
func ParityScan(groups map[string][]Page) ParityReport {
	rep := ParityReport{Resources: len(groups)}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		f := ParityCheck(groups[k])
		f.Resource = k
		rep.Findings = append(rep.Findings, f)
		switch f.Verdict {
		case ParityAgree:
			rep.Agreements++
		case ParityResolved:
			rep.Resolved++
		case ParityConflict:
			rep.Conflicts++
		}
	}
	return rep
}

// GroupByResource buckets pages by a caller-supplied logical-resource key. Pages whose key
// is empty (no resource identity available) are dropped — parity is only meaningful where
// identity is known. This is the seam the issue's "group only where recall already
// assembled candidates" note points at: the caller, holding the candidate set, supplies
// the key; recall does not scan the whole page table to invent one.
func GroupByResource(pages []Page, key func(Page) string) map[string][]Page {
	groups := map[string][]Page{}
	for _, p := range pages {
		if k := key(p); k != "" {
			groups[k] = append(groups[k], p)
		}
	}
	return groups
}

// replicaAuthoritative reports whether a replica can stand as a latest valid witness: it
// must carry a witness that is not currently refuted in the vDSO ledger. An unwitnessed
// replica has no trust anchor and is never an authority for parity resolution.
func replicaAuthoritative(p Page) bool {
	return p.Witness != "" && !vdso.Default.Revoked(p.Witness)
}

// distinctDigests returns the sorted distinct content addresses across replicas.
func distinctDigests(replicas []Page) []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range replicas {
		if !seen[p.Digest] {
			seen[p.Digest] = true
			out = append(out, p.Digest)
		}
	}
	sort.Strings(out)
	return out
}

// disagreements lists which integrity-relevant fields differ across the replicas — the
// evidence a human reads on a conflict row. Covers the five fields #786 names: digest,
// witness, trust epoch, durability, and descriptor class (here the benign-vs-sealed bit).
func disagreements(replicas []Page) []string {
	var out []string
	if !uniform(replicas, func(p Page) string { return p.Digest }) {
		out = append(out, "digest")
	}
	if !uniform(replicas, func(p Page) string { return p.Witness }) {
		out = append(out, "witness")
	}
	if !uniform(replicas, func(p Page) string { return strconv.FormatUint(p.TrustEpoch, 10) }) {
		out = append(out, "trust_epoch")
	}
	if !uniform(replicas, func(p Page) string { return p.Durability }) {
		out = append(out, "durability")
	}
	if !uniform(replicas, func(p Page) string { return strconv.FormatBool(p.Quarantined) }) {
		out = append(out, "quarantined")
	}
	return out
}

// uniform reports whether key(p) is identical across every replica.
func uniform(replicas []Page, key func(Page) string) bool {
	if len(replicas) < 2 {
		return true
	}
	first := key(replicas[0])
	for _, p := range replicas[1:] {
		if key(p) != first {
			return false
		}
	}
	return true
}
