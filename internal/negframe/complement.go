package negframe

// complement.go is the L2 rung of the negation operator (see
// docs/notes/GLOBAL-WORKSPACE-NEGATION-OPERATOR-AEO-2026-07.md): the positive-complement
// RESOLVER over a declarative domain registry. Where L0 (reframe.go) flips a fixed negative
// IDIOM in token space, L2 answers the harder question the program poses in its functional form
// -- "give me the positive form of not-X over domain D" -- by computing the complement of X in an
// ENUMERABLE domain and handing back the positive residual the model can select from, so the
// model never has to emulate the inversion in its scarce workspace.
//
// The registry is declarative and grounded in fak's own enumerable vocabularies -- the arbiter's
// lock modes, the router's lane kinds, boolean -- not invented toy domains: resolving "not
// shared" to "exclusive" is a real, true substitution the refusal-note and step-advice surfaces
// can make. Two of the program's fail-closed invariants live here:
//
//   - Fabrication is outside the contract. An unknown (or ambiguous) domain, or an X that is not
//     a member of the named domain, yields UNKNOWN with a reason -- never a made-up positive. A
//     confident wrong answer is the failure this rung exists to refuse.
//   - Exactness is honest. The complement is EXACT only when it collapses to a single member; a
//     wider complement is returned as a CANDIDATE SET the caller (or model) selects from, and a
//     domain that is exactly {X} has an empty positive residual (UNKNOWN), not a guess.
//
// Extend the registry with real enumerable domains rather than loosening the membership check --
// the same discipline the negation lexicon keeps with its "un-" allowlist (negframe.go).

import (
	"sort"
	"strings"
)

// ResolutionKind is the closed outcome vocabulary of a complement resolution.
type ResolutionKind string

const (
	// Exact: the domain is enumerable and its complement of X collapses to a single positive.
	// Positive carries it (and Members holds the one-element set).
	Exact ResolutionKind = "exact"
	// Candidates: the domain is enumerable but the complement is a SET (>1). Members carries the
	// positives the caller/model selects from; there is no single substitution.
	Candidates ResolutionKind = "candidates"
	// Unknown: the operator refuses to fabricate -- the domain is unknown or ambiguous, X is not a
	// member of it, or the residual after removing X is empty. Reason states which. Fail-closed.
	Unknown ResolutionKind = "unknown"
)

// Resolution is the outcome of resolving "not Negated" over a domain. It is JSON-shaped for the
// `fak negate resolve --json` surface and any downstream consumer.
type Resolution struct {
	Kind     ResolutionKind `json:"kind"`
	Domain   string         `json:"domain,omitempty"`   // the resolved domain's name ("" when no domain resolved)
	Negated  string         `json:"negated"`            // X in "not X", canonicalized to the domain's spelling when matched
	Positive string         `json:"positive,omitempty"` // the single complement member (Exact only)
	Members  []string       `json:"members,omitempty"`  // the complement set (Exact: 1 elem; Candidates: >1)
	Reason   string         `json:"reason,omitempty"`   // why Unknown (empty otherwise)
}

// Resolved reports whether the resolution produced a usable positive (Exact or Candidates), i.e.
// the operator did NOT fall back to Unknown. Callers gate on this before substituting.
func (r Resolution) Resolved() bool { return r.Kind == Exact || r.Kind == Candidates }

// Domain is one enumerable domain in the complement registry: a name and its ordered members. The
// order is the order the candidate set is reported in, so it is authored intentionally.
type Domain struct {
	Name    string   `json:"name"`
	Members []string `json:"members"`
}

// complementRegistry is the declarative L2 substrate: the enumerable domains over which "not X"
// resolves to an exact positive or a candidate set. Each domain is grounded in a real fak
// vocabulary, so a resolution is a TRUE substitution, not a plausible-looking guess:
//
//   - boolean       -- the two truth values; "not true" resolves exactly to "false".
//   - lock-mode     -- the arbiter's two lease lock modes (internal/leaseref, dos_arbitrate);
//                      "not shared" resolves exactly to "exclusive".
//   - lane-kind     -- the dispatch router's lane kinds (cluster/keyword/global); "not global"
//                      is a candidate set {cluster, keyword}, no single positive.
//   - lease-outcome -- the two admission verdicts a lease request ends in; "not refused" == "granted".
//   - polarity      -- the negation program's own axis; "not negative" == "positive".
//
// Members within a domain are disjoint across domains here, which is what lets DomainOf infer a
// domain from a bare member unambiguously. Keep it that way: a member shared by two domains makes
// the inference ambiguous (DomainOf reports not-found), which is fail-closed but less useful.
var complementRegistry = []Domain{
	{Name: "boolean", Members: []string{"true", "false"}},
	{Name: "lock-mode", Members: []string{"exclusive", "shared"}},
	{Name: "lane-kind", Members: []string{"cluster", "keyword", "global"}},
	{Name: "lease-outcome", Members: []string{"granted", "refused"}},
	{Name: "polarity", Members: []string{"positive", "negative"}},
}

// Domains returns a copy of the registry (stable order) for the `fak negate resolve --list`
// surface and tests. The copy keeps callers from mutating the package's registry.
func Domains() []Domain {
	out := make([]Domain, len(complementRegistry))
	for i, d := range complementRegistry {
		members := make([]string, len(d.Members))
		copy(members, d.Members)
		out[i] = Domain{Name: d.Name, Members: members}
	}
	return out
}

// negationPrefixes are the leading markers StripNegation recognizes as "a negative is in play".
// They are LEADING-only and deliberately narrow -- an embedded "not" mid-phrase is left for
// Classify, the full lexical detector. "no " is omitted on purpose: it over-strips ("no need to")
// where "not"/"non-" attach cleanly to a single atom.
var negationPrefixes = []string{"not-", "not ", "non-", "¬not ", "¬"}

// StripNegation removes a single leading negation marker from s and returns the bare atom plus
// whether a marker was found. It is the tiny "a negative is in play" detector the resolver's
// convenience path uses to turn `fak negate resolve "not shared"` into Resolve("shared", "").
func StripNegation(s string) (string, bool) {
	t := strings.TrimSpace(s)
	lower := strings.ToLower(t)
	for _, p := range negationPrefixes {
		if strings.HasPrefix(lower, p) {
			return strings.TrimSpace(t[len(p):]), true
		}
	}
	return t, false
}

// DomainOf returns the registered domain that contains member x (case-insensitive). If x sits in
// no domain, or in more than one, ok is false -- an ambiguous member must be resolved by NAMING
// the domain, never by a silent pick (fail-closed inference).
func DomainOf(x string) (Domain, bool) {
	x = strings.TrimSpace(x)
	var hit Domain
	n := 0
	for _, d := range complementRegistry {
		if _, ok := memberOf(d, x); ok {
			hit = d
			n++
		}
	}
	if n != 1 {
		return Domain{}, false
	}
	return hit, true
}

// Resolve computes the positive complement of "not negated" over the named domain. A blank domain
// asks Resolve to INFER the domain from negated via DomainOf. Every failure path is fail-closed:
// an unknown/ambiguous domain, an X the domain does not contain, or an empty residual all return
// Unknown with a reason -- the operator never fabricates a positive it cannot witness.
func Resolve(negated, domain string) Resolution {
	neg := strings.TrimSpace(negated)
	res := Resolution{Negated: neg}
	if neg == "" {
		res.Kind = Unknown
		res.Reason = "no term to negate"
		return res
	}

	var d Domain
	if strings.TrimSpace(domain) == "" {
		found, ok := DomainOf(neg)
		if !ok {
			res.Kind = Unknown
			res.Reason = "no single registered domain contains " + quoteTerm(neg) + "; name the domain with --domain"
			return res
		}
		d = found
	} else {
		found, ok := domainByName(domain)
		if !ok {
			res.Kind = Unknown
			res.Domain = strings.TrimSpace(domain)
			res.Reason = "unknown domain " + quoteTerm(domain)
			return res
		}
		d = found
	}
	res.Domain = d.Name

	canon, ok := memberOf(d, neg)
	if !ok {
		res.Kind = Unknown
		res.Reason = quoteTerm(neg) + " is not a member of domain " + quoteTerm(d.Name) + "; cannot compute its complement"
		return res
	}
	res.Negated = canon // canonicalize to the domain's spelling for a deterministic result

	comp := complementOf(d, canon)
	switch len(comp) {
	case 0:
		res.Kind = Unknown
		res.Reason = "domain " + quoteTerm(d.Name) + " has no positive residual after removing " + quoteTerm(canon)
	case 1:
		res.Kind = Exact
		res.Positive = comp[0]
		res.Members = comp
	default:
		res.Kind = Candidates
		res.Members = comp
	}
	return res
}

// domainByName looks a domain up by name (case-insensitive).
func domainByName(name string) (Domain, bool) {
	name = strings.TrimSpace(name)
	for _, d := range complementRegistry {
		if strings.EqualFold(d.Name, name) {
			return d, true
		}
	}
	return Domain{}, false
}

// memberOf reports whether x is a member of d (case-insensitive) and returns the canonical
// spelling the domain declares for it, so a resolution echoes the registry's casing, not the
// caller's.
func memberOf(d Domain, x string) (string, bool) {
	x = strings.TrimSpace(x)
	for _, m := range d.Members {
		if strings.EqualFold(m, x) {
			return m, true
		}
	}
	return "", false
}

// complementOf returns d's members minus x (canonical, case-insensitive), preserving the domain's
// authored order.
func complementOf(d Domain, x string) []string {
	var out []string
	for _, m := range d.Members {
		if !strings.EqualFold(m, x) {
			out = append(out, m)
		}
	}
	return out
}

// quoteTerm wraps a term in double quotes for a reason string, escaping any embedded quote so the
// message stays readable.
func quoteTerm(s string) string {
	return "\"" + strings.ReplaceAll(s, "\"", "'") + "\""
}

// sortedDomainNames returns the registry's domain names sorted, for stable help/list output.
func sortedDomainNames() []string {
	names := make([]string, 0, len(complementRegistry))
	for _, d := range complementRegistry {
		names = append(names, d.Name)
	}
	sort.Strings(names)
	return names
}
