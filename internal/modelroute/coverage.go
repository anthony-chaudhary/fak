package modelroute

import "sort"

// coverage.go — the MANIFEST <-> ROSTER cross-check: does a user's OWN account
// roster actually bind every model id their routing manifest can emit? modelroute.go
// picks abstract model ids; account.go binds ONE id to one account. Nothing tied the
// WHOLE routing policy to the WHOLE roster, so an operator could ship a manifest whose
// `guard-b` member has no binding and no default and only discover it when that route
// fails at dispatch. This closes the gap OFFLINE: enumerate every routed id and report,
// per id, whether it is explicitly bound, served only by the default account, or
// UNBOUND (a fail-at-dispatch hole). Pure, deterministic, stdlib-only — the same
// posture as the resolver it complements. It is the seam that makes "route any aspect
// to any model" answerable against "which of MY accounts serves it" for the whole
// policy at once, not one decision at a time.

// ModelIDs returns every distinct model id the Manifest can route to — every member
// and every scout across all rules AND the default plan — sorted. It is exactly the
// set an account roster must bind (or serve via its default) to cover the policy.
func (m Manifest) ModelIDs() []string {
	seen := map[string]bool{}
	add := func(id string) {
		if id != "" {
			seen[id] = true
		}
	}
	collect := func(p Plan) {
		for _, id := range p.Models() {
			add(id)
		}
		add(p.Scout)
	}
	for _, r := range m.Rules {
		collect(r.Plan)
	}
	collect(m.Default)
	out := make([]string, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// CoverageStatus is how a roster serves one routed model id.
type CoverageStatus string

const (
	// CoverageBound: the id has an explicit binding to a named account.
	CoverageBound CoverageStatus = "bound"
	// CoverageViaDefault: no explicit binding, but the roster's default account
	// serves it (the id is used verbatim as the upstream wire name).
	CoverageViaDefault CoverageStatus = "via-default"
	// CoverageUnbound: no binding AND no default — this route FAILS at dispatch.
	CoverageUnbound CoverageStatus = "unbound"
)

// CoverageRow is one routed id's disposition under a roster.
type CoverageRow struct {
	Model         string         `json:"model"`
	Status        CoverageStatus `json:"status"`
	Account       string         `json:"account,omitempty"`
	Kind          ProviderKind   `json:"kind,omitempty"`
	UpstreamModel string         `json:"upstream_model,omitempty"`
}

// Coverage is the whole-policy cross-check: one row per routed id (in the order
// passed, i.e. the sorted ModelIDs order) plus the bound / via-default / unbound
// tallies. Unbound > 0 means the roster does NOT cover the manifest — a fail-closed
// condition a caller should surface as an error, not a warning.
type Coverage struct {
	Rows    []CoverageRow `json:"rows"`
	Bound   int           `json:"bound"`
	Default int           `json:"via_default"`
	Unbound int           `json:"unbound"`
}

// hasBinding reports whether the roster carries an explicit binding for modelID.
func (r Roster) hasBinding(modelID string) bool {
	for _, b := range r.Bindings {
		if b.Model == modelID {
			return true
		}
	}
	return false
}

// Cover cross-checks a set of routed model ids against the roster, classifying each
// as explicitly bound, served only by the default account, or unbound. Pure: it
// assumes a validated roster and does no I/O (never os.Getenv, never a network probe).
func (r Roster) Cover(ids []string) Coverage {
	var cov Coverage
	for _, id := range ids {
		switch {
		case r.hasBinding(id):
			cov.Rows = append(cov.Rows, coverageRow(r, id, CoverageBound))
			cov.Bound++
		case r.Default != "":
			cov.Rows = append(cov.Rows, coverageRow(r, id, CoverageViaDefault))
			cov.Default++
		default:
			cov.Rows = append(cov.Rows, CoverageRow{Model: id, Status: CoverageUnbound})
			cov.Unbound++
		}
	}
	return cov
}

// coverageRow resolves id to its serving account for a covered (bound or default)
// disposition. A resolve error on a supposedly-covered id (only possible on an
// unvalidated roster) leaves the account fields empty rather than panicking.
func coverageRow(r Roster, id string, status CoverageStatus) CoverageRow {
	row := CoverageRow{Model: id, Status: status}
	if t, err := r.Resolve(id); err == nil {
		row.Account, row.Kind, row.UpstreamModel = t.Account, t.Kind, t.UpstreamModel
	}
	return row
}
