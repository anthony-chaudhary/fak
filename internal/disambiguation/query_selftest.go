package disambiguation

import (
	"errors"
	"fmt"
)

// QuerySelfTestReport is the machine-readable CLI witness for canonical and
// declared-alias query behavior.
type QuerySelfTestReport struct {
	Schema            string `json:"schema"`
	IndexVersion      string `json:"index_version"`
	CanonicalTerm     string `json:"canonical_term"`
	MatchedAlias      string `json:"matched_alias"`
	EntrySchema       string `json:"entry_schema"`
	Complete          bool   `json:"complete"`
	OverloadedTerm    string `json:"overloaded_term"`
	UnscopedAmbiguous bool   `json:"unscoped_ambiguous"`
	Scope             Scope  `json:"scope"`
}

// RunQuerySelfTest reads the public seed through both strict Query and Resolve.
// It proves alias resolution does not alter canonical ownership.
func RunQuerySelfTest() (QuerySelfTestReport, error) {
	canonical, err := Query("agent kernel")
	if err != nil {
		return QuerySelfTestReport{}, err
	}
	alias, err := Resolve("fused agent kernel")
	if err != nil {
		return QuerySelfTestReport{}, err
	}
	if canonical.Schema != QuerySchemaVersion || canonical.IndexVersion != PublicIndexVersion {
		return QuerySelfTestReport{}, fmt.Errorf("unexpected query versions %q %q", canonical.Schema, canonical.IndexVersion)
	}
	if canonical.Entry.Identity.CanonicalTerm != "agent kernel" || canonical.MatchedAlias != "" {
		return QuerySelfTestReport{}, fmt.Errorf("canonical query returned identity %q alias %q", canonical.Entry.Identity.CanonicalTerm, canonical.MatchedAlias)
	}
	if alias.Entry.Identity.CanonicalTerm != canonical.Entry.Identity.CanonicalTerm || alias.MatchedAlias != "fused agent kernel" {
		return QuerySelfTestReport{}, fmt.Errorf("alias query returned canonical %q alias %q", alias.Entry.Identity.CanonicalTerm, alias.MatchedAlias)
	}
	if err := alias.Entry.Validate(); err != nil {
		return QuerySelfTestReport{}, fmt.Errorf("query returned invalid entry: %w", err)
	}
	scoped, err := runScopeSelfTest()
	if err != nil {
		return QuerySelfTestReport{}, err
	}
	return QuerySelfTestReport{
		Schema:            alias.Schema,
		IndexVersion:      alias.IndexVersion,
		CanonicalTerm:     alias.Entry.Identity.CanonicalTerm,
		MatchedAlias:      alias.MatchedAlias,
		EntrySchema:       alias.Entry.Schema,
		Complete:          true,
		OverloadedTerm:    scoped.term,
		UnscopedAmbiguous: scoped.ambiguous,
		Scope:             scoped.scope,
	}, nil
}

type scopeSelfTestResult struct {
	term      string
	ambiguous bool
	scope     Scope
}

func runScopeSelfTest() (scopeSelfTestResult, error) {
	if _, err := Resolve("kernel"); !errors.Is(err, ErrScopeRequired) {
		return scopeSelfTestResult{}, fmt.Errorf("unscoped overloaded token error = %v, want ErrScopeRequired", err)
	}
	scope := Scope{Kind: "package", Value: "internal/disambiguation"}
	entry, err := ResolveScoped("kernel", scope)
	if err != nil {
		return scopeSelfTestResult{}, fmt.Errorf("resolve scoped overloaded token: %w", err)
	}
	if entry.Entry.Scope != scope {
		return scopeSelfTestResult{}, fmt.Errorf("scope changed from %#v to %#v", scope, entry.Entry.Scope)
	}
	return scopeSelfTestResult{term: "kernel", ambiguous: true, scope: entry.Entry.Scope}, nil
}

func scopeFixtureEntry(term string, scope Scope, target string) Entry {
	entry := SelfTestEntry()
	entry.Identity.CanonicalTerm = term
	entry.Identity.Aliases = []string{}
	entry.Scope = scope
	entry.Contrasts = []Contrast{{
		CanonicalTerm:       target,
		Explanation:         term + " and " + target + " are distinct in this fixture.",
		RequiredPair:        boolPointer(false),
		ForbiddenConflation: boolPointer(true),
	}}
	return entry
}
