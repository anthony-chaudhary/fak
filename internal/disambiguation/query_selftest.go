package disambiguation

import "fmt"

// QuerySelfTestReport is the machine-readable CLI witness for canonical and
// declared-alias query behavior.
type QuerySelfTestReport struct {
	Schema        string `json:"schema"`
	IndexVersion  string `json:"index_version"`
	CanonicalTerm string `json:"canonical_term"`
	MatchedAlias  string `json:"matched_alias"`
	EntrySchema   string `json:"entry_schema"`
	Complete      bool   `json:"complete"`
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
	return QuerySelfTestReport{
		Schema:        alias.Schema,
		IndexVersion:  alias.IndexVersion,
		CanonicalTerm: alias.Entry.Identity.CanonicalTerm,
		MatchedAlias:  alias.MatchedAlias,
		EntrySchema:   alias.Entry.Schema,
		Complete:      true,
	}, nil
}
