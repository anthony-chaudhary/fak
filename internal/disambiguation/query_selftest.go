package disambiguation

import "fmt"

// QuerySelfTestReport is the machine-readable CLI witness for the query seam.
type QuerySelfTestReport struct {
	Schema        string `json:"schema"`
	IndexVersion  string `json:"index_version"`
	CanonicalTerm string `json:"canonical_term"`
	EntrySchema   string `json:"entry_schema"`
	Complete      bool   `json:"complete"`
}

// RunQuerySelfTest reads the public seed through Query and validates the full
// record rather than inspecting the seed directly.
func RunQuerySelfTest() (QuerySelfTestReport, error) {
	response, err := Query("agent kernel")
	if err != nil {
		return QuerySelfTestReport{}, err
	}
	if response.Schema != QuerySchemaVersion || response.IndexVersion != PublicIndexVersion {
		return QuerySelfTestReport{}, fmt.Errorf("unexpected query versions %q %q", response.Schema, response.IndexVersion)
	}
	if response.Entry.Identity.CanonicalTerm != "agent kernel" {
		return QuerySelfTestReport{}, fmt.Errorf("query returned canonical term %q", response.Entry.Identity.CanonicalTerm)
	}
	if err := response.Entry.Validate(); err != nil {
		return QuerySelfTestReport{}, fmt.Errorf("query returned invalid entry: %w", err)
	}
	return QuerySelfTestReport{
		Schema:        response.Schema,
		IndexVersion:  response.IndexVersion,
		CanonicalTerm: response.Entry.Identity.CanonicalTerm,
		EntrySchema:   response.Entry.Schema,
		Complete:      true,
	}, nil
}
