package disambiguation

// ProvenanceSelfCheckReport is the JSON witness for strict public provenance.
type ProvenanceSelfCheckReport struct {
	Schema             string `json:"schema"`
	OK                 bool   `json:"ok"`
	RoundTrip          bool   `json:"round_trip"`
	RejectedAbsolute   bool   `json:"rejected_absolute"`
	RejectedEscape     bool   `json:"rejected_escape"`
	RejectedSourceKind bool   `json:"rejected_source_kind"`
}

// ProvenanceSelfCheck exercises the public valid and rejection seams without I/O.
func ProvenanceSelfCheck() ProvenanceSelfCheckReport {
	entry := cloneEntry(publicEntries[0])
	entry.Sources = []SourceWitness{{
		Kind: SourceKindGoSource, Locator: "internal/disambiguation/provenance.go#validateSourceProvenance",
		Revision: "public-selfcheck/1", CheckedAt: "2026-08-11T00:00:00Z", Probe: "fak-disambiguation/provenance-selfcheck",
	}}
	roundTrip := entry.Validate() == nil
	absolute := entry
	absolute.Sources = append([]SourceWitness(nil), entry.Sources...)
	absolute.Sources[0].Locator = "/private/source"
	escape := entry
	escape.Sources = append([]SourceWitness(nil), entry.Sources...)
	escape.Sources[0].Locator = "../fak-private/source"
	kind := entry
	kind.Sources = append([]SourceWitness(nil), entry.Sources...)
	kind.Sources[0].Kind = "private-repository"
	report := ProvenanceSelfCheckReport{
		Schema: "fak-disambiguation-provenance-selfcheck/1", RoundTrip: roundTrip,
		RejectedAbsolute:   ValidationCode(absolute.Validate()) == ErrProvenanceLocatorAbsolute,
		RejectedEscape:     ValidationCode(escape.Validate()) == ErrProvenanceLocatorEscape,
		RejectedSourceKind: ValidationCode(kind.Validate()) == ErrProvenanceSourceKind,
	}
	report.OK = report.RoundTrip && report.RejectedAbsolute && report.RejectedEscape && report.RejectedSourceKind
	return report
}
