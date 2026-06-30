package issuecatalog

import _ "embed"

// DefaultCatalogPath is the repo-relative path of the embedded catalog (for messages).
const DefaultCatalogPath = "internal/issuecatalog/catalog/perf-enablement.json"

//go:embed catalog/perf-enablement.json
var defaultCatalog []byte

// DefaultRows returns the rows of the embedded performance-enablement catalog. The
// binary is self-contained; --catalog FILE overrides it for iteration.
func DefaultRows() ([]Row, error) { return ParseCatalog(defaultCatalog) }
