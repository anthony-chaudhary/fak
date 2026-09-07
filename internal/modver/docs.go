package modver

// docs.go provides the lookup helpers and query seams for the docs/ prose
// keyspace (#2460).
//
// The docs/ keyspace versions prose pages so doc freshness is a derived rev
// rather than an ad-hoc guess. It uses a hybrid granularity matching doc structure:
//   - A top-level page (docs/<file>.md) stands alone, so it is file-keyed.
//   - A page in a section (docs/<dir>/...) belongs to a section revised as a unit,
//     so it is directory-keyed to docs/<dir> at any depth.
//   - Non-.md paths (such as the nightrun ledger docs/nightrun/module-versions.jsonl,
//     site config _config.yml, SVGs, and text files) are excluded from the module
//     keyspace so that writing or appending to data ledgers cannot bump any docs
//     module (keeping ledger stamping convergent).
//
// Consumers (like docfreshrsi or devfresh) holding a doc path can query its
// revision from an active Snapshot Report or directly from raw ledger JSONL bytes
// without shelling out to git or performing history walks.

import (
	"strings"
)

// DocModule maps a doc path (such as "docs/fak/edge-quickstart.md",
// "docs/architecture.md", "docs/fak", "docs:fak", or a doc-relative path like
// "fak/edge-quickstart.md" or "architecture.md") to its canonical module name in
// the docs keyspace. Non-markdown files (data ledgers, site config) and paths
// outside the docs keyspace return ok=false.
func DocModule(path string) (string, bool) {
	path = strings.TrimSpace(strings.ReplaceAll(path, "\\", "/"))
	if path == "" || path == "docs" {
		return "", false
	}
	if strings.HasPrefix(path, "docs:") {
		path = "docs/" + strings.TrimPrefix(path, "docs:")
	}

	// If it is already a full doc path that moduleOf recognizes, use that.
	if name, kind, ok := moduleOf(path); ok && kind == "docs" {
		return name, true
	}

	// If path starts with docs/, it might be a section directory module name
	// like "docs/fak" (which does not have a .md suffix).
	if strings.HasPrefix(path, "docs/") {
		rest := strings.TrimPrefix(path, "docs/")
		parts := strings.Split(rest, "/")
		if len(parts) > 0 && parts[0] != "" {
			// If the last segment has an extension and is not .md (e.g. docs/nightrun/ledger.jsonl),
			// it is data, not a docs module.
			last := parts[len(parts)-1]
			if dot := strings.LastIndex(last, "."); dot >= 0 {
				if last[dot:] != ".md" {
					return "", false
				}
			}
			if len(parts) == 1 && strings.HasSuffix(parts[0], ".md") {
				return "docs/" + parts[0], true
			}
			return "docs/" + parts[0], true
		}
		return "", false
	}

	// Try doc-relative path (e.g. "architecture.md" or "fak/edge-quickstart.md")
	// unless it clearly belongs to another known root.
	for _, root := range trackedRoots {
		if root != "docs" && (path == root || strings.HasPrefix(path, root+"/")) {
			return "", false
		}
	}
	if name, kind, ok := moduleOf("docs/" + path); ok && kind == "docs" {
		return name, true
	}

	// Also handle bare section name relative to docs, e.g. "fak" -> "docs/fak"
	if !strings.Contains(path, "/") {
		if dot := strings.LastIndex(path, "."); dot < 0 {
			return "docs/" + path, true
		}
	}

	return "", false
}

// DocRev returns the derived revision of the given doc path from the report.
// docPath may be a file path ("docs/architecture.md", "docs/fak/edge-quickstart.md"),
// a section directory ("docs/fak"), a prefixed form ("docs:fak"), or a relative
// doc path ("fak/edge-quickstart.md"). Returns (rev, true) if found in the report,
// or (0, false) if the path is not in the docs keyspace or not present in the report.
func (r Report) DocRev(docPath string) (int, bool) {
	m, ok := r.LookupDoc(docPath)
	if !ok {
		return 0, false
	}
	return m.Rev, true
}

// LookupDoc finds the Module for the given doc path in the report.
// docPath is resolved to its canonical module name via DocModule.
func (r Report) LookupDoc(docPath string) (Module, bool) {
	modName, ok := DocModule(docPath)
	if !ok {
		// Fallback: check if docPath directly matches a docs module in the report.
		modName = strings.TrimSpace(strings.ReplaceAll(docPath, "\\", "/"))
	}
	altName := ""
	if strings.HasPrefix(modName, "docs/") {
		altName = "docs:" + strings.TrimPrefix(modName, "docs/")
	} else if strings.HasPrefix(modName, "docs:") {
		altName = "docs/" + strings.TrimPrefix(modName, "docs:")
	}

	for _, m := range r.Modules {
		if (m.Name == modName || (altName != "" && m.Name == altName)) && m.Kind == "docs" {
			return m, true
		}
	}
	return Module{}, false
}

// LookupModule finds any module in the report by exact name.
func (r Report) LookupModule(name string) (Module, bool) {
	name = strings.TrimSpace(strings.ReplaceAll(name, "\\", "/"))
	for _, m := range r.Modules {
		if m.Name == name {
			return m, true
		}
	}
	return Module{}, false
}

// DocRevs returns a map of all docs modules in the report to their revisions.
func (r Report) DocRevs() map[string]int {
	revs := make(map[string]int)
	for _, m := range r.Modules {
		if m.Kind == "docs" {
			revs[m.Name] = m.Rev
		}
	}
	return revs
}

// DocRevFromReport returns the revision of the given doc path in rep.
func DocRevFromReport(rep Report, docPath string) (int, bool) {
	return rep.DocRev(docPath)
}

// LookupDocRev returns the revision of the given doc path in rep.
func LookupDocRev(rep Report, docPath string) (int, bool) {
	return rep.DocRev(docPath)
}

// LookupDoc returns the Module for the given doc path in rep.
func LookupDoc(rep Report, docPath string) (Module, bool) {
	return rep.LookupDoc(docPath)
}

// LatestLedgerRows returns a map from module name to its latest LedgerRow in the ledger.
func LatestLedgerRows(ledger []byte) map[string]LedgerRow {
	rows := parseLedgerRows(ledger)
	latest := make(map[string]LedgerRow, len(rows))
	for _, r := range rows {
		latest[r.Module] = r
	}
	return latest
}

// DocRevFromLedger queries the latest revision of a doc from raw ledger JSONL bytes.
// docPath is resolved to its canonical docs module via DocModule.
func DocRevFromLedger(ledger []byte, docPath string) (int, bool) {
	row, ok := LookupDocFromLedger(ledger, docPath)
	if !ok {
		return 0, false
	}
	return row.Rev, true
}

// LookupDocRevFromLedger is a synonym for DocRevFromLedger.
func LookupDocRevFromLedger(ledger []byte, docPath string) (int, bool) {
	return DocRevFromLedger(ledger, docPath)
}

// LookupDocFromLedger finds the latest LedgerRow for docPath in raw ledger JSONL bytes.
func LookupDocFromLedger(ledger []byte, docPath string) (LedgerRow, bool) {
	return LookupDocFromRows(parseLedgerRows(ledger), docPath)
}

// DocRevFromRows finds the latest revision for docPath across an already-parsed slice of LedgerRow.
func DocRevFromRows(rows []LedgerRow, docPath string) (int, bool) {
	row, ok := LookupDocFromRows(rows, docPath)
	if !ok {
		return 0, false
	}
	return row.Rev, true
}

// LookupDocFromRows finds the latest LedgerRow for docPath across an already-parsed slice of LedgerRow.
func LookupDocFromRows(rows []LedgerRow, docPath string) (LedgerRow, bool) {
	modName, ok := DocModule(docPath)
	if !ok {
		modName = strings.TrimSpace(strings.ReplaceAll(docPath, "\\", "/"))
	}
	altName := ""
	if strings.HasPrefix(modName, "docs/") {
		altName = "docs:" + strings.TrimPrefix(modName, "docs/")
	} else if strings.HasPrefix(modName, "docs:") {
		altName = "docs/" + strings.TrimPrefix(modName, "docs:")
	}

	for i := len(rows) - 1; i >= 0; i-- {
		r := rows[i]
		if (r.Module == modName || (altName != "" && r.Module == altName)) &&
			(r.Kind == "docs" || strings.HasPrefix(r.Module, "docs/") || strings.HasPrefix(r.Module, "docs:")) {
			return r, true
		}
	}
	return LedgerRow{}, false
}

// DocRevsFromLedger returns a map from docs module name to its latest revision
// across all rows in the ledger.
func DocRevsFromLedger(ledger []byte) map[string]int {
	return DocRevsFromRows(parseLedgerRows(ledger))
}

// DocRevsFromRows returns a map from docs module name to its latest revision
// across a slice of LedgerRow.
func DocRevsFromRows(rows []LedgerRow) map[string]int {
	revs := make(map[string]int)
	for _, r := range rows {
		if r.Kind == "docs" || strings.HasPrefix(r.Module, "docs/") || strings.HasPrefix(r.Module, "docs:") {
			revs[r.Module] = r.Rev
		}
	}
	return revs
}

// DocRevsForPaths returns a map from input doc path to its derived revision in rep.
// Paths that cannot be resolved or are not present in rep are omitted from the result.
func DocRevsForPaths(rep Report, paths []string) map[string]int {
	out := make(map[string]int, len(paths))
	for _, p := range paths {
		if rev, ok := rep.DocRev(p); ok {
			out[p] = rev
		}
	}
	return out
}

// DocRevsForPathsFromLedger returns a map from input doc path to its latest revision
// in the ledger. Paths not found in the ledger are omitted.
func DocRevsForPathsFromLedger(ledger []byte, paths []string) map[string]int {
	latest := LatestLedgerRows(ledger)
	out := make(map[string]int, len(paths))
	for _, p := range paths {
		modName, ok := DocModule(p)
		if !ok {
			modName = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
		}
		if row, found := latest[modName]; found {
			out[p] = row.Rev
			continue
		}
		altName := ""
		if strings.HasPrefix(modName, "docs/") {
			altName = "docs:" + strings.TrimPrefix(modName, "docs/")
		} else if strings.HasPrefix(modName, "docs:") {
			altName = "docs/" + strings.TrimPrefix(modName, "docs:")
		}
		if altName != "" {
			if row, found := latest[altName]; found {
				out[p] = row.Rev
			}
		}
	}
	return out
}
