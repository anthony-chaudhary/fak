package versionskew

import "github.com/anthony-chaudhary/fak/internal/appversion"

const BenchmarkConceptVersion = "fak.benchmark-concept.v1"

// Versioned returns a copy of row stamped with version. An existing version
// field wins; an empty version resolves through the shared application-version
// leaf.
func Versioned(row map[string]any, version string) map[string]any {
	out := make(map[string]any, len(row)+1)
	for key, value := range row {
		out[key] = value
	}
	if _, ok := out["version"]; !ok {
		if version == "" {
			version = appversion.Current()
		}
		out["version"] = version
	}
	return out
}

// VersionedRows stamps each row without mutating the input rows.
func VersionedRows(rows []map[string]any, version string) []map[string]any {
	out := make([]map[string]any, len(rows))
	for i, row := range rows {
		out[i] = Versioned(row, version)
	}
	return out
}
