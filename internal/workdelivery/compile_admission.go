package workdelivery

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// CompileSet is the explicit Go compile-set decision derived from delivery records.
// Admitted paths participate in compilation; excluded paths are overlaid away.
type CompileSet struct {
	Schema   string    `json:"schema"`
	Admitted []string  `json:"admitted,omitempty"`
	Excluded []string  `json:"excluded,omitempty"`
	Receipts []Receipt `json:"receipts"`
}

// CompileAdmissionError is a fail-closed declaration error suitable for machine routing.
type CompileAdmissionError struct {
	Code   string `json:"code"`
	UnitID string `json:"unit_id,omitempty"`
	Path   string `json:"path,omitempty"`
	Detail string `json:"detail"`
}

func (e *CompileAdmissionError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code, e.Detail)
}

// LoadCompileSet reads work-delivery records and derives the declared Go compile set.
func LoadCompileSet(paths ...string) (CompileSet, error) {
	units := make([]WorkUnit, 0, len(paths))
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return CompileSet{}, &CompileAdmissionError{Code: "DECLARATION_READ_FAILED", Path: filepath.ToSlash(path), Detail: err.Error()}
		}
		var unit WorkUnit
		if err := json.Unmarshal(data, &unit); err != nil {
			return CompileSet{}, &CompileAdmissionError{Code: "DECLARATION_MALFORMED", Path: filepath.ToSlash(path), Detail: err.Error()}
		}
		units = append(units, unit)
	}
	return DeriveCompileSet(units)
}

// DeriveCompileSet validates records and creates one receipt per compile-admission decision.
func DeriveCompileSet(units []WorkUnit) (CompileSet, error) {
	return DeriveCompileSetAt(units, time.Now().UTC())
}

// DeriveCompileSetAt is the deterministic compile-admission producer. Every receipt in one
// compile set carries the same caller-supplied observation time; runtime callers use
// DeriveCompileSet, while captures and replay can provide their witnessed event time.
func DeriveCompileSetAt(units []WorkUnit, observedAt time.Time) (CompileSet, error) {
	if observedAt.IsZero() {
		return CompileSet{}, &CompileAdmissionError{Code: "MISSING_OBSERVED_AT", Detail: "compile admission requires an observation time"}
	}
	observedAt = observedAt.UTC()
	set := CompileSet{Schema: Schema, Receipts: []Receipt{}}
	seen := map[string]string{}
	for _, unit := range units {
		if err := unit.Validate(); err != nil {
			return CompileSet{}, &CompileAdmissionError{Code: "DECLARATION_INVALID", UnitID: unit.ID, Detail: err.Error()}
		}
		if unit.Axes.Admission == AdmissionUndeclared {
			return CompileSet{}, &CompileAdmissionError{Code: "MISSING_DECLARATION", UnitID: unit.ID, Detail: "compile_admission must be admitted or excluded"}
		}
		if len(unit.Artifacts) == 0 {
			return CompileSet{}, &CompileAdmissionError{Code: "MISSING_ARTIFACT", UnitID: unit.ID, Detail: "delivery record has no artifacts to classify"}
		}
		for _, artifact := range unit.Artifacts {
			path := filepath.ToSlash(filepath.Clean(artifact.Path))
			if path == "." || filepath.IsAbs(artifact.Path) || path == ".." || len(path) >= 3 && path[:3] == "../" {
				return CompileSet{}, &CompileAdmissionError{Code: "INVALID_ARTIFACT_PATH", UnitID: unit.ID, Path: artifact.Path, Detail: "artifact path must be repository-relative"}
			}
			state := string(unit.Axes.Admission)
			if prior, ok := seen[path]; ok && prior != state {
				return CompileSet{}, &CompileAdmissionError{Code: "CONFLICTING_DECLARATION", UnitID: unit.ID, Path: path, Detail: fmt.Sprintf("artifact is both %s and %s", prior, state)}
			}
			seen[path] = state
			if unit.Axes.Admission == AdmissionAdmitted {
				set.Admitted = append(set.Admitted, path)
			} else {
				set.Excluded = append(set.Excluded, path)
			}
		}
		set.Receipts = append(set.Receipts, Receipt{
			Schema:     Schema,
			UnitID:     unit.ID,
			Gate:       "go.compile-set",
			Transition: Transition{Axis: AxisAdmission, From: string(AdmissionUndeclared), To: string(unit.Axes.Admission)},
			ObservedAt: observedAt,
			Evidence:   []Evidence{{Kind: "delivery-record", Reference: unit.ID, Witnessed: true}},
		})
	}
	set.Admitted = uniqueSorted(set.Admitted)
	set.Excluded = uniqueSorted(set.Excluded)
	return set, nil
}

func uniqueSorted(values []string) []string {
	sort.Strings(values)
	out := values[:0]
	for _, value := range values {
		if len(out) == 0 || out[len(out)-1] != value {
			out = append(out, value)
		}
	}
	return out
}

// IsCompileAdmissionError permits callers to route declaration failures without parsing prose.
func IsCompileAdmissionError(err error) bool {
	var target *CompileAdmissionError
	return errors.As(err, &target)
}
