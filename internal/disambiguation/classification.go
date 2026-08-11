package disambiguation

import (
	"fmt"
	"go/token"
	"sort"
	"strings"
)

const ClassificationSchemaVersion = "fak-disambiguation-classification/1"

const (
	// ClassificationIncidental marks a local exported implementation token as
	// coverage-accounted without admitting it as a canonical glossary identity.
	ClassificationIncidental = "incidental"
	// ClassificationReasonLocalImplementation is the stable reason code for an
	// exported Go identifier that is implementation vocabulary rather than a
	// public concept agents should query.
	ClassificationReasonLocalImplementation = "LOCAL_IMPLEMENTATION_TOKEN"
)

// TermClassification is public classification data consumed by the same Index
// that serves canonical queries and terminology coverage. Incidental terms are
// deliberately not inserted into canonical or alias query maps.
type TermClassification struct {
	SchemaVersion  string `json:"schema_version"`
	Term           string `json:"term"`
	Classification string `json:"classification"`
	Reason         string `json:"reason"`
}

// ValidateClassifications validates and returns a deterministic copy. The
// contract is intentionally closed: unknown schema versions, classes, reasons,
// malformed local Go identifiers, and duplicate identities are rejected.
func ValidateClassifications(items []TermClassification) ([]TermClassification, error) {
	out := append([]TermClassification(nil), items...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Term != out[j].Term {
			return out[i].Term < out[j].Term
		}
		if out[i].Classification != out[j].Classification {
			return out[i].Classification < out[j].Classification
		}
		return out[i].Reason < out[j].Reason
	})
	for i := range out {
		item := &out[i]
		if item.SchemaVersion == "" {
			item.SchemaVersion = ClassificationSchemaVersion
		}
		if item.SchemaVersion != ClassificationSchemaVersion {
			return nil, fmt.Errorf("classification[%d] %q: unsupported schema_version %q", i, item.Term, item.SchemaVersion)
		}
		if item.Term != strings.TrimSpace(item.Term) || !token.IsIdentifier(item.Term) || !token.IsExported(item.Term) {
			return nil, fmt.Errorf("classification[%d] %q: term must be an exported local Go identifier", i, item.Term)
		}
		if item.Classification != ClassificationIncidental {
			return nil, fmt.Errorf("classification[%d] %q: unsupported classification %q", i, item.Term, item.Classification)
		}
		if item.Reason != ClassificationReasonLocalImplementation {
			return nil, fmt.Errorf("classification[%d] %q: unsupported reason %q", i, item.Term, item.Reason)
		}
		if i > 0 && out[i-1].Term == item.Term {
			return nil, fmt.Errorf("classification[%d] %q: duplicate term", i, item.Term)
		}
	}
	return out, nil
}

// NewClassifiedIndex builds the single query/coverage index with strict local
// term classifications. It is the classification-aware form of NewIndex.
func NewClassifiedIndex(entries []Entry, classifications []TermClassification) (*Index, error) {
	validated, err := ValidateClassifications(classifications)
	if err != nil {
		return nil, err
	}
	index, err := newIndex(entries)
	if err != nil {
		return nil, err
	}
	index.classifications = validated
	index.incidental = make(map[string]TermClassification, len(validated))
	for _, item := range validated {
		index.incidental[item.Term] = item
	}
	return index, nil
}

// Classifications returns a stable copy of the classifications used by this
// index. Query results remain canonical-only.
func (i *Index) Classifications() []TermClassification {
	if i == nil {
		return nil
	}
	return append([]TermClassification(nil), i.classifications...)
}
