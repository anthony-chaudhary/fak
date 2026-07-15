package conceptcatalog

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const DataRel = "tools/concept_disambiguation_scorecard.data"

type Family struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	Roots   []string `json:"roots"`
	Ignore  []string `json:"ignore"`
	Exclude []string `json:"exclude"`
}

type Metadata struct {
	Families []Family `json:"families"`
}

type Row struct {
	ID             string   `json:"id"`
	Canonical      string   `json:"canonical"`
	Family         string   `json:"family"`
	Kind           string   `json:"kind"`
	Definition     string   `json:"definition"`
	Distinction    string   `json:"distinction"`
	DistinctFrom   []string `json:"distinct_from"`
	Aliases        []string `json:"aliases"`
	Grounding      string   `json:"grounding"`
	GroundingKind  string   `json:"grounding_kind"`
	GlossaryAnchor string   `json:"glossary_anchor"`
	Parent         string   `json:"parent"`
	Verdict        string   `json:"verdict"`
	Gaps           []string `json:"gaps"`
	Source         string   `json:"-"`
	Index          int      `json:"-"`
}

type Catalog struct {
	Meta Metadata
	Rows []Row
	Dir  string
}

type Diagnostic struct {
	File   string `json:"file"`
	RowID  string `json:"row_id,omitempty"`
	Field  string `json:"field"`
	Value  string `json:"value,omitempty"`
	Repair string `json:"repair"`
	Code   string `json:"code"`
}

func (d Diagnostic) Error() string {
	return fmt.Sprintf("%s: row %q field %s=%q: %s", d.File, d.RowID, d.Field, d.Value, d.Repair)
}

func Load(root string) (Catalog, error) {
	return LoadDir(filepath.Join(root, filepath.FromSlash(DataRel)))
}
func LoadDir(dir string) (Catalog, error) {
	c := Catalog{Dir: dir}
	b, err := os.ReadFile(filepath.Join(dir, "_meta.json"))
	if err != nil {
		return c, err
	}
	if err = json.Unmarshal(b, &c.Meta); err != nil {
		return c, fmt.Errorf("%s: %w", filepath.Join(dir, "_meta.json"), err)
	}
	files, err := filepath.Glob(filepath.Join(dir, "rows-*.json"))
	if err != nil {
		return c, err
	}
	sort.Strings(files)
	for _, p := range files {
		var x struct {
			Rows []Row `json:"rows"`
		}
		b, err = os.ReadFile(p)
		if err != nil {
			return c, err
		}
		if err = json.Unmarshal(b, &x); err != nil {
			return c, fmt.Errorf("%s: %w", p, err)
		}
		for i := range x.Rows {
			x.Rows[i].Source = filepath.ToSlash(p)
			x.Rows[i].Index = i
		}
		c.Rows = append(c.Rows, x.Rows...)
	}
	return c, nil
}

// ValidateTree adds the cross-tree grounding invariant: a row cannot be grounded
// exclusively by tests or build-tag-only source. It is separate from ValidateStrict
// because in-memory fixture catalogs do not necessarily have a repository root.
func ValidateTree(c Catalog, root string) []Diagnostic {
	out := Validate(c)
	groundings := make([]string, 0, len(c.Rows))
	for _, r := range c.Rows {
		groundings = append(groundings, r.Grounding)
	}
	found, err := ProductionCorpusMany(root, groundings)
	for _, r := range c.Rows {
		if strings.TrimSpace(r.Grounding) == "" {
			continue
		}
		if err != nil {
			out = append(out, diag(r, "grounding", r.Grounding, "grounding_check_failed", fmt.Sprintf("inspect the production corpus: %v", err)))
		} else if !found[token(r.Grounding)] {
			out = append(out, diag(r, "grounding", r.Grounding, "excluded_corpus_grounding", "add this grounding to production corpus; tests and build-tag-only files do not count"))
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		if out[i].RowID != out[j].RowID {
			return out[i].RowID < out[j].RowID
		}
		return out[i].Field < out[j].Field
	})
	return out
}

func norm(s string) string { return strings.ToLower(strings.TrimSpace(s)) }
func token(s string) string {
	var b strings.Builder
	for _, r := range norm(s) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}
func diag(r Row, field, value, code, repair string) Diagnostic {
	return Diagnostic{File: r.Source, RowID: r.ID, Field: field, Value: value, Code: code, Repair: repair}
}

func Validate(c Catalog) []Diagnostic {
	var out []Diagnostic
	families := map[string]bool{}
	ignored := map[string]map[string]bool{}
	for _, f := range c.Meta.Families {
		families[norm(f.ID)] = true
		m := map[string]bool{}
		for _, v := range append(append([]string{}, f.Ignore...), f.Exclude...) {
			m[token(v)] = true
		}
		ignored[norm(f.ID)] = m
	}
	ids := map[string]Row{}
	canon := map[string]Row{}
	for _, r := range c.Rows {
		rid := norm(r.ID)
		cn := norm(r.Canonical)
		if rid == "" {
			out = append(out, diag(r, "id", r.ID, "missing_id", "set a stable unique row ID"))
		} else if p, ok := ids[rid]; ok {
			out = append(out, diag(r, "id", r.ID, "duplicate_id", "choose a unique ID; already used by "+p.Source))
		} else {
			ids[rid] = r
		}
		if cn == "" {
			out = append(out, diag(r, "canonical", r.Canonical, "missing_canonical", "set the canonical concept name"))
		} else if p, ok := canon[cn]; ok {
			out = append(out, diag(r, "canonical", r.Canonical, "duplicate_canonical", "merge or rename this row; already used by "+p.ID))
		} else {
			canon[cn] = r
		}
		if !families[norm(r.Family)] {
			out = append(out, diag(r, "family", r.Family, "unknown_family", "use an ID declared in _meta.json families"))
		}
		validGK := map[string]bool{"symbol": true, "path": true, "metric": true, "doc": true, "verb": true, "claims": true}
		if !validGK[norm(r.GroundingKind)] {
			out = append(out, diag(r, "grounding_kind", r.GroundingKind, "wrong_grounding_kind", "use symbol, path, metric, doc, verb, or claims"))
		}
		if strings.TrimSpace(r.Grounding) == "" {
			out = append(out, diag(r, "grounding", r.Grounding, "missing_grounding", "name a production-corpus grounding token"))
		}
	}
	for _, r := range c.Rows {
		for _, ref := range r.DistinctFrom {
			n := norm(ref)
			if _, ok := ids[n]; ok {
				continue
			}
			// The legacy catalog permits canonical references. New/mutated catalogs
			// can opt into the strict ID-only contract through ValidateStrict.
			if _, ok := canon[n]; ok {
				continue
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		if out[i].RowID != out[j].RowID {
			return out[i].RowID < out[j].RowID
		}
		return out[i].Field < out[j].Field
	})
	return out
}

// ValidateStrict adds the authoring-time invariant that distinct_from references use stable IDs.
// Validate remains compatible with the inherited catalog while new writes are held to the stricter contract.
func ValidateStrict(c Catalog) []Diagnostic {
	out := Validate(c)
	ids := map[string]bool{}
	canon := map[string]Row{}
	ignored := map[string]map[string]bool{}
	for _, f := range c.Meta.Families {
		m := map[string]bool{}
		for _, v := range append(append([]string{}, f.Ignore...), f.Exclude...) {
			m[token(v)] = true
		}
		ignored[norm(f.ID)] = m
	}
	for _, r := range c.Rows {
		ids[norm(r.ID)] = true
		canon[norm(r.Canonical)] = r
	}
	aliases := map[string]Row{}
	groundings := map[string]Row{}
	for _, r := range c.Rows {
		if g := token(r.Grounding); g != "" {
			if p, ok := groundings[g]; ok {
				out = append(out, diag(r, "grounding", r.Grounding, "duplicate_grounding", "use a distinct production grounding; already owned by "+p.ID))
			} else {
				groundings[g] = r
			}
		}
		if ignored[norm(r.Family)][token(r.Grounding)] {
			out = append(out, diag(r, "grounding", r.Grounding, "classification_conflict", "remove this token from family ignore/exclude metadata because a catalog row positions it"))
		}
		cn := norm(r.Canonical)
		for _, a := range r.Aliases {
			n := norm(a)
			if n == "" {
				continue
			}
			if n == cn {
				out = append(out, diag(r, "aliases", a, "stale_alias", "remove aliases equal to the canonical name"))
			}
			if p, ok := aliases[n]; ok && p.ID != r.ID {
				out = append(out, diag(r, "aliases", a, "duplicate_alias", "remove or disambiguate alias also owned by "+p.ID))
			} else {
				aliases[n] = r
			}
		}
		for _, ref := range r.DistinctFrom {
			n := norm(ref)
			if ids[n] {
				continue
			}
			if p, ok := canon[n]; ok {
				out = append(out, diag(r, "distinct_from", ref, "canonical_reference", "replace canonical name with row ID "+p.ID))
			} else {
				out = append(out, diag(r, "distinct_from", ref, "unresolved_reference", "replace with an existing row ID or add the missing row"))
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		if out[i].RowID != out[j].RowID {
			return out[i].RowID < out[j].RowID
		}
		return out[i].Field < out[j].Field
	})
	return out
}
