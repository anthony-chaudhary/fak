package trajquery

import (
	"fmt"
	"strings"
)

// View is an operator-published scoped window onto a base relation. Agents query the
// view by name; the view's Scope predicates confine which rows they can ever see, and
// Columns is the allowlist of fields they may project or filter on. A view is the ONLY
// sanctioned way to query the corpus — a query against the base relation is an escape.
type View struct {
	Name    string      `json:"name"`    // the relation name a user query must target
	Base    string      `json:"base"`    // the underlying corpus relation
	Scope   []Predicate `json:"scope"`   // non-removable row filter (ANDed into every query)
	Columns []string    `json:"columns"` // allowlisted selectable/filterable columns
}

// Rewrite turns a user query written against the view into an executable query against
// the base relation with the scope enforced: it prepends the (non-removable) scope
// predicates to the user's WHERE and retargets FROM at the base. This is the view-inlining
// rewrite — `SELECT … FROM view WHERE u` becomes `SELECT … FROM base WHERE scope AND u`.
// Rewrite refuses first if Validate would (so you cannot execute an unsound rewrite).
func (v View) Rewrite(user Query) (Query, error) {
	if err := v.staticValidate(user); err != nil {
		return Query{}, err
	}
	cols := user.Columns
	if user.selectsAll() {
		// '*' under a view resolves to the allowlist, never the base's full column set —
		// otherwise a hidden column would leak through the star.
		cols = append([]string{}, v.Columns...)
	}
	rewritten := Query{
		Columns: cols,
		From:    v.Base,
		Where:   append(append([]Predicate{}, v.Scope...), user.Where...),
		Limit:   user.Limit,
	}
	return rewritten, nil
}

// Report is the validator's verdict: whether the (user query -> rewrite) pair is a sound
// scoping, and every reason it is not.
type Report struct {
	Valid      bool     `json:"valid"`
	Violations []string `json:"violations,omitempty"`
}

// Validate proves the scoping is sound. Static checks: the user query targets the view
// (not the base — the primary escape), the projection/WHERE reference only allowlisted
// columns (so a hidden column cannot be filtered-on to infer it), and the rewrite carries
// every scope predicate plus the user's own. Dynamic check: over corpus, every row the
// rewrite returns satisfies the scope. Pass a nil corpus to run static checks only.
func (v View) Validate(user Query, corpus []Row) Report {
	var viol []string
	if err := v.staticValidate(user); err != nil {
		viol = append(viol, err.Error())
	}
	rewritten, err := v.Rewrite(user)
	if err != nil {
		// staticValidate already recorded the reason; avoid a duplicate line.
		if len(viol) == 0 {
			viol = append(viol, err.Error())
		}
		return Report{Valid: false, Violations: viol}
	}
	// The rewrite must carry every scope predicate — the guarantee that scope survived.
	for _, sp := range v.Scope {
		if !containsPred(rewritten.Where, sp) {
			viol = append(viol, fmt.Sprintf("scope predicate dropped by rewrite: %s", sp))
		}
	}
	// ...and the user's own predicates (rewrite narrows, never discards the user filter).
	for _, up := range user.Where {
		if !containsPred(rewritten.Where, up) {
			viol = append(viol, fmt.Sprintf("user predicate lost by rewrite: %s", up))
		}
	}
	// The rewrite must read the base, never leave the user's view name dangling.
	if rewritten.From != v.Base {
		viol = append(viol, fmt.Sprintf("rewrite reads %q, want base %q", rewritten.From, v.Base))
	}
	// Dynamic: no returned row may violate the scope. This catches any gap the static
	// argument missed — belt and suspenders.
	if corpus != nil && len(viol) == 0 {
		got, err := rewritten.Execute(corpus)
		if err != nil {
			viol = append(viol, fmt.Sprintf("rewrite failed to execute: %v", err))
		} else {
			for i, row := range got {
				for _, sp := range v.Scope {
					ok, err := match(sp, reScope(row, corpus, v, i))
					if err != nil {
						viol = append(viol, err.Error())
						break
					}
					if !ok {
						viol = append(viol, fmt.Sprintf("returned row violates scope %s", sp))
						break
					}
				}
			}
		}
	}
	return Report{Valid: len(viol) == 0, Violations: viol}
}

// reScope returns a row to test scope predicates against. Because a projection may drop
// the scope columns from the output row, we validate scope against the FULL source row.
// The rewrite preserves order and never reorders/dedups, so returned row i under a star
// scope maps back by re-filtering the corpus; to stay simple and robust we instead check
// scope on the output row when the scope column survives projection, else on the matching
// source row found by identity of the projected fields.
func reScope(out Row, corpus []Row, v View, _ int) Row {
	// If every scope field is present in the projected row, test it directly.
	allPresent := true
	for _, sp := range v.Scope {
		if _, ok := out[sp.Field]; !ok {
			allPresent = false
			break
		}
	}
	if allPresent {
		return out
	}
	// Otherwise find a source row that projects to `out` and satisfies the scope; if such a
	// row exists the scope held for it. Fall back to `out` (fails closed) if none matches.
	for _, src := range corpus {
		if projectsTo(src, out) {
			return src
		}
	}
	return out
}

// projectsTo reports whether src projected onto out's keys equals out.
func projectsTo(src, out Row) bool {
	for k, v := range out {
		if stringify(src[k]) != stringify(v) {
			return false
		}
	}
	return true
}

func (v View) staticValidate(user Query) error {
	if v.Name == "" || v.Base == "" {
		return fmt.Errorf("view: name and base are required")
	}
	if v.Name == v.Base {
		return fmt.Errorf("view name and base must differ (else the view is the base)")
	}
	// The escape check: a user query must be written against the view, not the base.
	if !strings.EqualFold(user.From, v.Name) {
		return fmt.Errorf("query targets %q but must target view %q (querying the base is a scope escape)", user.From, v.Name)
	}
	allowed := map[string]bool{}
	for _, c := range v.Columns {
		allowed[c] = true
	}
	// '*' is fine (it resolves to the allowlist); named columns must be allowlisted.
	if !user.selectsAll() {
		for _, c := range user.Columns {
			if !allowed[c] {
				return fmt.Errorf("column %q is not in view %q's allowlist", c, v.Name)
			}
		}
	}
	// A WHERE may only filter on allowlisted columns — otherwise a hidden column could be
	// probed (e.g. WHERE secret LIKE 'a') to infer its value one query at a time.
	for _, p := range user.Where {
		if !allowed[p.Field] {
			return fmt.Errorf("WHERE references column %q outside view %q's allowlist", p.Field, v.Name)
		}
	}
	return nil
}

func containsPred(preds []Predicate, want Predicate) bool {
	for _, p := range preds {
		if p == want {
			return true
		}
	}
	return false
}
