package trajquery

import (
	"fmt"
	"strconv"
	"strings"
)

// Row is one trajectory record. Values are compared as numbers when both sides parse as
// numeric, else as strings — the corpus is JSON, so values arrive as strings/numbers/bools.
type Row map[string]any

// Op is a comparison operator in the supported WHERE subset.
type Op string

const (
	OpEq   Op = "="
	OpNe   Op = "!="
	OpLt   Op = "<"
	OpGt   Op = ">"
	OpLe   Op = "<="
	OpGe   Op = ">="
	OpLike Op = "LIKE" // substring containment
)

// Predicate is one `field <op> literal` term. Value is the literal as written (unquoted).
type Predicate struct {
	Field string `json:"field"`
	Op    Op     `json:"op"`
	Value string `json:"value"`
}

func (p Predicate) String() string {
	q := p.Value
	if _, err := strconv.ParseFloat(p.Value, 64); err != nil {
		q = "'" + p.Value + "'"
	}
	return fmt.Sprintf("%s %s %s", p.Field, p.Op, q)
}

// Query is a parsed SELECT over one relation: a projection, an AND-conjunction WHERE, and
// an optional LIMIT. It is both what Parse produces and what Rewrite emits.
type Query struct {
	Columns []string    `json:"columns"` // {"*"} for all
	From    string      `json:"from"`
	Where   []Predicate `json:"where"`
	Limit   int         `json:"limit"` // 0 => no limit
}

// selectsAll reports whether the projection is the star.
func (q Query) selectsAll() bool {
	return len(q.Columns) == 1 && q.Columns[0] == "*"
}

// String renders the query back to its SQL-ish source form (stable column and predicate
// order). It is used for audit/echo — e.g. showing a caller the scope-enforced rewritten
// query the server actually executed, so the enforced scope is visible, not hidden.
func (q Query) String() string {
	cols := "*"
	if !q.selectsAll() && len(q.Columns) > 0 {
		cols = strings.Join(q.Columns, ", ")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "SELECT %s FROM %s", cols, q.From)
	if len(q.Where) > 0 {
		parts := make([]string, len(q.Where))
		for i, p := range q.Where {
			parts[i] = p.String()
		}
		b.WriteString(" WHERE ")
		b.WriteString(strings.Join(parts, " AND "))
	}
	if q.Limit > 0 {
		fmt.Fprintf(&b, " LIMIT %d", q.Limit)
	}
	return b.String()
}

// Execute runs the query over rows and returns the projected, filtered, limited result.
func (q Query) Execute(rows []Row) ([]Row, error) {
	var out []Row
	for _, r := range rows {
		ok, err := matchAll(q.Where, r)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		out = append(out, project(q.Columns, r))
		if q.Limit > 0 && len(out) >= q.Limit {
			break
		}
	}
	return out, nil
}

func project(cols []string, r Row) Row {
	if len(cols) == 1 && cols[0] == "*" {
		out := make(Row, len(r))
		for k, v := range r {
			out[k] = v
		}
		return out
	}
	out := make(Row, len(cols))
	for _, c := range cols {
		if v, ok := r[c]; ok {
			out[c] = v
		}
	}
	return out
}

func matchAll(preds []Predicate, r Row) (bool, error) {
	for _, p := range preds {
		ok, err := match(p, r)
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
	}
	return true, nil
}

func match(p Predicate, r Row) (bool, error) {
	raw, present := r[p.Field]
	if !present {
		// A missing field satisfies nothing (and can't be LIKE-matched).
		return false, nil
	}
	cell := stringify(raw)
	// Numeric comparison when both sides are numbers; else lexicographic.
	cn, cerr := strconv.ParseFloat(cell, 64)
	vn, verr := strconv.ParseFloat(p.Value, 64)
	numeric := cerr == nil && verr == nil
	switch p.Op {
	case OpEq:
		return cell == p.Value, nil
	case OpNe:
		return cell != p.Value, nil
	case OpLike:
		return strings.Contains(cell, p.Value), nil
	case OpLt:
		if numeric {
			return cn < vn, nil
		}
		return cell < p.Value, nil
	case OpGt:
		if numeric {
			return cn > vn, nil
		}
		return cell > p.Value, nil
	case OpLe:
		if numeric {
			return cn <= vn, nil
		}
		return cell <= p.Value, nil
	case OpGe:
		if numeric {
			return cn >= vn, nil
		}
		return cell >= p.Value, nil
	default:
		return false, fmt.Errorf("unsupported operator %q", p.Op)
	}
}

func stringify(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case bool:
		if t {
			return "true"
		}
		return "false"
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	case int:
		return strconv.Itoa(t)
	case int64:
		return strconv.FormatInt(t, 10)
	case nil:
		return ""
	default:
		return fmt.Sprintf("%v", v)
	}
}
