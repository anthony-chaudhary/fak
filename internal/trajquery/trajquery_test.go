package trajquery

import (
	"strings"
	"testing"
)

func corpus() []Row {
	return []Row{
		{"id": "1", "session": "S1", "role": "user", "text": "fix the bug", "redacted": "false", "secret": "alpha"},
		{"id": "2", "session": "S1", "role": "agent", "text": "ran go test", "redacted": "false", "secret": "beta"},
		{"id": "3", "session": "S2", "role": "user", "text": "other session", "redacted": "false", "secret": "gamma"},
		{"id": "4", "session": "S1", "role": "agent", "text": "REDACTED", "redacted": "true", "secret": "delta"},
	}
}

func TestParse_Valid(t *testing.T) {
	q, err := Parse("SELECT id, text FROM traj WHERE session = 'S1' AND role = 'agent' LIMIT 5")
	if err != nil {
		t.Fatal(err)
	}
	if q.From != "traj" || len(q.Columns) != 2 || len(q.Where) != 2 || q.Limit != 5 {
		t.Fatalf("unexpected parse: %+v", q)
	}
	if q.Where[0] != (Predicate{Field: "session", Op: OpEq, Value: "S1"}) {
		t.Fatalf("pred0 = %+v", q.Where[0])
	}
}

func TestParse_StarAndNumbers(t *testing.T) {
	q, err := Parse("SELECT * FROM t WHERE n >= 3")
	if err != nil {
		t.Fatal(err)
	}
	if !q.selectsAll() || q.Where[0].Op != OpGe || q.Where[0].Value != "3" {
		t.Fatalf("unexpected: %+v", q)
	}
}

func TestParse_Rejects(t *testing.T) {
	bad := map[string]string{
		"or":            "SELECT * FROM t WHERE a = '1' OR b = '2'",
		"subquery":      "SELECT * FROM (SELECT * FROM t)",
		"star plus col": "SELECT *, id FROM t",
		"no from":       "SELECT id",
		"trailing":      "SELECT id FROM t garbage",
		"bad op":        "SELECT id FROM t WHERE a ~ '1'",
		"unterminated":  "SELECT id FROM t WHERE a = 'x",
	}
	for name, sql := range bad {
		if _, err := Parse(sql); err == nil {
			t.Errorf("%s: expected parse error for %q", name, sql)
		}
	}
}

func TestExecute_FilterProjectLimit(t *testing.T) {
	q, _ := Parse("SELECT id FROM traj WHERE session = 'S1' AND role = 'agent'")
	got, err := q.Execute(corpus())
	if err != nil {
		t.Fatal(err)
	}
	// rows 2 and 4 are S1/agent; projection keeps only id.
	if len(got) != 2 {
		t.Fatalf("got %d rows, want 2: %+v", len(got), got)
	}
	if _, hasSecret := got[0]["secret"]; hasSecret {
		t.Fatalf("projection leaked non-selected column: %+v", got[0])
	}
}

func TestExecute_NumericVsStringAndLike(t *testing.T) {
	rows := []Row{{"n": 2.0}, {"n": 10.0}, {"n": 9.0}}
	q, _ := Parse("SELECT n FROM t WHERE n > 9")
	got, _ := q.Execute(rows)
	if len(got) != 1 { // numeric: only 10 > 9 (string compare would also drop "10")
		t.Fatalf("numeric compare wrong: %+v", got)
	}
	ql, _ := Parse("SELECT text FROM t WHERE text LIKE 'test'")
	gl, _ := ql.Execute(corpus())
	if len(gl) != 1 || gl[0]["text"] != "ran go test" {
		t.Fatalf("LIKE wrong: %+v", gl)
	}
}

// sessionView scopes to session S1, non-redacted, and hides the secret column.
func sessionView() View {
	return View{
		Name:    "myturns",
		Base:    "traj",
		Scope:   []Predicate{{Field: "session", Op: OpEq, Value: "S1"}, {Field: "redacted", Op: OpEq, Value: "false"}},
		Columns: []string{"id", "session", "role", "text", "redacted"},
	}
}

func TestRewrite_EnforcesScope(t *testing.T) {
	v := sessionView()
	user, _ := Parse("SELECT id, text FROM myturns WHERE role = 'agent'")
	rw, err := v.Rewrite(user)
	if err != nil {
		t.Fatal(err)
	}
	if rw.From != "traj" {
		t.Fatalf("rewrite should read base, got %q", rw.From)
	}
	// scope predicates present and ahead of the user predicate.
	if len(rw.Where) != 3 || !containsPred(rw.Where, v.Scope[0]) || !containsPred(rw.Where, v.Scope[1]) {
		t.Fatalf("scope not enforced in rewrite: %+v", rw.Where)
	}
	// Executing the rewrite only ever returns S1, non-redacted rows.
	got, _ := rw.Execute(corpus())
	if len(got) != 1 { // row 2 (S1/agent/not-redacted); row 4 is redacted, excluded
		t.Fatalf("rewrite returned %d rows, want 1: %+v", len(got), got)
	}
}

func TestValidate_RejectsBaseEscape(t *testing.T) {
	v := sessionView()
	// Querying the base relation directly bypasses the view — the primary escape.
	user, _ := Parse("SELECT id, secret FROM traj")
	rep := v.Validate(user, corpus())
	if rep.Valid {
		t.Fatal("querying the base directly must be rejected")
	}
	if !anyContains(rep.Violations, "scope escape") {
		t.Fatalf("expected a scope-escape violation, got %v", rep.Violations)
	}
}

func TestValidate_RejectsHiddenColumnProbe(t *testing.T) {
	v := sessionView()
	// Filtering on the hidden `secret` column would let an agent infer it one query at a
	// time even though it's not projected — the validator must refuse.
	user, _ := Parse("SELECT id FROM myturns WHERE secret LIKE 'alp'")
	rep := v.Validate(user, corpus())
	if rep.Valid || !anyContains(rep.Violations, "allowlist") {
		t.Fatalf("hidden-column WHERE probe must be rejected: %+v", rep)
	}
	// And selecting the hidden column outright.
	user2, _ := Parse("SELECT id, secret FROM myturns")
	if rep2 := v.Validate(user2, corpus()); rep2.Valid {
		t.Fatal("selecting a non-allowlisted column must be rejected")
	}
}

func TestValidate_StarResolvesToAllowlistNotBase(t *testing.T) {
	v := sessionView()
	user, _ := Parse("SELECT * FROM myturns")
	rw, err := v.Rewrite(user)
	if err != nil {
		t.Fatal(err)
	}
	// '*' must NOT include the hidden secret column.
	got, _ := rw.Execute(corpus())
	for _, r := range got {
		if _, leaked := r["secret"]; leaked {
			t.Fatalf("star leaked hidden column: %+v", r)
		}
	}
	rep := v.Validate(user, corpus())
	if !rep.Valid {
		t.Fatalf("valid star query rejected: %v", rep.Violations)
	}
}

func TestValidate_ValidQueryPasses(t *testing.T) {
	v := sessionView()
	user, _ := Parse("SELECT id, text FROM myturns WHERE role = 'agent' LIMIT 10")
	rep := v.Validate(user, corpus())
	if !rep.Valid {
		t.Fatalf("expected valid, got violations: %v", rep.Violations)
	}
}

func anyContains(ss []string, sub string) bool {
	for _, s := range ss {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// enumView adds a per-column literal set on top of the allowlist.
func enumView() View {
	v := sessionView()
	v.Enums = map[string][]string{"role": {"agent", "user"}}
	return v
}

func TestValidate_EnumRejectsUndeclaredLiteral(t *testing.T) {
	v := enumView()
	// 'system' is allowlisted-column-legal but is NOT a declared role literal.
	user, _ := Parse("SELECT id FROM myturns WHERE role = 'system'")
	rep := v.Validate(user, corpus())
	if rep.Valid || !anyContains(rep.Violations, "allowed literals") {
		t.Fatalf("undeclared enum literal must be rejected: %+v", rep)
	}
	// A declared literal on the same column passes.
	ok, _ := Parse("SELECT id FROM myturns WHERE role = 'agent'")
	if rep := v.Validate(ok, corpus()); !rep.Valid {
		t.Fatalf("declared enum literal rejected: %v", rep.Violations)
	}
}

func TestValidate_EnumBlocksLikeSubstringProbe(t *testing.T) {
	v := enumView()
	// A partial LIKE substring is not a full declared literal — it would let a caller sniff
	// the closed set one character at a time, so it is refused.
	probe, _ := Parse("SELECT id FROM myturns WHERE role LIKE 'age'")
	if rep := v.Validate(probe, corpus()); rep.Valid {
		t.Fatal("LIKE substring probe of an enum column must be rejected")
	}
	// A LIKE whose value IS a full literal is fine.
	full, _ := Parse("SELECT id FROM myturns WHERE role LIKE 'agent'")
	if rep := v.Validate(full, corpus()); !rep.Valid {
		t.Fatalf("enum-literal LIKE rejected: %v", rep.Violations)
	}
}

func TestSchema_AdvertisesColumnsEnumsScope(t *testing.T) {
	s := enumView().Schema()
	if strings.Join(s.Columns, ",") != "id,session,role,text,redacted" {
		t.Fatalf("schema columns = %v", s.Columns)
	}
	if strings.Join(s.Enums["role"], ",") != "agent,user" {
		t.Fatalf("schema role enum = %v", s.Enums["role"])
	}
	// Scope is rendered as strings so a caller sees what is always enforced.
	if len(s.Scope) != 2 || !anyContains(s.Scope, "session = 'S1'") {
		t.Fatalf("schema scope = %v", s.Scope)
	}
}

func TestDescribe_IsDeterministicAndHidesNonAllowlisted(t *testing.T) {
	d := enumView().Describe()
	if strings.Contains(d, "secret") {
		t.Fatalf("Describe must not name a hidden column: %s", d)
	}
	for _, want := range []string{"columns:", "enums:", "role: agent, user", "scope"} {
		if !strings.Contains(d, want) {
			t.Fatalf("Describe missing %q: %s", want, d)
		}
	}
}

func TestQueryString_RendersScopeEnforcedRewrite(t *testing.T) {
	v := sessionView()
	user, _ := Parse("SELECT id, text FROM myturns WHERE role = 'agent' LIMIT 5")
	rw, err := v.Rewrite(user)
	if err != nil {
		t.Fatal(err)
	}
	got := rw.String()
	for _, want := range []string{"SELECT id, text FROM traj", "session = 'S1'", "role = 'agent'", "LIMIT 5"} {
		if !strings.Contains(got, want) {
			t.Fatalf("rewrite render missing %q: %s", want, got)
		}
	}
}
