package ciyaml

import "testing"

// TestPlainValueRegression pins the exact line that shipped .github/ISSUE_TEMPLATE/
// feature-request.yml unparseable in 45fafabb35. A new field's first line was welded
// onto the previous field's `required: true`, and every scanner already in Check
// passed it: the colons have their space, the quotes and brackets balance, and one
// line cannot dedent wrongly. This is the witness that the gate catches the thing it
// was written for, quoted from the commit rather than paraphrased.
func TestPlainValueRegression(t *testing.T) {
	const broken = "body:\n" +
		"  - type: textarea\n" +
		"    validations:\n" +
		"      required: true  - type: dropdown\n"

	issues := Check("feature-request.yml", []byte(broken))
	if len(issues) != 1 {
		t.Fatalf("want exactly 1 finding for the welded line, got %d:\n%s", len(issues), issues)
	}
	t.Logf("caught: %s", issues[0])
}

// TestPlainValueIssue separates the values YAML rejects from the ones it accepts.
// The accept half is the load-bearing half: a colon is ordinary inside a quoted
// scalar, a flow collection, a `${{ }}` expression and a URL, and a scanner that
// cannot tell those apart from a real second mapping key is one nobody can keep on.
// Every case below was confirmed against a real YAML parser before it was written
// down, so the table records observed behaviour rather than an assumption about it.
func TestPlainValueIssue(t *testing.T) {
	for _, tc := range []struct {
		name string
		line string
		want bool // true = must be reported
	}{
		{"welded mapping key", "a: true  - type: x", true},
		{"trailing colon", "a: see the following:", true},
		{"unquoted colon-space in prose", "about: note: this breaks", true},
		{"sequence item welded", "- a: b: c", true},

		{"quoted colon-space", `placeholder: "Today: …"`, false},
		{"flow mapping", "env: {A: 1, B: 2}", false},
		{"github expression", "if: ${{ github.ref == 'refs/heads/main' }}", false},
		{"url", "url: https://github.com/anthony-chaudhary/fak", false},
		{"colon with no space", "path: C:/work/fak", false},
		{"nested block opener", "attributes:", false},
		{"block scalar opener", "value: |", false},
		{"apostrophe in plain scalar", "name: don't cancel the run", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			issues := Check("t.yml", []byte(tc.line+"\n"))
			if got := len(issues) > 0; got != tc.want {
				t.Fatalf("Check(%q): reported=%v want=%v\n%s", tc.line, got, tc.want, issues)
			}
		})
	}
}
