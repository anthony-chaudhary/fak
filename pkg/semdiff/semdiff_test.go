package semdiff

import "testing"

// fak-test:runtime fast est=50ms
func TestSemdiffClassifyGo(t *testing.T) {
	const base = "package p\n\n// Add sums.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
	cases := []struct {
		name          string
		file          string
		before, after string
		class         Class
		reason        Reason
	}{
		{"identical", "p.go", base, base, ClassCommentOnly, ReasonNone},
		{"doc comment edit", "p.go", base,
			"package p\n\n// Add returns a+b.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n",
			ClassCommentOnly, ReasonNone},
		{"comment line added", "p.go", base,
			"package p\n\n// Add sums.\n// It never overflows silently.\nfunc Add(a, b int) int {\n\t// plain\n\treturn a + b\n}\n",
			ClassCommentOnly, ReasonNone},
		{"identifier changed", "p.go", base,
			"package p\n\n// Add sums.\nfunc Add(x, b int) int {\n\treturn x + b\n}\n",
			ClassLogic, ReasonCodeChanged},
		{"literal changed", "p.go",
			"package p\n\nconst N = 1\n", "package p\n\nconst N = 2\n",
			ClassLogic, ReasonCodeChanged},
		{"go:build changed", "p.go",
			"//go:build linux\n\npackage p\n", "//go:build darwin\n\npackage p\n",
			ClassLogic, ReasonDirectiveChanged},
		{"go:embed pattern changed", "p.go",
			"package p\n\nimport _ \"embed\"\n\n//go:embed a.txt\nvar s string\n",
			"package p\n\nimport _ \"embed\"\n\n//go:embed b.txt\nvar s string\n",
			ClassLogic, ReasonDirectiveChanged},
		{"go:noinline added", "p.go", base,
			"package p\n\n// Add sums.\n//go:noinline\nfunc Add(a, b int) int {\n\treturn a + b\n}\n",
			ClassLogic, ReasonDirectiveChanged},
		{"cgo preamble edit", "p.go",
			"package p\n\n// #include <stdio.h>\nimport \"C\"\n",
			"package p\n\n// #include <stdlib.h>\nimport \"C\"\n",
			ClassLogic, ReasonCgo},
		{"example output changed", "p_test.go",
			"package p\n\nfunc ExampleAdd() {\n\tprintln(Add(1, 2))\n\t// Output: 3\n}\n",
			"package p\n\nfunc ExampleAdd() {\n\tprintln(Add(1, 2))\n\t// Output: 4\n}\n",
			ClassLogic, ReasonExampleOutput},
		{"plain test comment edit", "p_test.go",
			"package p\n\n// TestAdd checks sums.\nfunc TestAdd() {}\n",
			"package p\n\n// TestAdd checks small sums.\nfunc TestAdd() {}\n",
			ClassCommentOnly, ReasonNone},
		{"broken after", "p.go", base,
			"package p\n\n/* Add sums.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n",
			ClassLogic, ReasonScanError},
		{"inline block comment replaced", "p.go",
			"package p\n\nvar x = a /*x*/ + b\n", "package p\n\nvar x = a /*y*/ + b\n",
			ClassCommentOnly, ReasonNone},
		{"multiline block comment inserts semicolon", "p.go",
			"package p\n\nvar x = a + b\n", "package p\n\nvar x = a /*\n*/ + b\n",
			ClassLogic, ReasonCodeChanged},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			class, reason := ClassifyGo(tc.file, []byte(tc.before), []byte(tc.after))
			if class != tc.class || reason != tc.reason {
				t.Fatalf("ClassifyGo = (%s, %s), want (%s, %s)", class, reason, tc.class, tc.reason)
			}
		})
	}
}
