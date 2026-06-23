package codelint

import (
	"context"
	"os/exec"
	"strings"
	"testing"
)

func bg() context.Context { return context.Background() }

// --- in-process packs (hermetic: no external binary) ------------------------

func TestGoPackReportsParseError(t *testing.T) {
	r := DefaultRegistry()
	bad := []byte("package x\n\nfunc (") // truncated func: a hard parse error
	fs, err := r.LintBytes(bg(), "broken.go", bad)
	if err != nil {
		t.Fatalf("LintBytes: %v", err)
	}
	if len(fs) == 0 {
		t.Fatal("want a GO_PARSE finding for truncated Go, got none")
	}
	f := fs[0]
	if f.Pack != "go" || f.Code != "GO_PARSE" || f.Severity != Error {
		t.Fatalf("want go/GO_PARSE/error, got %s/%s/%s", f.Pack, f.Code, f.Severity)
	}
	if f.File != "broken.go" {
		t.Fatalf("LintBytes must re-address findings to the given name, got %q", f.File)
	}
	if f.Line == 0 {
		t.Errorf("want a non-zero line for the parse error, got 0")
	}
}

func TestGoPackCleanFileHasNoOpinion(t *testing.T) {
	r := DefaultRegistry()
	good := []byte("package x\n\nfunc F() int { return 1 }\n")
	fs, err := r.LintBytes(bg(), "ok.go", good)
	if err != nil {
		t.Fatalf("LintBytes: %v", err)
	}
	if len(fs) != 0 {
		t.Fatalf("clean Go must yield no findings, got %v", fs)
	}
}

func TestGoPackIgnoresSemanticErrors(t *testing.T) {
	// undefined identifier is a TYPE error, not a parse error: a single file in
	// isolation has no package context, so codelint must NOT flag it (would be a
	// false positive on perfectly good code).
	r := DefaultRegistry()
	src := []byte("package x\n\nfunc F() int { return undefinedThing() }\n")
	fs, err := r.LintBytes(bg(), "sem.go", src)
	if err != nil {
		t.Fatalf("LintBytes: %v", err)
	}
	if len(fs) != 0 {
		t.Fatalf("semantic error must not be flagged by the parse-level pack, got %v", fs)
	}
}

func TestJSONPackReportsSyntaxError(t *testing.T) {
	r := DefaultRegistry()
	fs, err := r.LintBytes(bg(), "bad.json", []byte("{\"a\": }"))
	if err != nil {
		t.Fatalf("LintBytes: %v", err)
	}
	if len(fs) != 1 || fs[0].Code != "JSON_PARSE" || fs[0].Severity != Error {
		t.Fatalf("want one JSON_PARSE error, got %v", fs)
	}
	if fs[0].Line == 0 || fs[0].Col == 0 {
		t.Errorf("want a line:col for the JSON error, got %d:%d", fs[0].Line, fs[0].Col)
	}
}

func TestJSONPackCleanHasNoOpinion(t *testing.T) {
	r := DefaultRegistry()
	fs, err := r.LintBytes(bg(), "ok.json", []byte(`{"a":[1,2,3],"b":{"c":true}}`))
	if err != nil {
		t.Fatalf("LintBytes: %v", err)
	}
	if len(fs) != 0 {
		t.Fatalf("clean JSON must yield no findings, got %v", fs)
	}
}

// --- language detection -----------------------------------------------------

func TestPackForKnownAndUnknown(t *testing.T) {
	r := DefaultRegistry()
	for _, ext := range []string{"x.go", "y.PY", "k.cu", "z.json"} {
		if _, ok := r.PackFor(ext); !ok {
			t.Errorf("expected a pack for %q", ext)
		}
	}
	if _, ok := r.PackFor("notes.txt"); ok {
		t.Error("did not expect a pack for .txt")
	}
	// an unknown extension lints to no findings, never an error
	fs, err := r.LintBytes(bg(), "notes.txt", []byte("anything at all"))
	if err != nil || fs != nil {
		t.Fatalf("unknown ext must be (nil,nil), got (%v,%v)", fs, err)
	}
}

func TestLangsMenu(t *testing.T) {
	got := strings.Join(DefaultRegistry().Langs(), ",")
	for _, want := range []string{"cuda", "go", "json", "python"} {
		if !strings.Contains(got, want) {
			t.Errorf("Langs() %q missing %q", got, want)
		}
	}
}

// --- the external-checker diagnostic parser (hermetic: canned output) -------

func TestParseDiagnosticsGCCStyle(t *testing.T) {
	fs := parseDiagnostics("python", "PYTHON_SYNTAX", "",
		"bad.py:2:5: error: SyntaxError: invalid syntax")
	if len(fs) != 1 {
		t.Fatalf("want 1 finding, got %d (%v)", len(fs), fs)
	}
	f := fs[0]
	if f.File != "bad.py" || f.Line != 2 || f.Col != 5 || f.Severity != Error || f.Code != "PYTHON_SYNTAX" {
		t.Fatalf("bad parse: %+v", f)
	}
	if !strings.Contains(f.Detail, "invalid syntax") {
		t.Errorf("detail lost the message: %q", f.Detail)
	}
}

func TestParseDiagnosticsMSVCStyle(t *testing.T) {
	fs := parseDiagnostics("cuda", "CUDA_SYNTAX",
		`C:\k\bad.cu(12): error C2143: syntax error`, "")
	if len(fs) != 1 {
		t.Fatalf("want 1 finding, got %d (%v)", len(fs), fs)
	}
	f := fs[0]
	if f.Line != 12 || f.Severity != Error || f.Code != "CUDA_SYNTAX" {
		t.Fatalf("bad MSVC parse: %+v", f)
	}
}

func TestParseDiagnosticsWarningSeverityAndNoise(t *testing.T) {
	fs := parseDiagnostics("cuda", "CUDA_SYNTAX", "",
		"nvcc: usage banner line that is not a diagnostic\n"+
			"k.cu:3:1: warning: unused variable 'x'")
	if len(fs) != 1 {
		t.Fatalf("banner noise must be ignored; want 1 finding, got %d (%v)", len(fs), fs)
	}
	if fs[0].Severity != Warning || fs[0].Code != "CUDA_WARNING" {
		t.Fatalf("want a warning finding, got %+v", fs[0])
	}
}

// --- helpers ----------------------------------------------------------------

func TestHasErrorAndSummaryOrdersErrorsFirst(t *testing.T) {
	fs := []Finding{
		{Pack: "go", Code: "GO_WARN", File: "a.go", Line: 9, Severity: Warning, Detail: "soft"},
		{Pack: "go", Code: "GO_PARSE", File: "a.go", Line: 2, Severity: Error, Detail: "hard"},
	}
	if !HasError(fs) {
		t.Fatal("HasError should be true")
	}
	if HasError(fs[:1]) {
		t.Fatal("HasError should be false for warnings only")
	}
	sum := Summary(fs)
	if i, j := strings.Index(sum, "hard"), strings.Index(sum, "soft"); i < 0 || j < 0 || i > j {
		t.Fatalf("Summary must list the error before the warning:\n%s", sum)
	}
}

func TestSeverityString(t *testing.T) {
	if Error.String() != "error" || Warning.String() != "warning" {
		t.Fatalf("severity strings wrong: %q %q", Error, Warning)
	}
}

func TestOffsetToLineCol(t *testing.T) {
	src := []byte("ab\ncde\nf")
	for _, c := range []struct {
		off, line, col int
	}{
		{0, 1, 1}, {1, 1, 2}, {3, 2, 1}, {4, 2, 2}, {7, 3, 1},
	} {
		if l, k := offsetToLineCol(src, c.off); l != c.line || k != c.col {
			t.Errorf("offset %d: want %d:%d got %d:%d", c.off, c.line, c.col, l, k)
		}
	}
}

// --- presence-gated integration (skips when the toolchain is absent) --------

func TestPythonPackRealCompile(t *testing.T) {
	bin := firstOnPath([]string{"python3", "python"})
	if bin == "" {
		t.Skip("no python on PATH; the python pack degrades to no-opinion here")
	}
	r := DefaultRegistry()

	fs, err := r.LintBytes(bg(), "tensor_model.py", []byte("def f(:\n    return 1\n"))
	if err != nil {
		t.Fatalf("LintBytes: %v", err)
	}
	if !HasError(fs) {
		t.Fatalf("want a PYTHON_SYNTAX error from real python, got %v", fs)
	}
	if fs[0].Code != "PYTHON_SYNTAX" || fs[0].File != "tensor_model.py" {
		t.Fatalf("unexpected finding: %+v", fs[0])
	}

	clean, err := r.LintBytes(bg(), "ok.py", []byte("import math\n\ndef g(x):\n    return math.sqrt(x)\n"))
	if err != nil {
		t.Fatalf("LintBytes(clean): %v", err)
	}
	if len(clean) != 0 {
		t.Fatalf("clean python must yield no findings, got %v", clean)
	}
}

func TestCUDAPackRealCompile(t *testing.T) {
	if _, err := exec.LookPath("nvcc"); err != nil {
		t.Skip("no nvcc on PATH; the cuda pack degrades to no-opinion off a GPU host")
	}
	r := DefaultRegistry()
	fs, err := r.LintBytes(bg(), "kernel.cu", []byte("__global__ void k(float* a) { a[threadIdx.x] = ; }\n"))
	if err != nil {
		t.Fatalf("LintBytes: %v", err)
	}
	if !HasError(fs) {
		t.Fatalf("want a CUDA_SYNTAX error from real nvcc, got %v", fs)
	}
}
