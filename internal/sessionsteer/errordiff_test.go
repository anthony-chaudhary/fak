package sessionsteer

import (
	"strings"
	"testing"
)

func TestErrorDiff(t *testing.T) {
	t.Run("GoCompiler_UndefinedIdentifier", func(t *testing.T) {
		input := "main.go:12:5: undefined: foo"
		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.Compiler != "go" {
			t.Errorf("expected compiler 'go', got %q", diff.Compiler)
		}
		if diff.FilePath != "main.go" {
			t.Errorf("expected FilePath 'main.go', got %q", diff.FilePath)
		}
		if diff.Line != 12 {
			t.Errorf("expected Line 12, got %d", diff.Line)
		}
		if diff.Column != 5 {
			t.Errorf("expected Column 5, got %d", diff.Column)
		}
		if diff.OffendingToken != "foo" {
			t.Errorf("expected OffendingToken 'foo', got %q", diff.OffendingToken)
		}
		if !strings.Contains(diff.SuggestedFix, "foo") {
			t.Errorf("expected SuggestedFix to contain 'foo', got %q", diff.SuggestedFix)
		}
		if diff.CascadingCount != 0 {
			t.Errorf("expected 0 cascading errors, got %d", diff.CascadingCount)
		}

		expectedDiffPrefix := "--- a/main.go\n+++ b/main.go\n@@ -12,1 +12,1 @@\n"
		if !strings.HasPrefix(diff.FormattedDiff, expectedDiffPrefix) {
			t.Errorf("expected FormattedDiff prefix %q, got %q", expectedDiffPrefix, diff.FormattedDiff)
		}
		if !strings.Contains(diff.FormattedDiff, "+ // fix:") {
			t.Errorf("expected fix line in diff, got %q", diff.FormattedDiff)
		}
	})

	t.Run("GoCompiler_TypeMismatch", func(t *testing.T) {
		input := "bar.go:44:10: cannot use x (variable of type int) as type string"
		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.FilePath != "bar.go" || diff.Line != 44 || diff.Column != 10 {
			t.Errorf("unexpected location: %s:%d:%d", diff.FilePath, diff.Line, diff.Column)
		}
		if diff.OffendingToken != "x" {
			t.Errorf("expected OffendingToken 'x', got %q", diff.OffendingToken)
		}
		if !strings.Contains(diff.SuggestedFix, "string") || !strings.Contains(diff.SuggestedFix, "int") {
			t.Errorf("expected SuggestedFix to mention types, got %q", diff.SuggestedFix)
		}
		if !strings.Contains(diff.FormattedDiff, "--- a/bar.go") {
			t.Errorf("expected diff to contain '--- a/bar.go', got %q", diff.FormattedDiff)
		}
	})

	t.Run("GoCompiler_CascadingSuppression", func(t *testing.T) {
		input := `# command-line-arguments
./main.go:12:5: undefined: foo
./main.go:13:5: undefined: foo
./main.go:14:10: cannot use x (variable of type int) as type string
./main.go:15:2: not enough arguments in call to add`

		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// First error should be isolated: line 12
		if diff.Line != 12 {
			t.Errorf("expected isolated root line 12, got %d", diff.Line)
		}
		if diff.OffendingToken != "foo" {
			t.Errorf("expected root token 'foo', got %q", diff.OffendingToken)
		}
		// 4 total errors -> 3 cascading suppressed
		if diff.CascadingCount != 3 {
			t.Errorf("expected 3 cascading errors, got %d", diff.CascadingCount)
		}
	})

	t.Run("TypeScript_CannotFindName", func(t *testing.T) {
		input := "src/index.ts:15:7 - error TS2304: Cannot find name 'bar'."
		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.Compiler != "tsc" {
			t.Errorf("expected compiler 'tsc', got %q", diff.Compiler)
		}
		if diff.FilePath != "src/index.ts" {
			t.Errorf("expected FilePath 'src/index.ts', got %q", diff.FilePath)
		}
		if diff.Line != 15 {
			t.Errorf("expected Line 15, got %d", diff.Line)
		}
		if diff.Column != 7 {
			t.Errorf("expected Column 7, got %d", diff.Column)
		}
		if diff.OffendingToken != "bar" {
			t.Errorf("expected OffendingToken 'bar', got %q", diff.OffendingToken)
		}
		if !strings.Contains(diff.SuggestedFix, "bar") {
			t.Errorf("expected SuggestedFix to mention 'bar', got %q", diff.SuggestedFix)
		}
		if !strings.Contains(diff.FormattedDiff, "--- a/src/index.ts") {
			t.Errorf("expected diff to mention '--- a/src/index.ts', got %q", diff.FormattedDiff)
		}
	})

	t.Run("TypeScript_TypeMismatchAndVsFormat", func(t *testing.T) {
		input := `src/app.tsx(22,15): error TS2322: Type 'number' is not assignable to type 'string'.`
		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.FilePath != "src/app.tsx" || diff.Line != 22 || diff.Column != 15 {
			t.Errorf("unexpected location: %s:%d:%d", diff.FilePath, diff.Line, diff.Column)
		}
		if diff.OffendingToken != "number" {
			t.Errorf("expected OffendingToken 'number', got %q", diff.OffendingToken)
		}
		if !strings.Contains(diff.SuggestedFix, "string") {
			t.Errorf("expected SuggestedFix to mention target type, got %q", diff.SuggestedFix)
		}
	})

	t.Run("TypeScript_CascadingSuppression", func(t *testing.T) {
		input := `src/index.ts:15:7 - error TS2304: Cannot find name 'bar'.
src/index.ts:16:7 - error TS2304: Cannot find name 'bar'.
src/index.ts:20:12 - error TS2339: Property 'length' does not exist on type 'never'.
Found 3 errors in 1 file.`

		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.Line != 15 {
			t.Errorf("expected root line 15, got %d", diff.Line)
		}
		if diff.CascadingCount != 2 {
			t.Errorf("expected 2 cascading errors, got %d", diff.CascadingCount)
		}
	})

	t.Run("CargoRustc_CannotFindValue_SingleLine", func(t *testing.T) {
		input := "error[E0425]: cannot find value 'baz' in this scope\n --> src/main.rs:10:5"
		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.Compiler != "cargo" {
			t.Errorf("expected compiler 'cargo', got %q", diff.Compiler)
		}
		if diff.FilePath != "src/main.rs" {
			t.Errorf("expected FilePath 'src/main.rs', got %q", diff.FilePath)
		}
		if diff.Line != 10 {
			t.Errorf("expected Line 10, got %d", diff.Line)
		}
		if diff.Column != 5 {
			t.Errorf("expected Column 5, got %d", diff.Column)
		}
		if diff.OffendingToken != "baz" {
			t.Errorf("expected OffendingToken 'baz', got %q", diff.OffendingToken)
		}
		if !strings.Contains(diff.SuggestedFix, "baz") {
			t.Errorf("expected SuggestedFix to mention 'baz', got %q", diff.SuggestedFix)
		}
		if !strings.Contains(diff.FormattedDiff, "--- a/src/main.rs") {
			t.Errorf("expected diff to mention '--- a/src/main.rs', got %q", diff.FormattedDiff)
		}
	})

	t.Run("CargoRustc_BackticksAndSnippet", func(t *testing.T) {
		input := `error[E0425]: cannot find value ` + "`" + `baz` + "`" + ` in this scope
  --> src/main.rs:10:5
   |
10 |     baz();
   |     ^^^ not found in this scope`

		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.OffendingToken != "baz" {
			t.Errorf("expected OffendingToken 'baz', got %q", diff.OffendingToken)
		}
		if diff.OriginalSnippet != "    baz();" {
			t.Errorf("expected snippet '    baz();', got %q", diff.OriginalSnippet)
		}
		if !strings.Contains(diff.FormattedDiff, "-    baz();") {
			t.Errorf("expected diff to show snippet replacement, got %q", diff.FormattedDiff)
		}
		if !strings.Contains(diff.FormattedDiff, "+    // fix: bring 'baz' into scope or define it") {
			t.Errorf("expected aligned fix line, got %q", diff.FormattedDiff)
		}
	})

	t.Run("CargoRustc_CascadingSuppression", func(t *testing.T) {
		input := `error[E0425]: cannot find value 'baz' in this scope
 --> src/main.rs:10:5
  |
10 |     baz();
  |     ^^^ not found in this scope

error[E0425]: cannot find value 'qux' in this scope
 --> src/main.rs:15:5
  |
15 |     qux();
  |     ^^^ not found in this scope

error: could not compile 'my_app' (bin "my_app") due to 2 previous errors`

		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.Line != 10 {
			t.Errorf("expected root line 10, got %d", diff.Line)
		}
		if diff.CascadingCount != 1 {
			t.Errorf("expected 1 cascading error suppressed, got %d", diff.CascadingCount)
		}
	})

	t.Run("GenericCompilerFallback", func(t *testing.T) {
		input := "parser/tokens.py:55:12: error: unexpected token 'END_OF_FILE'"
		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.FilePath != "parser/tokens.py" || diff.Line != 55 || diff.Column != 12 {
			t.Errorf("unexpected location: %s:%d:%d", diff.FilePath, diff.Line, diff.Column)
		}
		if diff.OffendingToken != "END_OF_FILE" {
			t.Errorf("expected token 'END_OF_FILE', got %q", diff.OffendingToken)
		}
		if !strings.Contains(diff.FormattedDiff, "--- a/parser/tokens.py") {
			t.Errorf("expected unified diff header, got %q", diff.FormattedDiff)
		}
	})

	t.Run("EmptyAndNoCompilerError", func(t *testing.T) {
		_, errEmpty := SynthesizeErrorDiff("")
		if errEmpty == nil {
			t.Error("expected error on empty input")
		}

		_, errWhitespace := SynthesizeErrorDiff("   \n\t  ")
		if errWhitespace == nil {
			t.Error("expected error on whitespace input")
		}

		_, errNoMatch := SynthesizeErrorDiff("Everything compiled successfully without errors.")
		if errNoMatch == nil {
			t.Error("expected ErrNoCompilerError on non-error input")
		}
	})

	t.Run("Methods", func(t *testing.T) {
		input := "main.go:12:5: undefined: foo"
		diff, err := SynthesizeErrorDiff(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if diff.String() != diff.FormattedDiff {
			t.Errorf("expected String() to match FormattedDiff")
		}
		summary := diff.Summary()
		if !strings.Contains(summary, "[go]") || !strings.Contains(summary, "main.go:12:5") {
			t.Errorf("unexpected summary: %q", summary)
		}
	})
}
