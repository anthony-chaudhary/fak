package codelint

import (
	"encoding/json"
	"testing"
)

func TestFindingsFromLSPDiagnosticsMapsErrors(t *testing.T) {
	fs := FindingsFromLSPDiagnostics(" gopls ", "main.go", []LSPDiagnostic{
		{
			Severity: LSPSeverityWarning,
			Code:     LSPDiagnosticCode("unusedparams"),
			Message:  "unused parameter",
		},
		{
			Range: LSPRange{
				Start: LSPPosition{Line: 4, Character: 9},
			},
			Severity: LSPSeverityError,
			Code:     LSPDiagnosticCode("UndeclaredName"),
			Message:  " undefined: widget ",
		},
	})
	if len(fs) != 1 {
		t.Fatalf("want only the error diagnostic, got %d (%v)", len(fs), fs)
	}
	f := fs[0]
	if f.Pack != "gopls" || f.Code != "UndeclaredName" || f.File != "main.go" {
		t.Fatalf("unexpected identity fields: %+v", f)
	}
	if f.Line != 5 || f.Col != 10 || f.Severity != Error {
		t.Fatalf("unexpected location/severity: %+v", f)
	}
	if f.Detail != "undefined: widget" {
		t.Fatalf("message was not trimmed: %q", f.Detail)
	}
}

func TestFindingsFromLSPDiagnosticsErrorFloorAndFallbacks(t *testing.T) {
	for _, severity := range []int{0, LSPSeverityWarning, LSPSeverityInformation, LSPSeverityHint} {
		fs := FindingsFromLSPDiagnostics("gopls", "main.go", []LSPDiagnostic{
			{Severity: severity, Message: "not an error"},
		})
		if len(fs) != 0 {
			t.Fatalf("severity %d should be filtered, got %v", severity, fs)
		}
	}

	fs := FindingsFromLSPDiagnostics(" ", "broken.go", []LSPDiagnostic{
		{
			Range: LSPRange{
				Start: LSPPosition{Line: -1, Character: -2},
			},
			Severity: LSPSeverityError,
			Message:  " ",
		},
	})
	if len(fs) != 1 {
		t.Fatalf("want one fallback finding, got %d (%v)", len(fs), fs)
	}
	f := fs[0]
	if f.Pack != "lsp" || f.Code != "LSP_SEMANTIC" || f.Detail != "language server error" {
		t.Fatalf("fallback fields wrong: %+v", f)
	}
	if f.Line != 0 || f.Col != 0 {
		t.Fatalf("negative LSP positions should map to unknown coordinates, got %d:%d", f.Line, f.Col)
	}
}

func TestLSPDiagnosticCodeAcceptsIntegerJSON(t *testing.T) {
	var diagnostics []LSPDiagnostic
	err := json.Unmarshal([]byte(`[
		{
			"range": {"start": {"line": 1, "character": 2}, "end": {"line": 1, "character": 3}},
			"severity": 1,
			"code": 6133,
			"message": "declared and not used"
		}
	]`), &diagnostics)
	if err != nil {
		t.Fatalf("Unmarshal LSP diagnostics: %v", err)
	}
	fs := FindingsFromLSPDiagnostics("tsserver", "app.ts", diagnostics)
	if len(fs) != 1 {
		t.Fatalf("want one finding, got %d (%v)", len(fs), fs)
	}
	if fs[0].Code != "6133" || fs[0].Line != 2 || fs[0].Col != 3 {
		t.Fatalf("integer diagnostic code was not preserved: %+v", fs[0])
	}
}
