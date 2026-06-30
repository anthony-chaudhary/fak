package codelint

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// LSP diagnostic severities, matching the Language Server Protocol.
const (
	LSPSeverityError       = 1
	LSPSeverityWarning     = 2
	LSPSeverityInformation = 3
	LSPSeverityHint        = 4
)

// LSPPosition is a zero-based Language Server Protocol source coordinate.
type LSPPosition struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

// LSPRange is the source span reported by a language-server diagnostic.
type LSPRange struct {
	Start LSPPosition `json:"start"`
	End   LSPPosition `json:"end"`
}

// LSPDiagnosticCode is the LSP Diagnostic.code union: string | integer.
type LSPDiagnosticCode string

func (c *LSPDiagnosticCode) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*c = ""
		return nil
	}
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*c = LSPDiagnosticCode(s)
		return nil
	}
	var n int64
	if err := json.Unmarshal(b, &n); err == nil {
		*c = LSPDiagnosticCode(strconv.FormatInt(n, 10))
		return nil
	}
	return fmt.Errorf("lsp diagnostic code must be a string or integer")
}

func (c LSPDiagnosticCode) String() string {
	return string(c)
}

// LSPDiagnostic is the small, pack-local subset of publishDiagnostics that codelint
// needs to map a whole-package semantic checker back onto the common Finding shape.
type LSPDiagnostic struct {
	Range    LSPRange          `json:"range"`
	Severity int               `json:"severity"`
	Code     LSPDiagnosticCode `json:"code,omitempty"`
	Source   string            `json:"source,omitempty"`
	Message  string            `json:"message"`
}

// FindingsFromLSPDiagnostics maps language-server diagnostics into codelint findings.
// It deliberately keeps only severity=Error: semantic packs are advisory write-boundary
// signals, and hints/warnings would turn the self-correction channel into style noise.
func FindingsFromLSPDiagnostics(pack, file string, diagnostics []LSPDiagnostic) []Finding {
	pack = strings.TrimSpace(pack)
	if pack == "" {
		pack = "lsp"
	}
	codePrefix := strings.ToUpper(pack)
	var out []Finding
	for _, d := range diagnostics {
		if d.Severity != LSPSeverityError {
			continue
		}
		code := strings.TrimSpace(d.Code.String())
		if code == "" {
			code = codePrefix + "_SEMANTIC"
		}
		detail := strings.TrimSpace(d.Message)
		if detail == "" {
			detail = "language server error"
		}
		out = append(out, Finding{
			Pack:     pack,
			Code:     code,
			File:     file,
			Line:     lspOneBased(d.Range.Start.Line),
			Col:      lspOneBased(d.Range.Start.Character),
			Severity: Error,
			Detail:   detail,
		})
	}
	return out
}

func lspOneBased(n int) int {
	if n < 0 {
		return 0
	}
	return n + 1
}
