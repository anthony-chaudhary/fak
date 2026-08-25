package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// writeIndentedJSON encodes v as two-space-indented JSON to w. It is the single
// construction site for the `json.NewEncoder(w); enc.SetIndent("", "  ")` idiom
// that the command handlers repeat for every --json output path.
func writeIndentedJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// writeIndentedJSONNoEscape is writeIndentedJSON with HTML escaping disabled. The
// scorecard / report emitters use it so payloads with <, >, & survive verbatim.
func writeIndentedJSONNoEscape(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(v)
}

// writeIndentedJSONFile encodes v as indented, HTML-unescaped JSON and writes it to path,
// creating the parent directory (0o755) if it does not already exist. It is the shared body
// behind the per-report file writers (e.g. the affected-run and test-duration ledgers) that
// each buffered writeIndentedJSONNoEscape then MkdirAll + WriteFile in lockstep.
func writeIndentedJSONFile(path string, v any) error {
	var b bytes.Buffer
	if err := writeIndentedJSONNoEscape(&b, v); err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, b.Bytes(), 0o644)
}

// encodeJSONOrFail writes v as indented JSON to stdout, returning 0 on success.
// On an encode error it reports `<label>: encode json: <err>` to stderr and
// returns 1, matching the convention the per-command JSON branches use.
func encodeJSONOrFail(stdout, stderr io.Writer, v any, label string) int {
	return encodeJSONOrFailPrefixed(stdout, stderr, v, label+": encode json")
}

// encodeJSONOrFailPrefixed is encodeJSONOrFail for the --json branches whose stderr
// wording is not the "<cmd>: encode json:" convention. errPrefix is the exact text printed
// ahead of ": <err>" and is passed whole rather than assembled from a command name because
// the existing messages are genuinely inconsistent — most commands say
// "fak dispatch status: encode json", `fak lab` says "fak lab: encode", and `fak sweep`
// prints the bare command name — so every call site keeps the string it already printed.
func encodeJSONOrFailPrefixed(stdout, stderr io.Writer, v any, errPrefix string) int {
	if err := writeIndentedJSON(stdout, v); err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", errPrefix, err)
		return 1
	}
	return 0
}

// emitJSONOrRender centralizes the common command boundary where --json selects the
// indented machine payload and the default path invokes a command-specific renderer.
func emitJSONOrRender(stdout, stderr io.Writer, label string, asJSON bool, value any, render func(io.Writer)) int {
	return emitJSONOrRenderPrefixed(stdout, stderr, label+": encode json", asJSON, value, render)
}

func emitJSONOrRenderPrefixed(stdout, stderr io.Writer, errPrefix string, asJSON bool, value any, render func(io.Writer)) int {
	if asJSON {
		return encodeJSONOrFailPrefixed(stdout, stderr, value, errPrefix)
	}
	render(stdout)
	return 0
}

func emitReportGate(stdout io.Writer, asJSON bool, code int, message string, gated any) int {
	if asJSON {
		_ = writeIndentedJSONNoEscape(stdout, gated)
	} else {
		fmt.Fprintln(stdout, message)
	}
	return code
}

func checkAndEmitReportGate[T any](stdout io.Writer, asJSON bool, report T, check func(T) (int, string), withGate func(T, int, string) T) int {
	code, message := check(report)
	return emitReportGate(stdout, asJSON, code, message, withGate(report, code, message))
}
