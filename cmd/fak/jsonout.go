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
	if err := writeIndentedJSON(stdout, v); err != nil {
		fmt.Fprintf(stderr, "%s: encode json: %v\n", label, err)
		return 1
	}
	return 0
}
