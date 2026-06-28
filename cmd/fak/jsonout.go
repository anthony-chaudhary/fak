package main

import (
	"encoding/json"
	"fmt"
	"io"
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
