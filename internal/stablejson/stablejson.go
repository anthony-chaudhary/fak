// Package stablejson renders the canonical on-disk JSON form shared by fak's
// receipts, ledgers, and proposed manifests: two-space-indented encoding/json
// output with a single trailing newline. encoding/json sorts map keys, so the
// bytes are deterministic for a given value and byte-for-byte equality doubles
// as a content digest; the trailing newline keeps the written files stable
// under diff and concatenation.
package stablejson

import "encoding/json"

// Marshal returns v as stable JSON: json.MarshalIndent with a two-space indent
// plus one trailing newline. The marshal error is returned unwrapped so a
// caller's own context wraps it in one place.
func Marshal(v any) ([]byte, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
