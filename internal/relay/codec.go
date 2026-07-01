// Rung C2 (issue #1871): the deterministic wire codec for the Baton type defined in
// baton.go. C1 gave the shape; this file makes "inspect the parsed baton" a real
// equality check by giving the type a pure, no-I/O, no-clock Parse/Marshal pair whose
// output is byte-stable for a given value — the determinism that IS the relay witness
// (mirrors session/snapshot: same input -> byte-identical bytes).
//
// Determinism is structural, not incidental: the Baton type tree (baton.go) contains no
// Go maps, and encoding/json emits struct fields in declaration order, so json.Marshal
// over a projected baton is byte-stable across runs and processes. The only source of
// wire ambiguity the schema calls out — a nil slice serializing as `null` instead of the
// mandated `[]` — is removed by project() before encoding, so a baton built with nil
// slices and one built with empty slices Marshal to the SAME bytes.
package relay

import (
	"encoding/json"
	"fmt"
)

// project returns the canonical form of b: the four slice fields the schema doc requires
// to serialize as `[]` (open_questions, artifacts, do_not_rederive, progress_cursor.
// held_region) are forced non-nil so Marshal emits `[]` rather than `null`. It reads no
// clock and does no I/O, and it is idempotent — project(project(b)) == project(b) — so a
// round-trip is byte-identical regardless of whether the caller left those slices nil.
// The `,omitempty` scalar fields (ledger_ref, note) are deliberately left untouched: an
// empty value drops out of the wire form on both the write and the re-write, so the
// round-trip stays byte-identical for them too.
func project(b Baton) Baton {
	b.OpenQuestions = nonNilStrings(b.OpenQuestions)
	b.DoNotRederive = nonNilStrings(b.DoNotRederive)
	b.ProgressCursor.HeldRegion = nonNilStrings(b.ProgressCursor.HeldRegion)
	if b.Artifacts == nil {
		b.Artifacts = []Artifact{}
	}
	return b
}

// nonNilStrings maps a nil slice to an empty, non-nil one and returns any other slice
// unchanged — the minimal projection that turns a `null` into the schema-mandated `[]`
// without reordering or copying existing elements.
func nonNilStrings(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// Marshal encodes b as the canonical `fak.relay.baton.v1` wire bytes: it projects b to
// canonical form (empty slices as `[]`, not `null`) and encodes it with encoding/json.
// Because the Baton tree has no maps and json emits fields in declaration order, the
// output is deterministic — Marshal(b) called twice, or in two processes, yields the
// exact same bytes. It reads no clock and touches no I/O, and the output carries no
// trailing newline (json.Marshal, not an Encoder).
func Marshal(b Baton) ([]byte, error) {
	out, err := json.Marshal(project(b))
	if err != nil {
		return nil, fmt.Errorf("relay: marshal baton: %w", err)
	}
	return out, nil
}

// Parse decodes canonical baton bytes into a Baton and projects the result to canonical
// form, so Parse round-trips a Marshal output exactly: Marshal(Parse(data)) is
// byte-identical to a canonical `data`, and reflect.DeepEqual(project(b), Parse(Marshal(b)))
// holds. It rejects input that is not a JSON object (a bare number/string/array fails to
// decode into the struct). Content validation beyond decodability — schema-tag equality,
// the ArtifactKind vocabulary, ref resolution — is a later rung's job (the reader
// contract); Parse here is the pure, no-I/O, no-clock decode+project that the equality
// witness rests on.
func Parse(data []byte) (Baton, error) {
	var b Baton
	if err := json.Unmarshal(data, &b); err != nil {
		return Baton{}, fmt.Errorf("relay: parse baton: %w", err)
	}
	return project(b), nil
}
