// Rung C4 (issue #1873): the do_not_rederive index. The Baton type (baton.go, rung C1)
// already carries DoNotRederive as the wire-mandated `[]string` of closed-dead-end
// pointers, but nothing dedupes them — a leg that independently rediscovers the same
// dead end and appends it again would grow the baton without shedding any poison. This
// file adds a small in-memory index that a leg builds from a decoded baton (or from
// scratch), dedupes new pointers against, and projects back to the plain `[]string`
// the schema and codec (rung C2 #1871) already expect.
//
// The index stores pointers, never content (schema doc, "do_not_rederive"): callers add
// commit SHAs, issue refs, memory slugs, or file globs — the same closed-store-pointer
// vocabulary as Artifact — and the index treats each pointer string as its own stable
// key. It does not interpret, resolve, or rank pointers; consuming do_not_rederive
// advisorially is a later rung's job (out of scope here, per the issue).
package relay

// DoNotRederiveIndex is a deduplicated, insertion-ordered index of do_not_rederive
// pointers. The zero value is a usable empty index.
type DoNotRederiveIndex struct {
	seen  map[string]bool
	order []string
}

// NewDoNotRederiveIndex builds an index from pointers already present on a baton (or
// any other source), deduplicating on first occurrence and preserving that order. A nil
// or empty input yields an empty, usable index.
func NewDoNotRederiveIndex(pointers []string) *DoNotRederiveIndex {
	idx := &DoNotRederiveIndex{}
	for _, p := range pointers {
		idx.Add(p)
	}
	return idx
}

// Add records pointer in the index if it is non-empty and not already present.
// Re-adding a pointer already in the index is a no-op: it neither duplicates the entry
// nor moves it, so a dead end recorded once stays recorded once no matter how many
// legs rediscover it.
func (idx *DoNotRederiveIndex) Add(pointer string) {
	if pointer == "" || idx.seen[pointer] {
		return
	}
	if idx.seen == nil {
		idx.seen = make(map[string]bool)
	}
	idx.seen[pointer] = true
	idx.order = append(idx.order, pointer)
}

// Pointers returns the deduplicated pointers in first-seen order, ready to assign to
// Baton.DoNotRederive. It always returns a non-nil slice (empty when the index holds
// nothing), matching the schema's "empty array is valid" contract and codec.project's
// nil-to-`[]` canonicalization.
func (idx *DoNotRederiveIndex) Pointers() []string {
	out := make([]string, len(idx.order))
	copy(out, idx.order)
	return out
}

// Len reports how many distinct pointers the index holds.
func (idx *DoNotRederiveIndex) Len() int {
	return len(idx.order)
}
