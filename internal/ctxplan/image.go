package ctxplan

import (
	"encoding/json"
	"fmt"
)

// image.go — the PERSISTED image of a candidate Index: the SAFE-metadata snapshot a
// session writes alongside its recall core image so a resumed session RE-ATTACHES its
// index instead of rebuilding it (issue #558, half a).
//
// # Why an index is persistable at all (and why it is just the span table)
//
// An Index is pure SAFE metadata — the span table (the recency order), the inverted token
// posting lists, and the durable set (index.go). Two of those three are DERIVED, not
// primary: the posting lists and the durable set are a deterministic function of the span
// table (Add rebuilds them from each span's role+descriptor tokens and durability class).
// So the minimal faithful image is the span table ALONE, and BuildIndex re-derives the rest
// — exactly the equivalence maintain.go already proves (an incrementally-maintained index is
// reflect.DeepEqual to a fresh BuildIndex over the same span set). Serializing only the
// primary state and rederiving the derived state is both smaller AND safer than persisting
// the posting lists too: a hand-edited or version-skewed posting map can never silently
// disagree with the span table, because the loader never reads one — it rebuilds it.
//
// The image is therefore the same SAFE Span metadata the planner already reasons over —
// never the bytes of a sealed span, never the bytes of a benign one (a Span carries a
// content-address Digest, not content). Persisting it leaks nothing the in-memory index did
// not already hold: a sealed span persists with Sealed=true and its sealed-safe descriptor,
// so the trust invariant survives a save/load exactly as it survives a turn.
//
// # Re-attach == rebuild (the cost the persistence buys back)
//
// A resumed session that LoadIndexes its image pays O(spans) to rebuild the posting lists +
// durable set once, then Probes in O(c) per turn — versus rebuilding the index from the
// store every turn (O(N) per turn, Θ(N²) cumulative, the cost index.go exists to flatten).
// The persisted form is what lets the re-attach be a one-time O(N) instead of a per-resume
// re-scan of the durable store, and the round-trip is provably identical to the live index
// (image_test.go's RestoreIndex(ix.Image()) == ix witness), so re-attaching is never a
// behavior change — only a cost one.

// IndexImageVersion is the schema tag stamped into every IndexImage. A loader refuses an
// image whose version it does not recognize (RestoreIndex), so a forward-incompatible change
// to the Span shape fails closed at load instead of silently rebuilding a wrong index.
const IndexImageVersion = "ctxplan-index-v1"

// IndexImage is the serializable SAFE-metadata image of an Index: a version tag plus the
// span table in append (recency) order. The inverted posting lists and the durable set are
// NOT stored — RestoreIndex rederives them from the span table via BuildIndex, so the
// serialized form is minimal and can never disagree with itself. It is JSON-serializable
// (Span carries json tags) and carries no bytes, so it is safe to persist alongside a
// recall core image (recall.PersistIndex).
type IndexImage struct {
	Version string `json:"version"`
	Spans   []Span `json:"spans"`
}

// Image returns the persistable SAFE-metadata image of the index: the version tag plus a
// defensive copy of the span table (Spans() clones each span's Attrs, so the image is
// independent of the live index and safe to serialize while the index keeps mutating). It
// is the accessor a persistence layer serializes; RestoreIndex is its inverse.
func (ix *Index) Image() IndexImage {
	return IndexImage{Version: IndexImageVersion, Spans: ix.Spans()}
}

// RestoreIndex rebuilds a live Index from a persisted image, rederiving the inverted posting
// lists and the durable set from the span table via BuildIndex — so a restored index is
// STRUCTURALLY IDENTICAL to the one that produced the image (image_test.go), and a Probe
// over it is byte-identical to a Probe over a fresh BuildIndex of the same spans (re-attach
// == rebuild). It refuses an image whose Version it does not recognize (fail closed), so a
// schema skew surfaces as an error at load rather than a silently-wrong index.
func RestoreIndex(img IndexImage) (*Index, error) {
	if img.Version != IndexImageVersion {
		return nil, fmt.Errorf("ctxplan: index image version %q != %q", img.Version, IndexImageVersion)
	}
	return BuildIndex(img.Spans), nil
}

// MarshalIndexImage serializes an index's image to indented JSON — the bytes a persistence
// layer writes to disk. It is a thin convenience over json.MarshalIndent(ix.Image()), kept
// here so the on-disk form (and its indentation, matching recall's manifest.json/cas.json)
// has one definition.
func MarshalIndexImage(ix *Index) ([]byte, error) {
	return json.MarshalIndent(ix.Image(), "", "  ")
}

// UnmarshalIndexImage parses an index image from JSON bytes and restores it — the inverse of
// MarshalIndexImage. A malformed image is a parse error; a version-skewed one is refused by
// RestoreIndex. It is the one-call load path a persistence layer reads with.
func UnmarshalIndexImage(b []byte) (*Index, error) {
	var img IndexImage
	if err := json.Unmarshal(b, &img); err != nil {
		return nil, fmt.Errorf("ctxplan: bad index image: %w", err)
	}
	return RestoreIndex(img)
}
