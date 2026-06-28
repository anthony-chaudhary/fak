package capindex

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
)

// Digest is the ScaleMCP sync key: SHA-256 over a capability's bytes, rendered
// as "sha256:<hex>". It is the one hash function the whole loader keys on — a
// card's at-rest digest, a faulted body's digest, and the index's change
// detection all agree because they all call this. Two byte slices with the same
// contents always produce the same digest (stability); any change produces a
// different one (the hot-swap signal).
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// ChangeKind classifies one entry's transition between two index snapshots.
type ChangeKind int

const (
	// Added means the capability is present in the new snapshot but not the old.
	Added ChangeKind = iota
	// Removed means the capability vanished from the new snapshot.
	Removed
	// Changed means the capability is present in both but its digest differs.
	Changed
)

func (k ChangeKind) String() string {
	switch k {
	case Added:
		return "added"
	case Removed:
		return "removed"
	case Changed:
		return "changed"
	default:
		return "unknown"
	}
}

// Change is one CRUD-diff row: a capability ref and how it moved between two
// snapshots. OldDigest is empty for Added; NewDigest is empty for Removed.
type Change struct {
	Ref       CapRef
	Kind      ChangeKind
	OldDigest string
	NewDigest string
}

// Index is the at-rest, 0→∞ capability index. It holds CapCards only — never
// the full bodies — keyed by CapRef. The body is paged in lazily by a Resolver's
// Fault, so the at-rest cost is O(catalog cards), not O(catalog bodies).
//
// Register/Remove mutate the live set; Snapshot freezes a comparable view; Diff
// reports the CRUD delta between two snapshots so a re-index is O(changed), not
// O(catalog). That is the ScaleMCP hash-diff sync this whole epic stands on.
type Index struct {
	cards map[CapRef]CapCard
}

// NewIndex returns an empty index.
func NewIndex() *Index {
	return &Index{cards: make(map[CapRef]CapCard)}
}

// Register adds or replaces a card in the index. If the card has no Digest set,
// one is computed from its CardBytes so every registered card is digest-keyed.
// Re-registering an identical card is idempotent — the digest is unchanged, so a
// later Diff reports no Change for it (re-indexing an unchanged catalog is a
// no-op).
func (ix *Index) Register(card CapCard) {
	if card.Digest == "" {
		card.Digest = Digest(card.CardBytes)
	}
	ix.cards[card.Ref] = card
}

// RegisterAll registers every card from a Resolver's Index() in one call. This
// is the cheap at-rest sync: a resolver lists its cards, the index keys them by
// digest, and no body is paged.
func (ix *Index) RegisterAll(cards []CapCard) {
	for _, c := range cards {
		ix.Register(c)
	}
}

// Remove drops a card by ref. It returns true if a card was present.
func (ix *Index) Remove(ref CapRef) bool {
	if _, ok := ix.cards[ref]; !ok {
		return false
	}
	delete(ix.cards, ref)
	return true
}

// Len reports how many cards are held at rest.
func (ix *Index) Len() int { return len(ix.cards) }

// Get returns the card for a ref and whether it was present.
func (ix *Index) Get(ref CapRef) (CapCard, bool) {
	c, ok := ix.cards[ref]
	return c, ok
}

// Snapshot is a frozen, comparable view of an index: ref → digest. It carries
// only the hash per capability, so diffing two snapshots is a cheap map walk —
// the whole point of keying on a digest.
type Snapshot struct {
	digests map[CapRef]string
}

// Snapshot freezes the current index into a comparable digest map.
func (ix *Index) Snapshot() Snapshot {
	d := make(map[CapRef]string, len(ix.cards))
	for ref, card := range ix.cards {
		d[ref] = card.Digest
	}
	return Snapshot{digests: d}
}

// Diff reports the CRUD delta from old to new: each ref Added (in new only),
// Removed (in old only), or Changed (in both, different digest). Refs whose
// digest is identical in both snapshots produce NO row — that is the no-op an
// unchanged catalog yields, and the reason a re-index is O(changed). The result
// is sorted (Kind, then Name) so it is deterministic.
func (old Snapshot) Diff(next Snapshot) []Change {
	var out []Change

	for ref, newDigest := range next.digests {
		oldDigest, ok := old.digests[ref]
		switch {
		case !ok:
			out = append(out, Change{Ref: ref, Kind: Added, NewDigest: newDigest})
		case oldDigest != newDigest:
			out = append(out, Change{Ref: ref, Kind: Changed, OldDigest: oldDigest, NewDigest: newDigest})
		}
	}
	for ref, oldDigest := range old.digests {
		if _, ok := next.digests[ref]; !ok {
			out = append(out, Change{Ref: ref, Kind: Removed, OldDigest: oldDigest})
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		if out[i].Ref.Name != out[j].Ref.Name {
			return out[i].Ref.Name < out[j].Ref.Name
		}
		return out[i].Ref.Version < out[j].Ref.Version
	})
	return out
}
