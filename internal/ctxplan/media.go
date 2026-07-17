package ctxplan

import "strings"

// media.go — FIRST-CLASS MEDIA SPAN DESCRIPTORS (#5167).
//
// A Span carries a content-address Digest, not content, and every planner surface that
// scores, ages, or durability-classes a span is token/role-based — built for text
// (signalnoise.go, layout.go, index.go's inverted posting lists over role+descriptor).
// A raw media span (an image, an audio clip) has no descriptor tokens at all, so the
// planner is blind to it: it cannot be probed by relevance, it falls out of the layout's
// durability access path (unknown durability normalizes to turn), and its only residency
// path is recency. This file gives media a first-class SAFE descriptor so media context
// participates in the SAME recency/relevance/durability machinery as text — no new access
// path, no special-casing in the planner.
//
// The descriptor is three facts, all SAFE metadata (never media bytes):
//
//	media type — the coarse kind (image/audio/video; unknown fails closed to binary)
//	size class — small/medium/large, derived from Span.Bytes (the token-cost proxy)
//	alt text   — the caption/alt-text if the producer supplied one (extractive, human text)
//
// They are carried twice, deliberately: structured in Attrs (AttrMediaType /
// AttrMediaSizeClass / AttrMediaAlt) for programmatic readers, and rendered into
// Span.Descriptor as plain words so the EXISTING extractive tokenization
// (tokenSet(role+" "+descriptor)) indexes them with zero changes — a forecast intent of
// "the architecture diagram" reaches an image span exactly the way it reaches a text span.
// Durability defaults follow the package's fail-closed posture: a media span WITH a
// content-address Digest is recoverable from the durable store, so it defaults to bounded
// (reachable through the layout's deep durability path); one WITHOUT a digest is
// unrecoverable once elided and stays turn-scoped.

// Media type vocabulary — the closed coarse kinds NormMediaType folds into. Unknown input
// fails closed to MediaBinary (an honest "opaque bytes", never a guessed kind).
const (
	MediaImage  = "image"
	MediaAudio  = "audio"
	MediaVideo  = "video"
	MediaBinary = "binary"
)

// Size-class vocabulary — the coarse cost bands MediaSizeClassOf folds Span.Bytes into.
// Coarse on purpose: the planner needs "is this cheap or a budget event", not a byte count
// (Span.Bytes already carries the exact number).
const (
	MediaSizeSmall  = "small"  // <= 64 KiB — cheap to keep resident
	MediaSizeMedium = "medium" // <= 1 MiB — carried when it pulls weight
	MediaSizeLarge  = "large"  // > 1 MiB — a budget event; prefer the digest pointer
)

const (
	mediaSizeSmallMax  int64 = 64 << 10 // 64 KiB
	mediaSizeMediumMax int64 = 1 << 20  // 1 MiB
)

// Attrs keys for the structured half of the media descriptor. Attrs is the open bag the
// Span already serializes (json round-trips through IndexImage), so a media descriptor
// survives persist/re-attach with no schema change.
const (
	AttrMediaType      = "media_type"
	AttrMediaSizeClass = "media_size_class"
	AttrMediaAlt       = "media_alt"
)

// mediaTypes is the membership set behind NormMediaType — the same closed-vocabulary
// idiom durabilityRank/evidenceKinds use.
var mediaTypes = map[string]bool{
	MediaImage:  true,
	MediaAudio:  true,
	MediaVideo:  true,
	MediaBinary: true,
}

// NormMediaType folds any media-type string into the closed vocabulary, failing closed to
// MediaBinary ("opaque bytes") — the media analogue of NormDurability. Matching is
// case-insensitive and tolerant of MIME-style input ("image/png" -> image), because
// producers hand the planner MIME types far more often than bare kinds.
func NormMediaType(s string) string {
	t := strings.ToLower(strings.TrimSpace(s))
	if i := strings.IndexByte(t, '/'); i > 0 {
		t = t[:i]
	}
	if mediaTypes[t] {
		return t
	}
	return MediaBinary
}

// MediaSizeClassOf folds a byte size into the coarse size-class vocabulary. Non-positive
// sizes class as small (an empty asset costs nothing to keep).
func MediaSizeClassOf(bytes int64) string {
	switch {
	case bytes <= mediaSizeSmallMax:
		return MediaSizeSmall
	case bytes <= mediaSizeMediumMax:
		return MediaSizeMedium
	default:
		return MediaSizeLarge
	}
}

// MediaDescriptor is the first-class SAFE descriptor of one media span — the three facts
// that let an image/audio/video span be scored, aged, and durability-classed like text.
// It never carries media bytes; Alt is producer-supplied human text (caption/alt-text)
// and may be empty.
type MediaDescriptor struct {
	Type      string `json:"type"`
	SizeClass string `json:"size_class"`
	Alt       string `json:"alt,omitempty"`
}

// DescriptorText renders the descriptor as plain words for Span.Descriptor — the token
// surface the inverted index and the relevance ranker already consume. The leading
// "media <type> <size>" words make every media span reachable by kind ("the image",
// "that screenshot" forecasts overlapping "media image"), and the alt text contributes
// the same content-words a text span's extractive descriptor would.
func (d MediaDescriptor) DescriptorText() string {
	parts := []string{"media", d.Type, d.SizeClass}
	if d.Alt != "" {
		parts = append(parts, d.Alt)
	}
	return strings.Join(parts, " ")
}

// AnnotateMedia stamps a span as media in place and returns the descriptor it derived:
// type from NormMediaType (fail-closed), size class from the span's own Bytes, alt text
// trimmed as given. It writes both halves — the structured Attrs keys and the token-
// bearing Descriptor (prepended, preserving any existing descriptor text) — and applies
// the durability default: an empty Durability becomes bounded when the span carries a
// content-address Digest (the bytes are recoverable, so aging it out is safe) and turn
// when it does not (unrecoverable once elided; expire by default). An explicitly set
// Durability is never overridden. Call it once, at span creation.
func AnnotateMedia(s *Span, mediaType, alt string) MediaDescriptor {
	d := MediaDescriptor{
		Type:      NormMediaType(mediaType),
		SizeClass: MediaSizeClassOf(s.Bytes),
		Alt:       strings.TrimSpace(alt),
	}
	if s.Attrs == nil {
		s.Attrs = map[string]string{}
	}
	s.Attrs[AttrMediaType] = d.Type
	s.Attrs[AttrMediaSizeClass] = d.SizeClass
	if d.Alt != "" {
		s.Attrs[AttrMediaAlt] = d.Alt
	}
	if prev := strings.TrimSpace(s.Descriptor); prev != "" {
		s.Descriptor = d.DescriptorText() + " " + prev
	} else {
		s.Descriptor = d.DescriptorText()
	}
	if s.Durability == "" {
		if s.Digest != "" {
			s.Durability = DurabilityBounded
		} else {
			s.Durability = DurabilityTurn
		}
	}
	return d
}

// IsMediaSpan reports whether a span carries a media descriptor (was AnnotateMedia'd).
func IsMediaSpan(s Span) bool {
	_, ok := s.Attrs[AttrMediaType]
	return ok
}

// MediaDescriptorOf reads a span's media descriptor back from its Attrs — the inverse of
// AnnotateMedia, and the read path adapters use after a persist/re-attach round-trip.
// The boolean is false for a non-media span; the returned type/size are re-normalized on
// read (fail-closed), so a hand-edited image can never smuggle an open-vocabulary kind
// back into the planner.
func MediaDescriptorOf(s Span) (MediaDescriptor, bool) {
	t, ok := s.Attrs[AttrMediaType]
	if !ok {
		return MediaDescriptor{}, false
	}
	return MediaDescriptor{
		Type:      NormMediaType(t),
		SizeClass: normMediaSizeClass(s.Attrs[AttrMediaSizeClass], s.Bytes),
		Alt:       s.Attrs[AttrMediaAlt],
	}, true
}

// normMediaSizeClass folds a stored size-class string into the closed vocabulary, falling
// back to re-deriving it from the span's own Bytes when the stored value is unknown —
// the size class is DERIVED state, so rederiving is always honest.
func normMediaSizeClass(s string, bytes int64) string {
	switch s {
	case MediaSizeSmall, MediaSizeMedium, MediaSizeLarge:
		return s
	}
	return MediaSizeClassOf(bytes)
}
