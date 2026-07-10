package tokenizer

// vision.go — Qwen-VL vision placeholder recognition (#4031).
//
// A VLM prompt is still a plain id stream: the image is not tokenized into pixels,
// it is represented by a placeholder run the chat template emits —
// <|vision_start|> <|image_pad|>...(N copies) <|vision_end|> — and the serve seam
// (#4032) later splices the vision encoder's N feature rows in place of the pads.
// Two things have to be true for that to work, and both already hold in the base
// tokenizer: each placeholder must Encode to exactly ONE id (never BPE-split), and
// that id must be recoverable. The single-id property is already guaranteed —
// FromGGML marks these as CONTROL (or the isWrappedSpecial "<|...|>" fallback catches
// them when a conversion drops token_type), and specialByContent is matched
// longest-first before BPE. What #4031 adds is the typed, id-agnostic ACCESS surface
// the seam needs: getters that resolve each placeholder to whatever id THIS model's
// vocab assigned it (the Qwen reference vocab uses 151652.., but nothing here hardcodes
// that — the id is read from the loaded special table), plus predicates to scan a
// decoded id stream.
//
// Scope boundary (deliberate): #4031 only CLASSIFIES and EXPOSES ids. The expansion
// of one <|image_pad|> into N image-feature rows — and the reconciliation of that N
// against len(VisionEmbedding.Vectors) — is owned by the serve seam (#4032); N comes
// from the vision encoder's NumImageTokens (#4030), never from this file.

// Canonical Qwen-VL vision placeholder content strings. These string literals are the
// stable contract across Qwen2-VL / Qwen2.5-VL / Qwen3-VL; the numeric ids behind them
// are model-defined and resolved from the tokenizer's own special table at call time.
const (
	VisionStartToken = "<|vision_start|>"
	VisionEndToken   = "<|vision_end|>"
	VisionPadToken   = "<|vision_pad|>"
	ImagePadToken    = "<|image_pad|>"
	VideoPadToken    = "<|video_pad|>"
)

// visionPlaceholderTokens is the full set scanned by IsVisionPlaceholder — the two
// span markers plus the three pad kinds.
var visionPlaceholderTokens = [...]string{
	VisionStartToken, VisionEndToken, VisionPadToken, ImagePadToken, VideoPadToken,
}

// VisionPlaceholders is the capability the serve seam (#4032) depends on to locate
// and classify vision placeholder ids without importing the concrete tokenizer type.
// *Tokenizer satisfies it for both the tokenizer.json and GGUF load paths.
type VisionPlaceholders interface {
	VisionStartID() (int, bool)
	VisionEndID() (int, bool)
	ImagePadID() (int, bool)
	VideoPadID() (int, bool)
	IsVisionPlaceholder(id int) bool
	IsImagePad(id int) bool
}

// SpecialID returns the id registered for a special-token content string, resolved
// against the tokenizer's own added/control table — never a hardcoded value. ok is
// false when content is not a registered special (so a base-vocab piece that happens
// to equal content is not mistaken for one). The special set is tiny, so the linear
// scan over specialByContent is cheaper than holding a second reverse map.
func (t *Tokenizer) SpecialID(content string) (int, bool) {
	for _, sp := range t.specialByContent {
		if sp.content == content {
			return sp.id, true
		}
	}
	return 0, false
}

// VisionStartID / VisionEndID / VisionPadID / ImagePadID / VideoPadID resolve each
// canonical Qwen-VL placeholder to this model's id. ok is false when the vocab does
// not carry that placeholder (e.g. a text-only model), which is how the seam tells a
// VLM tokenizer from a plain one.
func (t *Tokenizer) VisionStartID() (int, bool) { return t.SpecialID(VisionStartToken) }
func (t *Tokenizer) VisionEndID() (int, bool)   { return t.SpecialID(VisionEndToken) }
func (t *Tokenizer) VisionPadID() (int, bool)   { return t.SpecialID(VisionPadToken) }
func (t *Tokenizer) ImagePadID() (int, bool)    { return t.SpecialID(ImagePadToken) }
func (t *Tokenizer) VideoPadID() (int, bool)    { return t.SpecialID(VideoPadToken) }

// IsImagePad reports whether id is the <|image_pad|> placeholder — the row the serve
// seam (#4032) replaces with one image-feature vector. It is false when the vocab has
// no <|image_pad|>, so a text-only model never spuriously matches.
func (t *Tokenizer) IsImagePad(id int) bool {
	got, ok := t.ImagePadID()
	return ok && got == id
}

// IsVisionPlaceholder reports whether id is any Qwen-VL vision placeholder — the span
// markers or an image/video pad. This is the set #4032 walks to find where encoder
// output splices into the prompt; it deliberately does NOT distinguish which kind
// (use the specific getters for that).
func (t *Tokenizer) IsVisionPlaceholder(id int) bool {
	for _, content := range visionPlaceholderTokens {
		if got, ok := t.SpecialID(content); ok && got == id {
			return true
		}
	}
	return false
}
