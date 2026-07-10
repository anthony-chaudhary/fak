package tokenizer

import "testing"

// visionVocab builds a tiny Qwen-VL-shaped GGML vocab: a byte-level BPE base plus the
// five vision placeholders. Ids are deliberately COMPACT and distinctive (3..7) rather
// than the Qwen2-VL reference values (vision_start=151652, vision_end=151653,
// vision_pad=151654, image_pad=151655, video_pad=151656) — the accessors resolve by
// content string, so the mechanism must return whatever id THIS vocab assigned, not a
// hardcoded one. markSpecial toggles CONTROL token_type vs. relying on the
// isWrappedSpecial "<|...|>" fallback.
func visionVocab(markSpecial bool) (*Tokenizer, error) {
	tokens := []string{
		"a", "b", "ab", // 0,1,2 — byte-level base
		VisionStartToken, // 3
		VisionEndToken,   // 4
		VisionPadToken,   // 5
		ImagePadToken,    // 6
		VideoPadToken,    // 7
	}
	var types []int32
	if markSpecial {
		types = []int32{1, 1, 1, ggmlTypeControl, ggmlTypeControl, ggmlTypeControl, ggmlTypeControl, ggmlTypeControl}
	}
	return FromGGML(tokens, []string{"a b"}, types, "qwen2")
}

// TestVisionPlaceholderIDs proves the typed accessors resolve each placeholder to the
// id its own vocab assigned, and that the predicates classify ids correctly.
func TestVisionPlaceholderIDs(t *testing.T) {
	tok, err := visionVocab(true)
	if err != nil {
		t.Fatalf("visionVocab: %v", err)
	}
	cases := []struct {
		name string
		got  func() (int, bool)
		want int
	}{
		{"vision_start", tok.VisionStartID, 3},
		{"vision_end", tok.VisionEndID, 4},
		{"vision_pad", tok.VisionPadID, 5},
		{"image_pad", tok.ImagePadID, 6},
		{"video_pad", tok.VideoPadID, 7},
	}
	for _, c := range cases {
		id, ok := c.got()
		if !ok || id != c.want {
			t.Errorf("%sID() = (%d,%v), want (%d,true)", c.name, id, ok, c.want)
		}
	}

	if !tok.IsImagePad(6) {
		t.Errorf("IsImagePad(6) = false, want true")
	}
	if tok.IsImagePad(3) {
		t.Errorf("IsImagePad(3 vision_start) = true, want false")
	}
	if tok.IsImagePad(2) {
		t.Errorf("IsImagePad(2 base token) = true, want false")
	}
	for _, id := range []int{3, 4, 5, 6, 7} {
		if !tok.IsVisionPlaceholder(id) {
			t.Errorf("IsVisionPlaceholder(%d) = false, want true", id)
		}
	}
	if tok.IsVisionPlaceholder(2) {
		t.Errorf("IsVisionPlaceholder(2 base token) = true, want false")
	}
}

// TestVisionPlaceholderEncodeSingleID proves the load-bearing property the serve seam
// depends on: a placeholder run Encodes to exactly one id per placeholder, never
// BPE-split, so an image is a contiguous run of pad ids to be replaced.
func TestVisionPlaceholderEncodeSingleID(t *testing.T) {
	tok, err := visionVocab(true)
	if err != nil {
		t.Fatalf("visionVocab: %v", err)
	}
	if ids, err := tok.Encode(ImagePadToken); err != nil || len(ids) != 1 || ids[0] != 6 {
		t.Fatalf("Encode(%q) = (%v,%v), want ([6],nil)", ImagePadToken, ids, err)
	}
	// The chat-template shape: <|vision_start|> then two <|image_pad|> then <|vision_end|>.
	prompt := VisionStartToken + ImagePadToken + ImagePadToken + VisionEndToken
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("Encode(prompt): %v", err)
	}
	want := []int{3, 6, 6, 4}
	if len(ids) != len(want) {
		t.Fatalf("Encode(prompt) = %v, want %v", ids, want)
	}
	for i, id := range ids {
		if id != want[i] {
			t.Fatalf("Encode(prompt) = %v, want %v", ids, want)
		}
	}
}

// TestVisionPlaceholderWrappedFallback proves that even when a GGUF conversion drops
// token_type, the isWrappedSpecial "<|...|>" fallback still registers the pads as
// single-id specials, so the accessors resolve them.
func TestVisionPlaceholderWrappedFallback(t *testing.T) {
	tok, err := visionVocab(false) // no token_type
	if err != nil {
		t.Fatalf("visionVocab: %v", err)
	}
	if id, ok := tok.ImagePadID(); !ok || id != 6 {
		t.Errorf("ImagePadID() via wrapped fallback = (%d,%v), want (6,true)", id, ok)
	}
	if ids, err := tok.Encode(ImagePadToken); err != nil || len(ids) != 1 || ids[0] != 6 {
		t.Errorf("Encode(%q) via wrapped fallback = (%v,%v), want ([6],nil)", ImagePadToken, ids, err)
	}
}

// TestVisionPlaceholderAbsent proves a text-only vocab reports no vision placeholders,
// so the seam can distinguish a VLM tokenizer from a plain one and never spuriously
// classifies a base id as a placeholder.
func TestVisionPlaceholderAbsent(t *testing.T) {
	tok, err := FromGGML([]string{"a", "b", "ab", "<|im_end|>"}, []string{"a b"}, nil, "qwen2")
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	for _, get := range []func() (int, bool){
		tok.VisionStartID, tok.VisionEndID, tok.VisionPadID, tok.ImagePadID, tok.VideoPadID,
	} {
		if id, ok := get(); ok {
			t.Errorf("text-only vocab resolved a vision placeholder to %d, want ok=false", id)
		}
	}
	for id := 0; id < tok.Vocab(); id++ {
		if tok.IsVisionPlaceholder(id) {
			t.Errorf("IsVisionPlaceholder(%d) = true in text-only vocab", id)
		}
	}
}

// TestVisionPlaceholdersInterface pins that *Tokenizer satisfies the capability the
// serve seam (#4032) depends on.
func TestVisionPlaceholdersInterface(t *testing.T) {
	var _ VisionPlaceholders = (*Tokenizer)(nil)
}
