package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// bigBase64 returns a base64-shaped blob of n characters — a stand-in for a real image's
// source.data. It is not valid base64 content-wise, but the estimators only ever measure its
// LENGTH, never decode it, so a repeated-char blob exercises the exact byte-weight path.
func bigBase64(n int) string { return strings.Repeat("A", n) }

// imageBlock builds one Anthropic image content block carrying a base64 source of the given size.
func imageBlock(dataLen int) map[string]any {
	return map[string]any{
		"type": "image",
		"source": map[string]any{
			"type":       "base64",
			"media_type": "image/png",
			"data":       bigBase64(dataLen),
		},
	}
}

// TestEstimateAnthropicTokensChargesImageRealCost pins the decoded-estimator half of the image
// accounting fix: an image block decodes to the "[image]" placeholder (~2 tokens), so before the
// fix an image-heavy request reported near-zero input tokens — telling any client that trusts
// count_tokens the window was empty right up to a real overflow. The estimate must now be at least
// imageTokenCost per image above the tiny text floor.
func TestEstimateAnthropicTokensChargesImageRealCost(t *testing.T) {
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"messages": []map[string]any{
			{"role": "user", "content": []map[string]any{
				{"type": "text", "text": "look at this"},
				imageBlock(50000), // a ~real screenshot's base64
			}},
		},
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := EstimateAnthropicTokens(req)
	if got < imageTokenCost {
		t.Fatalf("image request estimated at %d tokens; must be >= imageTokenCost (%d) — the image is not being charged", got, imageTokenCost)
	}
	// And the base64 length must NOT dominate: a 50k-char base64 would be ~12.5k tokens if weighed
	// raw. The decoded estimate weighs only the "[image]" placeholder + text, so the total is the
	// flat image charge plus a few text tokens — nowhere near 12.5k.
	if got > imageTokenCost+1000 {
		t.Fatalf("image request estimated at %d tokens; the base64 byte length is inflating it (want ~imageTokenCost=%d + small text)", got, imageTokenCost)
	}
}

// TestEstimateAnthropicTokensNoImagePlaceholderDoubleCount pins the #5166 fix: the decoder folds
// an image block to the "[image]" placeholder that the text walk already sums into chars, so before
// the fix imageTokenCost was charged ON TOP of the placeholder's ~len("[image]")/4 tokens — a small
// double-count per image. A message that is ONLY an image (no text) must therefore estimate to
// EXACTLY imageTokenCost: the placeholder now nets zero.
func TestEstimateAnthropicTokensNoImagePlaceholderDoubleCount(t *testing.T) {
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"messages": []map[string]any{
			{"role": "user", "content": []map[string]any{imageBlock(50000)}},
		},
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := EstimateAnthropicTokens(req)
	if got != imageTokenCost {
		t.Fatalf("image-only request estimated at %d tokens; want exactly imageTokenCost (%d) — the [image] placeholder is being double-counted on top of imageTokenCost", got, imageTokenCost)
	}
}

// TestEstimateElementTokensImageNotByteWeighted pins the byte-level half: an image element's
// giant base64 must collapse to the flat per-image charge, not len(rawJSON)/4. A 50k-char base64
// element is ~12.5k tokens by the old len/4 rule; the fix charges imageTokenCost (~1600) instead.
func TestEstimateElementTokensImageNotByteWeighted(t *testing.T) {
	el, err := json.Marshal(map[string]any{
		"role":    "user",
		"content": []map[string]any{imageBlock(50000)},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	raw := len(el) / 4
	got := estimateElementTokens(json.RawMessage(el))
	if got >= raw {
		t.Fatalf("estimateElementTokens=%d did not shrink the raw byte weight %d — the base64 is still inflating the estimate", got, raw)
	}
	if got < imageTokenCost || got > imageTokenCost+200 {
		t.Fatalf("estimateElementTokens=%d for a single-image turn; want ≈imageTokenCost (%d)", got, imageTokenCost)
	}
}

// TestEstimateElementTokensTextUnchanged is the no-regression guard: a text-only element must be
// byte-identical to the old len(el)/4, so every existing text-only compaction test is unaffected.
func TestEstimateElementTokensTextUnchanged(t *testing.T) {
	el, err := json.Marshal(map[string]any{
		"role":    "user",
		"content": []map[string]any{{"type": "text", "text": strings.Repeat("hello world ", 100)}},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if got, want := estimateElementTokens(json.RawMessage(el)), len(el)/4; got != want {
		t.Fatalf("text-only element estimate changed: got %d, want len/4=%d", got, want)
	}
}

// TestCompactionDoesNotWedgeOnLargeTrailingImage is the end-to-end wedge guard. Before the fix a
// single oversized trailing image made chooseKeptWindow overflow the budget on the very first
// (last) message — leaving an empty kept window and a permanent CompactReasonWindowNoDrop no-op,
// so the session sailed past the ceiling and never shed. With the image charged its flat cost, the
// last image fits under a realistic budget, the window walk keeps recent turns, and the old sprawled
// middle sheds normally (CompactReasonNone).
func TestCompactionDoesNotWedgeOnLargeTrailingImage(t *testing.T) {
	type block map[string]any
	// A realistic head-cached body: system with a breakpoint, a first user turn with a breakpoint,
	// a long text middle, then a final user turn carrying a big image.
	msgs := []map[string]any{
		{"role": "user", "content": []block{
			{"type": "text", "text": strings.Repeat("early cached context. ", 20), "cache_control": map[string]any{"type": "ephemeral"}},
		}},
	}
	// A sprawled middle worth shedding.
	for i := 1; i < 20; i++ {
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{
			"role":    role,
			"content": []block{{"type": "text", "text": strings.Repeat("middle turn text ", 60) + itoa(i)}},
		})
	}
	// The final user turn: a big image (base64 ~40k chars) plus a bit of text.
	msgs = append(msgs, map[string]any{
		"role": "user",
		"content": []block{
			{"type": "text", "text": "here is a screenshot"},
			imageBlock(40000),
		},
	})
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"system": []block{
			{"type": "text", "text": strings.Repeat("policy. ", 40), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": msgs,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	// Budget chosen so the sprawled middle clearly exceeds it but the trailing image (charged its
	// flat cost) plus a couple of recent turns fits — the shape that used to wedge.
	budget := 4000
	out, outcome := CompactAnthropicHistoryWithOutcome(raw, budget)

	// It must FIRE (shed the middle), not wedge into WindowNoDrop / identity.
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("compaction did not fire on a large-trailing-image session: reason=%q (want a clean fire; the trailing image must not wedge the window walk)", outcome.Reason)
	}
	if outcome.Dropped <= 0 {
		t.Fatalf("compaction fired but dropped %d turns; the sprawled middle must shed", outcome.Dropped)
	}
	// The fired body must still decode and must be strictly smaller than the input (real shrink).
	if len(out) >= len(raw) {
		t.Fatalf("compacted body (%d bytes) not smaller than input (%d)", len(out), len(raw))
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("compacted body no longer decodes: %v", err)
	}
	// The image must SURVIVE verbatim in the kept window (it is the most recent turn) — a kept
	// element is copied byte-for-byte, never stubbed.
	if !json.Valid(out) {
		t.Fatalf("compacted body is not valid JSON")
	}
	if !strings.Contains(string(out), `"image"`) {
		t.Fatalf("the trailing image was dropped; a recent large item must be kept, not shed")
	}
}

// TestCompactionMintsRestoreHandleForDroppedOriginatingImage pins the media-restore fix: when the
// session's ORIGINATING turn is an image (not text) and compaction drops it, the outcome must carry
// a restore handle (RestoreID + RestoreBytes) so a resuming model can page the image back in — not
// vanish into a bare turn count the way it did before. A text originating turn already got this;
// a media one must too.
func TestCompactionMintsRestoreHandleForDroppedOriginatingImage(t *testing.T) {
	type block map[string]any
	// First turn: an image (the originating turn). Then a sprawled text middle. No breakpoint on
	// the image turn, a system breakpoint anchors the (empty) protected prefix so the whole array
	// including the image is compactible.
	msgs := []map[string]any{
		{"role": "user", "content": []block{
			{"type": "text", "text": "analyze this diagram"},
			imageBlock(30000),
		}},
	}
	for i := 1; i < 24; i++ {
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{
			"role":    role,
			"content": []block{{"type": "text", "text": strings.Repeat("later turn text ", 50) + itoa(i)}},
		})
	}
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"system": []block{
			{"type": "text", "text": strings.Repeat("policy. ", 40), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": msgs,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	_, outcome := CompactAnthropicHistoryWithOutcome(raw, 3000)
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("compaction did not fire: reason=%q", outcome.Reason)
	}
	if outcome.Dropped <= 0 {
		t.Fatalf("nothing dropped; the originating image turn should have been shed")
	}
	// The originating image turn was dropped → a restore handle must be minted for it.
	if outcome.RestoreID == "" {
		t.Fatalf("dropped originating IMAGE turn minted no RestoreID — the image is unrecoverable (the bug)")
	}
	if len(outcome.RestoreBytes) == 0 {
		t.Fatalf("RestoreID set but no RestoreBytes stashed — nothing to page back in")
	}
	// The restore bytes must be the original image turn (carry the image block).
	if !strings.Contains(string(outcome.RestoreBytes), `"image"`) {
		t.Fatalf("RestoreBytes do not carry the image block; wrong turn was content-addressed")
	}
	// A resuming model must be able to SEE the handle in the emitted stub.
	if !strings.Contains(outcome.RestoreExcerpt, "media turn") {
		t.Fatalf("restore excerpt %q does not mark the dropped turn as media", outcome.RestoreExcerpt)
	}
}
