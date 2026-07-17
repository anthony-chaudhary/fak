package agent

// anthropic_token_estimate.go — an image-aware ~token estimate for a raw messages[]
// element, shared by every byte-level compaction/elision site that used to size a message
// as a flat len(rawJSON)/4.
//
// The problem it fixes: the byte-level rewrites (anthropic_compact.go, _goalpin.go, _view.go)
// estimate a message's resident cost as len(elems[i])/4 over its VERBATIM JSON bytes. For a
// plain text turn that ~4-chars/token heuristic is fine. For an IMAGE block it is badly wrong:
// the element JSON still carries the whole base64 `source.data` blob, so a single ~1.5k-token
// image (a small screenshot) is counted as its base64 length / 4 — often TENS OF THOUSANDS of
// phantom tokens. That over-count drives the compaction budget comparison (suffixTokens > budget)
// and the kept-window walk (chooseKeptWindow) off a number that has nothing to do with what the
// provider actually bills, so an image-heavy session compacts on byte weight, not token pressure,
// and one oversized image can wedge the window walk. The DECODED estimator (EstimateAnthropicTokens)
// has the mirror-image bug: an image decodes to the literal "[image]" (~2 tokens), so it UNDER-counts
// the same image to near zero. Neither equals the provider's real per-image cost.
//
// The fix is one currency for an image on both sides: a flat imageTokenCost per image block,
// substituted for the base64 bytes the raw estimate would otherwise weigh. Text is unchanged, so
// every existing text-only test is byte-for-byte identical — only a body that actually carries an
// image block moves, and it moves toward the truth.

import "encoding/json"

// imageTokenCost is the flat ~token cost fak charges a single image content block, in the same
// ~4-chars/token currency as the budget and the provider input_tokens. It is a deliberately
// conservative constant, not a per-image geometric computation: fak never decodes the image, so
// it cannot know the pixel dimensions Anthropic's true cost formula (~(w*h)/750, capped ~1600)
// keys on. 1600 is the documented per-image ceiling — charging the ceiling means the estimate
// never UNDER-counts an image (the failure mode that lets a real overflow slip past the budget),
// while still collapsing a multi-tens-of-thousands base64 over-count back to a real-order number.
// The exactness does not need to be perfect: this feeds a ~4-chars/token budget heuristic that was
// already approximate for text; the goal is to stop an image being off by one to two orders of
// magnitude in EITHER direction, not to bill it to the token.
const imageTokenCost = 1600

// estimateElementTokens is the image-aware replacement for len(el)/4 at the byte-level rewrite
// sites. For a message with no image blocks it returns exactly len(el)/4 — byte-identical to the
// old estimate, so every text-only caller and test is unchanged. For a message carrying one or more
// image blocks it charges imageTokenCost per image and weighs only the NON-image bytes at ~4
// chars/token, so a base64 blob no longer inflates the estimate. On any parse ambiguity it falls
// back to the plain len(el)/4 (never worse than today), because an over-count is a soft budget
// error, never a correctness one.
func estimateElementTokens(el json.RawMessage) int {
	base := len(el) / 4
	imgs, imgBytes, ok := imageBlockByteWeight(el)
	if !ok || imgs == 0 {
		return base
	}
	// Replace the image blocks' raw byte weight (imgBytes/4, dominated by base64) with a flat
	// per-image charge. Floor at zero so a defensively large imgBytes can never make the estimate
	// negative.
	adjusted := base - imgBytes/4 + imgs*imageTokenCost
	if adjusted < 0 {
		adjusted = 0
	}
	return adjusted
}

// imageBlockByteWeight reports, for one messages[] element, how many image blocks it carries and
// the total raw JSON byte length of those image blocks (source.data base64 included). ok is false
// when the element is not a block-array message (a bare-string content, a tool-call-only assistant
// turn, or unparseable) — those carry no image blocks and take the plain len/4 path. It walks the
// SAME nesting the decoder does: a top-level image block, and an image block nested inside a
// tool_result's content array (the screenshot-from-a-tool shape).
func imageBlockByteWeight(el json.RawMessage) (imgs, imgBytes int, ok bool) {
	var m struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil {
		return 0, 0, false
	}
	i, b := contentImageWeight(m.Content)
	return i, b, true
}

// contentImageWeight sums the image-block count and raw byte weight in a `content` value (a message
// content array or a nested tool_result content array). A bare-string content has no blocks. It
// recurses one level into a tool_result block's own content so an image returned BY a tool is
// weighed like a top-level image, matching parseAnthropicText's recursion.
func contentImageWeight(content json.RawMessage) (imgs, imgBytes int) {
	var blocks []json.RawMessage
	if json.Unmarshal(content, &blocks) != nil {
		return 0, 0
	}
	for _, blk := range blocks {
		var b struct {
			Type    string          `json:"type"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(blk, &b) != nil {
			continue
		}
		switch b.Type {
		case "image":
			imgs++
			imgBytes += len(blk)
		case "tool_result":
			// A tool_result can itself carry image blocks (a screenshot tool). Weigh those the
			// same way; its text blocks stay on the plain byte path (not subtracted here).
			ni, nb := contentImageWeight(b.Content)
			imgs += ni
			imgBytes += nb
		}
	}
	return imgs, imgBytes
}
