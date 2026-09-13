package model

import (
	"fmt"
	"strings"
)

// V4.1 text prompt encoding is pinned to
// deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277
// (inference/model.py and its sibling encoding/ directory).
//
// The tokenizer is byte-level BPE with a merge-rank table, matching the
// reference path. This file owns only encoding: it loads the pinned vocabulary
// and merge ranks, renders the chat/tool template, and encodes text fail-closed
// against the pinned vocabulary. It deliberately does NOT alias or reuse
// V4-Flash-0731 tokenization; that artifact is a separately-scoped control.

// DeepSeekV41TokenizerRevision pins the tokenizer to the published artifact.
const DeepSeekV41TokenizerRevision = DeepSeekV41FlashRevision

// V4.1 reference special-token spellings. A vocabulary that does not contain
// them fails admission rather than silently dropping them.
var DeepSeekV41SpecialTokens = []string{
	"<\uff5cbegin\u2581of\u2581sentence\uff5c>",
	"<\uff5cend\u2581of\u2581sentence\uff5c>",
	"\uff5cUser\uff5c>",
	"\uff5cAssistant\uff5c>",
	"\uff5cSystem\uff5c>",
	"<\uff5cTool\uff5c>",
	"<\uff5c/Tool\uff5c>",
}

// DeepSeekV41Tokenizer is a byte-level BPE encoder bound to an immutable
// vocabulary and merge-rank table loaded from the pinned revision. The zero
// value is invalid; construct with NewDeepSeekV41Tokenizer.
type DeepSeekV41Tokenizer struct {
	vocab      map[string]int
	ids        []string
	ranks      map[string]int
	byteToRune []rune
	runeToByte map[rune]byte
	special    map[string]int
	padID      int
	bosID      int
	eosID      int
}

// DeepSeekV41TokenizerSpec is the caller-supplied pinned artifact: the ordered
// token list (index == token id) and the merge-rank table. Both come from the
// pinned encoding/ directory.
type DeepSeekV41TokenizerSpec struct {
	Tokens []string
	Merges []string
}

// NewDeepSeekV41Tokenizer validates and freezes a pinned tokenizer. It fails
// closed on missing special tokens, duplicate ids, or malformed merges.
func NewDeepSeekV41Tokenizer(spec DeepSeekV41TokenizerSpec) (*DeepSeekV41Tokenizer, error) {
	if len(spec.Tokens) == 0 {
		return nil, fmt.Errorf("model: V4.1 tokenizer vocabulary is empty")
	}
	t := &DeepSeekV41Tokenizer{
		vocab:      make(map[string]int, len(spec.Tokens)),
		ids:        append([]string(nil), spec.Tokens...),
		ranks:      make(map[string]int, len(spec.Merges)),
		byteToRune: make([]rune, 256),
		runeToByte: make(map[rune]byte, 256),
		special:    make(map[string]int, len(DeepSeekV41SpecialTokens)),
		padID:      -1,
		bosID:      -1,
		eosID:      -1,
	}
	for id, tok := range spec.Tokens {
		if _, dup := t.vocab[tok]; dup {
			return nil, fmt.Errorf("model: V4.1 tokenizer duplicate token %q at id %d", tok, id)
		}
		t.vocab[tok] = id
	}
	for rank, merge := range spec.Merges {
		parts := strings.Fields(merge)
		if len(parts) != 2 {
			return nil, fmt.Errorf("model: V4.1 tokenizer malformed merge %q at rank %d", merge, rank)
		}
		key := parts[0] + " " + parts[1]
		if _, dup := t.ranks[key]; dup {
			return nil, fmt.Errorf("model: V4.1 tokenizer duplicate merge %q", key)
		}
		t.ranks[key] = rank
	}
	for _, tok := range DeepSeekV41SpecialTokens {
		id, ok := t.vocab[tok]
		if !ok {
			return nil, fmt.Errorf("model: V4.1 tokenizer missing pinned special token %q", tok)
		}
		t.special[tok] = id
	}
	t.buildByteAlphabet()
	t.bosID = t.special[DeepSeekV41SpecialTokens[0]]
	t.eosID = t.special[DeepSeekV41SpecialTokens[1]]
	t.padID = t.eosID
	return t, nil
}

// buildByteAlphabet maps the 256 raw bytes to the printable rune alphabet used
// by byte-level BPE.
func (t *DeepSeekV41Tokenizer) buildByteAlphabet() {
	used := make([]bool, 256)
	for b := 0x21; b <= 0x7e; b++ {
		t.byteToRune[b] = rune(b)
		used[b] = true
	}
	for b := 0xa1; b <= 0xac; b++ {
		t.byteToRune[b] = rune(b)
		used[b] = true
	}
	for b := 0xae; b <= 0xff; b++ {
		t.byteToRune[b] = rune(b)
		used[b] = true
	}
	next := rune(0x100)
	for b := 0; b < 256; b++ {
		if !used[b] {
			t.byteToRune[b] = next
			next++
		}
	}
	for b := 0; b < 256; b++ {
		t.runeToByte[t.byteToRune[b]] = byte(b)
	}
}

// VocabSize reports the pinned vocabulary size.
func (t *DeepSeekV41Tokenizer) VocabSize() int {
	if t == nil {
		return 0
	}
	return len(t.ids)
}

// SpecialID returns the pinned id for a special token, or -1 if the spelling is
// not a pinned special token.
func (t *DeepSeekV41Tokenizer) SpecialID(tok string) int {
	if t == nil {
		return -1
	}
	if id, ok := t.special[tok]; ok {
		return id
	}
	return -1
}

// encodeChunk applies byte-level BPE to one whitespace-delimited pre-token and
// fails closed if any produced symbol is absent from the pinned vocabulary.
func (t *DeepSeekV41Tokenizer) encodeChunk(chunk string) ([]int, error) {
	parts := make([]string, 0, len(chunk))
	for _, b := range []byte(chunk) {
		parts = append(parts, string(t.byteToRune[b]))
	}
	for len(parts) > 1 {
		bestI, bestRank := -1, -1
		for i := 0; i+1 < len(parts); i++ {
			if rank, ok := t.ranks[parts[i]+" "+parts[i+1]]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank, bestI = rank, i
				}
			}
		}
		if bestI < 0 {
			break
		}
		parts[bestI] = parts[bestI] + parts[bestI+1]
		parts = append(parts[:bestI+1], parts[bestI+2:]...)
	}
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		id, ok := t.vocab[p]
		if !ok {
			return nil, fmt.Errorf("model: V4.1 tokenizer produced token %q absent from pinned vocabulary", p)
		}
		out = append(out, id)
	}
	return out, nil
}

// Encode splits text at pinned special tokens (emitted verbatim) and
// BPE-encodes the remaining spans. Unknown bytes that yield a symbol absent
// from the vocabulary fail closed with a typed error; there is no best-effort
// fallback mapping.
func (t *DeepSeekV41Tokenizer) Encode(text string) ([]int, error) {
	if t == nil {
		return nil, fmt.Errorf("model: nil V4.1 tokenizer")
	}
	if text == "" {
		return nil, nil
	}
	type span struct {
		text    string
		special bool
	}
	var spans []span
	rest := text
	for rest != "" {
		matched := ""
		for _, tok := range DeepSeekV41SpecialTokens {
			if strings.HasPrefix(rest, tok) && len(tok) > len(matched) {
				matched = tok
			}
		}
		if matched != "" {
			spans = append(spans, span{text: matched, special: true})
			rest = rest[len(matched):]
			continue
		}
		next := len(rest)
		for _, tok := range DeepSeekV41SpecialTokens {
			if i := strings.Index(rest, tok); i >= 0 && i < next {
				next = i
			}
		}
		spans = append(spans, span{text: rest[:next]})
		rest = rest[next:]
	}
	var out []int
	for _, sp := range spans {
		if sp.special {
			out = append(out, t.special[sp.text])
			continue
		}
		for _, chunk := range splitV41PreTokens(sp.text) {
			ids, err := t.encodeChunk(chunk)
			if err != nil {
				return nil, err
			}
			out = append(out, ids...)
		}
	}
	return out, nil
}

// splitV41PreTokens segments text on the reference whitespace rule: each run
// of non-whitespace is prefixed by the leading whitespace that precedes it.
func splitV41PreTokens(s string) []string {
	var out []string
	var cur strings.Builder
	inSpace := false
	for _, r := range s {
		isSpace := r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' || r == '\v'
		if isSpace && !inSpace && cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
		cur.WriteRune(r)
		inSpace = isSpace
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

// Decode maps ids back to raw bytes. An id outside the vocabulary fails closed.
func (t *DeepSeekV41Tokenizer) Decode(ids []int) (string, error) {
	if t == nil {
		return "", fmt.Errorf("model: nil V4.1 tokenizer")
	}
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.ids) {
			return "", fmt.Errorf("model: V4.1 tokenizer id %d out of range", id)
		}
		tok := t.ids[id]
		if _, ok := t.special[tok]; ok {
			continue
		}
		for _, r := range tok {
			b, ok := t.runeToByte[r]
			if !ok {
				return "", fmt.Errorf("model: V4.1 tokenizer token %q contains non-byte rune", tok)
			}
			sb.WriteByte(b)
		}
	}
	return sb.String(), nil
}

// DeepSeekV41Message is one chat turn for the reference template.
type DeepSeekV41Message struct {
	Role    string
	Content string
}

// FormatDeepSeekV41Prompt renders messages into the pinned chat/tool template.
func FormatDeepSeekV41Prompt(messages []DeepSeekV41Message, addAssistantPrefix bool) string {
	var sb strings.Builder
	sb.WriteString("<\uff5cbegin\u2581of\u2581sentence\uff5c>")
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case "system":
			sb.WriteString("\uff5cSystem\uff5c>")
			sb.WriteString(msg.Content)
		case "user":
			sb.WriteString("\uff5cUser\uff5c>")
			sb.WriteString(msg.Content)
		case "assistant":
			sb.WriteString("\uff5cAssistant\uff5c>")
			sb.WriteString(msg.Content)
			sb.WriteString("<\uff5cend\u2581of\u2581sentence\uff5c>")
		case "tool", "observation":
			sb.WriteString("<\uff5cTool\uff5c>")
			sb.WriteString(msg.Content)
			sb.WriteString("<\uff5c/Tool\uff5c>")
		default:
			sb.WriteString("<\uff5c" + msg.Role + "\uff5c>")
			sb.WriteString(msg.Content)
		}
	}
	if addAssistantPrefix {
		sb.WriteString("\uff5cAssistant\uff5c>")
	}
	return sb.String()
}
