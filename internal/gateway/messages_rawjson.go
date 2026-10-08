package gateway

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func injectAnthropicUserTextRaw(raw []byte, text string) ([]byte, bool) {
	text = strings.TrimSpace(text)
	if len(raw) == 0 || text == "" {
		return nil, false
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil, false
	}
	messagesRaw, ok := obj["messages"]
	messagesTrimmed := bytes.TrimSpace(messagesRaw)
	if !ok || len(messagesTrimmed) == 0 || messagesTrimmed[0] != '[' {
		return nil, false
	}
	var elems []json.RawMessage
	if json.Unmarshal(messagesRaw, &elems) != nil {
		return nil, false
	}
	if raw, ok := mergeLastAnthropicUserTextRaw(raw, messagesRaw, elems, text); ok {
		return raw, true
	}
	return appendAnthropicUserTextRaw(raw, text)
}

func appendAnthropicUserTextRaw(raw []byte, text string) ([]byte, bool) {
	text = strings.TrimSpace(text)
	if len(raw) == 0 || text == "" {
		return nil, false
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil, false
	}
	messagesRaw, ok := obj["messages"]
	messagesTrimmed := bytes.TrimSpace(messagesRaw)
	if !ok || len(messagesTrimmed) == 0 || messagesTrimmed[0] != '[' {
		return nil, false
	}
	var elems []json.RawMessage
	if json.Unmarshal(messagesRaw, &elems) != nil {
		return nil, false
	}
	base := bytes.Index(raw, messagesRaw)
	if base < 0 || len(messagesRaw) < 2 {
		return nil, false
	}
	closeIdx := bytes.LastIndexByte(messagesRaw, ']')
	if closeIdx < 0 {
		return nil, false
	}
	insert := base + closeIdx // before the closing ']'
	msg, err := json.Marshal(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{Role: "user", Content: text})
	if err != nil {
		return nil, false
	}
	var out bytes.Buffer
	out.Grow(len(raw) + len(msg) + 1)
	out.Write(raw[:insert])
	if len(elems) > 0 {
		out.WriteByte(',')
	}
	out.Write(msg)
	out.Write(raw[insert:])
	b := out.Bytes()
	if _, err := agent.DecodeAnthropicMessagesRequest(b); err != nil {
		return nil, false
	}
	return b, true
}

func mergeLastAnthropicUserTextRaw(raw, messagesRaw []byte, elems []json.RawMessage, text string) ([]byte, bool) {
	messagesBase := bytes.Index(raw, messagesRaw)
	if messagesBase < 0 {
		return nil, false
	}
	for i := len(elems) - 1; i >= 0; i-- {
		var msg struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(elems[i], &msg) != nil || msg.Role != "user" {
			continue
		}
		elemOffset := bytes.LastIndex(messagesRaw, elems[i])
		if elemOffset < 0 {
			continue
		}
		contentOffset := bytes.Index(elems[i], msg.Content)
		if contentOffset < 0 {
			continue
		}
		start := messagesBase + elemOffset + contentOffset
		end := start + len(msg.Content)
		if existing, ok := asRawJSONString(msg.Content); ok && strings.TrimSpace(existing) != "" {
			merged := existing + "\n\n" + text
			repl, err := json.Marshal(merged)
			if err != nil {
				return nil, false
			}
			return replaceAnthropicRawRange(raw, start, end, repl)
		}
		var blocks []map[string]json.RawMessage
		if json.Unmarshal(msg.Content, &blocks) != nil {
			continue
		}
		for j := len(blocks) - 1; j >= 0; j-- {
			var typ string
			if json.Unmarshal(blocks[j]["type"], &typ) != nil || typ != "text" {
				continue
			}
			var existing string
			if json.Unmarshal(blocks[j]["text"], &existing) != nil || strings.TrimSpace(existing) == "" {
				continue
			}
			merged, err := json.Marshal(existing + "\n\n" + text)
			if err != nil {
				return nil, false
			}
			blocks[j]["text"] = merged
			content, err := json.Marshal(blocks)
			if err != nil {
				return nil, false
			}
			return replaceAnthropicRawRange(raw, start, end, content)
		}
		blocks = append(blocks, map[string]json.RawMessage{
			"type": json.RawMessage(`"text"`),
			"text": json.RawMessage(mustMarshalJSONString(text)),
		})
		content, err := json.Marshal(blocks)
		if err != nil {
			return nil, false
		}
		return replaceAnthropicRawRange(raw, start, end, content)
	}
	return nil, false
}

func asRawJSONString(raw json.RawMessage) (string, bool) {
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return "", false
	}
	return s, true
}

func mustMarshalJSONString(s string) []byte {
	raw, err := json.Marshal(s)
	if err != nil {
		return []byte(`""`)
	}
	return raw
}

func replaceAnthropicRawRange(raw []byte, start, end int, repl []byte) ([]byte, bool) {
	if start < 0 || end < start || end > len(raw) {
		return nil, false
	}
	var out bytes.Buffer
	out.Grow(len(raw) + len(repl) - (end - start))
	out.Write(raw[:start])
	out.Write(repl)
	out.Write(raw[end:])
	b := out.Bytes()
	if _, err := agent.DecodeAnthropicMessagesRequest(b); err != nil {
		return nil, false
	}
	return b, true
}

// spliceMaxTokens replaces the integer value of the top-level "max_tokens" key in an Anthropic
// /v1/messages body with cap, by a byte splice that touches ONLY that number — every other byte
// (and so the cache_control prefix) is preserved verbatim. It returns ok=false (caller leaves
// req.Raw unchanged) when the key is absent, the value is not a bare integer, or the splice
// would not re-decode to a valid request — fail-safe identity, never a cache-busting rewrite.
func spliceMaxTokens(raw []byte, cap int) ([]byte, bool) {
	// Locate the "max_tokens" key, then the JSON number that follows its colon. We scan for the
	// key bytes; a false match inside a string value is caught by the re-decode + value check.
	key := []byte(`"max_tokens"`)
	ki := bytes.Index(raw, key)
	if ki < 0 {
		return nil, false
	}
	i := ki + len(key)
	// Skip whitespace and the single ':' separator.
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	if i >= len(raw) || raw[i] != ':' {
		return nil, false
	}
	i++
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	// The value must be a bare JSON integer (digits, optional leading '-').
	start := i
	if i < len(raw) && raw[i] == '-' {
		i++
	}
	digitsStart := i
	for i < len(raw) && raw[i] >= '0' && raw[i] <= '9' {
		i++
	}
	if i == digitsStart { // no digits → not an integer value (e.g. it was a string) — bail
		return nil, false
	}
	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(raw[:start])
	b.WriteString(itoa(uint64(cap)))
	b.Write(raw[i:])
	out := b.Bytes()
	// Prove the splice produced a valid request before trusting it.
	if _, err := agent.DecodeAnthropicMessagesRequest(out); err != nil {
		return nil, false
	}
	return out, true
}
