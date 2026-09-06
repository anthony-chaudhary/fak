package mcpbroker

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// compactStructuredContent removes only insignificant JSON whitespace from a
// server-declared structured text mirror. It provides semantic JSON fidelity,
// not byte-exact recovery of the original formatting.
func compactStructuredContent(result, content json.RawMessage) json.RawMessage {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_COMPRESSOR"))) {
	case "noop", "none":
		return content
	}
	if len(content) < 48 {
		return content
	}
	fields, ok := compressionObjectFields(result)
	if !ok || !bytes.Equal(fields["content"].raw, content) {
		return content
	}
	for key := range fields {
		for _, reserved := range []string{"content", "structuredContent", "isError"} {
			if key != reserved && strings.EqualFold(key, reserved) {
				return content
			}
		}
	}
	if flag, exists := fields["isError"]; exists && string(flag.raw) != "false" {
		return content
	}
	structured := fields["structuredContent"].raw
	if len(structured) == 0 || structured[0] != '{' {
		return content
	}
	var blocks []json.RawMessage
	if json.Unmarshal(content, &blocks) != nil || len(blocks) != 1 {
		return content
	}
	block := blocks[0]
	parts, ok := compressionObjectFields(block)
	if !ok {
		return content
	}
	var kind, original string
	if json.Unmarshal(parts["type"].raw, &kind) != nil || kind != "text" || json.Unmarshal(parts["text"].raw, &original) != nil {
		return content
	}
	// Screen the original decoded payload: normalization must not hide anything
	// that the existing security admission screen would flag.
	if _, held := ctxmmu.ScreenBytes([]byte(original)); held {
		return content
	}
	var compact, declared bytes.Buffer
	if json.Compact(&compact, []byte(original)) != nil || compact.Len() == 0 || compact.Bytes()[0] != '{' || json.Compact(&declared, structured) != nil || !bytes.Equal(compact.Bytes(), declared.Bytes()) {
		return content
	}
	encoded, err := json.Marshal(compact.String())
	if err != nil {
		return content
	}
	// The raw block occurs exactly once: content is a single-element JSON array.
	start := bytes.Index(content, block) + parts["text"].start
	end := start + len(parts["text"].raw)
	newLen := len(content) - (end - start) + len(encoded)
	saved := len(content) - newLen
	if saved <= 0 || (saved < 256 && float64(saved)/float64(len(content)) < 0.15) {
		return content
	}
	out := make([]byte, 0, newLen)
	out = append(out, content[:start]...)
	out = append(out, encoded...)
	return append(out, content[end:]...)
}

type compressionField struct {
	raw   json.RawMessage
	start int
}

// Decode only the envelope keys and retain raw value spans. Ambiguous envelope
// keys are ineligible; duplicate keys inside structured values remain untouched.
func compressionObjectFields(raw []byte) (map[string]compressionField, bool) {
	if !json.Valid(raw) {
		return nil, false
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	first, err := d.Token()
	if err != nil || first != json.Delim('{') {
		return nil, false
	}
	fields := make(map[string]compressionField)
	seen := make(map[string]bool)
	for d.More() {
		token, err := d.Token()
		key, ok := token.(string)
		if err != nil || !ok || seen[strings.ToLower(key)] {
			return nil, false
		}
		seen[strings.ToLower(key)] = true
		var value json.RawMessage
		if d.Decode(&value) != nil {
			return nil, false
		}
		fields[key] = compressionField{raw: value, start: int(d.InputOffset()) - len(value)}
	}
	return fields, true
}
