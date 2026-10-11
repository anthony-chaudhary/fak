package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
)

// bindUnkeyedWireConversation uses the already-bounded body retained by the
// Anthropic or Gemini handler, before routing, normalization or admission. Their
// canonical transcripts can discard image bytes or flatten parts, so identity
// retains the original wire system, tools and prefix through the first wire-user
// turn. No raw payload is rewritten or retained in the session table.
//
// Object key order and whitespace do not affect the tuple. UseNumber preserves
// exact numeric values without float64 collisions or unbounded big-number work;
// different numeric spellings may conservatively start new accounting. Streaming,
// sampling, token limits, stop settings and other top-level generation options
// are excluded. Wire namespaces are distinct and the OpenAI key is unchanged.
//
// As with OpenAI, model/tools/system/root changes or LRU eviction reset budgets.
// Identical anonymous tuples share accounting; this is not agent identity or an
// abuse-resistant quota. Missing recognized user turns retain connection fallback.
func (s *Server) bindUnkeyedWireConversation(w http.ResponseWriter, r *http.Request, wire, model string, raw []byte) {
	if s == nil || r == nil || s.DefaultTraceID() != UnkeyedPerConnectionTraceID ||
		r.Context().Value(unkeyedTraceGeneratedKey{}) != true {
		return
	}
	var request struct {
		System            any              `json:"system"`
		SystemInstruction any              `json:"systemInstruction"`
		Tools             any              `json:"tools"`
		Messages          []map[string]any `json:"messages"`
		Contents          []map[string]any `json:"contents"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if decoder.Decode(&request) != nil {
		return // the existing adapter remains the sole validation/error owner
	}
	var system any
	var messages []map[string]any
	switch wire {
	case "anthropic":
		system, messages = request.System, request.Messages
	case "gemini":
		system, messages = request.SystemInstruction, request.Contents
	default:
		return
	}
	end := 0
	for i, message := range messages {
		role, _ := message["role"].(string)
		payload := message["content"]
		if wire == "gemini" {
			payload = message["parts"]
		}
		nonempty := false
		switch content := payload.(type) {
		case string:
			nonempty = strings.TrimSpace(content) != ""
		case []any:
			nonempty = len(content) > 0
		}
		if nonempty && (role == "user" || wire == "gemini" && (role == "" || strings.EqualFold(role, "user"))) {
			end = i + 1
			break
		}
	}
	if end == 0 {
		return
	}
	if model == "" {
		model = s.model
	}
	identity := struct {
		Version   string           `json:"version"`
		Wire      string           `json:"wire"`
		Principal string           `json:"principal"`
		Model     string           `json:"model"`
		System    any              `json:"system"`
		Tools     any              `json:"tools"`
		Root      []map[string]any `json:"root"`
	}{"unkeyed-wire-v1", wire, principalFromContext(r.Context()), model, system, request.Tools, messages[:end]}
	encoded, err := json.Marshal(identity)
	if err != nil {
		return
	}
	sum := sha256.Sum256(encoded)
	s.useHTTPTrace(w, r, UnkeyedTracePrefix+wire+"-"+hex.EncodeToString(sum[:]))
}
