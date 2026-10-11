package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// UnkeyedPerConnectionTraceID is a reserved DefaultTraceID: instead of charging
// every header-less request to one shared session, the gateway derives one
// session per client connection. Supported chat wires with a first user turn
// refine that fallback to conversation continuity after bounded body decoding.
// This is session accounting, not an authenticated agent identity (#13777).
const UnkeyedPerConnectionTraceID = "unkeyed:per-connection"

// UnkeyedTracePrefix prefixes every header-less session the gateway derives
// under UnkeyedPerConnectionTraceID.
const UnkeyedTracePrefix = "unkeyed-"

// IsUnkeyedTrace recognizes the derived-session namespace for budget seeding.
// The prefix alone is not proof of gateway provenance or caller identity.
func IsUnkeyedTrace(trace string) bool {
	return strings.HasPrefix(trace, UnkeyedTracePrefix) && len(trace) > len(UnkeyedTracePrefix)
}

// headerlessTrace resolves the trace for an HTTP request that named none: a
// per-connection identity under UnkeyedPerConnectionTraceID, else traceFor("").
func (s *Server) headerlessTrace(r *http.Request) string {
	if s.DefaultTraceID() == UnkeyedPerConnectionTraceID {
		if trace := unkeyedConnectionTrace(r); trace != "" {
			return trace
		}
	}
	return s.traceFor("")
}

// unkeyedConnectionTrace derives the per-connection identity from the peer
// address. The address is hashed so the trace id never echoes a client IP.
// It returns "" when the request carries no peer address.
func unkeyedConnectionTrace(r *http.Request) string {
	if r == nil || strings.TrimSpace(r.RemoteAddr) == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(r.RemoteAddr))
	return UnkeyedTracePrefix + hex.EncodeToString(sum[:8])
}

// Only metrics middleware can set this request-local marker when it generates
// a missing trace. An incoming header in the unkeyed namespace is not a marker.
type unkeyedTraceGeneratedKey struct{}

// bindUnkeyedChatConversation keeps an unchanged OpenAI conversation prefix on
// one budget across reconnections, and separates different prefixes sharing a
// proxy connection. Call after authentication and the existing bounded decode,
// before model/tool/message transformations or session admission.
//
// The root is the original message prefix through the first user turn. Changes
// to model, tools or root start new accounting, as does session-table eviction.
// Identical anonymous prefixes share accounting: this cannot identify an agent
// or prevent intentional budget resets. No-user requests retain the connection
// fallback. Other wires use their own versioned raw-prefix tuple; this OpenAI
// key and explicit/default traces remain unchanged.
func (s *Server) bindUnkeyedChatConversation(w http.ResponseWriter, r *http.Request, model string, tools []agent.ToolDef, messages []agent.Message) {
	if s == nil || r == nil || s.DefaultTraceID() != UnkeyedPerConnectionTraceID ||
		r.Context().Value(unkeyedTraceGeneratedKey{}) != true {
		return
	}
	end := 0
	for i, message := range messages {
		if message.Role == agent.RoleUser {
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
	// Use only the principal authenticated into context, never principalFor's
	// legacy header/body fallback. Structured encoding separates tuple fields.
	identity := struct {
		Version   string          `json:"version"`
		Principal string          `json:"principal"`
		Model     string          `json:"model"`
		Tools     []agent.ToolDef `json:"tools,omitempty"`
		Root      []agent.Message `json:"root"`
	}{"unkeyed-chat-v1", principalFromContext(r.Context()), model, tools, messages[:end]}
	encoded, err := json.Marshal(identity)
	if err != nil {
		return // no new body reader or alternate decode path
	}
	sum := sha256.Sum256(encoded)
	s.useHTTPTrace(w, r, UnkeyedTracePrefix+"conversation-"+hex.EncodeToString(sum[:]))
}
