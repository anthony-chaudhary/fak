package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	defaultResponsesElideThreshold = 1024
	elideRecentKeepMsgs            = 4
	responsesCASThreshold          = 32 << 10 // 32 KiB
)

// responsesElideThreshold returns the byte size threshold for eliding tool outputs on /v1/responses.
// Defaults to 1024 bytes, configurable via FAK_RESPONSES_ELIDE_THRESHOLD.
func (s *Server) responsesElideThreshold() int {
	if v := strings.TrimSpace(os.Getenv("FAK_RESPONSES_ELIDE_THRESHOLD")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultResponsesElideThreshold
}

// persistRestoreCAS writes large tool output bytes durably to the CAS directory.
func (s *Server) persistRestoreCAS(id string, body []byte) {
	persistRestoreCAS(strings.TrimPrefix(id, "sha256:"), body)
}

// maybeElideResponsesToolResults elides older oversized tool results for /v1/responses.
// It protects the active working set by keeping the most recent 4 tool result messages intact (elideRecentKeepMsgs = 4).
// Tool outputs exceeding the threshold are content-addressed via SHA-256 and stashed for recovery
// via fak_context_restore. Payloads >= 32 KiB are also persisted to durable CAS.
func (s *Server) maybeElideResponsesToolResults(trace string, messages []agent.Message, restoreToolName ...string) []agent.Message {
	if s == nil || len(messages) == 0 {
		return messages
	}
	toolName := "fak_context_restore"
	if len(restoreToolName) > 0 && strings.TrimSpace(restoreToolName[0]) != "" {
		toolName = strings.TrimSpace(restoreToolName[0])
	}
	trace = s.traceFor(trace)
	threshold := s.responsesElideThreshold()
	if threshold <= 0 {
		return messages
	}

	var toolIndices []int
	for i, m := range messages {
		if m.Role == agent.RoleTool {
			toolIndices = append(toolIndices, i)
		}
	}
	if len(toolIndices) <= elideRecentKeepMsgs {
		return messages
	}

	eligible := toolIndices[:len(toolIndices)-elideRecentKeepMsgs]
	var out []agent.Message
	for _, idx := range eligible {
		msg := messages[idx]
		if len(msg.Content) <= threshold || strings.HasPrefix(msg.Content, "...[fak: tool output elided") {
			continue
		}
		if out == nil {
			out = append([]agent.Message(nil), messages...)
		}
		body := []byte(msg.Content)
		sum := sha256.Sum256(body)
		digest := hex.EncodeToString(sum[:])
		excerpt := strings.TrimSpace(msg.Content)
		if len(excerpt) > 160 {
			excerpt = excerpt[:160]
		}
		s.stashRestore(trace, digest, excerpt, body)
		s.stashRestore(trace, "sha256:"+digest, excerpt, body)
		if len(body) >= responsesCASThreshold {
			s.persistRestoreCAS(digest, body)
		}
		out[idx].Content = fmt.Sprintf("...[fak: tool output elided (len=%d bytes); recover original via %s id=sha256:%s]...", len(body), toolName, digest)
	}
	if out != nil {
		return out
	}
	return messages
}
