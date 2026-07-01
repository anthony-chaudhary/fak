package contextq

import (
	"net/url"
	"strings"
)

const (
	MCPMissingContextSchema    = "fak-mcp-missing-context-resource/1"
	MCPMissingContextURIPrefix = "fak://context/missing/"
)

type MCPMissingContextRequest struct {
	Method      string `json:"method"`
	URI         string `json:"uri"`
	Key         string `json:"key"`
	Reason      string `json:"reason"`
	Audited     bool   `json:"audited"`
	BudgetBytes int64  `json:"budget_bytes,omitempty"`
}

func MCPMissingContextResourceRequest(uri string, budgetBytes int64) (MCPMissingContextRequest, bool) {
	uri = strings.TrimSpace(uri)
	if !strings.HasPrefix(uri, MCPMissingContextURIPrefix) {
		return MCPMissingContextRequest{}, false
	}
	key := strings.TrimPrefix(uri, MCPMissingContextURIPrefix)
	if decoded, err := url.PathUnescape(key); err == nil {
		key = decoded
	}
	key = strings.Trim(strings.TrimSpace(key), "/")
	if key == "" {
		return MCPMissingContextRequest{}, false
	}
	return MCPMissingContextRequest{
		Method:      "resources/read",
		URI:         uri,
		Key:         key,
		Reason:      "missing_context",
		Audited:     true,
		BudgetBytes: budgetBytes,
	}, true
}
