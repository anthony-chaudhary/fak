package systools

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The closed Code set for systools.
const (
	CodeMalformed   = "MALFORMED"
	CodeDefaultDeny = "DEFAULT_DENY"
	CodePolicyBlock = "POLICY_BLOCK"
	CodeSSRFBlock   = "SSRF_BLOCK"
	CodeIO          = "IO_ERROR"
	CodeCanceled    = "CANCELED"
)

// Refusal represents a typed refusal or error.
type Refusal struct {
	Code   string         `json:"code"`
	Reason abi.ReasonCode `json:"-"`
	Detail string         `json:"detail,omitempty"`
	Tool   string         `json:"tool,omitempty"`
}

func (r *Refusal) Error() string {
	if r == nil {
		return ""
	}
	if r.Detail == "" {
		return r.Code
	}
	return r.Code + ": " + r.Detail
}

func (r *Refusal) JSON() []byte {
	b, err := json.Marshal(map[string]any{"error": r})
	if err != nil {
		return []byte(`{"error":{"code":"IO_ERROR"}}`)
	}
	return b
}

func refuse(code, detail string) *Refusal {
	return &Refusal{Code: code, Reason: reasonFor(code), Detail: detail}
}

func reasonFor(code string) abi.ReasonCode {
	switch code {
	case CodePolicyBlock, CodeSSRFBlock:
		return abi.ReasonPolicyBlock
	case CodeDefaultDeny:
		return abi.ReasonDefaultDeny
	case CodeMalformed:
		return abi.ReasonMalformed
	default:
		return abi.ReasonNone
	}
}
