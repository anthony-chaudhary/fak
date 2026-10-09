package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
)

// UnkeyedPerConnectionTraceID is a reserved DefaultTraceID: instead of charging
// every header-less request to one shared session, the gateway derives one
// session per client connection, so one agent exhausting a budget cannot 409
// every other header-less client on the endpoint (fak-private#930).
const UnkeyedPerConnectionTraceID = "unkeyed:per-connection"

// UnkeyedTracePrefix prefixes every per-connection identity the gateway derives
// under UnkeyedPerConnectionTraceID.
const UnkeyedTracePrefix = "unkeyed-"

// IsUnkeyedTrace reports whether trace is a gateway-derived per-connection
// identity for a caller that sent no trace header.
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
