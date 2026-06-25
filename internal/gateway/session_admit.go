package gateway

// session_admit.go — the PROXY-path enforcement of the session control surface. The
// control routes (#620) let an operator write a served session's DRIVE state; fak's
// OWN agent turn loop honors it at each turn boundary (agent.RunArm + WithSessionTable).
// On the PROXIED serve/guard path, though, each /v1/{chat,messages,generateContent}
// request is a single upstream round-trip driven by an EXTERNAL agent — there is no
// in-process turn loop to gate. The natural analog is ADMISSION: when an operator has
// paused, drained, or stopped a session, the gateway refuses that session's next
// request instead of forwarding it upstream. That is what makes "cancel a request in
// flight" real on the flagship path — the operator POSTs draining/stopped and the
// agent's subsequent calls are refused at the boundary, cleanly, with the reason.
//
// HONEST SCOPE. This refuses the NEXT request for the session; an already-open upstream
// round-trip rides its own request context (the operator's stop takes effect at the
// next call boundary, never mid-stream — the same boundary discipline the design owns).
// It keys on the request TraceID, so an operator can only target a session whose agent
// sends a stable X-Trace-Id (a minted per-request gw-<n> is, by construction, not
// externally addressable). THROTTLED is admitted (pace shapes fak's own loop, not proxy
// admission); only the non-advancing states refuse.

import (
	"context"
	"net/http"
)

// sessionAdmits reports whether a proxied request for trace may proceed under the
// operator-controlled DRIVE state. Fail-OPEN: with no observeSession wired (the route
// disabled) or an empty trace, and for the advancing states (running/throttled/unknown),
// it returns true and the request path is byte-for-byte the pre-control behavior. It
// returns false only for the operator-set non-advancing states (paused/draining/stopped),
// carrying the state so the refusal can name why.
func (s *Server) sessionAdmits(ctx context.Context, trace string) (bool, SessionState) {
	if s.observeSession == nil || trace == "" {
		return true, SessionState{}
	}
	st := s.observeSession(ctx, trace)
	switch st.Run {
	case "paused", "draining", "stopped":
		return false, st
	default:
		return true, st
	}
}

// writeSessionRefusal emits the 409 a proxied request gets when an operator has held or
// stopped its session. 409 Conflict (not 503): the request is well-formed; the session
// STATE refuses it — the same status the control routes return for a terminal/stale-CAS
// write. The error code is "session_<state>" so a client can branch on it, and the
// operator's reason token (if any) rides the message.
func writeSessionRefusal(w http.ResponseWriter, st SessionState) {
	msg := "session " + st.TraceID + " is " + st.Run + " (operator control); request refused"
	if st.Reason != "" {
		msg += ": " + st.Reason
	}
	writeErrCode(w, http.StatusConflict, "session_"+st.Run, msg)
}
